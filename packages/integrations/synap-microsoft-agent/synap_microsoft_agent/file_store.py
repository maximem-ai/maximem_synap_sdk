"""``SynapAgentFileStore`` — the MAF harness file-memory surface, backed by Synap.

MAF's harness exposes a second, separate memory seam: ``AgentFileStore``, a
small async file interface behind two providers.

``FileMemoryProvider`` (``file_memory_store=``)
    Seven tools the model calls directly — ``file_memory_write``, ``_read``,
    ``_delete``, ``_ls``, ``_grep``, ``_replace``, ``_replace_lines``. MAF
    describes it as *session-scoped working memory*: somewhere to park plans,
    intermediate results, and downloaded data so they survive context
    compaction.

``FileAccessProvider`` (``file_access_store=``)
    The same interface, for the agent's general file access.

Backing it with Synap turns that working memory into something that outlives
the session and is searchable by meaning rather than by regex.

What maps cleanly, and what does not
------------------------------------

===================  =========================================================
Method               Behaviour
===================  =========================================================
``write``            ``memories.create``, plus a local write-through cache so
                     the file is readable back immediately.
``read``             Cache first. On a miss, the file name is treated as a
                     question and answered from Synap recall. ``MEMORY.md``
                     (configurable) is the synthesized recall document.
``file_exists``      ``read`` returning something.
``delete``           **Real.** ``ingestion_id`` -> ``status().memory_ids`` ->
                     ``memories.delete``. Verified against a live instance.
``search``           Regex over cached files, **plus** a semantic hit against
                     Synap attributed to the recall file.
``list_children``    Cache only — see below.
``create_directory`` No-op. There are no directories.
===================  =========================================================

``list_children`` is the honest gap
-----------------------------------

There is no list-memories-by-scope call in the Synap SDK — ``memories`` offers
create, batch_create, status, wait_for_completion, get, update,
create_from_file, and delete. So ``list_children`` can only report what this
process wrote.

That matters more here than it first looks. ``FileMemoryProvider`` builds its
own ``memories.md`` index from ``list_children`` and injects it into the prompt
each run, so on a cold process the agent's *listing* of its files is empty even
though the *content* is still in Synap and still reachable through ``read`` and
``search``. Files come back by name or by question, not by browsing.

Within one session — which is what ``FileMemoryProvider`` is scoped to — the
cache covers every file the agent wrote, so this is invisible. Across a
restart it is not, and the docs say so rather than letting it look like data
loss.

Internal bookkeeping stays local
--------------------------------

``FileMemoryProvider`` writes two kinds of file the model never sees: the
``memories.md`` index it rebuilds after every write, and a
``<name>_description.md`` sidecar per file. Those are MAF's bookkeeping, not
the user's memory, and sending them through an extraction pipeline would fill
the scope with noise about file names. They are kept in the cache and never
submitted. Real content still goes to Synap.

Wiring
------

    from agent_framework import create_harness_agent
    from synap_microsoft_agent import SynapAgentFileStore

    agent = create_harness_agent(
        client,
        file_memory_store=SynapAgentFileStore(sdk, user_id="alice"),
    )

``FileMemoryProvider(scope=...)`` decides which folder the files land in —
``None`` isolates per session, an explicit value groups across sessions. Line
it up with the scope this store reads and writes, or the tool surface and the
memory will disagree about whose files they are.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

from synap_microsoft_agent._harness_compat import require_harness

require_harness()

from agent_framework import (  # noqa: E402 — must follow the version guard
    AgentFileStore,
    FileSearchMatch,
    FileSearchResult,
    FileStoreEntry,
)

logger = logging.getLogger(__name__)

DEFAULT_RECALL_FILENAME = "MEMORY.md"
"""File name that maps to the synthesized Synap recall document."""

DEFAULT_CACHE_TTL_SECONDS = 3600
"""How long a written file stays readable from the local cache.

Longer than the deepagents backend's five minutes on purpose. There, the cache
only had to cover read-after-write inside a turn. Here it is also the only
thing ``list_children`` can see, so a short TTL would make the agent's own
files vanish from its listing part-way through a session.
"""

MAX_SEARCH_PATTERN_LENGTH = 256
"""Matches MAF's own cap so an over-long pattern fails the same way."""

_INDEX_FILE_NAME = "memories.md"
_DESCRIPTION_SUFFIX = "_description.md"


def _normalize(path: str, *, is_directory: bool = False) -> str:
    """Normalize a relative store path, matching MAF's own rules.

    MAF's ``_normalize_relative_path`` lives on a private module path (D5), so
    the rules are reimplemented here rather than imported: trim, convert
    backslashes, collapse separators, and reject rooted paths, drive letters,
    and ``.``/``..`` segments.

    Raises:
        ValueError: On an empty file path, a rooted or drive-rooted path, a
            traversal segment, or a file path ending in a separator.
    """
    if not path or not path.strip():
        if not is_directory:
            raise ValueError("A file path must not be empty or whitespace-only.")
        return ""

    stripped = path.strip()
    converted = stripped.replace("\\", "/")

    if not is_directory and converted.endswith("/"):
        raise ValueError(
            f"Invalid path: {path!r}. A file path must not end with a path separator."
        )

    normalized = converted.strip("/")
    if (
        stripped.startswith(("/", "\\"))
        or (len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":")
    ):
        raise ValueError(
            f"Invalid path: {path!r}. Paths must be relative and must not start "
            "with '/', '\\', or a drive root."
        )

    segments: List[str] = []
    for segment in normalized.split("/"):
        if not segment:
            continue
        if segment in (".", ".."):
            raise ValueError(
                f"Invalid path: {path!r}. Paths must not contain '.' or '..' segments."
            )
        segments.append(segment)

    result = "/".join(segments)
    if not is_directory and not result:
        raise ValueError(f"Invalid path: {path!r}. A file path must not be empty.")
    return result


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_internal(file_name: str) -> bool:
    """Is this MAF's own bookkeeping rather than the user's memory?"""
    lowered = _basename(file_name).lower()
    return lowered.endswith(_DESCRIPTION_SUFFIX) or lowered == _INDEX_FILE_NAME


class _FileCache:
    """Per-process, TTL'd record of what this store has written.

    Two jobs. It is the read-after-write guarantee — Synap ingestion is queued,
    so a file written a moment ago is not yet retrievable and the agent would
    read back nothing. And it is the only thing ``list_children`` can enumerate.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._enabled = ttl_seconds > 0
        self._entries: Dict[str, Tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def put(self, path: str, content: str) -> None:
        if not self._enabled:
            return
        async with self._lock:
            self._entries[path] = (time.monotonic(), content)

    async def get(self, path: str) -> Optional[str]:
        if not self._enabled:
            return None
        async with self._lock:
            entry = self._entries.get(path)
            if entry is None:
                return None
            written_at, content = entry
            if (time.monotonic() - written_at) > self._ttl:
                self._entries.pop(path, None)
                return None
            return content

    async def items(self) -> List[Tuple[str, str]]:
        if not self._enabled:
            return []
        async with self._lock:
            now = time.monotonic()
            live: List[Tuple[str, str]] = []
            for path, (written_at, content) in list(self._entries.items()):
                if (now - written_at) > self._ttl:
                    self._entries.pop(path, None)
                    continue
                live.append((path, content))
            return sorted(live)

    async def drop(self, path: str) -> bool:
        async with self._lock:
            return self._entries.pop(path, None) is not None


class SynapAgentFileStore(AgentFileStore):
    """An ``AgentFileStore`` backed by Synap memory.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        user_id: External user ID. At least one of ``user_id`` or
            ``customer_id`` is required.
        customer_id: External customer ID.
        recall_filename: Base name that maps to the synthesized recall
            document. Defaults to ``"MEMORY.md"``.
        max_results: ``max_results`` for every ``fetch``.
        mode: Retrieval mode for ``read``. ``"fast"`` — reads sit on the
            agent's path.
        search_mode: Retrieval mode for ``search``. ``"accurate"`` — a search
            is a deliberate question.
        precision_level: ``"high"`` (default) or ``"medium"``.
        document_type: ``document_type`` for writes.
        ingest_mode: ``"fast"`` (default) or ``"long-range"``.
        cache_ttl_seconds: Cache lifetime. Setting this to ``0`` disables the
            cache and with it both read-after-write and ``list_children``;
            do not, unless you have measured that you can live without them.

    Raises:
        ValueError: If neither ``user_id`` nor ``customer_id`` is given.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        recall_filename: str = DEFAULT_RECALL_FILENAME,
        max_results: int = 20,
        mode: str = "fast",
        search_mode: str = "accurate",
        precision_level: str = "high",
        document_type: str = "document",
        ingest_mode: str = "fast",
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapAgentFileStore requires a non-None sdk")
        if not user_id and not customer_id:
            raise ValueError(
                "SynapAgentFileStore requires at least one of user_id or "
                "customer_id — a store with no scope cannot read or write memory"
            )
        if not recall_filename or not recall_filename.strip():
            raise ValueError("SynapAgentFileStore requires a non-empty recall_filename")

        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.recall_filename = recall_filename.strip()
        self.max_results = max_results
        self.mode = mode
        self.search_mode = search_mode
        self.precision_level = precision_level
        self.document_type = document_type
        self.ingest_mode = ingest_mode
        self._cache = _FileCache(cache_ttl_seconds)
        # path -> ingestion ids. `create` returns an ingestion id and `delete`
        # needs a memory id; `status(ingestion_id).memory_ids` is the bridge.
        # Held separately from the cache because it must outlive the TTL — a
        # file can be deletable long after its content has aged out.
        self._ingestions: Dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_recall(self, path: str) -> bool:
        return _basename(path).lower() == self.recall_filename.lower()

    async def _fetch_text(self, query: Optional[str], *, mode: str) -> str:
        """Return formatted context from Synap, or ``""``. Never raises."""
        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id,
                search_query=[query] if query else None,
                max_results=self.max_results,
                mode=mode,
                precision_level=self.precision_level,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — read path must not raise
            logger.error(
                "microsoft_agent.file_store.fetch failed: query=%s user_id=%s error=%s",
                query,
                self.user_id,
                exc,
                exc_info=True,
            )
            return ""
        return (getattr(response, "formatted_context", None) or "").strip()

    # ------------------------------------------------------------------
    # AgentFileStore
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, *, overwrite: bool = True) -> None:
        """Write ``content``, and submit it to Synap unless it is bookkeeping.

        Raises:
            FileExistsError: When ``overwrite`` is ``False`` and the file is
                already held, matching ``InMemoryAgentFileStore``.
            SynapIntegrationError: When the Synap submission fails.
        """
        normalized = _normalize(path)
        if not overwrite and await self._cache.get(normalized) is not None:
            raise FileExistsError(f"File already exists: {path!r}")

        await self._cache.put(normalized, content)

        if _is_internal(normalized) or not content.strip():
            # MAF's index and description sidecars, or an empty file. Cached so
            # the agent can read them back; not ingested, because neither is
            # something the user asked to remember.
            return

        async with wrap_sdk_errors_async(
            "microsoft_agent.file_store.write",
            logger,
            path=normalized,
            user_id=self.user_id,
        ):
            response = await self.sdk.memories.create(
                document=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
                document_type=self.document_type,
                mode=self.ingest_mode,
            )
        ingestion_id = getattr(response, "ingestion_id", None)
        if ingestion_id is not None:
            self._ingestions.setdefault(normalized, []).append(ingestion_id)

    async def read(self, path: str) -> Optional[str]:
        """Return the file content, or ``None`` if there is nothing there.

        Cache first, so a file this store wrote always reads back exactly.
        Beyond that, the recall file is answered by an unqueried scope fetch
        and any other name is treated as a question — ``deployment-notes.md``
        becomes the query ``deployment notes``. That is a relevance search, so
        it can miss; it does not invent content.
        """
        normalized = _normalize(path)
        cached = await self._cache.get(normalized)
        if cached is not None:
            return cached

        if _is_internal(normalized):
            # Bookkeeping is cache-only by construction. Answering a missing
            # index with a recall block would hand MAF a "file list" made of
            # memory prose, which it would then parse as file names.
            return None

        if self._is_recall(normalized):
            return await self._fetch_text(None, mode=self.mode) or None

        stem = _basename(normalized).rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
        return await self._fetch_text(stem, mode=self.mode) or None

    async def delete(self, path: str) -> bool:
        """Delete the file and the Synap memories it produced.

        Returns ``True`` when something was actually removed — a cached file,
        or memories resolved from this store's own ingestion ids. Returns
        ``False`` otherwise, including for a file written by an earlier
        process, because there is no way to resolve that back to a memory id
        and reporting success would be a lie the agent acts on.

        The chain was verified against a live instance: one write produced two
        memories, both were deleted, and both were confirmed gone.
        """
        normalized = _normalize(path)
        removed = await self._cache.drop(normalized)

        ingestion_ids = self._ingestions.pop(normalized, [])
        for ingestion_id in ingestion_ids:
            try:
                status = await self.sdk.memories.status(ingestion_id)
            except Exception as exc:  # noqa: BLE001 — best effort, logged
                logger.error(
                    "microsoft_agent.file_store.delete: status failed "
                    "ingestion_id=%s error=%s",
                    ingestion_id,
                    exc,
                    exc_info=True,
                )
                continue
            for memory_id in getattr(status, "memory_ids", []) or []:
                try:
                    await self.sdk.memories.delete(memory_id)
                    removed = True
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "microsoft_agent.file_store.delete: delete failed "
                        "memory_id=%s error=%s",
                        memory_id,
                        exc,
                        exc_info=True,
                    )
        return removed

    async def list_children(self, directory: str = "") -> List[FileStoreEntry]:
        """Return the direct children of ``directory``, from the cache.

        Directories first, then files, matching ``InMemoryAgentFileStore``.
        Names are relative to ``directory``.

        This sees only what this process wrote. See the module docstring —
        it is the one method Synap cannot back today, and it is why a restarted
        agent can read its files by name but cannot browse them.
        """
        prefix = _normalize(directory, is_directory=True)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        files: List[str] = []
        directories: List[str] = []
        seen: set = set()
        for path, _ in await self._cache.items():
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            separator = remainder.find("/")
            if separator == -1:
                files.append(remainder)
            elif separator > 0:
                segment = remainder[:separator]
                if segment in seen:
                    continue
                seen.add(segment)
                directories.append(segment)

        entries = [FileStoreEntry(name, FileStoreEntry.DIRECTORY) for name in directories]
        entries.extend(FileStoreEntry(name, FileStoreEntry.FILE) for name in files)
        return entries

    async def file_exists(self, path: str) -> bool:
        """Return whether ``read`` would return content for ``path``."""
        return await self.read(path) is not None

    async def search(
        self,
        directory: str,
        regex_pattern: str,
        glob_pattern: Optional[str] = None,
        *,
        recursive: bool = False,
    ) -> List[FileSearchResult]:
        """Search by regex over cached files, and by meaning over Synap.

        Both, deliberately. The regex half is exact and is what the model
        expects from a tool called ``grep``; it covers everything written this
        session. The semantic half is what a memory service is for, and it
        reaches memories no regex could match because they are worded
        differently from the pattern. Semantic hits are attributed to the
        recall file so the model can read that file for the full context.

        Raises:
            ValueError: When ``regex_pattern`` is longer than
                :data:`MAX_SEARCH_PATTERN_LENGTH`, matching MAF's cap.
            re.error: When ``regex_pattern`` is not a valid regular expression.
                Left unwrapped on purpose — the model sees it and retries.
        """
        if len(regex_pattern) > MAX_SEARCH_PATTERN_LENGTH:
            raise ValueError(
                f"Regex pattern is too long ({len(regex_pattern)} characters). "
                f"Maximum supported length is {MAX_SEARCH_PATTERN_LENGTH} characters."
            )
        regex = re.compile(regex_pattern, flags=re.IGNORECASE)

        prefix = _normalize(directory, is_directory=True)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        results: List[FileSearchResult] = []
        matched_names: set = set()

        for path, content in await self._cache.items():
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            if not recursive and "/" in relative:
                continue
            if glob_pattern and not fnmatch.fnmatch(relative.lower(), glob_pattern.lower()):
                continue
            matches = [
                FileSearchMatch(number, line)
                for number, line in enumerate(content.splitlines(), start=1)
                if regex.search(line)
            ]
            if not matches:
                continue
            matched_names.add(relative)
            results.append(
                FileSearchResult(
                    file_name=relative,
                    snippet=matches[0].line[:200],
                    matching_lines=matches,
                )
            )

        # The semantic half. The pattern is passed as a natural-language query,
        # which is the whole point of routing this at a memory service.
        recall_name = self.recall_filename
        if recall_name in matched_names:
            return results
        if glob_pattern and not fnmatch.fnmatch(recall_name.lower(), glob_pattern.lower()):
            return results

        text = await self._fetch_text(regex_pattern, mode=self.search_mode)
        if not text:
            return results

        lines = [
            FileSearchMatch(number, line)
            for number, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
        if lines:
            results.append(
                FileSearchResult(
                    file_name=recall_name,
                    snippet=lines[0].line[:200],
                    matching_lines=lines,
                )
            )
        return results

    async def create_directory(self, path: str) -> None:
        """No-op. Synap has no directories; paths are keys."""
        _normalize(path, is_directory=True)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"SynapAgentFileStore(user_id={self.user_id!r}, "
            f"customer_id={self.customer_id!r}, "
            f"recall_filename={self.recall_filename!r})"
        )


__all__ = [
    "SynapAgentFileStore",
    "DEFAULT_RECALL_FILENAME",
    "DEFAULT_CACHE_TTL_SECONDS",
    "MAX_SEARCH_PATTERN_LENGTH",
]
