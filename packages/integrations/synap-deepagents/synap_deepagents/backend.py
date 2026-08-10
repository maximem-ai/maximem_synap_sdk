"""``SynapBackend`` — a deepagents backend that stores memory in Synap.

deepagents talks to storage through :class:`~deepagents.backends.protocol.BackendProtocol`,
a *filesystem* interface: ``ls``, ``read``, ``write``, ``edit``, ``grep``, ``glob``,
``upload_files``, ``download_files``. Synap is not a filesystem, so this backend
maps those verbs onto memory operations by intent rather than one-to-one:

===================  ==========================================================
Backend call         What it does
===================  ==========================================================
``read`` of the      Calls ``sdk.fetch()`` and returns the formatted context as
recall file          the file body. The file is synthesized per call; it is
                     never stored anywhere.
``grep``             Calls ``sdk.fetch(search_query=[pattern])``. **This is a
                     semantic search, not a regex match** — the pattern is
                     treated as a natural-language query.
``write``            Calls ``sdk.memories.create()``.
``edit``             Stores the added text as a new memory. Synap supersedes
                     rather than rewrites, so this is an append, not a patch.
``ls`` / ``glob``    List the synthesized recall file plus anything written in
                     this session.
``delete``           **Not implemented** — see below.
===================  ==========================================================

Mount it under a route, never as the default backend::

    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend
    from synap_deepagents import SynapBackend

    backend = CompositeBackend(
        default=FilesystemBackend(root_dir="/path/to/repo"),
        routes={"/memories/": SynapBackend(sdk, user_id="alice")},
    )

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5",
        backend=backend,
        memory=["/memories/AGENTS.md"],
    )

Passing ``SynapBackend`` as ``backend=`` on its own would route the agent's
source-code reads and writes through a memory API. The constructor cannot stop
you, but you will lose your working tree. Always mount it on a route.

**Why there is no ``delete``.** ``sdk.memories.create()`` returns an
``ingestion_id``, not a ``memory_id``, and ``sdk.memories.delete()`` needs a
``memory_id``. A path written through this backend therefore cannot be resolved
back to a durable memory, so a path-addressed delete cannot be honoured.
``delete`` is optional in ``BackendProtocol``, so this backend inherits the
default that raises ``NotImplementedError``, and ``CompositeBackend`` reports it
cleanly. Remove memories through the Synap API or dashboard with a memory id.

(If you go looking for deepagents' capability guard: its ``delete`` docstring
points at ``supports_delete``, but the function that actually exists in 0.7.4 is
``_supports_delete``. Catch ``NotImplementedError`` rather than depending on
either name.)

**Error policy.** Reads never raise: ``MemoryMiddleware`` raises ``ValueError``
on any download error code other than ``file_not_found``, so a Synap outage that
surfaced any other code would kill the agent run. Every read failure is logged at
ERROR and reported as ``file_not_found``, which degrades to an empty memory block.
Writes raise :class:`~synap_integrations_common.SynapIntegrationError`, per the
Synap integration error policy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import normalize_read_bounds
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)

DEFAULT_RECALL_FILENAME = "AGENTS.md"
"""Basename treated as the synthesized recall document."""

DEFAULT_CACHE_TTL_SECONDS = 300
"""How long a written entry stays readable from the local cache."""


def _normalize(path: str) -> str:
    """Return ``path`` as a leading-slash absolute path with no trailing slash."""
    cleaned = "/" + (path or "").strip().lstrip("/")
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/")
    return cleaned or "/"


def _basename(path: str) -> str:
    return _normalize(path).rsplit("/", 1)[-1]


class _WriteCache:
    """Per-scope, TTL'd record of what this session has written.

    Synap ingestion is queued, so a memory written a moment ago is not yet
    retrievable. Without this cache an agent that writes a file and then reads
    it back in the same turn gets nothing, which reads as data loss. The cache
    is a read-after-write guarantee for this process only — it is not a dedup
    layer and it does not survive a restart.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._enabled = ttl_seconds > 0
        self._entries: Dict[str, Tuple[float, str]] = {}

    def put(self, path: str, content: str) -> None:
        if not self._enabled:
            return
        self._entries[_normalize(path)] = (time.monotonic(), content)

    def get(self, path: str) -> Optional[str]:
        if not self._enabled:
            return None
        key = _normalize(path)
        entry = self._entries.get(key)
        if entry is None:
            return None
        written_at, content = entry
        if (time.monotonic() - written_at) > self._ttl:
            self._entries.pop(key, None)
            return None
        return content

    def paths(self) -> List[str]:
        if not self._enabled:
            return []
        live = []
        now = time.monotonic()
        for path, (written_at, _) in list(self._entries.items()):
            if (now - written_at) > self._ttl:
                self._entries.pop(path, None)
                continue
            live.append(path)
        return sorted(live)

    def drop(self, path: str) -> None:
        self._entries.pop(_normalize(path), None)


class SynapBackend(BackendProtocol):
    """A ``BackendProtocol`` implementation backed by Synap memory.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        user_id: External user ID. At least one of ``user_id`` or
            ``customer_id`` is required — a backend with neither has no scope
            to read or write.
        customer_id: External customer ID.
        conversation_id: Optional conversation scope for recall.
        recall_filename: Basename that maps to the synthesized recall document.
            Defaults to ``"AGENTS.md"`` so it lines up with the path you pass to
            ``create_deep_agent(memory=[...])``.
        max_results: ``max_results`` passed to ``sdk.fetch``.
        mode: ``"fast"`` (default) or ``"accurate"``. ``read`` of the recall
            file sits on the agent's startup path, so ``"fast"`` is the sane
            default; ``grep`` overrides it with ``grep_mode``.
        grep_mode: Retrieval mode for ``grep``. Defaults to ``"accurate"``,
            because a grep is a deliberate question and worth the latency.
        precision_level: ``"high"`` (default) or ``"medium"``.
        document_type: ``document_type`` for writes. Defaults to ``"document"``
            because a backend write is explicit intent, not chat transcript.
        ingest_mode: ``"fast"`` (default) or ``"long-range"``.
        include_conversation_context: Whether recall includes compacted
            conversation history. Defaults to ``False``.
        cache_ttl_seconds: Read-after-write cache lifetime. ``0`` disables it —
            only do that if you can tolerate a write not being readable back.

    Raises:
        ValueError: If neither ``user_id`` nor ``customer_id`` is given.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        recall_filename: str = DEFAULT_RECALL_FILENAME,
        max_results: int = 20,
        mode: str = "fast",
        grep_mode: str = "accurate",
        precision_level: str = "high",
        document_type: str = "document",
        ingest_mode: str = "fast",
        include_conversation_context: bool = False,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapBackend requires a non-None sdk")
        if not user_id and not customer_id:
            raise ValueError(
                "SynapBackend requires at least one of user_id or customer_id — "
                "a backend with no scope cannot read or write memory"
            )
        if not recall_filename or not recall_filename.strip():
            raise ValueError("SynapBackend requires a non-empty recall_filename")

        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id
        self.recall_filename = recall_filename.strip()
        self.max_results = max_results
        self.mode = mode
        self.grep_mode = grep_mode
        self.precision_level = precision_level
        self.document_type = document_type
        self.ingest_mode = ingest_mode
        self.include_conversation_context = include_conversation_context
        self._cache = _WriteCache(cache_ttl_seconds)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_recall(self, path: str) -> bool:
        return _basename(path) == self.recall_filename

    def _recall_path(self) -> str:
        return f"/{self.recall_filename}"

    async def _fetch_text(self, query: Optional[str], *, mode: str) -> str:
        """Return formatted context from Synap, or ``""`` if there is none.

        Never raises. Callers on the read path translate an empty string into
        ``file_not_found``; see the module docstring for why nothing else is
        safe to return.
        """
        try:
            response = await self.sdk.fetch(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
                search_query=[query] if query else None,
                max_results=self.max_results,
                mode=mode,
                precision_level=self.precision_level,
                include_conversation_context=self.include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read path must not raise
            logger.error(
                "Synap recall failed: op=deepagents.backend.fetch error=%s "
                "user_id=%s customer_id=%s query=%s",
                exc,
                self.user_id,
                self.customer_id,
                query,
                exc_info=True,
            )
            return ""
        return (getattr(response, "formatted_context", None) or "").strip()

    async def _body_for(self, path: str) -> Optional[str]:
        """Return the body for ``path``, or ``None`` if there is nothing there."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        if self._is_recall(path):
            return await self._fetch_text(None, mode=self.mode) or None
        # An arbitrary path we never wrote: treat the basename as a question.
        stem = _basename(path).rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
        return await self._fetch_text(stem, mode=self.mode) or None

    @staticmethod
    def _read_result(path: str, body: str, offset: int, limit: int) -> ReadResult:
        """Slice ``body`` into a ``ReadResult`` with valid pagination fields."""
        offset, limit = normalize_read_bounds(offset, limit)
        if limit == 0:
            return ReadResult(no_lines_requested=True)

        lines = body.splitlines()
        total = len(lines)
        if total == 0:
            # An inspected but genuinely empty document. No window to report.
            return ReadResult(file_data=FileData(content="", encoding="utf-8"))
        if offset >= total:
            return ReadResult(error=FILE_NOT_FOUND)

        window = lines[offset : offset + limit]
        end_line = offset + len(window)
        return ReadResult(
            file_data=FileData(content="\n".join(window), encoding="utf-8"),
            total_lines=total,
            start_line=offset + 1,
            end_line=end_line,
            next_offset=end_line,
        )

    def _file_info(self, path: str, body: Optional[str] = None) -> FileInfo:
        info: FileInfo = {"path": _normalize(path), "is_dir": False}
        if body is not None:
            info["size"] = len(body.encode("utf-8"))
        return info

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        body = await self._body_for(file_path)
        if body is None:
            return ReadResult(error=FILE_NOT_FOUND)
        return self._read_result(file_path, body, offset, limit)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return run_async(self.aread(file_path, offset, limit))

    # ------------------------------------------------------------------
    # grep — the query seam
    # ------------------------------------------------------------------

    async def agrep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        *,
        max_count: Optional[int] = None,
    ) -> GrepResult:
        """Search memory semantically. ``pattern`` is a query, not a regex.

        Returns one :class:`GrepMatch` per line of the retrieved context, so the
        result renders the way the agent expects a grep result to render.
        """
        if not pattern or not pattern.strip():
            return GrepResult(matches=[])

        text = await self._fetch_text(pattern, mode=self.grep_mode)
        if not text:
            return GrepResult(matches=[])

        # Attribute matches to the recall file, not to a directory. When
        # mounted on a route, ``CompositeBackend`` strips its prefix before
        # calling us and re-adds it to every returned path — so a grep of
        # ``/memories/`` arrives here as ``/`` and must not go back out as
        # ``/memories/``. Returning ``/AGENTS.md`` remaps to
        # ``/memories/AGENTS.md``, which is a path the agent can actually read.
        normalized = _normalize(path) if path else "/"
        target = self._recall_path() if normalized == "/" else normalized
        matches: List[GrepMatch] = []
        for index, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            matches.append({"path": target, "line": index, "text": line})
            if max_count is not None and len(matches) >= max_count:
                return GrepResult(matches=matches, truncated=True)
        return GrepResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        *,
        max_count: Optional[int] = None,
    ) -> GrepResult:
        return run_async(self.agrep(pattern, path, glob, max_count=max_count))

    # ------------------------------------------------------------------
    # write / edit
    # ------------------------------------------------------------------

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        path = _normalize(file_path)
        async with wrap_sdk_errors_async(
            "deepagents.backend.write",
            logger,
            path=path,
            user_id=self.user_id,
        ):
            await self.sdk.memories.create(
                document=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
                document_type=self.document_type,
                mode=self.ingest_mode,
            )
        self._cache.put(path, content)
        return WriteResult(path=path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return run_async(self.awrite(file_path, content))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Store the text ``new_string`` adds, as a new memory.

        deepagents' stock memory prompt tells the model to call ``edit_file`` to
        save what it has learned, so this path carries real intent and must
        work. It is an append, not a patch: Synap supersedes memories rather
        than rewriting a document in place, so ``old_string`` is used only to
        work out what is new, never to locate a byte range.
        """
        path = _normalize(file_path)
        addition = new_string
        if old_string and old_string in new_string:
            addition = new_string.replace(old_string, "", 1).strip()
        addition = (addition or "").strip()

        if not addition:
            # A pure deletion or a no-op rewrite. Nothing to remember, and
            # inventing a write would store noise.
            return EditResult(path=path, occurrences=0)

        async with wrap_sdk_errors_async(
            "deepagents.backend.edit",
            logger,
            path=path,
            user_id=self.user_id,
        ):
            await self.sdk.memories.create(
                document=addition,
                user_id=self.user_id,
                customer_id=self.customer_id,
                document_type=self.document_type,
                mode=self.ingest_mode,
            )

        cached = self._cache.get(path)
        if cached is not None:
            self._cache.put(path, f"{cached}\n{addition}".strip())
        else:
            self._cache.put(path, addition)
        return EditResult(path=path, occurrences=1)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return run_async(self.aedit(file_path, old_string, new_string, replace_all))

    # ------------------------------------------------------------------
    # ls / glob
    # ------------------------------------------------------------------

    def _visible_paths(self) -> List[str]:
        paths = set(self._cache.paths())
        paths.add(self._recall_path())
        return sorted(paths)

    async def als(self, path: str) -> LsResult:
        prefix = _normalize(path)
        entries: List[FileInfo] = []
        for candidate in self._visible_paths():
            if prefix in ("/", "") or candidate.startswith(prefix):
                entries.append(self._file_info(candidate, self._cache.get(candidate)))
        return LsResult(entries=entries)

    def ls(self, path: str) -> LsResult:
        return run_async(self.als(path))

    async def aglob(
        self, pattern: str, path: Optional[str] = None
    ) -> GlobResult:
        import fnmatch

        base = _normalize(path) if path else "/"
        matches: List[FileInfo] = []
        for candidate in self._visible_paths():
            if base not in ("/", "") and not candidate.startswith(base):
                continue
            relative = candidate[len(base) :].lstrip("/") if base != "/" else candidate.lstrip("/")
            if fnmatch.fnmatch(relative, pattern.lstrip("/")) or fnmatch.fnmatch(
                candidate, pattern
            ):
                matches.append(self._file_info(candidate, self._cache.get(candidate)))
        return GlobResult(matches=matches)

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        return run_async(self.aglob(pattern, path))

    # ------------------------------------------------------------------
    # batch transfer — the path MemoryMiddleware uses
    # ------------------------------------------------------------------

    async def adownload_files(
        self, paths: List[str]
    ) -> List[FileDownloadResponse]:
        """Return one response per path, in the order given.

        ``MemoryMiddleware`` calls this on every agent run and raises
        ``ValueError`` on any error code except ``file_not_found``. Nothing else
        is ever returned here — see the module docstring.
        """
        responses: List[FileDownloadResponse] = []
        for path in paths:
            body = await self._body_for(path)
            if body is None:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=FILE_NOT_FOUND)
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=body.encode("utf-8"))
                )
        return responses

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        return run_async(self.adownload_files(paths))

    async def aupload_files(
        self, files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        responses: List[FileUploadResponse] = []
        for path, blob in files:
            try:
                content = blob.decode("utf-8")
            except UnicodeDecodeError:
                logger.error(
                    "Synap upload rejected non-text payload: path=%s bytes=%d",
                    path,
                    len(blob),
                )
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error="Synap stores text memories; binary payloads are not supported",
                    )
                )
                continue
            result = await self.awrite(path, content)
            responses.append(FileUploadResponse(path=path, error=result.error))
        return responses

    def upload_files(
        self, files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        return run_async(self.aupload_files(files))

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"SynapBackend(user_id={self.user_id!r}, "
            f"customer_id={self.customer_id!r}, "
            f"recall_filename={self.recall_filename!r})"
        )


__all__ = ["SynapBackend", "DEFAULT_RECALL_FILENAME", "DEFAULT_CACHE_TTL_SECONDS"]
