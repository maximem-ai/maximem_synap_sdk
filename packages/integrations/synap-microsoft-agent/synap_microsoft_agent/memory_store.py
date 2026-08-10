"""``SynapMemoryStore`` — the MAF harness topic-memory subsystem, backed by Synap.

MAF's harness keeps memory as a *notebook*: a ``MEMORY.md`` index of pointer
lines, one topic record per subject, an LLM extraction pass per turn, and a
periodic consolidation rewrite. ``MemoryStore`` is the storage seam under all
of that, and ``MemoryContextProvider(store=...)`` is a public constructor
argument — so plugging Synap in needs no fork and no upstream change.

What this replaces, and what it deliberately does not
-----------------------------------------------------

Extraction, merging, topic selection, and consolidation stay in
``MemoryContextProvider``. They are MAF's reasoning about memory and they are
good. This store replaces the two things underneath: **where records live** and
**how recall is retrieved**.

===========================  ================================================
``MemoryStore`` method       What it does here
===========================  ================================================
``write_topic``              Records the topic exactly, and submits its
                             content to Synap for durable semantic recall.
``get_topic``                Returns the exact record. Raises
                             ``FileNotFoundError`` when there isn't one — see
                             "Why records are not read back from Synap".
``list_topics``              The recorded topics.
``delete_topic``             Drops the record and deletes the Synap memories
                             it produced.
``rebuild_index``            Derives index entries from the records. Nothing
                             is stored; the index is a projection.
``get_index_text``           MAF's pointer lines **plus a Synap recall
                             block**. This is the differentiator: the block is
                             durable across restarts and across agents, so the
                             agent sees memory the record layer has forgotten.
``read_state`` /             Maintenance state (consolidation bookkeeping).
``write_state``              Kept with the records.
``search_transcripts``       ``sdk.fetch(search_query=[q])`` — retrieval by
                             meaning, replacing substring search over turn
                             files.
``get_transcripts_directory``A scratch path. See "Transcripts" below.
===========================  ================================================

Why records are not read back from Synap
----------------------------------------

This was measured against a live instance rather than assumed, and the result
set the design.

``MemoryContextProvider._merge_memory`` does read-modify-write: it reads a
topic record, adds a memory line, and writes the record back. That requires the
record to come back **exactly** as it went in.

Synap does not work that way, by design. ``memories.create`` runs an extraction
pipeline: it rewrites, splits, and merges what you submit. Submitting a topic
record as JSON and reading it back returned four extracted memories, no JSON
envelope, and the text rephrased into the third person. The *substance*
survived; the *record* did not.

Reading records back from Synap would therefore feed ``_merge_memory`` a
rewritten record, which it would rewrite again on the next turn, and again on
the next consolidation pass. That is not a quality trade-off — it is
progressive corruption of the user's memory. So records are held exactly, by a
:class:`TopicRecordStore`, and Synap holds the durable *content*.

The default record store lives for the life of the process. When it is cold —
a restart, a new worker — ``get_topic`` raises ``FileNotFoundError``, which is
MAF's own not-found contract; ``_merge_memory`` catches it and starts a fresh
record. The topic looks new, and no mangled record ever enters the loop. That
is safe degradation, and it is deliberate. Meanwhile the content written on
every previous run is still in Synap and still reaches the prompt, through the
recall block in ``get_index_text``.

Pass your own :class:`TopicRecordStore` to survive restarts. When Synap gains a
verbatim record API this class gets a durable default and nothing else changes.

Transcripts
-----------

``get_transcripts_directory`` has to return a ``Path`` — the ABC demands it.
Pair this store with :class:`SynapMemoryContextProvider` (or build it via
:func:`create_synap_harness_memory`) and that path is never read and never
created: the subclass overrides the only two public methods that touch it, and
transcripts go to Synap instead. Pair it with the stock
``MemoryContextProvider`` and the path becomes real, so transcripts land on
local disk while topics stay in the cloud. That is a documented fallback, not a
failure — but know which one you are running.

Wiring
------

    from agent_framework import create_harness_agent
    from synap_microsoft_agent import create_synap_harness_memory

    agent = create_harness_agent(
        client,
        history_provider=create_synap_harness_memory(
            sdk, user_id="alice", customer_id="acme",
        ),
    )
"""

from __future__ import annotations

import contextvars
import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    default_scope,
    run_async,
    wrap_sdk_errors_async,
)

from synap_microsoft_agent._harness_compat import require_harness

require_harness()

from agent_framework import (  # noqa: E402 — must follow the version guard
    MemoryIndexEntry,
    MemoryStore,
    MemoryTopicRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_RECALL_HEADER = "## Durable memory (Synap)"
"""Heading for the recall block appended to ``MEMORY.md``."""

DEFAULT_OWNER_STATE_KEY = "synap_memory_owner"
"""Session-state key carrying the resolved owner, mirroring ``MemoryFileStore``."""

_NO_TOPICS_TEXT = "(no topics yet)"
_INDEX_HEADER = "# Memory Index"

_RECALL_QUERY: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "synap_recall_query", default=None
)
"""The question the current turn is asking, for the duration of that turn.

``MemoryStore.get_index_text`` is handed a line limit and a line length and
nothing else — MAF never passes it the user's question, because the file store
it was designed against has nothing to do with one. Without the question the
recall block is an *unqueried* scope fetch, and an unqueried fetch returns
whatever the scope considers generally salient rather than what was asked.

Measured on the conformance bench: asked "when do we send invoices", an
unqueried block came back with the user's code-review preferences. Correct, and
useless.

:class:`~synap_microsoft_agent.harness_memory.SynapMemoryContextProvider` sets
this from ``context.input_messages`` around its ``before_run``, which is public
and which it already overrides. A ``ContextVar`` rather than an attribute so
concurrent sessions on one store cannot read each other's question.
"""


class TopicRecordStore(Protocol):
    """Exact storage for topic records and maintenance state.

    Split out from :class:`SynapMemoryStore` so the fidelity requirement is a
    seam rather than a hidden assumption. Anything implementing this — a Redis
    hash, a Postgres table, a durable KV — makes topic records survive a
    restart without touching the rest of the store.

    Implementations must return records **byte-identical** to what was written.
    That is the whole point: ``_merge_memory`` reads, edits, and writes back.
    """

    def put(self, owner: str, slug: str, payload: Dict[str, Any]) -> None:
        """Store one topic record under ``owner``."""

    def get(self, owner: str, slug: str) -> Optional[Dict[str, Any]]:
        """Return one topic record, or ``None`` if it is not held."""

    def delete(self, owner: str, slug: str) -> bool:
        """Drop one topic record; return whether anything was removed."""

    def list(self, owner: str) -> List[Dict[str, Any]]:
        """Return every topic record held for ``owner``."""

    def put_state(self, owner: str, state: Dict[str, Any]) -> None:
        """Store maintenance state for ``owner``."""

    def get_state(self, owner: str) -> Optional[Dict[str, Any]]:
        """Return maintenance state for ``owner``, or ``None``."""


class InMemoryTopicRecordStore:
    """The default :class:`TopicRecordStore`: exact, in-process, not durable.

    Records are held as plain dicts and copied on the way in and out, so a
    caller that mutates what it got back cannot corrupt the stored record.

    It is thread-safe because MAF's ``MemoryStore`` methods are synchronous and
    a harness agent may drive several sessions from a thread pool.
    """

    def __init__(self) -> None:
        self._topics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def put(self, owner: str, slug: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._topics.setdefault(owner, {})[slug] = json.loads(json.dumps(payload))

    def get(self, owner: str, slug: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            found = self._topics.get(owner, {}).get(slug)
            return json.loads(json.dumps(found)) if found is not None else None

    def delete(self, owner: str, slug: str) -> bool:
        with self._lock:
            return self._topics.get(owner, {}).pop(slug, None) is not None

    def list(self, owner: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(v)) for v in self._topics.get(owner, {}).values()]

    def put_state(self, owner: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state[owner] = json.loads(json.dumps(state))

    def get_state(self, owner: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            found = self._state.get(owner)
            return json.loads(json.dumps(found)) if found is not None else None


class SynapMemoryStore(MemoryStore):
    """A ``MemoryStore`` whose durable half is Synap.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        user_id: External user ID. At least one of ``user_id`` or
            ``customer_id`` is required — a store with neither has no scope.
        customer_id: External customer ID.
        record_store: Exact storage for topic records. Defaults to
            :class:`InMemoryTopicRecordStore`, which does not survive the
            process. Read the module docstring before accepting that default in
            production.
        include_recall: Append a Synap recall block to ``MEMORY.md``. On by
            default — it is the only path by which memory written on a previous
            run reaches the prompt. Costs one ``fetch`` per turn (~0.5 s
            measured on prod).
        recall_header: Heading for that block.
        max_results: ``max_results`` for every ``fetch``.
        mode: Retrieval mode for the per-turn recall block. ``"fast"``, since
            it sits on the turn path.
        search_mode: Retrieval mode for ``search_transcripts``. ``"accurate"``,
            because a search is a deliberate question.
        precision_level: ``"high"`` (default) or ``"medium"``.
        document_type: ``document_type`` for writes.
        ingest_mode: ``"fast"`` (default) or ``"long-range"``.
        owner_state_key: Session-state key carrying the owner ID, for hosts
            that route sessions to different owners.
        scope_resolver: Optional callable over the ``AgentSession`` returning
            an owner string, for multi-tenant hosts.
        transcripts_root: Where the scratch transcript path points. Only ever
            used when this store is paired with the stock
            ``MemoryContextProvider``.

    Raises:
        ValueError: If neither ``user_id`` nor ``customer_id`` is given.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        record_store: Optional[TopicRecordStore] = None,
        include_recall: bool = True,
        recall_header: str = DEFAULT_RECALL_HEADER,
        max_results: int = 20,
        mode: str = "fast",
        search_mode: str = "accurate",
        precision_level: str = "high",
        document_type: str = "document",
        ingest_mode: str = "fast",
        owner_state_key: str = DEFAULT_OWNER_STATE_KEY,
        scope_resolver: Optional[Any] = None,
        transcripts_root: Optional[str] = None,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryStore requires a non-None sdk")
        if not user_id and not customer_id:
            raise ValueError(
                "SynapMemoryStore requires at least one of user_id or customer_id — "
                "a store with no scope cannot read or write memory"
            )

        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.records: TopicRecordStore = record_store or InMemoryTopicRecordStore()
        self.include_recall = include_recall
        self.recall_header = recall_header
        self.max_results = max_results
        self.mode = mode
        self.search_mode = search_mode
        self.precision_level = precision_level
        self.document_type = document_type
        self.ingest_mode = ingest_mode
        self.owner_state_key = owner_state_key
        self.scope_resolver = scope_resolver
        self.transcripts_root = transcripts_root or str(
            Path(tempfile.gettempdir()) / "synap-maf-transcripts"
        )
        # path -> ingestion ids, so delete_topic can reach the memories a write
        # produced. `memories.create` returns an ingestion id, not a memory id;
        # the bridge is `status(ingestion_id).memory_ids`.
        self._ingestions: Dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def get_owner_id(self, session: Any) -> Optional[str]:
        """Return the owner key for ``session``.

        Concrete on the ABC rather than abstract, so overriding is a choice.
        We override because the owner is what partitions records, and getting
        it from the session is what lets one store serve many tenants.
        """
        if self.scope_resolver is not None:
            resolved = self.scope_resolver(session)
            if resolved:
                return str(resolved)
        state = getattr(session, "state", None)
        if isinstance(state, dict):
            from_state = state.get(self.owner_state_key)
            if from_state:
                return str(from_state)
        return self._default_owner()

    def _default_owner(self) -> str:
        if self.user_id:
            return default_scope(self.user_id, self.customer_id)
        return f"/{self.customer_id}"

    def _owner(self, session: Any) -> str:
        return self.get_owner_id(session) or self._default_owner()

    def export_provider_state(self, session: Any) -> Dict[str, Any]:
        """Carry the owner across the temporary sessions MAF builds.

        ``MemoryContextProvider.get_messages`` and ``save_messages`` do not get
        the live session — they get a state dict and rebuild a session from it.
        Without the owner in that dict, a transcript read would resolve to the
        wrong scope.
        """
        return {self.owner_state_key: self._owner(session)}

    def import_provider_state(self, session: Any, *, state: Mapping[str, Any]) -> None:
        """Apply exported routing state back onto a rebuilt session."""
        owner = state.get(self.owner_state_key)
        if owner and isinstance(getattr(session, "state", None), dict):
            session.state[self.owner_state_key] = owner

    def _scope_kwargs(self, session: Any) -> Dict[str, Any]:
        """Return the ``user_id`` / ``customer_id`` pair for a Synap call.

        A ``scope_resolver`` changes the *record* partition. It does not
        re-scope the Synap call, because Synap's scope chain is expressed as
        IDs rather than as a path string, and inventing IDs from a path would
        be guessing. Hosts that need per-tenant Synap scoping construct one
        store per tenant.
        """
        del session
        return {"user_id": self.user_id, "customer_id": self.customer_id}

    # ------------------------------------------------------------------
    # Topics — exact, from the record store
    # ------------------------------------------------------------------

    def list_topics(self, session: Any, *, source_id: str) -> List[MemoryTopicRecord]:
        """Return every topic record for this owner.

        Pure-local: no network call, so the per-turn cost is zero.
        """
        del source_id
        owner = self._owner(session)
        records = [MemoryTopicRecord.from_dict(dict(p)) for p in self.records.list(owner)]
        return sorted(records, key=lambda r: (r.topic.lower(), r.updated_at))

    def get_topic(self, session: Any, *, source_id: str, topic: str) -> MemoryTopicRecord:
        """Return one topic record exactly as written.

        Raises:
            FileNotFoundError: When no record is held. This is MAF's own
                not-found contract — ``_merge_memory`` and ``_run_consolidation``
                both catch it — and raising it is what keeps a rewritten record
                from ever reaching the read-modify-write loop.
        """
        del source_id
        owner = self._owner(session)
        payload = self.records.get(owner, _slug(topic))
        if payload is None:
            raise FileNotFoundError(
                f"No memory topic named '{topic}' was found for this owner."
            )
        return MemoryTopicRecord.from_dict(dict(payload))

    def write_topic(self, session: Any, record: MemoryTopicRecord, *, source_id: str) -> None:
        """Record the topic exactly, then submit its content to Synap.

        The record write happens first and unconditionally. If the Synap
        submission then fails it raises, per the integration error policy —
        but the local record is already consistent, so the agent's next turn
        sees the memory it just wrote rather than losing it to a network blip.
        """
        del source_id
        owner = self._owner(session)
        slug = _slug(record.slug or record.topic)
        self.records.put(owner, slug, record.to_dict())

        document = _record_to_document(record)
        if not document.strip():
            return

        async def _submit() -> Any:
            async with wrap_sdk_errors_async(
                "microsoft_agent.memory_store.write_topic",
                logger,
                topic=record.topic,
                owner=owner,
            ):
                return await self.sdk.memories.create(
                    document=document,
                    document_type=self.document_type,
                    mode=self.ingest_mode,
                    **self._scope_kwargs(session),
                )

        response = run_async(_submit())
        ingestion_id = getattr(response, "ingestion_id", None)
        if ingestion_id is not None:
            self._ingestions.setdefault(f"{owner}::{slug}", []).append(ingestion_id)

    def delete_topic(self, session: Any, *, source_id: str, topic: str) -> None:
        """Drop the record, and delete the Synap memories that topic produced.

        Raises:
            FileNotFoundError: When the topic is not held, matching
                ``MemoryFileStore``. The harness's ``delete_memory_topic`` tool
                surfaces this to the model as a plain error.
        """
        del source_id
        owner = self._owner(session)
        slug = _slug(topic)
        if not self.records.delete(owner, slug):
            raise FileNotFoundError(
                f"No memory topic named '{topic}' was found for this owner."
            )

        ingestion_ids = self._ingestions.pop(f"{owner}::{slug}", [])
        if not ingestion_ids:
            # Written by an earlier process, so there is no ingestion id to
            # resolve. The record is gone; the Synap memories are not. Say so
            # rather than reporting a clean delete.
            logger.warning(
                "microsoft_agent.memory_store.delete_topic: dropped the record for "
                "topic=%s but no ingestion ids were held, so its Synap memories "
                "remain. Delete them by memory id via the API or dashboard.",
                topic,
            )
            return

        async def _purge() -> None:
            for ingestion_id in ingestion_ids:
                try:
                    status = await self.sdk.memories.status(ingestion_id)
                except Exception as exc:  # noqa: BLE001 — best effort, logged
                    logger.error(
                        "microsoft_agent.memory_store.delete_topic: status failed "
                        "ingestion_id=%s error=%s",
                        ingestion_id,
                        exc,
                        exc_info=True,
                    )
                    continue
                for memory_id in getattr(status, "memory_ids", []) or []:
                    try:
                        await self.sdk.memories.delete(memory_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "microsoft_agent.memory_store.delete_topic: delete failed "
                            "memory_id=%s error=%s",
                            memory_id,
                            exc,
                            exc_info=True,
                        )

        run_async(_purge())

    # ------------------------------------------------------------------
    # Index — the differentiator
    # ------------------------------------------------------------------

    def rebuild_index(
        self, session: Any, *, source_id: str, line_limit: int, line_length: int
    ) -> List[MemoryIndexEntry]:
        """Derive index entries from the records. Nothing is stored.

        ``MemoryFileStore`` writes ``MEMORY.md`` to disk here. There is nothing
        to write: the index is a projection of the records, so it is rebuilt on
        demand and is never stale.
        """
        del line_length
        topics = self.list_topics(session, source_id=source_id)
        return [MemoryIndexEntry.from_topic_record(t) for t in topics][:line_limit]

    def get_index_text(
        self,
        session: Any,
        *,
        source_id: str,
        line_limit: int,
        line_length: int,
        index_entries: Optional[Sequence[MemoryIndexEntry]] = None,
    ) -> str:
        """Return ``MEMORY.md``: MAF's pointer lines, plus Synap recall.

        The pointer lines are MAF's own format, unchanged, so topic selection
        keeps working exactly as it does on the file store. The recall block is
        the addition, and it is the reason to use this integration: it carries
        memory the record layer never had — written on a previous run, by a
        previous process, or by a different agent against the same scope.

        Never raises. This text goes into the system prompt on **every** turn,
        so a retrieval outage degrades to pointer lines with no recall block
        rather than ending the run.
        """
        if index_entries is None:
            index_entries = self.rebuild_index(
                session,
                source_id=source_id,
                line_limit=line_limit,
                line_length=line_length,
            )

        pointer_lines = [
            entry.to_pointer_line(max_length=line_length)
            for entry in index_entries[:line_limit]
        ]
        lines = [_INDEX_HEADER, ""]
        lines.extend(pointer_lines if pointer_lines else [_NO_TOPICS_TEXT])

        if self.include_recall:
            # Conditioned on the turn's question when one is available. Without
            # it this degrades to an unqueried scope fetch, which returns what
            # the scope finds generally salient rather than what was asked —
            # still useful, measurably less so.
            recall = self._fetch_text(
                session, query=_RECALL_QUERY.get(), mode=self.mode
            )
            if recall:
                lines.extend(["", self.recall_header, "", recall])

        return "\n".join(lines).rstrip()

    @staticmethod
    def set_recall_query(query: Optional[str]) -> Any:
        """Scope the next ``get_index_text`` recall block to ``query``.

        Returns the token to pass to :meth:`reset_recall_query`. Called by
        :class:`SynapMemoryContextProvider` around ``before_run``; a caller
        driving the store directly can use it too.
        """
        return _RECALL_QUERY.set((query or "").strip() or None)

    @staticmethod
    def reset_recall_query(token: Any) -> None:
        """Undo a :meth:`set_recall_query`, so the query never outlives its turn."""
        _RECALL_QUERY.reset(token)

    # ------------------------------------------------------------------
    # Maintenance state
    # ------------------------------------------------------------------

    def read_state(self, session: Any, *, source_id: str) -> Dict[str, Any]:
        """Return consolidation bookkeeping for this owner.

        Kept alongside the records rather than in Synap. It is opaque
        machine state with no recall value, and round-tripping it through an
        extraction pipeline would corrupt it for no benefit.
        """
        del source_id
        stored = self.records.get_state(self._owner(session))
        state = {"last_consolidated_at": None, "sessions_since_consolidation": []}
        if isinstance(stored, dict):
            state.update(stored)
        if not isinstance(state.get("sessions_since_consolidation"), list):
            state["sessions_since_consolidation"] = []
        if not isinstance(state.get("last_consolidated_at"), (str, type(None))):
            state["last_consolidated_at"] = None
        return state

    def write_state(self, session: Any, state: Mapping[str, Any], *, source_id: str) -> None:
        """Persist consolidation bookkeeping for this owner."""
        del source_id
        self.records.put_state(self._owner(session), dict(state))

    # ------------------------------------------------------------------
    # Transcripts
    # ------------------------------------------------------------------

    def get_transcripts_directory(self, session: Any, *, source_id: str) -> Path:
        """Return a scratch path. Read the module docstring before relying on it.

        With :class:`SynapMemoryContextProvider` nothing reads or creates this.
        With the stock ``MemoryContextProvider`` it becomes the real transcript
        root and you get a hybrid: topics in Synap, transcripts on disk.
        """
        owner = _safe_component(self._owner(session))
        return Path(self.transcripts_root) / _safe_component(source_id) / owner

    def search_transcripts(
        self,
        session: Any,
        *,
        source_id: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search memory by meaning, in place of substring search over turn files.

        ``MemoryFileStore`` casefolds the query and looks for it inside stored
        message text, so a question worded differently from the transcript
        finds nothing. This asks Synap instead.

        The rows keep MAF's shape — ``session_id`` / ``line_number`` / ``role``
        / ``text`` — because ``_format_search_results`` renders them for the
        model. ``line_number`` is the line's position in the retrieved block,
        which is what makes the rendered output readable; it is not an offset
        into any stored file, and there is no file it could point into.
        """
        del source_id
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        text = self._fetch_text(session, query=query.strip(), mode=self.search_mode)
        if not text:
            return []

        rows: List[Dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "line_number": line_number,
                    "role": "memory",
                    "text": line,
                }
            )
            if len(rows) >= limit:
                break
        return rows

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_text(self, session: Any, *, query: Optional[str], mode: str) -> str:
        """Return formatted context from Synap, or ``""``.

        Never raises. Both callers sit on paths where an exception would end
        the agent turn: ``get_index_text`` feeds the system prompt every run,
        and ``search_transcripts`` backs a model-facing tool.
        """

        async def _call() -> Any:
            return await self.sdk.fetch(
                search_query=[query] if query else None,
                max_results=self.max_results,
                mode=mode,
                precision_level=self.precision_level,
                include_conversation_context=False,
                **self._scope_kwargs(session),
            )

        try:
            response = run_async(_call())
        except Exception as exc:  # noqa: BLE001 — turn path must not raise
            logger.error(
                "microsoft_agent.memory_store.fetch failed: query=%s user_id=%s error=%s",
                query,
                self.user_id,
                exc,
                exc_info=True,
            )
            return ""
        return (getattr(response, "formatted_context", None) or "").strip()

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"SynapMemoryStore(user_id={self.user_id!r}, "
            f"customer_id={self.customer_id!r}, "
            f"records={type(self.records).__name__})"
        )


def _slug(value: str) -> str:
    """Normalize a topic name to a stable key.

    Mirrors MAF's ``_slugify_topic`` closely enough for keying without
    importing it — it lives on a private path (D5). Topic names come from an
    LLM extraction pass, so casing and spacing vary between turns for what is
    meant to be the same topic.
    """
    cleaned = "".join(c if c.isalnum() else "-" for c in (value or "").strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "untitled"


def _safe_component(value: str) -> str:
    """Make ``value`` safe to use as a single path segment."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-") or "default"


def _record_to_document(record: MemoryTopicRecord) -> str:
    """Render a topic record as prose for ingestion.

    Deliberately **not** ``record.to_dict()`` as JSON. Measured against a live
    instance, both shapes come back rewritten — the pipeline extracts either
    way — but a JSON envelope adds punctuation and key names that show up as
    noise in the extracted memories. Prose gives the extractor plain sentences,
    which is what it is built for.

    The record itself is preserved exactly by the record store, so nothing here
    needs to be reversible.
    """
    parts = [f"{record.topic}: {record.summary}".strip().rstrip(":")]
    parts.extend(str(memory).strip() for memory in record.memories if str(memory).strip())
    return "\n".join(part for part in parts if part)


__all__ = [
    "SynapMemoryStore",
    "TopicRecordStore",
    "InMemoryTopicRecordStore",
    "DEFAULT_RECALL_HEADER",
    "DEFAULT_OWNER_STATE_KEY",
]
