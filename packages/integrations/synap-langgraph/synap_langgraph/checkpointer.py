"""SynapCheckpointSaver — best-effort fuzzy ``BaseCheckpointSaver``.

Synap's retrieval is semantic-search shaped, not exact key-value. This saver
trades off checkpoint fidelity for the benefits of storing thread state in
Synap: observability, audit, and cross-thread memory sitting next to the
execution history.

For production durability, pair with a proper KV-backed saver
(``SqliteSaver`` / ``PostgresSaver`` in a subgraph or a composite saver).

Storage strategy:

- ``aput(config, checkpoint, metadata, new_versions)`` serialises the
  checkpoint blob via ``self.serde.dumps_typed`` → ``(type, bytes)``, then
  ingests as a Synap memory via ``sdk.memories.create`` with the raw bytes
  base64-encoded in ``document`` and metadata tagging the thread/checkpoint
  coordinates.
- ``aget_tuple(config)`` fetches by thread id and returns the latest
  matching checkpoint.
- ``alist(...)`` yields the filtered matches from a single fetch.
- ``aput_writes`` ingests pending writes with a distinguishing metadata type.
- ``adelete_thread`` warns + no-ops (Synap has no public delete API).

Error policy: writes surface ``SynapIntegrationError`` on SDK failure. Reads
degrade gracefully (return ``None`` / empty iterator).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)


_MARKER = "lg_cp"
_WRITE_MARKER = "lg_cp_writes"


class SynapCheckpointSaver(BaseCheckpointSaver):
    """Synap-backed LangGraph checkpoint saver (best-effort fuzzy)."""

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        *,
        mode: str = "accurate",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapCheckpointSaver requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapCheckpointSaver requires a non-empty user_id")
        super().__init__()
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.mode = mode
        self._delete_warned = False

    # ── put ─────────────────────────────────────────────────────────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = dict(config.get("configurable", {}))
        thread_id = str(configurable.get("thread_id") or "")
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        checkpoint_id = str(checkpoint.get("id") or configurable.get("checkpoint_id") or "")
        parent_checkpoint_id = str(configurable.get("checkpoint_id") or "")

        serde_type, raw_bytes = self.serde.dumps_typed(checkpoint)
        document = _encode(serde_type, raw_bytes, metadata, new_versions)

        meta = {
            _MARKER: True,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
        }

        async with wrap_sdk_errors_async(
            "langgraph.checkpointer.put",
            logger,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        ):
            await self.sdk.memories.create(
                document=document,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                metadata=meta,
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return run_async(self.aput(config, checkpoint, metadata, new_versions))

    # ── put_writes ──────────────────────────────────────────────────────────

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = dict(config.get("configurable", {}))
        thread_id = str(configurable.get("thread_id") or "")
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        checkpoint_id = str(configurable.get("checkpoint_id") or "")

        # Serialise each write individually so deserialisation fidelity survives
        # per-channel. Each write is (channel, value).
        encoded_writes: list[dict[str, Any]] = []
        for channel, value in writes:
            t, b = self.serde.dumps_typed(value)
            encoded_writes.append({
                "channel": channel,
                "type": t,
                "value_b64": base64.b64encode(b).decode("ascii"),
            })

        document = json.dumps({
            "task_id": task_id,
            "task_path": task_path,
            "writes": encoded_writes,
        })

        meta = {
            _WRITE_MARKER: True,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
        }

        async with wrap_sdk_errors_async(
            "langgraph.checkpointer.put_writes",
            logger,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            task_id=task_id,
        ):
            await self.sdk.memories.create(
                document=document,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                metadata=meta,
            )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        run_async(self.aput_writes(config, writes, task_id, task_path))

    # ── get_tuple ───────────────────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        configurable = dict(config.get("configurable", {}))
        thread_id = str(configurable.get("thread_id") or "")
        if not thread_id:
            return None
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        target_cp_id = configurable.get("checkpoint_id")

        tuples = await self._fetch_checkpoint_tuples(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            target_cp_id=str(target_cp_id) if target_cp_id else None,
            limit=10,
        )
        if not tuples:
            return None
        return tuples[0]

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return run_async(self.aget_tuple(config))

    # ── list ────────────────────────────────────────────────────────────────

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        configurable = dict((config or {}).get("configurable", {}))
        thread_id = str(configurable.get("thread_id") or "")
        if not thread_id:
            return
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        before_id: Optional[str] = None
        if before:
            before_id = str(
                (before.get("configurable") or {}).get("checkpoint_id") or ""
            ) or None

        tuples = await self._fetch_checkpoint_tuples(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            target_cp_id=None,
            limit=max(limit or 25, 25),
        )

        count = 0
        for tup in tuples:
            if before_id and tup.config["configurable"].get("checkpoint_id") == before_id:
                break
            if filter and not _metadata_matches_filter(tup.metadata, filter):
                continue
            yield tup
            count += 1
            if limit and count >= limit:
                return

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        async def _collect() -> list[CheckpointTuple]:
            out: list[CheckpointTuple] = []
            async for t in self.alist(config, filter=filter, before=before, limit=limit):
                out.append(t)
            return out

        return iter(run_async(_collect()))

    # ── delete_thread ───────────────────────────────────────────────────────

    async def adelete_thread(self, thread_id: str) -> None:
        if not self._delete_warned:
            logger.warning(
                "SynapCheckpointSaver.delete_thread: Synap has no public delete "
                "API; this is a no-op. This warning fires once."
            )
            self._delete_warned = True

    def delete_thread(self, thread_id: str) -> None:
        run_async(self.adelete_thread(thread_id))

    # ── fetch helpers ───────────────────────────────────────────────────────

    async def _fetch_checkpoint_tuples(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        target_cp_id: Optional[str],
        limit: int,
    ) -> list[CheckpointTuple]:
        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[f"{thread_id}"],
                max_results=limit,
                mode=self.mode,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SynapCheckpointSaver: sdk.fetch failed thread_id=%s error=%s",
                thread_id, exc, exc_info=True,
            )
            return []

        # Collect candidate facts: must be our checkpoint marker + same thread.
        candidates: list[tuple[Any, dict[str, Any]]] = []
        for fact in getattr(response, "facts", None) or []:
            md = getattr(fact, "metadata", None) or {}
            if not md.get(_MARKER):
                continue
            if md.get("thread_id") != thread_id:
                continue
            if checkpoint_ns and md.get("checkpoint_ns") != checkpoint_ns:
                continue
            if target_cp_id and md.get("checkpoint_id") != target_cp_id:
                continue
            candidates.append((fact, md))

        # Most recent first by extracted_at.
        candidates.sort(
            key=lambda pair: getattr(pair[0], "extracted_at", 0) or 0,
            reverse=True,
        )

        tuples: list[CheckpointTuple] = []
        for fact, md in candidates:
            try:
                checkpoint, cp_metadata = _decode(
                    getattr(fact, "content", "") or "",
                    self.serde,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "SynapCheckpointSaver: failed to decode checkpoint "
                    "thread_id=%s id=%s error=%s",
                    thread_id, md.get("checkpoint_id"), exc, exc_info=True,
                )
                continue

            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": md.get("checkpoint_ns") or "",
                    "checkpoint_id": md.get("checkpoint_id") or "",
                }
            }
            parent_cfg: Optional[RunnableConfig] = None
            parent_id = md.get("parent_checkpoint_id")
            if parent_id:
                parent_cfg = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": md.get("checkpoint_ns") or "",
                        "checkpoint_id": parent_id,
                    }
                }
            tuples.append(
                CheckpointTuple(
                    config=cfg,
                    checkpoint=checkpoint,
                    metadata=cp_metadata,
                    parent_config=parent_cfg,
                    pending_writes=None,
                )
            )
        return tuples


# ── encode / decode helpers ─────────────────────────────────────────────────


def _encode(
    serde_type: str,
    raw_bytes: bytes,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> str:
    """Pack the checkpoint + metadata into a single JSON string document."""
    return json.dumps({
        "serde_type": serde_type,
        "checkpoint_b64": base64.b64encode(raw_bytes).decode("ascii"),
        "metadata": _safe_json(dict(metadata or {})),
        "new_versions": _safe_json(dict(new_versions or {})),
    })


def _decode(document: str, serde: Any) -> tuple[Checkpoint, CheckpointMetadata]:
    payload = json.loads(document)
    t = payload["serde_type"]
    b = base64.b64decode(payload["checkpoint_b64"])
    checkpoint = serde.loads_typed((t, b))
    metadata: CheckpointMetadata = payload.get("metadata") or {}  # type: ignore[assignment]
    return checkpoint, metadata


def _safe_json(obj: Any) -> Any:
    """Coerce an arbitrary dict into json-serialisable form (best effort)."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return {k: str(v) for k, v in obj.items()} if isinstance(obj, dict) else str(obj)


def _metadata_matches_filter(
    metadata: CheckpointMetadata,
    filter_: dict[str, Any],
) -> bool:
    md = dict(metadata or {})
    for k, v in filter_.items():
        if md.get(k) != v:
            return False
    return True
