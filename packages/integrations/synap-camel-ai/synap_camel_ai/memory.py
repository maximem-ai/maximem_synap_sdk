"""SynapAgentMemory — a CAMEL-AI ``AgentMemory`` backed by Synap.

Why subclass :class:`~camel.memories.ChatHistoryMemory` rather than ``AgentMemory``
directly? CAMEL's ``AgentMemory.get_context()`` is the *sole* source of the model's
input — it returns whatever ``retrieve()`` yields. A memory that returned only Synap
recall would strip the system prompt and the user's current question, so the agent
could not answer. ``ChatHistoryMemory`` already keeps the live conversation;
``SynapAgentMemory`` layers Synap on top:

- ``retrieve()``: the real conversation, with Synap's long-term memories prepended as
  a system context block (fetched using the latest user turn as the query).
- ``write_records()``: after CAMEL stores a completed turn locally, the accumulated
  transcript is ingested into Synap for server-side extraction.

Error policy:
- recall (read) degrades: a Synap failure logs at ERROR and returns the plain local
  history — the turn still runs.
- persistence (write) is best-effort by default (``on_error="fallback"``): it runs
  inside the agent loop, so a transient ingestion failure must not discard the
  model's just-produced response. Set ``on_error="raise"`` for strict environments.
- ``clear()`` is a silent local reset — Synap has no public delete API, and CAMEL
  calls ``clear()`` on every construction/summarization, so warning would be noise.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import uuid4

from camel.memories import (
    ChatHistoryMemory,
    ContextRecord,
    MemoryRecord,
    ScoreBasedContextCreator,
)
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

# Metadata marker on every Synap doc this memory creates.
_MARKER = "camel_ai_conversation"
_OPEN = "<synap_long_term_memory>"
_CLOSE = "</synap_long_term_memory>"

_CONVERSATIONAL = (OpenAIBackendRole.USER, OpenAIBackendRole.ASSISTANT)
_SYSTEM_ROLES = (OpenAIBackendRole.SYSTEM, OpenAIBackendRole.DEVELOPER)


class SynapAgentMemory(ChatHistoryMemory):
    """CAMEL ``AgentMemory`` that augments the local conversation with Synap.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        customer_id: Optional customer/org scope; empty means customer-less.
        agent_id: Optional CAMEL agent id. ``ChatAgent`` overwrites this with its
            own id after construction; it seeds the Synap document id.
        mode: Synap fetch mode — ``"accurate"`` (default) or ``"fast"``.
        max_results: Cap on recalled memories per turn.
        token_limit: Context-window token budget for CAMEL's context creator.
        token_counter: Override the token counter. Defaults to an offline
            ``OpenAITokenCounter`` for GPT-4o-mini (no API key needed).
        window_size: CAMEL recency window over local history (``None`` = unbounded).
        storage: CAMEL key-value storage for local history (``None`` = in-memory).
        on_error: ``"fallback"`` (default) makes the persistence write best-effort;
            ``"raise"`` propagates :class:`SynapIntegrationError`.

    Attach via ``ChatAgent(memory=SynapAgentMemory(...))`` — the constructor path
    only. Do **not** assign ``agent.memory = ...`` after construction; that setter
    re-runs retrieve/clear/write and would amplify the injected context.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        *,
        customer_id: str = "",
        agent_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 5,
        token_limit: int = 4096,
        token_counter=None,
        window_size: Optional[int] = None,
        storage=None,
        on_error: str = "fallback",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapAgentMemory requires a non-None sdk")
        if not user_id or not str(user_id).strip():
            raise ValueError("SynapAgentMemory requires a non-empty user_id")
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                f"SynapAgentMemory: on_error must be 'fallback' or 'raise', "
                f"got {on_error!r}"
            )
        counter = token_counter or OpenAITokenCounter(ModelType.GPT_4O_MINI)
        context_creator = ScoreBasedContextCreator(counter, token_limit)
        super().__init__(
            context_creator,
            storage=storage,
            window_size=window_size,
            agent_id=agent_id,
        )
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.mode = mode
        self.max_results = max_results
        self.on_error = on_error
        self._turns: List[dict] = []
        self._fallback_doc = f"camel-{uuid4().hex}"

    # ── document id (lazy: ChatAgent sets agent_id AFTER __init__) ───────────

    @property
    def _doc_id(self) -> str:
        aid = self.agent_id
        return f"camel-{aid}" if aid else self._fallback_doc

    # ── writes: preserve local history, then persist completed turns ─────────

    def write_records(self, records: List[MemoryRecord]) -> None:
        # Local history first — the agent's own context depends on it.
        super().write_records(records)

        for record in records:
            if record.role_at_backend in _CONVERSATIONAL:
                role = (
                    "user"
                    if record.role_at_backend == OpenAIBackendRole.USER
                    else "assistant"
                )
                self._turns.append(
                    {"role": role, "content": record.message.content or ""}
                )

        # Persist the accumulated transcript once per completed turn (triggered by
        # the assistant message). Never per-message to a stable doc (that would
        # clobber), and never at construction (system messages are skipped above).
        if any(r.role_at_backend == OpenAIBackendRole.ASSISTANT for r in records):
            self._persist()

    def _persist(self) -> None:
        transcript = self._render_transcript()
        if not transcript:
            return
        try:
            run_async(self._apersist(transcript))
        except SynapIntegrationError:
            # wrap_sdk_errors_async already logged at ERROR. In-loop write: swallow
            # by default so a transient outage doesn't discard the model response.
            if self.on_error == "raise":
                raise

    async def _apersist(self, transcript: str) -> None:
        async with wrap_sdk_errors_async(
            "camel_ai.write_records",
            logger,
            doc_id=self._doc_id,
            user_id=self.user_id,
        ):
            await self.sdk.memories.create(
                document=transcript,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                document_type="ai-chat-conversation",
                document_id=self._doc_id,
                metadata={_MARKER: True},
            )

    def _render_transcript(self) -> str:
        lines = []
        for turn in self._turns:
            content = (turn["content"] or "").strip()
            if not content:
                continue
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n\n".join(lines)

    # ── reads: local conversation + prepended Synap recall ───────────────────

    def retrieve(self) -> List[ContextRecord]:
        local = super().retrieve()
        query = self._last_user_text(local)
        if not query:
            return local
        try:
            block = run_async(self._arecall(query))
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapAgentMemory: recall fetch failed user_id=%s error=%s",
                self.user_id,
                exc,
                exc_info=True,
            )
            return local
        if not block:
            return local
        # role_at_backend (not the BaseMessage factory) drives the emitted role, so
        # build the block with make_user_message and tag it SYSTEM at the backend —
        # a clean system context block that preserves user/assistant alternation.
        recall = ContextRecord(
            memory_record=MemoryRecord(
                message=BaseMessage.make_user_message(
                    "system", f"{_OPEN}\n{block}\n{_CLOSE}"
                ),
                role_at_backend=OpenAIBackendRole.SYSTEM,
            ),
            score=1.0,
        )
        split = self._first_non_system_index(local)
        return local[:split] + [recall] + local[split:]

    async def _arecall(self, query: str) -> str:
        response = await self.sdk.fetch(
            user_id=self.user_id,
            customer_id=self.customer_id or None,
            search_query=[query],
            max_results=self.max_results,
            mode=self.mode,
            include_conversation_context=False,
        )
        return (getattr(response, "formatted_context", None) or "").strip()

    def _last_user_text(self, records: List[ContextRecord]) -> str:
        for record in reversed(records):
            mr = record.memory_record
            if mr.role_at_backend == OpenAIBackendRole.USER:
                text = (mr.message.content or "").strip()
                if text and not text.startswith(_OPEN):
                    return text
        return ""

    def _first_non_system_index(self, records: List[ContextRecord]) -> int:
        for i, record in enumerate(records):
            if record.memory_record.role_at_backend not in _SYSTEM_ROLES:
                return i
        return len(records)

    # ── clear: silent local reset (Synap has no public delete API) ───────────

    def clear(self) -> None:
        super().clear()
        self._turns.clear()
