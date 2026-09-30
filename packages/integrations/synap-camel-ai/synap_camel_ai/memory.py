"""SynapAgentMemory — a CAMEL-AI ``AgentMemory`` backed by Synap.

Why subclass :class:`~camel.memories.ChatHistoryMemory` rather than ``AgentMemory``
directly? CAMEL's ``AgentMemory.get_context()`` is the *sole* source of the model's
input — it returns whatever ``retrieve()`` yields. A memory that returned only Synap
recall would strip the system prompt and the user's current question, so the agent
could not answer. ``ChatHistoryMemory`` already keeps the live conversation;
``SynapAgentMemory`` layers Synap on top:

- ``retrieve()``: the real conversation, with Synap's long-term memories prepended as
  a system context block (fetched using the latest user turn as the query).
- ``write_records()``: reports the turn on Synap's live stream when one is open, and
  otherwise ingests the accumulated transcript for server-side extraction.

Error policy:
- recall (read) degrades: a Synap failure logs at ERROR and returns the plain local
  history — the turn still runs.
- persistence (write) is best-effort by default (``on_error="fallback"``): it runs
  inside the agent loop, so a transient ingestion failure must not discard the
  model's just-produced response. Set ``on_error="raise"`` for strict environments.
- ``clear()`` is a silent local reset — Synap has no public delete API, and CAMEL
  calls ``clear()`` on every construction/summarization, so warning would be noise.

## Reporting the turn on the live stream

CAMEL has no callback protocol. ``ChatAgent`` exposes exactly one usage hook
(``on_request_usage``) and nothing for turns, tools or reasoning, so the memory's
own ``write_records`` is the only place an integration can see what the agent is
doing — and it happens to see *everything*, because ``ChatAgent.update_memory`` is
how every one of those things is recorded:

=========================  =================================================
what CAMEL writes          what goes on the stream
=========================  =================================================
``USER`` record            the user turn
``ASSISTANT`` record with  a ``tool_call`` per entry, carrying the parsed
``meta_dict["tool_calls"]``  arguments and the provider's ``tool_call_id``
``FUNCTION`` record        the ``tool_result``, under that same id
``ASSISTANT`` record with  the reasoning step (when the provider returned
``reasoning_content``      one) and then the assistant turn
=========================  =================================================

**Stream first, REST only as a fallback, never both.** The server persists
``user_message`` and ``assistant_message`` from the stream itself
(``grpc/servicer.py``) and extracts memories from the conversation it builds. The
transcript ingest below does the same job over REST, so doing both writes the
conversation twice and extracts it twice.
:func:`~synap_integrations_common.stream_events.report_turn` returns whether the
event actually went out, which is what makes the choice decidable rather than
guessed: with no stream open, or with a send that failed, ``_persist`` runs exactly
as it always has.

Two shapes in the table above are traps, and both fail *silently* rather than
loudly:

- CAMEL stores tool arguments the way OpenAI sends them, as a JSON **string** under
  ``function.arguments``. ``report_tool_call`` keeps a dict and drops anything else,
  so passing the string through reports the call with no arguments at all.
- A tool result is whatever the tool returned, and the SDK JSON-encodes it. A
  ``ToolResult`` (CAMEL's text-plus-images wrapper) is not encodable, and
  ``stream_events`` swallows the failure by design, so the result would never be
  reported and nothing would say so.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
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
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    run_async,
    stream_is_active,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

# Metadata marker on every Synap doc this memory creates.
_MARKER = "camel_ai_conversation"
_OPEN = "<synap_long_term_memory>"
_CLOSE = "</synap_long_term_memory>"

_CONVERSATIONAL = (OpenAIBackendRole.USER, OpenAIBackendRole.ASSISTANT)
_SYSTEM_ROLES = (OpenAIBackendRole.SYSTEM, OpenAIBackendRole.DEVELOPER)
_TOOL_ROLES = (OpenAIBackendRole.FUNCTION, OpenAIBackendRole.TOOL)


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    """The tool calls CAMEL recorded on an assistant message.

    ``_record_assistant_tool_calls_from_requests`` puts them in ``meta_dict``
    in OpenAI's wire shape, which is also where the streaming path
    (``_record_assistant_tool_calls_message``) puts them.
    """
    meta = getattr(message, "meta_dict", None)
    calls = meta.get("tool_calls") if isinstance(meta, dict) else None
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _tool_args(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The arguments of one recorded tool call, as a dict.

    ⚠ CAMEL stores ``function.arguments`` as the JSON **string** the provider
    sent. ``report_tool_call`` keeps a dict and drops everything else, so
    handing the string over reports the call with no arguments — the tool name
    arrives, what it was asked for does not, and nothing fails. A string that
    is not JSON at all is passed through under one key rather than dropped:
    the model wrote it, so it says something.
    """
    function = call.get("function")
    raw = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"arguments": raw}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


def _tool_result(result: Any) -> Any:
    """What a tool returned, in a shape the SDK can put on the wire.

    The SDK calls ``json.dumps`` on a tool result and ``stream_events``
    swallows whatever that raises, so an un-encodable result is not an error
    anywhere — it is a result that never arrives. ``ToolResult``, which is what
    every CAMEL tool returning images hands back, is one of those. Falling back
    to ``str`` is what CAMEL's own ``_serialize_tool_result`` does before it
    puts a result in front of the model, so the anticipation agent reads what
    the model read.
    """
    if isinstance(result, str):
        return result
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        return str(result)
    return result


class SynapAgentMemory(ChatHistoryMemory):
    """CAMEL ``AgentMemory`` that augments the local conversation with Synap.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        conversation_id: Synap conversation this agent's turns belong to, used
            only when a live stream is open. Defaults to the document id, which
            is derived from the CAMEL agent id and is stable for the life of the
            agent. Pass the same id you gave
            :func:`~synap_camel_ai.synap_st_system_message` to keep short-term
            context and the reported turns on one conversation.
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
        conversation_id: str = "",
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
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_results = max_results
        self.on_error = on_error
        self._turns: List[dict] = []
        self._fallback_doc = f"camel-{uuid4().hex}"
        # ⚠ Which memory records have already gone out on the stream.
        #
        # ``write_records`` is normally called once per message, so a turn is
        # reported once. ``ChatAgent.load_memory`` is the exception: it replays
        # another memory's records into this one, and without this guard every
        # historical turn would be reported again — and extracted again — as if
        # the user had just said it. Keyed on the record uuid CAMEL stamps, and
        # kept across ``clear()`` because a replay can happen after one. Per
        # instance, not per class: two agents must not silence each other.
        self._reported: set = set()

    # ── document id (lazy: ChatAgent sets agent_id AFTER __init__) ───────────

    @property
    def _doc_id(self) -> str:
        aid = self.agent_id
        return f"camel-{aid}" if aid else self._fallback_doc

    @property
    def _conversation_id(self) -> str:
        return self.conversation_id or self._doc_id

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

        # Report the whole write on the live stream, if one is open. Returns
        # whether the stream took responsibility for it.
        streamed = self._report(records)

        # Persist the accumulated transcript once per completed turn (triggered by
        # the assistant message). Never per-message to a stable doc (that would
        # clobber), and never at construction (system messages are skipped above).
        #
        # ⚠ Not when the stream already carried the turn. The server persists
        # `user_message` / `assistant_message` from the stream itself and
        # extracts from the conversation it builds, so doing both writes this
        # conversation twice and extracts it twice.
        if (
            any(r.role_at_backend == OpenAIBackendRole.ASSISTANT for r in records)
            and not streamed
        ):
            self._persist()

    # ── the live stream ──────────────────────────────────────────────────────

    def _report(self, records: List[MemoryRecord]) -> bool:
        """Report a write on Synap's live stream.

        Returns whether the stream has taken responsibility for this write, and
        therefore whether the REST transcript ingest should stand down. ``False``
        whenever there is no stream, so a caller who never opened one behaves
        exactly as they did before.

        Never raises: this runs inside ``ChatAgent.step``, and no amount of
        telemetry is worth ending somebody's agent run.
        """
        if not stream_is_active(self.sdk):
            return False
        coro = self._areport(records)
        try:
            return bool(run_async(coro))
        except Exception as exc:  # noqa: BLE001 — a write must never break a step
            close = getattr(coro, "close", None)
            if callable(close):
                # A coroutine the bridge never started would warn on GC.
                close()
            logger.debug(
                "SynapAgentMemory: stream report failed: %s", exc, exc_info=True
            )
            return False

    async def _areport(self, records: List[MemoryRecord]) -> bool:
        """Send one write's events. Returns whether every turn in it went out."""
        owned = True
        for record in records:
            if record.uuid in self._reported:
                continue
            self._reported.add(record.uuid)
            role = record.role_at_backend
            message = record.message
            if role == OpenAIBackendRole.USER:
                text = (getattr(message, "content", "") or "").strip()
                if text:
                    owned = await self._report_turn("user", text) and owned
            elif role == OpenAIBackendRole.ASSISTANT:
                await self._report_assistant(message)
                text = (getattr(message, "content", "") or "").strip()
                if text:
                    owned = await self._report_turn("assistant", text) and owned
            elif role in _TOOL_ROLES:
                await report_tool_result(
                    self.sdk,
                    result=_tool_result(getattr(message, "result", None)),
                    tool_name=str(getattr(message, "func_name", "") or ""),
                    tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
                    conversation_id=self._conversation_id,
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                )
        return owned

    async def _report_assistant(self, message: Any) -> None:
        """The reasoning and the tool calls carried on one assistant message.

        Reasoning first: it is what the model did before it decided anything.
        The tool calls then go out under the provider's own ``id``, which is the
        same id CAMEL puts on the ``FunctionCallingMessage`` holding the result,
        so the two halves of one invocation can be paired.
        """
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if reasoning:
            await report_reasoning(
                self.sdk,
                content=reasoning,
                conversation_id=self._conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
        for call in _tool_calls(message):
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            await report_tool_call(
                self.sdk,
                tool_name=str(name or "tool"),
                tool_args=_tool_args(call),
                tool_call_id=str(call.get("id") or ""),
                conversation_id=self._conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )

    async def _report_turn(self, role: str, content: str) -> bool:
        return await report_turn(
            self.sdk,
            role=role,
            content=content,
            conversation_id=self._conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    def _persist(self) -> None:
        transcript = self._render_transcript()
        if not transcript:
            return
        coro = self._apersist(transcript)
        try:
            run_async(coro)
        except SynapIntegrationError:
            # wrap_sdk_errors_async already logged at ERROR. In-loop write: swallow
            # by default so a transient outage doesn't discard the model response.
            if self.on_error == "raise":
                raise
        except Exception as exc:  # noqa: BLE001 — the bridge, not the SDK
            # ⚠ `run_async` can fail on its own: there is a loop running that
            # `nest_asyncio` cannot patch, and no coroutine ever starts. That is
            # not a `SynapIntegrationError`, so it used to travel straight out
            # of `write_records` into `ChatAgent.step` and end the run — the one
            # thing `on_error="fallback"` exists to prevent. Same policy as an
            # ingestion failure: logged, and propagated only when the caller
            # asked for strict.
            close = getattr(coro, "close", None)
            if callable(close):
                # A coroutine the bridge never started would warn on GC.
                close()
            logger.error(
                "SynapAgentMemory: could not run the transcript ingest "
                "doc_id=%s user_id=%s error=%s",
                self._doc_id,
                self.user_id,
                exc,
                exc_info=True,
            )
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
