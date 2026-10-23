"""SynapAgentMemory — a CAMEL-AI ``AgentMemory`` backed by Synap.

Why subclass :class:`~camel.memories.ChatHistoryMemory` rather than ``AgentMemory``
directly? CAMEL's ``AgentMemory.get_context()`` is the *sole* source of the model's
input — it returns whatever ``retrieve()`` yields. A memory that returned only Synap
recall would strip the system prompt and the user's current question, so the agent
could not answer. ``ChatHistoryMemory`` already keeps the live conversation;
``SynapAgentMemory`` layers Synap on top:

- ``retrieve()``: the real conversation, with Synap's long-term memories prepended as
  a system context block (fetched using the latest user turn as the query). Once
  Synap has compacted the conversation, every turn before the current one is
  replaced by Synap's compaction and the turns after it (see "What the model
  sees" below).
- ``write_records()``: writes each user and assistant turn as a conversation
  message (on Synap's live stream, opened on the first write, over REST when it
  cannot be), and reports tool calls, tool results and reasoning on the stream.

Error policy:
- recall (read) degrades: a Synap failure logs at ERROR and returns the plain local
  history — the turn still runs.
- persistence (write) is best-effort by default (``on_error="fallback"``): it runs
  inside the agent loop, so a transient write failure must not discard the
  model's just-produced response. Set ``on_error="raise"`` for strict environments.
- ``clear()`` is a silent local reset — Synap has no public delete API, and CAMEL
  calls ``clear()`` on every construction/summarization, so warning would be noise.

## What a write sends

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

**Turns are conversation messages, written once.** Each user and assistant
turn goes through :func:`~synap_integrations_common.record_turn`: on the stream
(the first write opens it; the server persists ``user_message`` /
``assistant_message`` from the event itself), otherwise
``sdk.conversation.record_message`` over REST. Never both. Because the server
keeps a real conversation either way, it compacts it, promotes it to long-term
memory ingestion, and the SDK's local history search can archive it.

This replaced a REST ``memories.create`` of the accumulated transcript. That
path fed extraction directly but left no conversation behind, so compaction and
history search had nothing to work on. It is gone rather than kept alongside:
the server already promotes recorded messages to long-term ingestion, so
writing both would extract the same text twice.

Tool calls, tool results and reasoning are stream-only: they are anticipation
hints, not turns, and are silent no-ops when the stream cannot open.

Three kinds of record are never written as turns: the compaction CAMEL's own
summarizer writes back (an assistant ``[CONTEXT_SUMMARY] ...`` and the user
message it rewrites as ``Based on the previous CONTEXT_SUMMARY, continue with
my current message: ...``), because the real turns were already written and the
summary is built from a context that includes Synap's own recall block; and
Synap's history block, for the same reason.

## What the model sees

``ChatAgent`` reads its input from ``get_context()``, which is
``context_creator.create_context(retrieve())``, so ``retrieve()`` is the seam.
When :func:`~synap_integrations_common.fetch_compacted_history` returns a
compaction, :func:`~synap_integrations_common.splice_history` swaps every record
before the current turn for: CAMEL's system messages, Synap's compaction block,
and Synap's turns after it. CAMEL's own summarizer stays installed; on a
history that short it does not fire, and when Synap has no compaction yet
(``None``) the records are left untouched.

``ScoreBasedContextCreator`` orders everything except the first system record
by timestamp, so every record built here is stamped to sort exactly where it was
placed: a recall block stamped "now" would land after the user's question.

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

import asyncio
import dataclasses
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4, uuid5

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
    CompactedHistory,
    SynapIntegrationError,
    fetch_compacted_history,
    is_synap_history,
    report_reasoning,
    report_tool_call,
    record_turn,
    report_tool_result,
    run_async,
    splice_history,
)

logger = logging.getLogger(__name__)

# Fixed namespace for the agent-id derived conversation id. Never change it:
# doing so would remap every existing agent to a new conversation.
_CAMEL_NAMESPACE = UUID("6f1c4f0e-3b57-5c1e-9a0b-7d2f4c8e1a35")

_OPEN = "<synap_long_term_memory>"
_CLOSE = "</synap_long_term_memory>"

_SYSTEM_ROLES = (OpenAIBackendRole.SYSTEM, OpenAIBackendRole.DEVELOPER)
_TOOL_ROLES = (OpenAIBackendRole.FUNCTION, OpenAIBackendRole.TOOL)

# What ChatAgent._update_memory_with_summary writes after CAMEL's own
# summarization: an assistant message starting with the first, and the last user
# message again under the second.
_SUMMARY_MARK = "[CONTEXT_SUMMARY]"
_SUMMARY_USER_PREFIX = "Based on the previous CONTEXT_SUMMARY, continue with my current message: "
# The block synap_st_system_message puts in the system message.
_ST_BLOCK = re.compile(
    r"<synap_short_term_context>.*?</synap_short_term_context>\s*", re.DOTALL
)
# Records built here are stamped this far apart so the context creator's
# timestamp sort keeps the order they were placed in.
_STAMP_STEP = 1e-5


def _is_internal(text: str) -> bool:
    """Text CAMEL or Synap generated about the conversation, not a turn in it."""
    stripped = (text or "").lstrip()
    return (
        stripped.startswith(_SUMMARY_MARK)
        or stripped.startswith(_SUMMARY_USER_PREFIX)
        or stripped.startswith(_OPEN)
        or is_synap_history(stripped)
    )


def _unprefixed(text: str) -> str:
    """The user's own words, without CAMEL's summarizer prefix."""
    return text[len(_SUMMARY_USER_PREFIX):] if text.startswith(_SUMMARY_USER_PREFIX) else text


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
        conversation_id: Synap conversation this agent's turns are written to.
            Defaults to ``camel-{agent_id}`` (a random ``camel-<hex>`` until
            ``ChatAgent`` assigns its id), which is stable for the life of the
            agent but new for every agent — pass your own id to continue one
            conversation across agents or processes. Pass the same id you gave
            :func:`~synap_camel_ai.synap_st_system_message` to keep short-term
            context and the reported turns on one conversation.
        agent_id: Optional CAMEL agent id. ``ChatAgent`` overwrites this with its
            own id after construction; it seeds the default conversation id.
        mode: Synap fetch mode — ``"accurate"`` (default) or ``"fast"``.
        max_results: Cap on recalled memories per turn.
        token_limit: Accepted for compatibility and handed to CAMEL's context
            creator, which no longer filters on it. What decides when CAMEL
            summarizes is ``ChatAgent(token_limit=...)``; set it there.
        token_counter: Override the token counter. Defaults to an offline
            ``OpenAITokenCounter`` for GPT-4o-mini (no API key needed).
        window_size: CAMEL recency window over local history (``None`` = unbounded).
        storage: CAMEL key-value storage for local history (``None`` = in-memory).
        on_error: ``"fallback"`` (default) makes the turn write best-effort;
            ``"raise"`` raises :class:`SynapIntegrationError` when a turn could
            not be written.

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
        self._fallback_doc = str(uuid4())
        # ⚠ Which memory records have already been written.
        #
        # ``write_records`` is normally called once per message, so a turn is
        # written once. ``ChatAgent.load_memory`` is the exception: it replays
        # another memory's records into this one, and without this guard every
        # historical turn would be written again — and extracted again — as if
        # the user had just said it. Keyed on the record uuid CAMEL stamps, and
        # kept across ``clear()`` because a replay can happen after one. Per
        # instance, not per class: two agents must not silence each other.
        self._reported: set = set()

    # ── default conversation id (lazy: ChatAgent sets agent_id AFTER __init__)

    @property
    def _doc_id(self) -> str:
        aid = self.agent_id
        # The SDK requires a UUID conversation_id; uuid5 keeps one agent id
        # mapping to the same conversation across restarts.
        return str(uuid5(_CAMEL_NAMESPACE, str(aid))) if aid else self._fallback_doc

    @property
    def _conversation_id(self) -> str:
        return self.conversation_id or self._doc_id

    # ── writes: preserve local history, then persist completed turns ─────────

    def write_records(self, records: List[MemoryRecord]) -> None:
        # Local history first — the agent's own context depends on it.
        super().write_records(records)
        if not self._has_work(records):
            return
        coro = self._awrite(records)
        try:
            failed = run_async(coro)
        except Exception as exc:  # noqa: BLE001 — the bridge, not the SDK
            # ⚠ `run_async` can fail on its own: there is a loop running that
            # `nest_asyncio` cannot patch, and no coroutine ever starts. That is
            # not a `SynapIntegrationError`, so it would travel straight out of
            # `write_records` into `ChatAgent.step` and end the run — the one
            # thing `on_error="fallback"` exists to prevent.
            close = getattr(coro, "close", None)
            if callable(close):
                # A coroutine the bridge never started would warn on GC.
                close()
            logger.error(
                "SynapAgentMemory: could not write the turn "
                "conversation_id=%s user_id=%s error=%s",
                self._conversation_id,
                self.user_id,
                exc,
                exc_info=True,
            )
            if self.on_error == "raise":
                raise
            return
        if failed and self.on_error == "raise":
            raise SynapIntegrationError(
                "camel_ai.write_records",
                f"{failed} turn(s) were not written to Synap",
                {"conversation_id": self._conversation_id, "user_id": self.user_id},
            )

    def _has_work(self, records: List[MemoryRecord]) -> bool:
        """Whether this write needs the async bridge at all.

        ``run_async`` applies ``nest_asyncio`` to whatever loop it finds, which
        patches that loop's class globally. Anything but a system prompt can be
        reported (the first one opens the stream), so only a write of system
        prompts, or of records already written, skips the bridge.
        """
        return any(
            record.uuid not in self._reported
            and record.role_at_backend not in _SYSTEM_ROLES
            for record in records
        )

    async def _awrite(self, records: List[MemoryRecord]) -> int:
        """Write one batch of records. Returns how many turns were not written."""
        failed = 0
        for record in records:
            if record.uuid in self._reported:
                continue
            self._reported.add(record.uuid)
            role = record.role_at_backend
            message = record.message
            text = (getattr(message, "content", "") or "").strip()
            if role == OpenAIBackendRole.USER:
                if text and not _is_internal(text) and not await self._record_turn("user", text):
                    failed += 1
            elif role == OpenAIBackendRole.ASSISTANT:
                await self._report_assistant(message)
                if text and not _is_internal(text) and not await self._record_turn("assistant", text):
                    failed += 1
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
        return failed

    async def _report_assistant(self, message: Any) -> None:
        """The reasoning and the tool calls carried on one assistant message.

        Stream only: both are silent no-ops without ``listen()``.
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

    async def _record_turn(self, role: str, content: str) -> bool:
        return await record_turn(
            self.sdk,
            role=role,
            content=content,
            conversation_id=self._conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    # ── reads: local conversation + prepended Synap recall ───────────────────

    def retrieve(self) -> List[ContextRecord]:
        local = super().retrieve()
        query = self._last_user_text(local)
        if not query:
            return local
        try:
            block, history = run_async(self._arecall(query))
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapAgentMemory: recall fetch failed user_id=%s error=%s",
                self.user_id,
                exc,
                exc_info=True,
            )
            return self._stamped(local)
        records = self._splice(local, history) if history else local
        if block:
            # role_at_backend (not the BaseMessage factory) drives the emitted
            # role, so build the block with make_user_message and tag it SYSTEM
            # at the backend: a clean system context block that preserves
            # user/assistant alternation.
            recall = self._record(
                "system", f"{_OPEN}\n{block}\n{_CLOSE}", score=1.0
            )
            split = self._first_non_system_index(records)
            records = records[:split] + [recall] + records[split:]
        return self._stamped(records)

    async def _arecall(self, query: str) -> Tuple[str, Optional[CompactedHistory]]:
        """Synap's recall for the latest question, and its compaction if any."""
        history = asyncio.ensure_future(
            fetch_compacted_history(self.sdk, self._conversation_id)
        )
        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[query],
                max_results=self.max_results,
                mode=self.mode,
                include_conversation_context=False,
            )
        except BaseException:
            history.cancel()
            raise
        block = (getattr(response, "formatted_context", None) or "").strip()
        return block, await history

    # ── the seam: Synap's compaction in place of CAMEL's earlier turns ───────

    def _splice(
        self, records: List[ContextRecord], history: CompactedHistory
    ) -> List[ContextRecord]:
        def role_of(record: ContextRecord) -> str:
            role = record.memory_record.role_at_backend
            if role in _SYSTEM_ROLES:
                return "system"
            if role == OpenAIBackendRole.USER:
                return "user"
            if role == OpenAIBackendRole.ASSISTANT:
                return "assistant"
            return "tool"

        def text_of(record: ContextRecord) -> str:
            # The summarizer's rewritten user message is the user's own question
            # that Synap already holds; match it by what the user said.
            return _unprefixed(record.memory_record.message.content or "")

        # synap_st_system_message put a snapshot of the compaction into the
        # system message at construction; the live one replaces it.
        cleaned = [self._without_st_block(r) for r in records]
        return splice_history(
            cleaned,
            history,
            role_of=role_of,
            text_of=text_of,
            make_message=lambda role, text: self._record(role, text),
        )

    @staticmethod
    def _without_st_block(record: ContextRecord) -> ContextRecord:
        mr = record.memory_record
        if mr.role_at_backend not in _SYSTEM_ROLES:
            return record
        content = mr.message.content or ""
        if "<synap_short_term_context>" not in content:
            return record
        message = dataclasses.replace(mr.message, content=_ST_BLOCK.sub("", content))
        return record.model_copy(
            update={"memory_record": mr.model_copy(update={"message": message})}
        )

    def _record(self, role: str, text: str, *, score: float = 0.5) -> ContextRecord:
        """A context record of ``role`` ("system", "user" or "assistant")."""
        if role == "assistant":
            message = BaseMessage.make_assistant_message("assistant", text)
            backend = OpenAIBackendRole.ASSISTANT
        elif role == "user":
            message = BaseMessage.make_user_message("user", text)
            backend = OpenAIBackendRole.USER
        else:
            message = BaseMessage.make_user_message("system", text)
            backend = OpenAIBackendRole.SYSTEM
        return ContextRecord(
            memory_record=MemoryRecord(message=message, role_at_backend=backend),
            score=score,
        )

    @staticmethod
    def _stamped(records: List[ContextRecord]) -> List[ContextRecord]:
        """The records with timestamps that sort in list order.

        ``ScoreBasedContextCreator`` keeps the first system record first and
        sorts every other record by timestamp. A record built now would sort
        after the user's question however early it was placed, so walking back
        from the end each timestamp is pulled below the next one.
        """
        out = list(records)
        ceiling = None
        for i in range(len(out) - 1, -1, -1):
            if ceiling is not None and out[i].timestamp >= ceiling:
                out[i] = out[i].model_copy(update={"timestamp": ceiling - _STAMP_STEP})
            ceiling = out[i].timestamp
        return out

    def _last_user_text(self, records: List[ContextRecord]) -> str:
        for record in reversed(records):
            mr = record.memory_record
            if mr.role_at_backend == OpenAIBackendRole.USER:
                text = _unprefixed((mr.message.content or "").strip())
                if text and not _is_internal(text):
                    return text
        return ""

    def _first_non_system_index(self, records: List[ContextRecord]) -> int:
        for i, record in enumerate(records):
            if record.memory_record.role_at_backend not in _SYSTEM_ROLES:
                return i
        return len(records)

    # clear(): inherited. A silent local reset — Synap has no public delete
    # API, and CAMEL calls clear() on every construction/summarization.
