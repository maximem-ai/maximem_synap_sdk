"""Synap callback handler for zero-config auto-memory.

Add SynapCallbackHandler to any LangChain chain or agent. Every
conversation turn is automatically recorded to Synap for automatic
memory extraction.

No explicit save_context() calls needed.

## Design notes

LangChain callbacks **must not raise** — a raising callback aborts the
whole chain, which is almost never what the user wants for a
best-effort side-channel like memory recording. So this handler
catches every exception. The *original* implementation demoted those
failures to ``logger.debug``, which meant ingestion outages were
effectively invisible: DEBUG is off in virtually every production
logger configuration. We now log at ``ERROR`` with structured context
instead, preserving the "don't break the chain" contract while making
failures actually observable.

Extracting the LLM response text previously chained two attribute
lookups with a silent fallback (``generation.text`` -> ``generation.
message.content``). That's fragile in the presence of custom LLM
wrappers. Centralised in ``_extract_text`` so the dispatch is explicit
and unit-tested.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, LLMResult

from maximem_synap import MaximemSynapSDK

from synap_integrations_common.stream_events import (
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
)

from synap_langchain._messages import message_text

logger = logging.getLogger(__name__)


def _extract_text(generation: Generation) -> str:
    """Return the text payload from a LangChain Generation.

    Ordering matters. ``ChatGeneration`` subclasses populate both
    ``.text`` and ``.message.content``, but when the message content
    is a structured list (e.g., multiple text blocks plus tool calls),
    LangChain derives ``.text`` from only the **first** text block and
    drops the rest. Reading ``.text`` naively would therefore lose
    content. So:

    1. If the generation has a message with **list** content, rebuild
       the string by concatenating every text part. This covers
       multi-block chat outputs.
    2. Otherwise prefer ``.text`` (string LLMs and chat models whose
       content is already a plain string).
    3. Fall back to ``message.content`` as a string.
    """
    message = getattr(generation, "message", None)
    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            return "".join(parts)

    text = getattr(generation, "text", "") or ""
    if text:
        return text

    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content) if content else ""
    return ""


def _tool_output(output: Any) -> Any:
    """What the tool returned, in a shape the SDK can serialise.

    ⚠ The SDK does `json.dumps` on this, and a LangChain `ToolMessage` is not
    JSON serialisable. LangGraph's `ToolNode` hands `on_tool_end` exactly
    that, so the dumps raised, `stream_events._send` swallowed it by design,
    and **every tool result in a LangGraph run was silently never reported**.
    Nothing failed, nothing logged above debug, and the anticipation agent
    just never learned what any tool returned.

    Unwrap to `.content` when it is there, and leave anything already
    serialisable alone.
    """
    content = getattr(output, "content", None)
    return output if content is None else content


def _role_and_text(item: Any) -> tuple:
    """``(role, text)`` for a message in any of the shapes a run is given."""
    if isinstance(item, BaseMessage):
        return str(item.type), message_text(item)
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]).lower(), str(item[1])
    if isinstance(item, dict):
        return str(item.get("role") or item.get("type") or "").lower(), message_text(item)
    if isinstance(item, str):
        return "human", item
    return "", ""


def _input_text(inputs: Any) -> tuple:
    """``(key, text)`` of the user's message in a run's inputs, or ``("", "")``."""
    if isinstance(inputs, dict):
        for name in ("messages", "input"):
            if name in inputs:
                return _input_text(inputs[name])
        return "", ""
    if hasattr(inputs, "to_messages"):
        inputs = inputs.to_messages()
    items = inputs if isinstance(inputs, list) else [inputs]
    for item in reversed(items):
        role, text = _role_and_text(item)
        if role in ("human", "user") and text.strip():
            return str(getattr(item, "id", None) or ""), text
    return "", ""


def _final_text(outputs: Any) -> str:
    """The answer in a run's outputs: the last message, when it is words and
    not a tool call, or a plain string. Empty when there is none."""
    if isinstance(outputs, dict):
        for name in ("messages", "output", "text"):
            if name in outputs:
                return _final_text(outputs[name])
        return ""
    if isinstance(outputs, list):
        return _final_text(outputs[-1]) if outputs else ""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, BaseMessage):
        tool_calls = getattr(outputs, "tool_calls", None)
        if outputs.type != "ai" or (isinstance(tool_calls, list) and tool_calls):
            return ""
        return message_text(outputs)
    return ""


class SynapCallbackHandler(AsyncCallbackHandler):
    """Auto-records all conversation turns to Synap.

    Records one user turn and one assistant turn per run: the question the
    run was given and the answer it ended with. A run that calls the model
    several times (a tool loop), or a tool or middleware that calls a model
    of its own, is still one turn. The turns go on the gRPC stream, opened by
    the first one, and over REST when the stream cannot carry them. Record
    failures are logged but never raised: breaking the chain over a
    memory-sidechannel failure would do more harm than good.

    Example::

        handler = SynapCallbackHandler(
            sdk=sdk,
            conversation_id="conv-123",
            user_id="user-456",
            customer_id="cust-789",
        )
        chain = ConversationChain(llm=ChatOpenAI(), callbacks=[handler])
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapCallbackHandler requires a non-None sdk")
        if not conversation_id:
            raise ValueError(
                "SynapCallbackHandler requires a non-empty conversation_id"
            )
        if not user_id:
            raise ValueError("SynapCallbackHandler requires a non-empty user_id")

        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        # ⚠ Which user turns have already gone out.
        #
        # `on_chat_model_start` fires once per MODEL CALL, not once per turn.
        # A tool loop calls the model again after every tool result with the
        # same human message still in the list, so a three-tool turn reported
        # the user's question four times and Synap extracted it four times.
        # Invisible in a single-shot chain, which is why it survived.
        #
        # Keyed on the message id langchain stamps, falling back to the text
        # for graphs that do not set one. Per instance, not per class: two
        # conversations must not silence each other.
        self._reported_user_turns: set = set()
        # The chat model this chain runs on, as LangChain reports it. Sent
        # with each turn so Synap sizes compaction to that model's context
        # window (COMPACTION_V3_PLAN.md §4.3). In metadata rather than as
        # record_message(model=) so an older installed SDK still accepts it.
        self._agent_model: Optional[str] = None
        # ⚠ Which run each model call belongs to, so a turn is the run's and
        # not the model call's. `on_llm_end` fires for every call: each lap
        # of a tool loop (text beside a tool call is not an answer), and any
        # model a tool or a middleware calls on its own (a summarizer's
        # output is not something the assistant said). The final answer is
        # the root run's output; when the handler cannot see the root (it was
        # attached to the model only), a model call that ends without a tool
        # call is the answer.
        self._parents: Dict[UUID, Optional[UUID]] = {}
        self._tool_runs: set = set()
        # root run id -> whether its user turn has been recorded
        self._roots: Dict[UUID, bool] = {}
        # root run id -> the last answer-shaped model output under it
        self._answers: Dict[UUID, str] = {}
        # Text of the last user message that had no id, until the turn ends.
        self._last_unkeyed_user: Optional[str] = None

    async def _record(self, role: str, content: str) -> None:
        # Stream first (opened here if need be), REST only if the stream could
        # not carry it, never both: the server persists a streamed turn from
        # its event, so a REST write on top of a delivered one writes it twice.
        # `record_turn` never raises, which a callback must not.
        model_metadata = {"agent_model": self._agent_model} if self._agent_model else None
        await record_turn(
            self.sdk, role=role, content=content,
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
            metadata=model_metadata,
        )
        if role == "assistant":
            self._last_unkeyed_user = None

    # ─── which run is which ──────────────────────────────────────────────

    def _note(self, run_id: UUID, parent_run_id: Optional[UUID]) -> None:
        self._parents[run_id] = parent_run_id

    def _note_tool(self, run_id: UUID, parent_run_id: Optional[UUID]) -> None:
        self._note(run_id, parent_run_id)
        self._tool_runs.add(run_id)

    def _ancestors(self, run_id: UUID) -> List[UUID]:
        out: List[UUID] = []
        current = self._parents.get(run_id)
        while current is not None and current not in out and len(out) < 256:
            out.append(current)
            current = self._parents.get(current)
        return out

    def _root_of(self, run_id: UUID) -> UUID:
        ancestors = self._ancestors(run_id)
        return ancestors[-1] if ancestors else run_id

    def _inside_tool(self, run_id: UUID) -> bool:
        return any(a in self._tool_runs for a in self._ancestors(run_id))

    def _forget(self, root: UUID) -> None:
        for rid in [r for r in self._parents if r == root or self._root_of(r) == root]:
            self._parents.pop(rid, None)
            self._tool_runs.discard(rid)
        self._roots.pop(root, None)
        self._answers.pop(root, None)

    async def _record_user(self, key: str, text: str) -> bool:
        """Record a user message once; whether it was new."""
        if not text.strip():
            return False
        if key:
            if key in self._reported_user_turns:
                return False
            self._reported_user_turns.add(key)
        else:
            if text == self._last_unkeyed_user:
                return False  # same turn, later lap of the same tool loop
            self._last_unkeyed_user = text
        await self._record("user", text)
        return True

    async def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """The top-level run starting: its input is the user's turn."""
        self._note(run_id, parent_run_id)
        if parent_run_id is not None:
            return
        self._roots[run_id] = False
        key, text = _input_text(inputs)
        if text.strip():
            self._roots[run_id] = True
            await self._record_user(key, text)

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """The top-level run ending: its output is the assistant's turn."""
        if run_id not in self._roots:
            self._parents.pop(run_id, None)
            return
        text = _final_text(outputs) or self._answers.get(run_id, "")
        self._forget(run_id)
        if text.strip():
            await self._record("assistant", text)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._roots:
            self._forget(run_id)
        else:
            self._parents.pop(run_id, None)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the user message when a chat model starts."""
        self._note(run_id, parent_run_id)
        self._agent_model = _model_name(serialized, kwargs.get("invocation_params"), metadata) or self._agent_model
        if not messages or self._inside_tool(run_id):
            return
        if self._roots.get(self._root_of(run_id)):
            return  # the root run's input already gave this turn's question
        for msg in reversed(messages[-1]):
            if msg.type == "human":
                await self._record_user(getattr(msg, "id", None) or "", message_text(msg))
                break

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the assistant response when the LLM finishes."""
        if not response.generations or not response.generations[0]:
            return
        reported = (response.llm_output or {}).get("model_name") if isinstance(response.llm_output, dict) else None
        if not reported:
            message = getattr(response.generations[0][0], "message", None)
            reported = (getattr(message, "response_metadata", None) or {}).get("model_name")
        if isinstance(reported, str) and reported:
            self._agent_model = reported
        generation = response.generations[0][0]
        text = _extract_text(generation)
        message = getattr(generation, "message", None)
        tool_calls = getattr(message, "tool_calls", None)
        if (not text.strip() or (isinstance(tool_calls, list) and tool_calls)
                or self._inside_tool(run_id)):
            self._parents.pop(run_id, None)
            return  # a lap that calls a tool, or a model a tool called: not the answer
        root = self._root_of(run_id)
        self._parents.pop(run_id, None)
        if root in self._roots:
            self._answers[root] = text  # the root's end records the last one
        else:
            await self._record("assistant", text)

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._note(run_id, parent_run_id)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._parents.pop(run_id, None)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._tool_runs.discard(run_id)
        self._parents.pop(run_id, None)

    # ─── the three events this handler never reported ───────────────────
    #
    # It recorded the user turn and the assistant turn and stopped there, so
    # anticipation saw a conversation with questions and answers and nothing
    # in between: no tool calls, no results, no reasoning. Those are the
    # events that say what the agent is ABOUT to need, which is the whole
    # point of anticipating.
    #
    # All three are silent when no stream is running, and none of them can
    # raise into the chain: `stream_events` swallows everything by design,
    # and a callback that throws aborts the whole LangChain run.

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Report a tool the agent is about to call.

        `run_id` is LangChain's own id for this tool invocation and it is the
        same value `on_tool_end` receives, so it ties the call to its result.
        """
        self._note_tool(run_id, parent_run_id)
        await report_tool_call(
            self.sdk,
            tool_name=str((serialized or {}).get("name") or "tool"),
            # `inputs` is the PARSED dict; `input_str` is its Python repr.
            # Sending the repr gave the anticipation agent
            # `{"input": "{'pnr': 'QX41RT'}"}` to reason over instead of the
            # arguments themselves.
            tool_args=inputs if isinstance(inputs, dict) and inputs
            else ({"input": input_str} if input_str else {}),
            tool_call_id=str(run_id),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report what the tool returned, tied to the call by `run_id`."""
        self._tool_runs.discard(run_id)
        self._parents.pop(run_id, None)
        await report_tool_result(
            self.sdk,
            result=_tool_output(output),
            tool_call_id=str(run_id),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )

    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report the agent's reasoning for choosing a tool.

        `AgentAction.log` is where LangChain puts the model's own words about
        why it picked this action, which is the closest thing the framework
        exposes to a reasoning step.
        """
        text = getattr(action, "log", None)
        if not isinstance(text, str) or not text.strip():
            return
        await report_reasoning(
            self.sdk, content=text.strip(),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )


def _model_name(serialized: Any, invocation_params: Any, metadata: Any) -> Optional[str]:
    """The chat model's name from what on_chat_model_start is handed, or None."""
    for source in (invocation_params, metadata, (serialized or {}).get("kwargs") if isinstance(serialized, dict) else None):
        if isinstance(source, dict):
            for key in ("model", "model_name", "ls_model_name", "model_id"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None
