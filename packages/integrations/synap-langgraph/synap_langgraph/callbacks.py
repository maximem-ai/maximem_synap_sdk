"""Report a LangGraph run on Synap's live stream.

LangGraph runs on LangChain's callback machinery. A handler passed in
``config={"callbacks": [...]}`` reaches the chat model and the tool nodes
inside a compiled graph, so :class:`synap_langchain.SynapCallbackHandler` is
the base here rather than a second copy of it. Measured against
``langgraph==1.2.11`` with a ``StateGraph`` + ``ToolNode`` loop:
``on_chat_model_start``, ``on_llm_end``, ``on_tool_start`` and
``on_tool_end`` all fire, and ``on_tool_start`` / ``on_tool_end`` share a
``run_id``, which is what ties a tool call to its result.

Two things behave differently in a graph than in a chain, and they are the
whole reason this subclass exists. Both lose an event outright.

**Reasoning never arrives as an agent action.** ``on_agent_action`` is an
``AgentExecutor`` callback. It does not fire once in a LangGraph run, so the
reasoning event the base handler reports is dead weight here. A graph's
reasoning arrives instead on the AIMessage, either as a standard
``{"type": "reasoning", "reasoning": ...}`` content block or, for the
providers that use it, as ``additional_kwargs["reasoning_content"]``.
``AIMessage.content_blocks`` normalises both into the first shape, which is
what :func:`_reasoning_steps` reads in ``on_llm_end``.

**``ToolNode`` hands ``on_tool_end`` the whole ``ToolMessage``,** and the SDK
JSON-encodes a tool result before sending it. ``json.dumps`` cannot encode a
``ToolMessage``, so passing it straight through means the send throws, the
stream helper swallows it, and every tool result in the run is silently never
reported. :func:`_tool_output` unwraps it first.

The user and assistant turns are the base handler's, and a ReAct graph is why
it works the way it does: the model is called again after every tool result
with the human message still in the list, so a turn is read from the root
run's input and output (once per turn), not from each model call. See
:class:`synap_langchain.SynapCallbackHandler`.

Everything in here obeys the same two rules as the base handler: silent when
no stream is running, and never raises into the graph. A callback that
throws aborts the whole run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    report_reasoning,
    report_tool_result,
)
from synap_langchain.callbacks import SynapCallbackHandler

logger = logging.getLogger(__name__)


def _tool_output(output: Any) -> Any:
    """What a tool returned, in a shape the SDK can put on the wire.

    ``ToolNode`` hands ``on_tool_end`` the ``ToolMessage`` it just built, and
    the SDK JSON-encodes a tool result. ``json.dumps`` cannot encode a
    ``ToolMessage``, and the failure is silent by design: the stream helper
    swallows everything so a hook can never abort a run. The result of
    passing it through unchanged is a run that reports every tool call and
    not one tool result. Unwrap to the content, which is the string or the
    block list the tool actually produced.
    """
    content = getattr(output, "content", None)
    if isinstance(output, BaseMessage) or (
        content is not None and hasattr(output, "tool_call_id")
    ):
        return content
    return output


def _reasoning_steps(response: LLMResult) -> List[str]:
    """Every reasoning step on the first generation of an LLM result.

    Only the first generation is read, which is the one the base handler
    reports as the assistant turn. Reporting reasoning from candidates whose
    text is thrown away would describe deliberation that never reached the
    user.
    """
    generations = getattr(response, "generations", None) or []
    if not generations or not generations[0]:
        return []
    message = getattr(generations[0][0], "message", None)
    if message is None:
        return []

    # `content_blocks` is langchain-core 1.x and normalises the provider
    # spellings (a native reasoning block, or Ollama/DeepSeek/XAI/Groq's
    # `additional_kwargs["reasoning_content"]`) into one shape. On an older
    # core, or a message object that does not implement it, fall back to the
    # raw content list.
    try:
        blocks: Any = message.content_blocks
    except Exception:  # noqa: BLE001 — a custom message wrapper, not a failure
        logger.debug(
            "synap langgraph: no content_blocks on %s, reading raw content",
            type(message).__name__, exc_info=True,
        )
        blocks = getattr(message, "content", None)
    if not isinstance(blocks, list):
        return []

    steps: List[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        text = str(block.get("reasoning") or "").strip()
        if text:
            steps.append(text)
    return steps


class SynapLangGraphCallbackHandler(SynapCallbackHandler):
    """Reports a LangGraph run's five events on Synap's live stream.

    Pass it the same way as any LangChain callback, in the run config::

        from synap_langgraph import SynapLangGraphCallbackHandler

        handler = SynapLangGraphCallbackHandler(
            sdk=sdk,
            conversation_id="conv-123",
            user_id="user-456",
            customer_id="cust-789",
        )
        await app.ainvoke(
            {"messages": [HumanMessage("where is my order")]},
            config={"callbacks": [handler]},
        )

    One handler covers one conversation. Build a fresh one per conversation
    rather than sharing one across several: the user-turn de-duplication is
    per instance.

    All five events, all verified firing inside a compiled graph: the user
    turn (``on_chat_model_start``), the reasoning step (read off the
    AIMessage in ``on_llm_end``, because ``on_agent_action`` never fires
    here), the tool call (``on_tool_start``), the tool result
    (``on_tool_end``) and the assistant turn (``on_llm_end``). The call and
    the result share ``run_id`` as their ``tool_call_id``, which is what lets
    the anticipation agent pair them when several tools are in flight.

    The turn reporting is the base handler's: one user turn and one assistant
    turn per run, on the stream (opened by the first), REST only as a fallback,
    never both.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        super().__init__(
            sdk=sdk,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Report the reasoning, then hand the assistant turn to the base.

        Reasoning first: it is what the model did before it answered, and the
        assistant turn is the event anticipation acts on, so it goes out
        last.
        """
        steps = [] if self._inside_tool(run_id) else _reasoning_steps(response)
        for step_index, thought in enumerate(steps):
            await report_reasoning(
                self.sdk,
                content=thought,
                step_index=step_index,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
        await super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report what the tool returned, tied to the call by ``run_id``."""
        self._tool_runs.discard(run_id)
        self._parents.pop(run_id, None)
        await report_tool_result(
            self.sdk,
            result=_tool_output(output),
            tool_name=str(getattr(output, "name", "") or ""),
            tool_call_id=str(run_id),
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )


__all__ = ["SynapLangGraphCallbackHandler"]
