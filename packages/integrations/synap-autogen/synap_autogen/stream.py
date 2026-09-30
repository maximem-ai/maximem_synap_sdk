"""Report an AutoGen run to Synap, so the anticipation agent sees the whole turn.

This package gave an agent two tools and a short-term-context wrapper. Synap
saw whatever the model chose to search for and never saw the run itself: not
the user's turn, not the reply, not the tools, not the reasoning. Anticipation
had nothing to work with, so every fetch was a cold retrieval.

``SynapStreamingChatContext`` fills that in. It wraps the agent's
``model_context``, which is the one place every message an ``AssistantAgent``
produces passes through:

===============================  =========================================
What ``add_message`` receives     What it is
===============================  =========================================
``UserMessage``                   the user's turn
``AssistantMessage`` (str)        the assistant's turn
``AssistantMessage`` (calls)      one or more tool calls
``AssistantMessage.thought``      the model's reasoning for this step
``FunctionExecutionResultMessage`` what the tools returned
===============================  =========================================

``FunctionCall.id`` and ``FunctionExecutionResult.call_id`` are the same value,
so a call and its result travel with a shared ``tool_call_id`` and the
anticipation agent can tell they belong together.

⚠ **One gap, and it is on AutoGen's default path.** When
``reflect_on_tool_use`` is ``False`` (the default whenever ``output_content_type``
is unset) a turn that ends in a tool call answers with a
``ToolCallSummaryMessage``, and AutoGen never writes that message into the
model context. The tool calls and results are reported, but the
``assistant_message`` that tells anticipation the turn has ended is not,
because nothing hands it to us. :meth:`SynapStreamingChatContext.report_reply`
is the one line that closes it, and it is a no-op when the reply was already
reported, so adding it is always safe.

Two rules hold throughout, and they come from ``stream_events``.

*Silent without a stream.* Everything here needs an active
``sdk.instance.listen()``. Most callers do not have one, and for them every
report is a no-op. Adding this context must not change behaviour for someone
who has not opted into streaming.

*Never raises.* ``add_message`` runs inside the agent's own loop, where an
exception from telemetry would end the run. No context is worth that.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Mapping, Optional

from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)


class SynapStreamingChatContext(ChatCompletionContext):
    """An AG2 ``ChatCompletionContext`` that reports the turn on the Synap stream.

    Wraps an inner context (defaults to
    :class:`~autogen_core.model_context.UnboundedChatCompletionContext`) and
    reports every message the agent adds. Reads are pass-through: this class
    changes nothing about what the model sees.

    Compose it with :class:`~synap_autogen.short_term.SynapShortTermChatContext`
    when you want short-term context injected as well::

        from synap_autogen import SynapShortTermChatContext, SynapStreamingChatContext

        model_context = SynapStreamingChatContext(
            sdk, conversation_id="conv_abc", user_id="user-1",
            inner=SynapShortTermChatContext(sdk, conversation_id="conv_abc"),
        )
        agent = AssistantAgent(name="support", model_client=client,
                               model_context=model_context)

    Args:
        sdk: A configured :class:`MaximemSynapSDK`. Without an active
            ``listen()`` stream every report is a no-op.
        conversation_id: The conversation these turns belong to. **Required.**
            Without it the server cannot say which conversation an event is
            part of, and no session is opened.
        user_id: Synap user scope. **Required.**
        customer_id: B2B only. Not accepted on a B2C instance.
        inner: Optional inner context to wrap.
        report_tool_results: Whether tool results are reported. On by default,
            because a result is what tells anticipation how the turn is
            actually going. ⚠ A tool result is usually your own customer's
            data. It is an anticipation hint and never becomes a long-term
            memory, but it does leave your process. Turn this off if that is
            not something you want to send.
        report_thoughts: Whether ``AssistantMessage.thought`` is reported as a
            reasoning step. On by default. Most providers return nothing here,
            and a turn with no reasoning is an ordinary turn.

    Note:
        Attach this to **one** agent. In a team every agent receives every
        other agent's reply as a ``UserMessage``, so two agents sharing one
        ``conversation_id`` would report the same text twice, once as an
        assistant turn and once as a user turn.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
        *,
        inner: Optional[ChatCompletionContext] = None,
        report_tool_results: bool = True,
        report_thoughts: bool = True,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStreamingChatContext requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapStreamingChatContext requires a non-empty conversation_id"
            )
        if not user_id or not str(user_id).strip():
            raise ValueError("SynapStreamingChatContext requires a non-empty user_id")

        super().__init__()
        self._sdk = sdk
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._customer_id = customer_id
        self._inner = inner or UnboundedChatCompletionContext()
        self._report_tool_results = report_tool_results
        self._report_thoughts = report_thoughts
        self._step = 0
        self._last_assistant_text = ""

    # ── the ids every event carries ─────────────────────────────────────
    def _ids(self) -> dict:
        return {
            "conversation_id": self._conversation_id,
            "user_id": self._user_id,
            "customer_id": self._customer_id,
        }

    # ── ChatCompletionContext: reads and state are pass-through ─────────

    async def get_messages(self) -> List[LLMMessage]:
        return await self._inner.get_messages()

    async def clear(self) -> None:
        await self._inner.clear()

    async def save_state(self) -> Mapping[str, Any]:
        return {"inner": await self._inner.save_state()}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        # Deliberately not reported. Restoring a saved conversation is not a
        # turn happening now, and replaying it onto the stream would have
        # anticipation react to history as if it were live.
        await self._inner.load_state(dict(state).get("inner", {}))

    # ── the tap ─────────────────────────────────────────────────────────

    async def add_message(self, message: LLMMessage) -> None:
        """Store the message, then report it.

        The inner context is updated first, so a slow or failing report can
        never leave the model looking at an incomplete conversation.
        """
        await self._inner.add_message(message)
        await self._report(message)

    async def _report(self, message: LLMMessage) -> None:
        """Turn one AutoGen message into Synap stream events.

        Wrapped whole. ``stream_events`` already swallows everything it does,
        but the shape-reading here is ours, and an exception raised inside
        ``add_message`` would abort the agent's run.
        """
        try:
            await self._dispatch(message)
        except Exception:  # noqa: BLE001 — a tap must never break the run
            logger.debug("synap_autogen: reporting a message failed", exc_info=True)

    async def _dispatch(self, message: LLMMessage) -> None:
        if isinstance(message, SystemMessage):
            # The system prompt is the instructions, not a turn.
            return

        if isinstance(message, UserMessage):
            text = _user_text(message.content)
            if text:
                await report_turn(
                    self._sdk, role="user", content=text, **self._ids()
                )
            return

        if isinstance(message, AssistantMessage):
            await self._report_assistant(message)
            return

        if isinstance(message, FunctionExecutionResultMessage):
            if not self._report_tool_results:
                return
            for result in message.content or []:
                await report_tool_result(
                    self._sdk,
                    result=getattr(result, "content", result),
                    tool_name=str(getattr(result, "name", "") or ""),
                    tool_call_id=str(getattr(result, "call_id", "") or ""),
                    **self._ids(),
                )
            return

    async def _report_assistant(self, message: AssistantMessage) -> None:
        # The thought comes first because it is why the model did what it is
        # about to do, and it rides on the same message as the action.
        thought = (getattr(message, "thought", None) or "").strip()
        if thought and self._report_thoughts:
            self._step += 1
            await report_reasoning(
                self._sdk,
                content=thought,
                step_index=self._step,
                thought_type="model_thought",
                **self._ids(),
            )

        content = message.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                self._last_assistant_text = text
                await report_turn(
                    self._sdk, role="assistant", content=text, **self._ids()
                )
            return

        for call in content or []:
            await report_tool_call(
                self._sdk,
                tool_name=str(getattr(call, "name", "") or "tool"),
                tool_args=_tool_args(getattr(call, "arguments", None)),
                # AutoGen's own id for this invocation, and the same value
                # comes back on the result as `call_id`. The field can be
                # empty for some models, and an empty id is dropped rather
                # than sent, which is better than inventing one that ties
                # the call to nothing.
                tool_call_id=str(getattr(call, "id", "") or ""),
                **self._ids(),
            )

    # ── the one gap the model context cannot see ────────────────────────

    async def report_reply(self, result: Any) -> bool:
        """Report the agent's final reply, if it was not reported already.

        Call it after ``agent.run(...)`` or ``team.run(...)``::

            result = await agent.run(task="where is my order")
            await ctx.report_reply(result)

        Needed because of one AutoGen default. With ``reflect_on_tool_use``
        set to ``False`` (which is the default unless ``output_content_type``
        is set) a turn that ends in a tool call answers with a
        ``ToolCallSummaryMessage``, and AutoGen never adds that message to the
        model context. Without this line, such a turn reports its tool calls
        and results and then stops: anticipation never sees the
        ``assistant_message`` that says the turn has ended, which is the event
        it acts on.

        Returns whether an event went out. ``False`` when there was no stream,
        when there was nothing to report, or when this exact reply already
        went out through ``add_message`` — so calling it on every turn is
        safe and never writes the reply twice.
        """
        try:
            text = _reply_text(result)
        except Exception:  # noqa: BLE001 — never break the caller's turn
            logger.debug("synap_autogen: reading the reply failed", exc_info=True)
            return False

        if not text or text == self._last_assistant_text:
            return False

        self._last_assistant_text = text
        return await report_turn(
            self._sdk, role="assistant", content=text, **self._ids()
        )


def _user_text(content: Any) -> str:
    """The text of a ``UserMessage``, whose content may be multimodal.

    Images and anything else that is not a string are skipped rather than
    stringified: the repr of an image is not something a person said.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(part for part in content if isinstance(part, str)).strip()
    return ""


def _tool_args(arguments: Any) -> Optional[dict]:
    """``FunctionCall.arguments`` as a dict.

    AutoGen hands them over as a JSON string. A model can emit something that
    is not an object, or not valid JSON at all, and in that case the raw text
    is passed through under a single key rather than dropped.
    """
    if arguments is None:
        return None
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    if not arguments.strip():
        return None
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {"arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"arguments": arguments}


def _reply_text(result: Any) -> str:
    """The final reply text out of whatever ``run`` returned.

    Handles a ``TaskResult`` (``.messages``), a ``Response``
    (``.chat_message``), a bare message, and a plain string, because the two
    entry points people use return two different shapes.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()

    message = None
    messages = getattr(result, "messages", None)
    if messages:
        message = messages[-1]
    elif getattr(result, "chat_message", None) is not None:
        message = result.chat_message
    else:
        message = result

    to_text = getattr(message, "to_text", None)
    if callable(to_text):
        text = to_text()
        return text.strip() if isinstance(text, str) else ""

    content = getattr(message, "content", None)
    return content.strip() if isinstance(content, str) else ""


__all__ = ["SynapStreamingChatContext"]
