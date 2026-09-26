"""Report an OpenAI Agents run to Synap, so the next turn starts warm.

This package gave an agent two tools and nothing else. Synap saw whatever the
model chose to search for and never saw the run itself: not the reply, not the
tools, not the reasoning. Anticipation had nothing to work with, so every fetch
was a cold retrieval.

``SynapRunHooks`` fills that in. Pass it to ``Runner.run(..., hooks=...)`` and
the run reports itself.

Two rules hold throughout.

*Silent without a stream.* Everything here needs an active
``sdk.instance.listen()``. Most callers do not have one, and for them every
method is a no-op. Adding these hooks must not change behaviour for someone
who has not opted in.

*Never raises.* A hook runs inside the agent's own loop, where an exception
from telemetry would end the run. No context is worth that.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by the presence of the dependency
    from agents import RunHooks
except Exception:  # noqa: BLE001 - keeps the module importable without the SDK
    RunHooks = object  # type: ignore[assignment,misc]


class SynapRunHooks(RunHooks):  # type: ignore[misc,valid-type]
    """Run hooks that report the agent's activity on the Synap stream.

    Args:
        sdk: A configured ``MaximemSynapSDK`` with an active ``listen()``
            stream. Without one, every hook is a no-op.
        user_id: Synap user scope. Required.
        customer_id: B2B only. Not accepted on a B2C instance.
        conversation_id: The conversation these runs belong to. Required for
            the events to be useful: without it the server cannot say which
            conversation they are part of, and no session is opened.
        report_tool_results: Whether tool results are reported. On by default,
            because a result is what tells anticipation how the turn is
            actually going. ⚠ A tool result is usually your own customer's
            data. It is an anticipation hint and never becomes a long-term
            memory, but it does leave your process. Turn this off if that is
            not something you want to send.

    Example::

        hooks = SynapRunHooks(sdk, user_id="u1", conversation_id=conv_id)
        result = await Runner.run(agent, "where is my order", hooks=hooks)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        *,
        report_tool_results: bool = True,
        report_model_reasoning: bool = True,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapRunHooks requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapRunHooks requires a non-empty user_id")
        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id or ""
        self._report_tool_results = report_tool_results
        self._report_model_reasoning = report_model_reasoning
        # Tool calls and their results arrive as separate hooks with no id
        # tying them together, so the pairing is kept here: one tool in flight
        # per name, which is what a sequential agent loop produces.
        self._calls: dict[str, str] = {}
        self._step = 0

    # ── what the ids are for every event ────────────────────────────────
    def _ids(self) -> dict[str, str]:
        return {
            "conversation_id": self._conversation_id,
            "user_id": self._user_id,
            "customer_id": self._customer_id,
        }

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        name = str(getattr(tool, "name", "") or "")
        call_id = f"{name}:{self._step}"
        self._step += 1
        self._calls[name] = call_id
        await report_tool_call(
            self._sdk, tool_name=name, tool_call_id=call_id, **self._ids(),
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        if not self._report_tool_results:
            return
        name = str(getattr(tool, "name", "") or "")
        await report_tool_result(
            self._sdk,
            result=result,
            tool_name=name,
            tool_call_id=self._calls.pop(name, ""),
            **self._ids(),
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        """Report the model's reasoning, when the provider returned any.

        Most do not. OpenAI's reasoning models hand back a summary if one was
        asked for and an encrypted item otherwise, so this is frequently a
        no-op — which is correct. A turn with no reasoning to report is an
        ordinary turn.
        """
        if not self._report_model_reasoning:
            return
        text = _reasoning_text(response)
        if not text:
            return
        await report_reasoning(
            self._sdk, content=text, thought_type="model_reasoning", **self._ids(),
        )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Report the reply.

        This is the event anticipation acts on: the turn has ended, so the next
        one can be predicted and its context pushed before the user speaks
        again. An integration that reports tools and not this gets no
        anticipation at all.
        """
        text = _output_text(output)
        if not text:
            return
        await report_turn(
            self._sdk, role="assistant", content=text, **self._ids(),
        )


async def report_user_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Report the user's turn before you run the agent.

    There is no hook for this: ``on_agent_start`` fires with the agent, not
    with what the user said, and the run's input is yours rather than the
    SDK's. One line before ``Runner.run`` is the whole cost, and without it
    the conversation Synap sees has answers and no questions.
    """
    return await report_turn(
        sdk, role="user", content=content or "",
        conversation_id=conversation_id, user_id=user_id, customer_id=customer_id,
    )


def _output_text(output: Any) -> str:
    """The agent's final text, from whatever shape the run returned."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    for attr in ("final_output", "output_text", "text", "content"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(output)


def _reasoning_text(response: Any) -> str:
    """Reasoning summary text off a ModelResponse, if the provider sent any.

    Deliberately conservative: it reads the documented shape and gives up
    rather than guessing. Reporting the wrong field as a customer's reasoning
    is worse than reporting nothing.
    """
    items = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in items:
        if str(getattr(item, "type", "")) != "reasoning":
            continue
        for entry in getattr(item, "summary", None) or []:
            text = getattr(entry, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)
