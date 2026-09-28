"""Report a Pydantic AI run on Synap's live stream.

Pydantic AI's one hook into a run as it happens is
``event_stream_handler``: an async callable taking ``(RunContext,
AsyncIterable[AgentStreamEvent])``. The agent calls it once per graph node,
so a run with one tool produces four calls: the model request, the tool
execution, the second model request, and a trailing empty one. Everything
the anticipation agent needs between the two turns arrives there.

Two surfaces, because the hook does not carry the turns:

- :class:`SynapEventStreamHandler` reports the three events in between — the
  tool call, the tool result and the reasoning step. Pass it wherever your
  version of Pydantic AI accepts an ``event_stream_handler``.
- :func:`synap_run` runs the agent with that handler installed and reports
  the user turn before and the assistant turn after, so all five events go
  out. This is the one to reach for.

**The id that pairs a call with its result** is ``event.tool_call_id``,
which Pydantic AI exposes as a property on both the call event and the
result event. It is the id the model itself issued, so it also matches the
``ToolCallPart`` in the message history.

**Version drift is handled by reading, never by assuming.** Checked against
``pydantic-ai-slim`` 2.51.0 (run) and 1.0.10 (source): the result event
renamed its payload from ``result`` to ``part``, and ``PartEndEvent`` does
not exist before 2.x, so reasoning is accumulated from ``PartStartEvent`` +
``PartDeltaEvent`` and merely corrected by ``PartEndEvent`` when there is
one. Events are matched on their ``event_kind`` discriminator rather than by
importing classes, so this module imports nothing from ``pydantic_ai`` and a
version that spells a class differently degrades to reporting less rather
than to an ImportError.

Both rules from ``stream_events`` hold throughout: silent when no
``sdk.instance.listen()`` is running, and never raises. A handler that
throws inside the agent's own loop takes down the run.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, AsyncIterable, Dict, Optional

from synap_integrations_common import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    run_async,
)

from synap_pydantic_ai.deps import SynapDeps

logger = logging.getLogger(__name__)

# Logged once per process rather than once per run: a version without the
# hook is a property of the install, not of the call.
_warned_no_hook = False


# ---------------------------------------------------------------------------
# Shape readers — each one tolerates a version that spells a field differently
# ---------------------------------------------------------------------------


def _sendable(value: Any) -> Any:
    """``value``, or its string form when it cannot go on the wire.

    The SDK JSON-encodes a tool result, and the stream helper swallows every
    failure so a hook can never abort a run. An object ``json`` cannot encode
    is therefore not an error anybody sees: it is a tool result that silently
    never gets reported. A string is worse than the structure and far better
    than nothing.
    """
    if isinstance(value, str):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _tool_args(part: Any) -> Optional[Dict[str, Any]]:
    """A tool call's arguments as a dict.

    ``ToolCallPart.args`` is a dict when the provider sent structured
    arguments and the raw JSON string the model produced when it did not.
    ``args_as_dict()`` parses the second case and exists in every version
    that has ``ToolCallPart``.
    """
    args = getattr(part, "args", None)
    if isinstance(args, dict):
        return args
    as_dict = getattr(part, "args_as_dict", None)
    if callable(as_dict):
        try:
            parsed = as_dict()
        except Exception:  # noqa: BLE001 — malformed args are not our error
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _result_part(event: Any) -> Any:
    """The payload of a tool-result event.

    Named ``part`` from Pydantic AI 2.x and ``result`` before it. Both
    versions keep ``tool_call_id`` on the event itself, which is what the
    pairing actually depends on.
    """
    part = getattr(event, "part", None)
    if part is not None:
        return part
    return getattr(event, "result", None)


def _thinking_text(part: Any) -> Optional[str]:
    """The text of a reasoning part, or ``None`` if this is not one."""
    if getattr(part, "part_kind", None) != "thinking":
        return None
    return str(getattr(part, "content", "") or "")


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


class SynapEventStreamHandler:
    """Reports a Pydantic AI run's tool calls, tool results and reasoning.

    Pass it where your version of Pydantic AI takes an
    ``event_stream_handler``: per run on 2.x, and either per run or on the
    ``Agent`` constructor on 1.x. :func:`synap_run` does this for you and
    adds the two turns, which this handler deliberately does not report --
    the event stream has no "the run ended" event, and guessing at one is
    how a turn gets written twice.

    Example::

        handler = SynapEventStreamHandler(
            sdk, conversation_id="conv_abc", user_id="u1", customer_id="c1",
        )
        result = await agent.run("where is my order", event_stream_handler=handler)

    Build one per run. The reasoning step counter is per instance.
    """

    def __init__(
        self,
        sdk: Any,
        *,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapEventStreamHandler requires a non-None sdk")
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        self._step = 0

    async def __call__(self, ctx: Any, events: AsyncIterable[Any]) -> None:
        # One reasoning accumulator per node. Pydantic AI calls the handler
        # once per graph node, and a part index only means anything within
        # the one model response that produced it.
        thoughts: Dict[Any, str] = {}
        try:
            async for event in events:
                try:
                    await self._dispatch(event, thoughts)
                except Exception:  # noqa: BLE001 — never abort the agent run
                    logger.debug(
                        "synap pydantic-ai: event report failed", exc_info=True,
                    )
        except Exception:  # noqa: BLE001 — never abort the agent run
            logger.debug("synap pydantic-ai: event stream failed", exc_info=True)
        try:
            await self._report_reasoning(thoughts)
        except Exception:  # noqa: BLE001 — never abort the agent run
            logger.debug("synap pydantic-ai: reasoning report failed", exc_info=True)

    async def _dispatch(self, event: Any, thoughts: Dict[Any, str]) -> None:
        kind = getattr(event, "event_kind", "")
        if kind == "function_tool_call":
            await self._report_tool_call(event)
        elif kind == "function_tool_result":
            await self._report_tool_result(event)
        elif kind in ("part_start", "part_end"):
            # A repeated start at the same index replaces what was there, and
            # an end carries the finished part, so both are authoritative
            # over whatever the deltas built up.
            text = _thinking_text(getattr(event, "part", None))
            if text is not None:
                thoughts[getattr(event, "index", 0)] = text
        elif kind == "part_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "part_delta_kind", None) != "thinking":
                return
            index = getattr(event, "index", 0)
            thoughts[index] = thoughts.get(index, "") + str(
                getattr(delta, "content_delta", "") or ""
            )

    async def _report_tool_call(self, event: Any) -> None:
        part = getattr(event, "part", None)
        await report_tool_call(
            self.sdk,
            tool_name=str(getattr(part, "tool_name", "") or "tool"),
            tool_args=_tool_args(part),
            tool_call_id=str(getattr(event, "tool_call_id", "") or ""),
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    async def _report_tool_result(self, event: Any) -> None:
        part = _result_part(event)
        await report_tool_result(
            self.sdk,
            result=_sendable(getattr(part, "content", None)),
            tool_name=str(getattr(part, "tool_name", "") or ""),
            tool_call_id=str(getattr(event, "tool_call_id", "") or ""),
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    async def _report_reasoning(self, thoughts: Dict[Any, str]) -> None:
        """Send the node's reasoning once the node's stream has ended.

        A reasoning part arrives in pieces, so there is nothing worth
        reporting until the stream that carries it is done. ``step_index``
        counts across the whole run, not within one node, which is why it
        lives on the instance.
        """
        for index in sorted(thoughts, key=lambda i: (str(type(i)), i)):
            content = thoughts[index].strip()
            if not content:
                continue
            await report_reasoning(
                self.sdk,
                content=content,
                step_index=self._step,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
            self._step += 1


# ---------------------------------------------------------------------------
# The two turns
# ---------------------------------------------------------------------------


def _prompt_text(user_prompt: Any) -> str:
    """The user's turn as text.

    ``user_prompt`` is a string, or a sequence of ``UserContent`` when the
    turn carries images or documents. There is no text for the parts that
    are not text, and inventing one would put a description of the caller's
    attachment into their memory, so the sequence is reported as the
    framework renders it.
    """
    if user_prompt is None:
        return ""
    return user_prompt if isinstance(user_prompt, str) else str(user_prompt)


def _output_text(result: Any) -> str:
    """The assistant's turn as text.

    ``output`` is a string for a plain agent and the validated object for a
    structured one. Both are reported: a structured answer is still what the
    agent said.
    """
    output = getattr(result, "output", None)
    if output is None:
        return ""
    return output if isinstance(output, str) else str(output)


async def _report_turn(deps: SynapDeps, role: str, content: str) -> None:
    """Stream first, REST only as a fallback, never both.

    The server persists ``user_message`` and ``assistant_message`` from the
    stream itself (``grpc/servicer.py``), so a ``record_message`` on top of a
    delivered stream event writes the same turn twice and extracts it twice.
    ``report_turn`` returns whether it actually went out, which is what makes
    the choice decidable rather than guessed.
    """
    if not content:
        return
    if await report_turn(
        deps.sdk,
        role=role,
        content=content,
        conversation_id=deps.conversation_id or "",
        user_id=deps.user_id,
        customer_id=deps.customer_id,
    ):
        return
    if not deps.conversation_id:
        # REST records against a conversation and the stream opens its own,
        # so a run with no conversation_id has nowhere to fall back to.
        return
    try:
        await deps.sdk.conversation.record_message(
            conversation_id=deps.conversation_id,
            role=role,
            content=content,
            user_id=deps.user_id,
            customer_id=deps.customer_id or None,
        )
    except Exception as exc:  # noqa: BLE001 — never break the agent run
        logger.error(
            "synap pydantic-ai: record_message failed role=%s "
            "conversation_id=%s error=%s",
            role, deps.conversation_id, exc, exc_info=True,
        )


def _accepts_event_stream_handler(agent: Any) -> bool:
    """Whether this Pydantic AI version takes the hook per run.

    Named explicitly or not at all: a signature that only has ``**kwargs``
    is not evidence that it would accept this one, and passing an argument
    the agent does not know is a TypeError inside the caller's run.
    """
    try:
        parameters = inspect.signature(agent.run).parameters
    except (TypeError, ValueError):  # a callable with no inspectable signature
        return False
    return "event_stream_handler" in parameters


async def synap_run(
    agent: Any,
    user_prompt: Any = None,
    *,
    deps: SynapDeps,
    **kwargs: Any,
) -> Any:
    """Run a Pydantic AI agent and report the whole turn to Synap.

    A drop-in for ``await agent.run(...)``. Everything else is passed
    through, so ``message_history``, ``output_type``, ``model_settings`` and
    the rest behave exactly as they do on the agent.

    Reported, in the order they happen: the user turn, then each reasoning
    step, tool call and tool result as the run produces them, then the
    assistant turn. The middle three come from a
    :class:`SynapEventStreamHandler` installed for this run; pass your own
    ``event_stream_handler=`` and it is left alone.

    Every report is silent when no ``sdk.instance.listen()`` is running, and
    none of them can break the run. A failure to report is logged, not
    raised.

    Args:
        agent: A ``pydantic_ai.Agent``.
        user_prompt: What the user said, as you would pass it to
            ``agent.run``.
        deps: :class:`SynapDeps` carrying the SDK and the scope. It is
            forwarded to the agent as its ``deps``, so your tools and system
            prompts see it unchanged. Set ``conversation_id`` on it for the
            turn to be attributed to a conversation.

    Returns:
        Whatever ``agent.run`` returned, untouched.

    Example::

        from pydantic_ai import Agent
        from synap_pydantic_ai import SynapDeps, synap_run

        agent = Agent("openai:gpt-4o", deps_type=SynapDeps)
        deps = SynapDeps(
            sdk=sdk, user_id="u1", customer_id="c1",
            conversation_id="conv_abc",
        )
        result = await synap_run(agent, "where is my order", deps=deps)
    """
    global _warned_no_hook

    if agent is None:
        raise ValueError("synap_run requires a non-None agent")
    if deps is None:
        raise ValueError("synap_run requires SynapDeps (it carries the sdk)")

    await _report_turn(deps, "user", _prompt_text(user_prompt))

    if "event_stream_handler" not in kwargs:
        if _accepts_event_stream_handler(agent):
            kwargs["event_stream_handler"] = SynapEventStreamHandler(
                deps.sdk,
                conversation_id=deps.conversation_id or "",
                user_id=deps.user_id,
                customer_id=deps.customer_id,
            )
        elif not _warned_no_hook:
            _warned_no_hook = True
            logger.warning(
                "synap pydantic-ai: this version of Agent.run does not take "
                "event_stream_handler, so tool calls, tool results and "
                "reasoning will not be reported. The user and assistant "
                "turns still are. Upgrade pydantic-ai for the whole turn."
            )

    result = await agent.run(user_prompt, deps=deps, **kwargs)

    await _report_turn(deps, "assistant", _output_text(result))
    return result


def synap_run_sync(
    agent: Any,
    user_prompt: Any = None,
    *,
    deps: SynapDeps,
    **kwargs: Any,
) -> Any:
    """Synchronous :func:`synap_run`, for callers outside an event loop.

    It drives the async path rather than ``agent.run_sync``, because the
    reporting either side of the run is async either way.
    """
    return run_async(synap_run(agent, user_prompt, deps=deps, **kwargs))


__all__ = ["SynapEventStreamHandler", "synap_run", "synap_run_sync"]
