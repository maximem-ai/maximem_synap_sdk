"""Report an OpenAI Agents run to Synap, so the next turn starts warm.

This package gave an agent two tools and nothing else. Synap saw whatever the
model chose to search for and never saw the run itself: not the reply, not the
tools, not the reasoning. Anticipation had nothing to work with, so every fetch
was a cold retrieval.

``SynapRunHooks`` fills that in. Pass it to ``Runner.run(..., hooks=...)`` and
the run reports itself: the user's turn (read from the input of the first model
call), tool calls and results, reasoning, and the reply.

Two rules hold throughout.

*The stream is the default path.* The first report opens ``sdk.instance.listen()``
if the caller has not, and everything goes on it. When the stream cannot carry
a turn (``SYNAP_AUTO_LISTEN=false``, or it will not open) the user and assistant
turns go over REST (``sdk.conversation.record_message``, which needs a
``conversation_id``); tool calls, results and reasoning exist only on the stream.

*Never raises.* A hook runs inside the agent's own loop, where an exception
from telemetry would end the run. No context is worth that.

Hooks cannot change what the model is sent, so Synap's compaction takes effect
through the run config: ``Runner.run(agent, x, **hooks.run_kwargs())``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import weakref
from typing import Any, Dict, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    is_synap_history,
    report_reasoning,
    report_tool_call,
    report_tool_result,
    record_turn,
)

from synap_openai_agents.history import (
    item_role,
    item_text,
    synap_call_model_input_filter,
)

logger = logging.getLogger(__name__)

# What `report_user_turn` wrote last, per SDK and conversation, so a hook that
# reads the same turn off the model input does not write it a second time.
_manual_turns: "weakref.WeakKeyDictionary[Any, Dict[str, str]]" = weakref.WeakKeyDictionary()

try:  # pragma: no cover - exercised by the presence of the dependency
    from agents import RunHooks
except Exception:  # noqa: BLE001 - keeps the module importable without the SDK
    RunHooks = object  # type: ignore[assignment,misc]


class SynapRunHooks(RunHooks):  # type: ignore[misc,valid-type]
    """Run hooks that report the agent's activity on the Synap stream.

    Args:
        sdk: A configured ``MaximemSynapSDK``. The first report opens the
            ``listen()`` stream if you have not (``SYNAP_AUTO_LISTEN=false``
            stops that, and turns then go over REST); tool and reasoning
            hooks need the stream.
        user_id: Synap user scope. Required.
        customer_id: B2B only. Not accepted on a B2C instance.
        conversation_id: The conversation these runs belong to. Pass it:
            Synap's compaction replaces the SDK's earlier turns only for a
            known conversation, and the REST fallback cannot write a turn
            without one.
        report_tool_results: Whether tool results are reported. On by default,
            because a result is what tells anticipation how the turn is
            actually going. ⚠ A tool result is usually your own customer's
            data. It is an anticipation hint and never becomes a long-term
            memory, but it does leave your process. Turn this off if that is
            not something you want to send.

    Example::

        hooks = SynapRunHooks(sdk, user_id="u1", conversation_id=conv_id)
        result = await Runner.run(agent, "where is my order", **hooks.run_kwargs())

    ``run_kwargs()`` is ``hooks=`` plus a ``run_config=`` whose
    ``call_model_input_filter`` sends the model Synap's compaction in place of
    the SDK's earlier turns once the conversation has been compacted. With
    ``hooks=hooks`` alone the run is reported but the model still gets the
    SDK's own history.
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

    def run_config(self, **kwargs: Any) -> Any:
        """A ``RunConfig`` that sends the model Synap's compacted history.

        ``kwargs`` are the rest of the ``RunConfig`` you wanted. A
        ``call_model_input_filter`` of your own runs after this one.
        """
        from agents import RunConfig

        kwargs["call_model_input_filter"] = synap_call_model_input_filter(
            self._sdk, self._conversation_id, then=kwargs.get("call_model_input_filter"),
        )
        return RunConfig(**kwargs)

    def run_kwargs(self, **run_config_kwargs: Any) -> Dict[str, Any]:
        """``{"hooks": self, "run_config": self.run_config(...)}`` for ``Runner.run``."""
        return {"hooks": self, "run_config": self.run_config(**run_config_kwargs)}

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any,
    ) -> None:
        """Write the user's turn, once per run, from the model's input.

        The turn is the last user message in the input of the run's first model
        call. Later calls (after a tool, after a handoff) see the same message
        and are skipped.
        """
        try:
            if _flag(context, f"_synap_user_turn_{id(self)}"):
                return
            _set_flag(context, f"_synap_user_turn_{id(self)}")
            text = _last_user_text(input_items)
            if not text:
                return
            if _manual_turns.get(self._sdk, {}).pop(self._conversation_id, None) == text:
                return  # report_user_turn already wrote it
            await record_turn(
                self._sdk, role="user", content=text, **self._ids(),
                metadata=_model_metadata(getattr(agent, "model", None)),
            )
        except Exception:  # noqa: BLE001 — a hook must never end the run
            logger.debug("synap user turn report failed", exc_info=True)

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
        await record_turn(
            self._sdk, role="assistant", content=text, **self._ids(),
            metadata=_model_metadata(getattr(agent, "model", None)),
        )


async def report_user_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
    model: Any = None,
) -> bool:
    """Write the user's turn yourself. Returns whether it was written.

    ``SynapRunHooks`` already writes it, from the input of the run's first model
    call, so most callers never need this. It is for a run that does not use the
    hooks; if you call it and also use them, the turn is written once.

    ``model`` is the agent's model (``agent.model``); pass it so Synap sizes
    compaction to that model's context window.
    """
    written = await record_turn(
        sdk, role="user", content=content or "",
        conversation_id=conversation_id, user_id=user_id, customer_id=customer_id,
        metadata=_model_metadata(model),
    )
    if written:
        try:
            _manual_turns.setdefault(sdk, {})[conversation_id] = content
        except TypeError:  # an SDK that cannot be weakly referenced
            pass
    return written


def _model_metadata(model: Any) -> Optional[Dict[str, str]]:
    """``{"agent_model": name}`` for ``agent.model`` (a name or a Model), or None.

    Synap sizes compaction to the agent model's context window
    (COMPACTION_V3_PLAN.md §4.3); the server knows common models' windows.
    """
    name = model if isinstance(model, str) else getattr(model, "model", None)
    return {"agent_model": name} if isinstance(name, str) and name else None


def _flag(obj: Any, name: str) -> bool:
    try:
        return bool(vars(obj).get(name))
    except TypeError:
        return False


def _set_flag(obj: Any, name: str) -> None:
    try:
        setattr(obj, name, True)
    except Exception:  # noqa: BLE001 — a context that refuses attributes is written for every call
        pass


def _last_user_text(input_items: Any) -> str:
    """The text of the last user message, unless it is history Synap put there."""
    if isinstance(input_items, str):
        return input_items
    for item in reversed(list(input_items or [])):
        if item_role(item) == "user":
            text = item_text(item)
            return "" if is_synap_history(text) else text
    return ""


def _output_text(output: Any) -> str:
    """The agent's final text, from whatever shape the run returned.

    A structured output (a Pydantic model, a dataclass, a dict) is written as
    JSON, not as its ``repr``.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        if hasattr(output, "model_dump_json"):
            return output.model_dump_json()
        if dataclasses.is_dataclass(output) and not isinstance(output, type):
            return json.dumps(dataclasses.asdict(output), default=str)
        if isinstance(output, (dict, list, tuple)):
            return json.dumps(output, default=str)
    except Exception:  # noqa: BLE001 — fall through to the generic readers
        pass
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
