"""Synap turn recorder for Smolagents.

`create_synap_recorder` returns a callback for
``step_callbacks={MemoryStep: create_synap_recorder(...)}``. Smolagents fires it at
the end of every completed step; the recorder reports the step on Synap's live
stream and ingests the step's output into Synap so future sessions can recall
what the agent did.

Three behaviours are load-bearing:

- **Register for ``MemoryStep``, not ``ActionStep``.** Smolagents' registry walks
  ``memory_step.__class__.__mro__``, so a callback registered against the base
  class receives every step type: ``PlanningStep`` (the plan, reported as
  reasoning), ``ActionStep`` (model output, tool calls, observations) and
  ``FinalAnswerStep`` (the assistant turn). Registered against ``ActionStep``
  alone -- which is what this module used to document -- the final answer never
  arrives, and the assistant turn is the one event anticipation acts on. Use
  :func:`synap_step_callbacks` and the wiring is right by construction.

- **Per-step ``document_id``.** Each step is its own memory document
  (``smolagents-{conversation_id}-{step_number}``); a single stable id would clobber
  each step down to the last.

- **Log, never raise.** The recorder runs inside the agent loop, in a ``finally``
  with no surrounding try/except, so a raising callback would abort the whole run. A
  failed ingest is logged at ERROR and swallowed -- same contract as the Strands
  stream hook. The stream reports cannot raise either: ``stream_events`` swallows
  everything by design, and the sync-to-async bridge around them is wrapped too.

## What goes on the stream

Everything here is silent unless ``sdk.instance.listen()`` is running, so adding
it changes nothing for a caller who has not opted into streaming.

===================  ==========================================================
Event                Where it comes from
===================  ==========================================================
user turn            ``agent.task`` -- see the caveat below
assistant turn       ``FinalAnswerStep.output``, or an ``ActionStep`` whose
                     ``is_final_answer`` is set (whichever arrives first; the
                     second one is suppressed)
tool call            ``ActionStep.tool_calls[].id`` / ``.name`` / ``.arguments``
tool result          ``ActionStep.observations``
reasoning            ``PlanningStep.plan`` and ``ActionStep.model_output``
===================  ==========================================================

⚠ **The user turn is derived, not hooked.** Smolagents appends a ``TaskStep`` to
memory but never passes it to a step callback, so there is no hook that carries
what the user asked. ``Agent.run()`` does set ``self.task`` before the first
step, and the registry hands the callback ``agent=self``, so the task is read
from there and reported once per task. That means the user turn is reported at
the end of the first step rather than before it. Pass
``report_user_task=False`` and report it yourself with
``synap_integrations_common.report_turn`` if you need it earlier.

⚠ **Parallel tool calls share one observation.** ``ToolCallingAgent`` merges every
tool's output for a step into the single ``ActionStep.observations`` string, so
when a step made more than one call there is no per-call result to report. The
merged text still goes out, but with no ``tool_call_id``: attributing it to one
of the calls would be a guess. A step with exactly one call is reported with
that call's id, which is the common case.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

_MARKER = "smolagents_step"


def _as_text(output: Any) -> str:
    """Coerce an ``ActionStep.model_output`` (``str | list[dict] | None``) to text."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(output)


def _tool_args(arguments: Any) -> Optional[Dict[str, Any]]:
    """A tool call's arguments as a dict, which is the only shape the stream takes.

    ``ToolCallingAgent`` produces a dict. ``CodeAgent`` produces the code blob as
    a plain string, which is still the argument the tool was called with, so it
    is carried under a key rather than dropped.
    """
    if arguments is None:
        return None
    if isinstance(arguments, dict):
        return arguments
    return {"input": str(arguments)}


def _final_answer_text(memory_step: Any) -> str:
    """The assistant turn, if this step is the one that ends the run.

    Two steps can carry it. ``_run_stream`` finalises the last ``ActionStep``
    with ``is_final_answer`` set and then finalises a ``FinalAnswerStep`` holding
    the same answer, so whichever the caller's registration delivers, the turn is
    reported; the recorder suppresses the second.
    """
    if getattr(memory_step, "is_final_answer", False):
        return _stringify(getattr(memory_step, "action_output", None))
    # FinalAnswerStep carries `output` and nothing else. An ActionStep has
    # `action_output`, so requiring its absence keeps the two apart.
    if hasattr(memory_step, "output") and not hasattr(memory_step, "action_output"):
        return _stringify(getattr(memory_step, "output", None))
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def create_synap_recorder(
    sdk: MaximemSynapSDK,
    user_id: str,
    conversation_id: str,
    *,
    customer_id: str = "",
    report_user_task: bool = True,
    report_tool_results: bool = True,
    ingest_steps: bool = True,
) -> Callable[..., None]:
    """Build a Smolagents step callback that reports each step to Synap.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`. Every stream report is a no-op
            unless ``sdk.instance.listen()`` is running.
        user_id: Synap user scope. **Required.**
        conversation_id: Conversation id; seeds each step's document id. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        report_user_task: Whether the user turn is reported from ``agent.task``.
            Turn it off if you report the user turn yourself before the run.
        report_tool_results: Whether tool results are reported. ⚠ A tool result is
            usually your own customer's data. It is an anticipation hint and never
            becomes a long-term memory, but it does leave your process.
        ingest_steps: Whether each step's model output is also ingested as a
            long-term memory document. This is the recorder's original behaviour
            and stays on by default. It is a separate write from the stream: the
            stream carries the turn, this carries the step. Turn it off if you do
            not want the same text extracted twice while a stream is running.

    Returns:
        A callable ``recorder(memory_step, agent=None)`` for ``step_callbacks``.
        Register it against ``MemoryStep`` (see :func:`synap_step_callbacks`) so it
        receives planning steps and the final answer, not action steps alone.
    """
    if sdk is None:
        raise ValueError("create_synap_recorder requires a non-None sdk")
    if not user_id or not str(user_id).strip():
        raise ValueError("create_synap_recorder requires a non-empty user_id")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError("create_synap_recorder requires a non-empty conversation_id")

    ids = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "customer_id": customer_id,
    }
    # One task and one answer per run, remembered so neither is reported twice.
    # `task` also resets `answer`, so a second `agent.run()` on the same recorder
    # reports its own final answer even when the text repeats.
    seen: Dict[str, Optional[str]] = {"task": None, "answer": None}

    async def _report(memory_step: Any, agent: Optional[Any]) -> None:
        """Everything this step puts on the stream, in one coroutine.

        One bridged call per step rather than one per event: the bridge is the
        expensive part, and the order here is the order the anticipation agent
        sees -- question, thinking, tools, answer.
        """
        if report_user_task:
            task = getattr(agent, "task", None)
            if isinstance(task, str) and task.strip() and task != seen["task"]:
                seen["task"] = task
                seen["answer"] = None
                await report_turn(sdk, role="user", content=task.strip(), **ids)

        step_number = getattr(memory_step, "step_number", None)

        plan = getattr(memory_step, "plan", None)
        if isinstance(plan, str) and plan.strip():
            await report_reasoning(
                sdk, content=plan.strip(), step_index=step_number,
                thought_type="plan", **ids,
            )

        model_output = _as_text(getattr(memory_step, "model_output", None)).strip()
        if model_output:
            await report_reasoning(
                sdk, content=model_output, step_index=step_number,
                thought_type="action", **ids,
            )

        tool_calls = list(getattr(memory_step, "tool_calls", None) or [])
        for call in tool_calls:
            await report_tool_call(
                sdk,
                tool_name=str(getattr(call, "name", "") or "tool"),
                tool_args=_tool_args(getattr(call, "arguments", None)),
                tool_call_id=str(getattr(call, "id", "") or ""),
                **ids,
            )

        observations = getattr(memory_step, "observations", None)
        if report_tool_results and observations:
            # Only a lone call can own the merged observation string; see the
            # module docstring for why a step with several calls sends no id.
            only = tool_calls[0] if len(tool_calls) == 1 else None
            await report_tool_result(
                sdk,
                result=observations,
                tool_name=str(getattr(only, "name", "") or "") if only else "",
                tool_call_id=str(getattr(only, "id", "") or "") if only else "",
                **ids,
            )

        answer = _final_answer_text(memory_step)
        if answer and answer != seen["answer"]:
            seen["answer"] = answer
            await report_turn(sdk, role="assistant", content=answer, **ids)

    def recorder(memory_step: Any, agent: Optional[Any] = None) -> None:
        # Stream first. Reported before the ingest so a slow or failing ingest
        # cannot delay the assistant turn, which is what anticipation waits on.
        try:
            run_async(_report(memory_step, agent))
        except Exception as exc:  # noqa: BLE001: in-loop callback, log and never raise
            # stream_events swallows its own failures; this guards the bridge.
            logger.debug(
                "SynapStepRecorder: stream report failed error=%s", exc, exc_info=True
            )

        if not ingest_steps:
            return
        # Skip failed steps: the callback fires for them too.
        if getattr(memory_step, "error", None) is not None:
            return
        text = _as_text(getattr(memory_step, "model_output", None)).strip()
        if not text:
            return
        step_number = getattr(memory_step, "step_number", 0)
        try:
            run_async(
                sdk.memories.create(
                    document=text,
                    user_id=user_id,
                    customer_id=customer_id or None,
                    document_type="ai-chat-conversation",
                    document_id=f"smolagents-{conversation_id}-{step_number}",
                    metadata={_MARKER: True, "step": step_number},
                )
            )
        except Exception as exc:  # noqa: BLE001: in-loop callback, log and never raise
            logger.error(
                "SynapStepRecorder: ingest failed step=%s error=%s",
                step_number,
                exc,
                exc_info=True,
            )

    return recorder


def synap_step_callbacks(
    sdk: MaximemSynapSDK,
    user_id: str,
    conversation_id: str,
    **kwargs: Any,
) -> Dict[Any, Callable[..., None]]:
    """The ``step_callbacks`` mapping to pass straight to a Smolagents agent.

    Registers the recorder against ``MemoryStep`` so it receives planning steps,
    action steps and the final answer. Registering against ``ActionStep`` instead
    loses the assistant turn, and without that there is no anticipation.

    Example::

        agent = CodeAgent(
            model=InferenceClientModel(),
            tools=create_synap_tools(sdk, user_id="alice"),
            step_callbacks=synap_step_callbacks(
                sdk, user_id="alice", conversation_id="conv_abc"),
        )
    """
    from smolagents.memory import MemoryStep

    return {
        MemoryStep: create_synap_recorder(sdk, user_id, conversation_id, **kwargs)
    }
