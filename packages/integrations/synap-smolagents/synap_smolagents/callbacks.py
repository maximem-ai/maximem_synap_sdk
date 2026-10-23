"""Synap turn recorder for Smolagents.

`create_synap_recorder` returns a callback for
``step_callbacks={MemoryStep: create_synap_recorder(...)}``. Smolagents fires it at
the end of every completed step; the recorder writes the user and assistant turns
to Synap as conversation messages and reports the rest of the step (reasoning,
tool calls, tool results) on Synap's live stream when one is open.

Turns are written with :func:`synap_integrations_common.record_turn`: on the
stream (the first report opens ``sdk.instance.listen()`` for you; the server
persists the turn from the event), otherwise ``sdk.conversation.record_message``
over REST, never both. ``SYNAP_AUTO_LISTEN=false`` keeps it on REST. Either way
the server holds a real conversation under ``conversation_id``, which it compacts
and promotes to long-term memory, and which the SDK's local history search can
archive.

The recorder also attaches :func:`synap_smolagents.attach_synap_history` to the
agent it is handed, so once Synap has compacted the conversation the model sees
Synap's compaction instead of the agent's earlier steps.

Three behaviours are load-bearing:

- **Register for ``MemoryStep``, not ``ActionStep``.** Smolagents' registry walks
  ``memory_step.__class__.__mro__``, so a callback registered against the base
  class receives every step type: ``PlanningStep`` (the plan, reported as
  reasoning), ``ActionStep`` (model output, tool calls, observations) and
  ``FinalAnswerStep`` (the assistant turn). Registered against ``ActionStep``
  alone -- which is what this module used to document -- the final answer never
  arrives, and the assistant turn is the one event anticipation acts on. Use
  :func:`synap_step_callbacks` and the wiring is right by construction.

- **No per-step memory documents by default.** The recorder used to ingest every
  step's ``model_output`` with ``memories.create``. That left no conversation on
  the server, so compaction and history search had nothing to work on, and now
  that the turns are recorded (and promoted to long-term memory by the server)
  it would extract the final answer twice. ``ingest_steps=True`` still turns it
  on, one document per step (``smolagents-{conversation_id}-{step_number}``).

- **Log, never raise.** The recorder runs inside the agent loop, in a ``finally``
  with no surrounding try/except, so a raising callback would abort the whole run. A
  failed write is logged and swallowed -- same contract as the Strands stream
  hook. ``stream_events`` swallows everything by design, and the sync-to-async
  bridge around it is wrapped too.

## What is sent

The two turns are always written (stream or REST). Everything else is
stream-only and dropped when the stream cannot open.

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
``report_user_task=False`` and write it yourself with
``synap_integrations_common.record_turn`` if you need it earlier.

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
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
)

from synap_smolagents.history import _current_task_step, attach_synap_history

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
    ingest_steps: bool = False,
    compact_history: bool = True,
) -> Callable[..., None]:
    """Build a Smolagents step callback that records each run to Synap.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`. The first report opens the
            gRPC stream; turns go over REST when it cannot open, and every
            other report is dropped.
        user_id: Synap user scope. **Required.**
        conversation_id: The Synap conversation the turns are written to. Reuse
            it across ``agent.run()`` calls that belong to one conversation.
            **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        report_user_task: Whether the user turn is written from ``agent.task``.
            Turn it off if you write the user turn yourself before the run.
        report_tool_results: Whether tool results are reported. ⚠ A tool result is
            usually your own customer's data. It is an anticipation hint and never
            becomes a long-term memory, but it does leave your process.
        ingest_steps: Also ingest each step's model output as its own memory
            document via ``memories.create``. Off by default: the recorded turns
            already reach long-term memory through the server, so this extracts
            the agent's intermediate thoughts and, again, its final answer. Turn
            it on only if you want the step-by-step reasoning remembered.
        compact_history: Attach :func:`attach_synap_history` to the agent on the
            first step, so the model sees Synap's compaction in place of the
            agent's earlier steps once the conversation has one.

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
    # A run is told apart by the TaskStep it appended to memory, a new object each
    # time, so a second `agent.run()` reports its own task and answer even when
    # the text repeats. An agent with no memory falls back to comparing the task.
    seen: Dict[str, Any] = {"run": None, "task": None, "answer": None}

    async def _report(memory_step: Any, agent: Optional[Any]) -> None:
        """Everything this step puts on the stream, in one coroutine.

        One bridged call per step rather than one per event: the bridge is the
        expensive part, and the order here is the order the anticipation agent
        sees -- question, thinking, tools, answer.
        """
        if report_user_task:
            task = getattr(agent, "task", None)
            run = _current_task_step(agent)
            new_run = (run is not seen["run"]) if run is not None else (task != seen["task"])
            if isinstance(task, str) and task.strip() and new_run:
                seen["run"], seen["task"], seen["answer"] = run, task, None
                await record_turn(sdk, role="user", content=task.strip(), **ids)

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
            await record_turn(sdk, role="assistant", content=answer, **ids)

    def recorder(memory_step: Any, agent: Optional[Any] = None) -> None:
        if compact_history and agent is not None:
            try:
                attach_synap_history(agent, sdk, conversation_id)
            except Exception:  # noqa: BLE001: in-loop callback, log and never raise
                logger.warning("SynapStepRecorder: history attach failed", exc_info=True)
        # Turns and stream events first, so an opt-in step ingest cannot delay
        # the assistant turn, which is what anticipation waits on.
        try:
            run_async(_report(memory_step, agent))
        except Exception as exc:  # noqa: BLE001: in-loop callback, log and never raise
            # stream_events swallows its own failures; this guards the bridge.
            logger.warning(
                "SynapStepRecorder: turn write failed error=%s", exc, exc_info=True
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
