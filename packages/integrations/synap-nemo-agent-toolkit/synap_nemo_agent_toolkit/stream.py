"""Report a NAT workflow run onto Synap's live gRPC stream.

NAT does not hand an integration per-hook callbacks the way LangChain does.
Everything a workflow does is published on one reactive stream of
:class:`~nat.data_models.intermediate_step.IntermediateStep` events, and the
supported way to read that stream is a **telemetry exporter**: a
:class:`~nat.observability.exporter.base_exporter.BaseExporter` that NAT's
``ExporterManager`` starts and stops around every run.

So that is what this is. Drop ``_type: synap_stream`` into a workflow's
``general.telemetry.tracing`` block and the whole turn is reported; there is
nothing to change in the workflow itself.

## What maps to what

===================  =============================================
NAT event            Synap event
===================  =============================================
``WORKFLOW_START``   ``report_turn(role="user")``
``WORKFLOW_END``     ``report_turn(role="assistant")``
``TOOL_START``       ``report_tool_call``
``TOOL_END``         ``report_tool_result``
``LLM_END``          ``report_reasoning`` (see below)
===================  =============================================

**The turn is the workflow run, not the model call.** ``WORKFLOW_START`` and
``WORKFLOW_END`` are emitted once each by ``nat.runtime.runner``, whatever
happens in between. ``LLM_START`` fires once per lap of a tool loop, so
reading the user turn off it would report the same question three times for a
three-tool turn. There is nothing to de-duplicate here because nothing repeats.

**Reasoning is every LLM output except the last one.** NAT has no reasoning
event: ``IntermediateStepType`` has no member for it and no framework wrapper
emits one. But an agent loop's LLM outputs are deliberation right up until the
final one, which is the answer — and the answer already goes out as the
assistant turn. So an ``LLM_END`` is held, released as a reasoning step when
the run does something else (another ``LLM_END``, or a ``TOOL_START``), and
dropped at ``WORKFLOW_END`` where it would have duplicated the answer. In the
ordinary ReAct shape the release happens on the tool call that the reasoning
led to, which is the same moment LangChain reports ``on_agent_action``.

⚠ One buffer for the run, so under parallel branches a reasoning step can be
released by a tool call from the other branch. A reasoning event carries no
id tying it to a tool, so this mis-times a hint and never mis-attributes one.

## Order

Every event goes through one queue drained by one worker, so the user turn
reaches the server before the tool calls that follow it. ``export()`` is
called synchronously from inside the agent's own loop (NAT's ``Subject``
delivers to subscribers inline), and firing a task per event let them
interleave at their first await.

The queue is drained on ``stop()`` before the subscription is dropped.
Without that the ``WORKFLOW_END`` turn — the one event the anticipation agent
actually acts on — is the one most likely to be lost, because it is queued
last and ``stop()`` follows it immediately.

## Stream only

There is deliberately no REST fallback here. The server persists
``user_message`` and ``assistant_message`` from the stream itself, so
reporting a turn *and* calling ``conversation.record_message`` would write it
twice and extract it twice. With no stream running, every call in
``stream_events`` is a silent no-op and this exporter costs nothing — which is
the right behaviour for something a user switched on under ``telemetry``.
A workflow that wants turns written to long-term memory has
:class:`~synap_nemo_agent_toolkit.editor.SynapMemoryEditor` for that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any, Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import OptionalSecretStr, get_secret_value
from nat.data_models.intermediate_step import IntermediateStep, IntermediateStepType
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.exporter.base_exporter import BaseExporter, IsolatedAttribute

from maximem_synap import MaximemSynapSDK
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

# How long stop() waits for queued events to reach the stream. Short, because
# it runs on the workflow's teardown path: a Synap blip must not hold a NAT
# run open. Long enough for the last turn on a healthy stream.
_DEFAULT_DRAIN_TIMEOUT_S = 5.0


# ── reading a NAT event ────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """``value`` if the SDK can serialise it, otherwise its string form.

    ⚠ The SDK does ``json.dumps`` on a tool result and on tool args, and
    ``stream_events._send`` swallows every exception by design. A framework
    object that ``json.dumps`` refuses is therefore **silently never
    reported**: nothing fails, nothing logs above debug, and the anticipation
    agent simply never learns what the tool returned. A string is passed
    through untouched — the SDK sends one as itself rather than as a quoted
    JSON string.
    """
    if isinstance(value, str):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _unwrap_result(output: Any) -> Any:
    """What a tool returned, unwrapped from the framework's envelope.

    NAT's framework wrappers put the raw object on ``data.output``: LangChain
    hands over a ``ToolMessage``, LlamaIndex a ``ToolOutput``. Both carry the
    payload on ``.content`` and neither is JSON serialisable. Anything already
    plain is left alone.
    """
    content = getattr(output, "content", None)
    return _json_safe(output if content is None else content)


def _user_text(value: Any) -> str:
    """The user's words out of a ``WORKFLOW_START`` input.

    ``data.input`` is whatever the workflow's entry type is: a bare string, a
    pydantic request model, or a dict. Mirrors NAT's own
    ``nat.utils.atif_converter._extract_user_input`` so the two agree on what
    the user said.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    obj: Any = value
    if hasattr(value, "model_dump"):
        try:
            obj = value.model_dump()
        except Exception:  # noqa: BLE001 — a model that will not dump is not fatal
            obj = value
    if isinstance(obj, dict):
        if obj.get("input_message"):
            return str(obj["input_message"])
        messages = obj.get("messages")
        if isinstance(messages, list):
            last_user = ""
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    last_user = message.get("content", "")
            if last_user:
                return str(last_user)
    return str(value)


def _assistant_text(value: Any) -> str:
    """The agent's answer out of a ``WORKFLOW_END`` output.

    ``ainvoke`` puts the result there directly. ``astream`` puts a list of the
    chunks it yielded, because the runner collects a preview rather than the
    joined string, so a streamed answer arrives as its pieces and has to be
    put back together.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_assistant_text(part) for part in value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:  # noqa: BLE001
            dumped = None
        if isinstance(dumped, dict):
            for key in ("value", "output", "content", "text"):
                if dumped.get(key):
                    return str(dumped[key])
    return str(value)


def _tool_args(step: IntermediateStep) -> dict:
    """The tool's arguments as a dict, not as a repr of one.

    ⚠ ``data.input`` on a ``TOOL_START`` is the framework's *string* rendering
    of the arguments: NAT's LangChain wrapper puts ``input_str`` there and the
    parsed dict on ``metadata.tool_inputs``. Sending the string gives the
    anticipation agent ``{"input": "{'pnr': 'QX41RT'}"}`` to reason over
    instead of the arguments themselves, so the parsed side wins wherever the
    wrapper supplies it.
    """
    metadata = step.metadata
    parsed = (
        metadata.get("tool_inputs")
        if isinstance(metadata, dict)
        else getattr(metadata, "tool_inputs", None)
    )
    if isinstance(parsed, dict) and parsed:
        return {key: _json_safe(value) for key, value in parsed.items()}

    raw = step.data.input if step.data is not None else None
    if isinstance(raw, dict) and raw:
        return {key: _json_safe(value) for key, value in raw.items()}
    if isinstance(raw, str) and raw:
        with contextlib.suppress(ValueError, TypeError):
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return {key: _json_safe(value) for key, value in decoded.items()}
        return {"input": raw}
    if raw is None:
        return {}
    return {"input": _json_safe(raw)}


def _conversation_id_from(step: IntermediateStep) -> str:
    """The conversation id NAT stamped on a workflow event, if it did.

    ``nat.runtime.runner`` copies it into ``metadata.provided_metadata`` on
    ``WORKFLOW_START`` and ``WORKFLOW_END``. It is not on any other event.
    """
    metadata = step.metadata
    provided = (
        metadata.get("provided_metadata")
        if isinstance(metadata, dict)
        else getattr(metadata, "provided_metadata", None)
    )
    if isinstance(provided, dict):
        value = provided.get("conversation_id")
        if value:
            return str(value)
    return ""


# ── the exporter ───────────────────────────────────────────────────────────


class SynapStreamExporter(BaseExporter):
    """Reports a NAT run's events on an open Synap gRPC stream.

    Silent when no stream is running: every call in ``stream_events`` no-ops,
    so a workflow that has not started one behaves exactly as it did before.
    Nothing here can raise into the run.

    Args:
        sdk: A :class:`MaximemSynapSDK`. For the reports to go anywhere,
            ``await sdk.instance.listen()`` must already have been called on
            it — the YAML factory below does that for you.
        conversation_id: Used when NAT's own conversation id is unset. NAT
            carries one per run (set by the HTTP front end, or by
            ``session.run``), and that one wins where it exists.
        user_id: Used when NAT's context has no user id.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on
            a B2C instance (``user_context_isolation=equals_customer``): the
            server refuses a call carrying one. NAT has no equivalent of it,
            so it can only be configured here.
        drain_timeout_s: How long ``stop()`` waits for queued events.
        context_state: NAT context state. Defaults to the ambient one.
    """

    # Reset on every isolated copy. ExporterManager gives each concurrent run
    # its own exporter via `copy.copy`, which is shallow: a plain attribute
    # would be one queue and one reasoning buffer shared by every run in
    # flight, and two conversations would report into each other.
    _events: IsolatedAttribute[asyncio.Queue] = IsolatedAttribute(asyncio.Queue)
    _pending_reasoning: IsolatedAttribute[list] = IsolatedAttribute(list)
    _worker: IsolatedAttribute[list] = IsolatedAttribute(list)

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        drain_timeout_s: float = _DEFAULT_DRAIN_TIMEOUT_S,
        context_state: Optional[ContextState] = None,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStreamExporter requires a non-None sdk")
        super().__init__(context_state)
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        self.drain_timeout_s = drain_timeout_s

    # ── ids ────────────────────────────────────────────────────────────────

    def _ids(self, step: IntermediateStep) -> dict:
        """Scope for one event: NAT's own ids first, configured ones second."""
        conversation_id = _conversation_id_from(step)
        if not conversation_id:
            with contextlib.suppress(Exception):
                conversation_id = self._context_state.conversation_id.get() or ""
        user_id = ""
        with contextlib.suppress(Exception):
            user_id = self._context_state.user_id.get() or ""
        return {
            "conversation_id": conversation_id or self.conversation_id,
            "user_id": user_id or self.user_id,
            "customer_id": self.customer_id,
        }

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def _pre_start(self) -> None:
        """Remember the loop and start the single drain worker."""
        self._loop = asyncio.get_running_loop()
        if not self._worker:
            self._worker.append(self._loop.create_task(self._drain()))

    async def _cleanup(self) -> None:
        """Let what is queued reach the stream, then stop the worker.

        ``BaseExporter.stop`` calls this before dropping the subscription, and
        the last thing queued is the assistant turn.
        """
        if not self._worker:
            return
        worker = self._worker.pop()
        try:
            await asyncio.wait_for(self._events.join(), timeout=self.drain_timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.debug(
                "synap stream exporter: %d event(s) still queued at shutdown",
                self._events.qsize(),
            )
        except Exception:  # noqa: BLE001 — teardown must not mask the run
            logger.debug("synap stream exporter: drain failed", exc_info=True)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker

    # ── the subscription callback ──────────────────────────────────────────

    def export(self, event: IntermediateStep) -> None:
        """Queue one NAT event. Called inline from the agent's own loop."""
        if not isinstance(event, IntermediateStep) or not self._running:
            return
        loop = self._loop
        if loop is None:
            return
        try:
            if _running_loop() is loop:
                self._events.put_nowait(event)
            else:
                # A framework wrapper whose callbacks run on a worker thread.
                # asyncio.Queue is not thread safe, so hop to the loop first.
                loop.call_soon_threadsafe(self._events.put_nowait, event)
        except Exception:  # noqa: BLE001 — never break the agent run
            logger.debug("synap stream exporter: could not queue event", exc_info=True)

    async def _drain(self) -> None:
        while True:
            event = await self._events.get()
            try:
                await self._report(event)
            except Exception:  # noqa: BLE001 — one bad event is not the run
                logger.debug("synap stream exporter: report failed", exc_info=True)
            finally:
                self._events.task_done()

    # ── one event ──────────────────────────────────────────────────────────

    async def _report(self, step: IntermediateStep) -> None:
        event_type = step.event_type
        ids = self._ids(step)

        if event_type == IntermediateStepType.WORKFLOW_START:
            text = _user_text(step.data.input if step.data is not None else None)
            if text:
                await report_turn(self.sdk, role="user", content=text, **ids)
            return

        if event_type == IntermediateStepType.WORKFLOW_END:
            # Whatever is held is the final answer, which is about to go out
            # as the assistant turn. Reporting it as reasoning first would say
            # the same thing twice.
            self._pending_reasoning.clear()
            text = _assistant_text(step.data.output if step.data is not None else None)
            if text:
                await report_turn(self.sdk, role="assistant", content=text, **ids)
            return

        if event_type == IntermediateStepType.TOOL_START:
            await self._flush_reasoning(ids)
            await report_tool_call(
                self.sdk,
                tool_name=step.name or "tool",
                tool_args=_tool_args(step),
                # START and END of one NAT step share a UUID, which is what
                # ties the call to its result.
                tool_call_id=step.UUID,
                **ids,
            )
            return

        if event_type == IntermediateStepType.TOOL_END:
            await report_tool_result(
                self.sdk,
                result=_unwrap_result(
                    step.data.output if step.data is not None else None
                ),
                tool_name=step.name or "",
                tool_call_id=step.UUID,
                **ids,
            )
            return

        if event_type == IntermediateStepType.LLM_END:
            await self._flush_reasoning(ids)
            text = _assistant_text(step.data.output if step.data is not None else None)
            if text.strip():
                self._pending_reasoning.append(text.strip())

    async def _flush_reasoning(self, ids: dict) -> None:
        """Release the held LLM output: the run went on, so it was not the answer."""
        if not self._pending_reasoning:
            return
        content = self._pending_reasoning.pop()
        self._pending_reasoning.clear()
        # No thought_type: the release is triggered by "the run continued",
        # which is a tool decision in a ReAct loop and another model call in
        # other shapes. Labelling it one of those would be a guess.
        await report_reasoning(self.sdk, content=content, **ids)


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


# ── NAT YAML-wired plugin ──────────────────────────────────────────────────


class SynapStreamExporterConfig(TelemetryExporterBaseConfig, name="synap_stream"):
    """YAML-wired config for Synap stream reporting.

    Example YAML::

        general:
          telemetry:
            tracing:
              synap:
                _type: synap_stream
                customer_id: "acme"     # B2B instances only
    """

    conversation_id: str = Field(
        default="",
        description=(
            "Fallback conversation id, used only when NAT's own conversation "
            "id is unset for the run."
        ),
    )
    user_id: str = Field(
        default="",
        description=(
            "Fallback user id, used only when NAT's context carries none."
        ),
    )
    customer_id: str = Field(
        default="",
        description=(
            "B2B ONLY. Required on an instance with "
            "user_context_isolation=strict; NOT accepted on a B2C instance, "
            "where the server refuses a call that carries one."
        ),
    )
    drain_timeout_s: float = Field(
        default=_DEFAULT_DRAIN_TIMEOUT_S,
        description=(
            "Seconds to wait on shutdown for queued events to reach Synap."
        ),
    )
    api_key: OptionalSecretStr = Field(
        default=None,
        description=(
            "Synap API key. When omitted, falls back to SYNAP_API_KEY env var."
        ),
    )
    instance_id: str = Field(
        default="",
        description=(
            "Optional Synap instance ID. When empty the SDK resolves it from "
            "the API key via /auth/whoami. Falls back to SYNAP_INSTANCE_ID env."
        ),
    )


@register_telemetry_exporter(config_type=SynapStreamExporterConfig)
async def synap_stream_exporter(config: SynapStreamExporterConfig, builder: Builder):
    """Open a Synap gRPC stream and yield an exporter that reports onto it."""
    api_key: Optional[str] = get_secret_value(config.api_key) or os.environ.get(
        "SYNAP_API_KEY"
    )
    if api_key is None:
        raise RuntimeError(
            "Synap API key is not set. Provide it via "
            "SynapStreamExporterConfig.api_key or the SYNAP_API_KEY "
            "environment variable."
        )

    sdk = MaximemSynapSDK(
        instance_id=config.instance_id or os.environ.get("SYNAP_INSTANCE_ID", ""),
        api_key=api_key,
        _force_new=True,
    )
    await sdk.initialize()
    # Without this there is no stream and every report is a silent no-op, so
    # the exporter would look healthy and do nothing at all.
    await sdk.instance.listen()

    try:
        yield SynapStreamExporter(
            sdk=sdk,
            conversation_id=config.conversation_id,
            user_id=config.user_id,
            customer_id=config.customer_id,
            drain_timeout_s=config.drain_timeout_s,
        )
    finally:
        # stop_listening closes the session the stream opened. The telemetry
        # row for a turn is written when the NEXT user message arrives, so
        # without it the last turn of the workflow is never committed.
        try:
            await sdk.instance.stop_listening()
        except Exception:  # noqa: BLE001 — teardown must not mask workflow errors
            logger.exception("synap_stream_exporter: stop_listening raised")
        teardown = getattr(sdk, "shutdown", None) or getattr(sdk, "close", None)
        if teardown is not None:
            try:
                await teardown()
            except Exception:  # noqa: BLE001
                logger.exception("synap_stream_exporter: SDK teardown raised")


__all__ = [
    "SynapStreamExporter",
    "SynapStreamExporterConfig",
    "synap_stream_exporter",
]
