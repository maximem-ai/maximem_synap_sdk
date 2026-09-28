"""Report an Agno run to Synap, so the next turn starts warm.

This package backed Agno's *user memories* and reported nothing about the run
itself. Synap saw whatever the memory manager chose to store, never the
question, the reply, the tools or the model's reasoning. Anticipation had
nothing to work with, so every fetch stayed a cold retrieval.

``create_synap_hooks`` returns the hook keyword arguments an ``Agent`` takes::

    agent = Agent(
        db=SynapDb(sdk, customer_id="acme"),
        model=OpenAIChat(id="gpt-4o-mini"),
        **create_synap_hooks(sdk, customer_id="acme"),
    )
    agent.run("Where is my order?", user_id="alice")

Two rules hold throughout.

*Silent without a stream.* Everything here needs an active
``sdk.instance.listen()``. Without one every hook is a no-op, so adding them
changes nothing for someone who has not opted into streaming. Nothing here
writes over REST either: this package has never recorded conversation turns, and
a hook that quietly started doing so would double-extract against the stream and
bill for turns the caller never asked to store.

*Never raises.* Agno logs and continues when a hook raises, but a hook that
throws inside a guardrail chain is still a hazard, and telemetry has no business
interrupting a run. ``stream_events`` swallows everything by design and the
bridge around it is wrapped too.

## What goes on the stream

===================  ==========================================================
Event                Where it comes from
===================  ==========================================================
user turn            pre-hook, ``run_input.input_content_string()``
assistant turn       post-hook, ``run_output.content``
tool call            post-hook, ``run_output.tools[].tool_call_id`` / ``.tool_name``
                     / ``.tool_args``
tool result          post-hook, ``run_output.tools[].result``, under the same
                     ``tool_call_id``
reasoning            post-hook, ``run_output.reasoning_content`` and
                     ``run_output.reasoning_steps[]``
===================  ==========================================================

The tool and reasoning events are reported *before* the assistant turn, because
``assistant_message`` is the event anticipation acts on: it means the turn has
ended, and everything that explains the turn should already be in front of it.

⚠ **The tool events arrive after the run, not during it.** Agno's ``tool_hooks``
do fire around each call as it happens, but the arguments Agno builds for a tool
hook (``agent``, ``team``, ``run_context``, ``name``/``function_name``,
``function``/``func``/``function_call``, ``args``/``arguments``) carry no call id
-- ``function_call`` is bound to the *next* link in the middleware chain, not to
the ``FunctionCall`` object that holds ``call_id``. Pairing a call with its
result from there would mean inventing an id. ``run_output.tools`` carries Agno's
own ``tool_call_id`` on both halves, so the pairing is real; the cost is that the
whole batch lands at the end of the run.

## Sync or async

Agno's sync path (``agent.run()``) **skips** coroutine hooks with a warning; its
async path (``agent.arun()``) runs both kinds. :func:`create_synap_hooks`
therefore returns plain functions, which work on both paths, and bridges to the
async SDK through ``run_async``.

⚠ On ``arun()`` that bridge drives the coroutine on the *running* loop via
``nest_asyncio``, which is the same trick every sync surface in these
integrations uses but does not work on every loop implementation (uvloop, in
particular). If you only ever call ``arun()``, use
:func:`create_synap_async_hooks` instead: the hooks are awaited natively and no
loop is patched.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    """A content field as text. Agno types ``RunOutput.content`` as ``Any``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        try:
            return str(dump(exclude_none=True)).strip()
        except Exception:  # noqa: BLE001: a model that will not dump is not fatal
            pass
    return str(value).strip()


def _reasoning_step_text(step: Any) -> str:
    """One ``ReasoningStep`` as a line of reasoning.

    Its four text fields are each optional and each carry a different part of
    the thought, so they are joined rather than picked between.
    """
    parts = [
        _text(getattr(step, field, None))
        for field in ("title", "reasoning", "action", "result")
    ]
    return "\n".join(p for p in parts if p)


def _ids(
    run_context: Any,
    user_id: Optional[str],
    conversation_id: str,
    default_user_id: str,
    customer_id: str,
) -> Dict[str, str]:
    """The scope every event on this run carries.

    ``conversation_id`` defaults to Agno's ``session_id``, which is the id that
    actually groups a conversation's runs; an explicit one overrides it for
    callers who key conversations their own way. ``user_id`` comes off the run
    Agno is executing, so one agent serving many users reports each under their
    own scope.
    """
    return {
        "conversation_id": conversation_id or _text(getattr(run_context, "session_id", "")),
        "user_id": user_id or _text(getattr(run_context, "user_id", "")) or default_user_id,
        "customer_id": customer_id,
    }


async def _report_input(
    sdk: Any, run_input: Any, ids: Dict[str, str]
) -> None:
    content = ""
    if run_input is not None:
        as_string = getattr(run_input, "input_content_string", None)
        if callable(as_string):
            try:
                content = _text(as_string())
            except Exception:  # noqa: BLE001: telemetry never breaks the run
                content = ""
        if not content:
            content = _text(getattr(run_input, "input_content", None))
    if content:
        await report_turn(sdk, role="user", content=content, **ids)


async def _report_output(
    sdk: Any,
    run_output: Any,
    ids: Dict[str, str],
    *,
    report_tool_results: bool,
) -> None:
    if run_output is None:
        return

    # Reasoning first: it is what led to the tools and the answer.
    reasoning = _text(getattr(run_output, "reasoning_content", None))
    if reasoning:
        await report_reasoning(
            sdk, content=reasoning, thought_type="reasoning_content", **ids
        )
    steps: List[Any] = list(getattr(run_output, "reasoning_steps", None) or [])
    for index, step in enumerate(steps):
        text = _reasoning_step_text(step)
        if text:
            await report_reasoning(
                sdk, content=text, step_index=index,
                thought_type="reasoning_step", **ids,
            )

    for execution in list(getattr(run_output, "tools", None) or []):
        call_id = _text(getattr(execution, "tool_call_id", None))
        name = _text(getattr(execution, "tool_name", None)) or "tool"
        args = getattr(execution, "tool_args", None)
        await report_tool_call(
            sdk,
            tool_name=name,
            tool_args=args if isinstance(args, dict) else None,
            tool_call_id=call_id,
            **ids,
        )
        result = getattr(execution, "result", None)
        if report_tool_results and result is not None:
            await report_tool_result(
                sdk, result=result, tool_name=name, tool_call_id=call_id, **ids
            )

    # The assistant turn last: it is the moment the turn ends, and anticipation
    # reads it as the cue to predict the next one.
    content = _text(getattr(run_output, "content", None))
    if content:
        await report_turn(sdk, role="assistant", content=content, **ids)


def _validate(sdk: Any, site: str) -> None:
    if sdk is None:
        raise ValueError(f"{site} requires a non-None sdk")


def create_synap_async_hooks(
    sdk: MaximemSynapSDK,
    *,
    customer_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    report_tool_results: bool = True,
) -> Dict[str, List[Callable[..., Any]]]:
    """``pre_hooks`` / ``post_hooks`` as coroutine functions, for ``agent.arun()``.

    ⚠ Agno's synchronous ``agent.run()`` **skips coroutine hooks** with a warning
    and reports nothing. Use :func:`create_synap_hooks` unless every call site is
    ``arun()``.

    Args:
        sdk: A configured SDK. Without an active ``listen()`` stream every hook
            is a no-op.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        conversation_id: Overrides Agno's ``session_id`` as the conversation
            these runs belong to. Leave it unset and the session id is used.
        user_id: Fallback Synap user scope, used only when the run itself names
            no user. Prefer passing ``user_id=`` to ``agent.run()``.
        report_tool_results: Whether tool results are reported. ⚠ A tool result
            is usually your own customer's data. It is an anticipation hint and
            never becomes a long-term memory, but it does leave your process.

    Returns:
        A dict to splat into ``Agent(...)``.
    """
    _validate(sdk, "create_synap_async_hooks")
    default_user_id = user_id

    async def synap_pre_hook(
        run_input: Any = None,
        run_context: Any = None,
        user_id: Optional[str] = None,  # noqa: A002: Agno matches hook args by name
    ) -> None:
        ids = _ids(run_context, user_id, conversation_id, default_user_id, customer_id)
        await _report_input(sdk, run_input, ids)

    async def synap_post_hook(
        run_output: Any = None,
        run_context: Any = None,
        user_id: Optional[str] = None,  # noqa: A002: Agno matches hook args by name
    ) -> None:
        ids = _ids(run_context, user_id, conversation_id, default_user_id, customer_id)
        await _report_output(
            sdk, run_output, ids, report_tool_results=report_tool_results
        )

    return {"pre_hooks": [synap_pre_hook], "post_hooks": [synap_post_hook]}


def create_synap_hooks(
    sdk: MaximemSynapSDK,
    *,
    customer_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    report_tool_results: bool = True,
) -> Dict[str, List[Callable[..., Any]]]:
    """``pre_hooks`` / ``post_hooks`` that work on both ``run()`` and ``arun()``.

    Agno's sync path skips coroutine hooks, so these are plain functions that
    bridge to the async SDK through ``run_async``. See the module docstring for
    the ``nest_asyncio`` caveat on ``arun()``.

    Args and return value are the same as :func:`create_synap_async_hooks`.

    Example::

        agent = Agent(
            db=SynapDb(sdk, customer_id="acme"),
            model=OpenAIChat(id="gpt-4o-mini"),
            **create_synap_hooks(sdk, customer_id="acme"),
        )
        agent.run("Where is my order?", user_id="alice")
    """
    _validate(sdk, "create_synap_hooks")
    default_user_id = user_id

    def synap_pre_hook(
        run_input: Any = None,
        run_context: Any = None,
        user_id: Optional[str] = None,  # noqa: A002: Agno matches hook args by name
    ) -> None:
        ids = _ids(run_context, user_id, conversation_id, default_user_id, customer_id)
        _bridge(_report_input(sdk, run_input, ids), "pre_hook")

    def synap_post_hook(
        run_output: Any = None,
        run_context: Any = None,
        user_id: Optional[str] = None,  # noqa: A002: Agno matches hook args by name
    ) -> None:
        ids = _ids(run_context, user_id, conversation_id, default_user_id, customer_id)
        _bridge(
            _report_output(
                sdk, run_output, ids, report_tool_results=report_tool_results
            ),
            "post_hook",
        )

    return {"pre_hooks": [synap_pre_hook], "post_hooks": [synap_post_hook]}


def _bridge(coro: Any, site: str) -> None:
    """Drive one report coroutine from a sync hook, swallowing everything.

    ``stream_events`` already swallows its own failures; this guards the bridge
    itself, which can fail on a loop ``nest_asyncio`` cannot patch.
    """
    try:
        run_async(coro)
    except Exception as exc:  # noqa: BLE001: telemetry never breaks a run
        close = getattr(coro, "close", None)
        if callable(close):
            # A coroutine the bridge never started would otherwise warn on GC.
            close()
        logger.debug("synap %s stream report failed: %s", site, exc, exc_info=True)


__all__ = ["create_synap_hooks", "create_synap_async_hooks"]
