"""Tests for create_synap_hooks: Agno pre/post hooks that report the turn.

Documented contract (from hooks.py):
- 'Silent without a stream': no `listen()`, no events, and nothing over REST.
- 'Never raises': a hook is telemetry and must not interrupt a run.
- A tool call and its result share Agno's own `tool_call_id`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_agno import create_synap_async_hooks, create_synap_hooks


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def _streaming_sdk(is_listening=True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _run_context(session_id="sess-1", user_id="alice"):
    return SimpleNamespace(session_id=session_id, user_id=user_id, run_id="run-1")


def _run_input(content="where is my order"):
    return SimpleNamespace(
        input_content=content,
        input_content_string=lambda: content,
    )


def _tool_execution(tool_call_id="call_7", tool_name="lookup_order",
                    tool_args=None, result="status: shipped"):
    return SimpleNamespace(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args if tool_args is not None else {"pnr": "QX41RT"},
        result=result,
    )


def _run_output(content="it ships tomorrow", tools=None,
                reasoning_content=None, reasoning_steps=None):
    return SimpleNamespace(
        content=content,
        tools=tools,
        reasoning_content=reasoning_content,
        reasoning_steps=reasoning_steps,
    )


def _hooks(sdk, **kwargs):
    h = create_synap_hooks(sdk, customer_id="acme", **kwargs)
    return h["pre_hooks"][0], h["post_hooks"][0]


def _sent(sdk, event_type):
    return [
        c.kwargs for c in sdk.instance.send_message.await_args_list
        if c.kwargs["event_type"] == event_type
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        create_synap_hooks(None)


def test_returns_the_kwargs_an_agent_takes():
    hooks = create_synap_hooks(MagicMock())
    assert set(hooks) == {"pre_hooks", "post_hooks"}
    assert len(hooks["pre_hooks"]) == 1 and len(hooks["post_hooks"]) == 1


# ---------------------------------------------------------------------------
# Stream first, and never a REST write on top of it
# ---------------------------------------------------------------------------


class TestStreamOnlyNoRestDoubleWrite:
    def test_no_stream_means_no_events_at_all(self):
        sdk = _streaming_sdk(is_listening=False)
        pre, post = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        post(run_output=_run_output(), run_context=_run_context(), user_id="alice")
        sdk.instance.send_message.assert_not_awaited()
        sdk.instance.record_tool_call.assert_not_awaited()

    def test_a_reported_turn_is_NOT_also_written_over_rest(self):
        """The server persists user_message/assistant_message from the stream
        itself, so a record_message here would store and extract it twice."""
        sdk = _streaming_sdk()
        pre, post = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        post(run_output=_run_output(), run_context=_run_context(), user_id="alice")
        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The five events
# ---------------------------------------------------------------------------


class TestTheFiveEvents:
    def test_the_user_turn_is_reported_from_the_run_input(self):
        sdk = _streaming_sdk()
        pre, _ = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        [turn] = _sent(sdk, "user_message")
        assert turn["content"] == "where is my order"
        assert turn["role"] == "user"

    def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(), run_context=_run_context(), user_id="alice")
        [turn] = _sent(sdk, "assistant_message")
        assert turn["content"] == "it ships tomorrow"
        assert turn["role"] == "assistant"

    def test_a_tool_call_is_reported_with_agnos_own_id(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(tools=[_tool_execution()]),
             run_context=_run_context(), user_id="alice")
        call = sdk.instance.record_tool_call.await_args
        assert call.args[0] == "lookup_order"
        assert call.args[1] == {"pnr": "QX41RT"}
        assert call.kwargs["tool_call_id"] == "call_7"

    def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(tools=[_tool_execution()]),
             run_context=_run_context(), user_id="alice")
        assert sdk.instance.record_tool_result.await_args.args[0] == "status: shipped"

    def test_the_call_and_its_result_share_an_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(tools=[_tool_execution()]),
             run_context=_run_context(), user_id="alice")
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        )

    def test_every_tool_in_the_run_is_reported(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(
            run_output=_run_output(tools=[
                _tool_execution(tool_call_id="a"),
                _tool_execution(tool_call_id="b"),
            ]),
            run_context=_run_context(), user_id="alice",
        )
        ids = [c.kwargs["tool_call_id"]
               for c in sdk.instance.record_tool_call.await_args_list]
        assert ids == ["a", "b"]

    def test_report_tool_results_can_be_turned_off(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk, report_tool_results=False)
        post(run_output=_run_output(tools=[_tool_execution()]),
             run_context=_run_context(), user_id="alice")
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_not_awaited()

    def test_reasoning_content_is_reported(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(reasoning_content="I should check the booking"),
             run_context=_run_context(), user_id="alice")
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "I should check the booking"
        )

    def test_each_reasoning_step_is_reported_with_its_index(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        steps = [
            SimpleNamespace(title="Look it up", reasoning="The PNR is known",
                            action=None, result=None),
            SimpleNamespace(title="Answer", reasoning=None,
                            action="I will reply", result="Replied"),
        ]
        post(run_output=_run_output(reasoning_steps=steps),
             run_context=_run_context(), user_id="alice")
        calls = sdk.instance.record_thinking.await_args_list
        assert [c.kwargs["step_index"] for c in calls] == [0, 1]
        assert calls[0].args[0] == "Look it up\nThe PNR is known"

    def test_an_empty_reasoning_step_reports_nothing(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(
            run_output=_run_output(reasoning_steps=[
                SimpleNamespace(title=None, reasoning=None, action=None, result=None)
            ]),
            run_context=_run_context(), user_id="alice",
        )
        sdk.instance.record_thinking.assert_not_awaited()

    def test_an_empty_reply_reports_no_assistant_turn(self):
        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=_run_output(content=None),
             run_context=_run_context(), user_id="alice")
        assert _sent(sdk, "assistant_message") == []


class TestTheAssistantTurnComesLast:
    def test_tools_and_reasoning_are_reported_before_the_turn_ends(self):
        """assistant_message means the turn is over, so what explains it has
        to already be on the wire."""
        sdk = _streaming_sdk()
        order = []
        sdk.instance.record_thinking = AsyncMock(
            side_effect=lambda *a, **k: order.append("thinking"))
        sdk.instance.record_tool_call = AsyncMock(
            side_effect=lambda *a, **k: order.append("tool_call"))
        sdk.instance.record_tool_result = AsyncMock(
            side_effect=lambda *a, **k: order.append("tool_result"))
        sdk.instance.send_message = AsyncMock(
            side_effect=lambda *a, **k: order.append(k["event_type"]))
        _, post = _hooks(sdk)
        post(
            run_output=_run_output(
                tools=[_tool_execution()], reasoning_content="thinking hard"),
            run_context=_run_context(), user_id="alice",
        )
        assert order == [
            "thinking", "tool_call", "tool_result", "assistant_message"
        ]


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    def test_conversation_id_defaults_to_the_agno_session(self):
        sdk = _streaming_sdk()
        pre, _ = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(session_id="sess-9"),
            user_id="alice")
        assert _sent(sdk, "user_message")[0]["conversation_id"] == "sess-9"

    def test_an_explicit_conversation_id_wins(self):
        sdk = _streaming_sdk()
        pre, _ = _hooks(sdk, conversation_id="conv-mine")
        pre(run_input=_run_input(), run_context=_run_context(session_id="sess-9"),
            user_id="alice")
        assert _sent(sdk, "user_message")[0]["conversation_id"] == "conv-mine"

    def test_the_user_id_comes_off_the_run(self):
        """One agent serving many users reports each under their own scope."""
        sdk = _streaming_sdk()
        pre, _ = _hooks(sdk, user_id="fallback")
        pre(run_input=_run_input(), run_context=_run_context(user_id="bob"),
            user_id="bob")
        assert _sent(sdk, "user_message")[0]["user_id"] == "bob"

    def test_the_constructor_user_id_is_only_a_fallback(self):
        sdk = _streaming_sdk()
        pre, _ = _hooks(sdk, user_id="fallback")
        pre(run_input=_run_input(),
            run_context=SimpleNamespace(session_id="s", user_id=None),
            user_id=None)
        assert _sent(sdk, "user_message")[0]["user_id"] == "fallback"

    def test_an_empty_customer_id_is_not_sent(self):
        """A B2C instance rejects a call carrying customer_id."""
        sdk = _streaming_sdk()
        pre = create_synap_hooks(sdk)["pre_hooks"][0]
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        assert "customer_id" not in _sent(sdk, "user_message")[0]


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------


class TestNoneOfItBreaksTheRun:
    def test_a_dead_stream_does_not_raise(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        pre, post = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        post(run_output=_run_output(tools=[_tool_execution()]),
             run_context=_run_context(), user_id="alice")

    def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        sdk = MagicMock(spec=[])
        pre, post = _hooks(sdk)
        pre(run_input=_run_input(), run_context=_run_context(), user_id="alice")
        post(run_output=_run_output(), run_context=_run_context(), user_id="alice")

    def test_a_run_input_whose_stringifier_throws_does_not_raise(self):
        sdk = _streaming_sdk()
        def boom():
            raise RuntimeError("bad input")
        pre, _ = _hooks(sdk)
        pre(
            run_input=SimpleNamespace(
                input_content="fallback text", input_content_string=boom),
            run_context=_run_context(), user_id="alice",
        )
        assert _sent(sdk, "user_message")[0]["content"] == "fallback text"

    def test_a_run_output_whose_field_explodes_does_not_raise(self):
        """stream_events swallows its own failures; this covers the bridge
        around them, which reads the run output's fields."""
        class Exploding:
            reasoning_content = None
            reasoning_steps = None
            content = "it ships tomorrow"

            @property
            def tools(self):
                raise RuntimeError("boom")

        sdk = _streaming_sdk()
        _, post = _hooks(sdk)
        post(run_output=Exploding(), run_context=_run_context(), user_id="alice")

    def test_hooks_called_with_nothing_do_not_raise(self):
        sdk = _streaming_sdk()
        pre, post = _hooks(sdk)
        pre()
        post()
        sdk.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Async variant, for callers who only ever use arun()
# ---------------------------------------------------------------------------


class TestAsyncHooks:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_async_hooks(None)

    @pytest.mark.asyncio
    async def test_they_are_coroutine_functions(self):
        from inspect import iscoroutinefunction

        hooks = create_synap_async_hooks(MagicMock())
        assert iscoroutinefunction(hooks["pre_hooks"][0])
        assert iscoroutinefunction(hooks["post_hooks"][0])

    @pytest.mark.asyncio
    async def test_they_report_the_same_events(self):
        sdk = _streaming_sdk()
        hooks = create_synap_async_hooks(sdk, customer_id="acme")
        await hooks["pre_hooks"][0](
            run_input=_run_input(), run_context=_run_context(), user_id="alice")
        await hooks["post_hooks"][0](
            run_output=_run_output(tools=[_tool_execution()]),
            run_context=_run_context(), user_id="alice")
        assert _sent(sdk, "user_message")[0]["content"] == "where is my order"
        assert _sent(sdk, "assistant_message")[0]["content"] == "it ships tomorrow"
        assert sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"] == "call_7"

    @pytest.mark.asyncio
    async def test_a_dead_stream_does_not_raise(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        hooks = create_synap_async_hooks(sdk)
        await hooks["pre_hooks"][0](
            run_input=_run_input(), run_context=_run_context(), user_id="alice")


# ---------------------------------------------------------------------------
# Guards against upstream renames: these are the names the hooks read
# ---------------------------------------------------------------------------


def test_agno_still_passes_the_hook_args_we_declare():
    """Agno filters hook arguments by parameter NAME, so a rename upstream
    silently stops feeding the hook rather than failing."""
    from agno.utils.hooks import filter_hook_args

    pre, post = _hooks(_streaming_sdk())
    pre_args = filter_hook_args(
        pre,
        {"run_input": 1, "run_context": 2, "agent": 3, "session": 4,
         "user_id": 5, "debug_mode": 6, "metadata": 7},
    )
    assert set(pre_args) == {"run_input", "run_context", "user_id"}

    post_args = filter_hook_args(
        post,
        {"run_output": 1, "agent": 2, "session": 3, "run_context": 4,
         "user_id": 5, "debug_mode": 6, "metadata": 7},
    )
    assert set(post_args) == {"run_output", "run_context", "user_id"}


def test_the_agno_fields_the_hooks_read_still_exist():
    from agno.models.response import ToolExecution
    from agno.run.agent import RunInput, RunOutput
    from agno.run.base import RunContext

    assert {"tool_call_id", "tool_name", "tool_args", "result"} <= set(
        ToolExecution.__dataclass_fields__
    )
    assert {"content", "tools", "reasoning_content", "reasoning_steps"} <= set(
        RunOutput.__dataclass_fields__
    )
    assert "input_content" in RunInput.__dataclass_fields__
    assert callable(RunInput.input_content_string)
    assert {"session_id", "user_id"} <= set(RunContext.__dataclass_fields__)


def test_agno_skips_coroutine_hooks_on_the_sync_path():
    """Why create_synap_hooks returns plain functions: agno's sync run()
    logs a warning and skips an async hook, reporting nothing."""
    import inspect

    from agno.agent import _hooks as agno_hooks

    source = inspect.getsource(agno_hooks.execute_pre_hooks)
    assert "iscoroutinefunction(hook)" in source
    assert "Skipping hook" in source
