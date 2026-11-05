"""Tests for :mod:`synap_harnessx.extension`, driving the real HarnessX engine."""

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from harnessx import Agent, AgentConfig, PermissionLevel, ProviderResponse, ToolCall
from harnessx.providers import LLMProvider

from synap_harnessx import SYNAP_MEMORY_PROMPT, SynapExtension


class ScriptedProvider(LLMProvider):
    """Replays fixed responses and remembers the system prompt of each call."""

    def __init__(self, responses: Iterable[ProviderResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        return next(self._responses)

    async def count_tokens(self, **kwargs: Any) -> int:
        return 0


def make_agent(sdk, *responses, **ext_kwargs):
    provider = ScriptedProvider(responses or [ProviderResponse(text="hello back")])
    ext = SynapExtension(sdk=sdk, user_id="alice", **ext_kwargs)
    agent = Agent(
        config=AgentConfig(model="claude-sonnet-4-6", system_prompt="BASE"),
        provider=provider,
        extensions=[ext],
    )
    return agent, provider, ext


def system_of(provider, index=0):
    system = provider.calls[index]["system"]
    return system if isinstance(system, str) else str(system)


# -- construction ----------------------------------------------------------


def test_requires_sdk():
    with pytest.raises(ValueError, match="sdk"):
        SynapExtension(sdk=None, user_id="alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError, match="user_id"):
        SynapExtension(sdk=mock_sdk, user_id="")


def test_prompt_must_be_str_or_none(mock_sdk):
    with pytest.raises(TypeError):
        SynapExtension(sdk=mock_sdk, user_id="a", memory_prompt=5)  # type: ignore[arg-type]


def test_prompt_needs_slot(mock_sdk):
    with pytest.raises(ValueError, match="synap_memory"):
        SynapExtension(sdk=mock_sdk, user_id="a", memory_prompt="no slot")


def test_default_conversation_id_is_agent_session(mock_sdk):
    agent, _, ext = make_agent(mock_sdk)
    assert ext._conversation_id == agent.session_id


def test_explicit_conversation_id_wins(mock_sdk):
    agent, _, ext = make_agent(mock_sdk, conversation_id="conv-fixed")
    assert ext._conversation_id == "conv-fixed"


# -- recall ----------------------------------------------------------------


async def test_fetches_with_user_message_and_injects_block(mock_sdk):
    agent, provider, _ = make_agent(mock_sdk)
    await agent.run("what do I like?")

    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["search_query"] == ["what do I like?"]
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] is None
    assert kwargs["mode"] == "fast"

    system = system_of(provider)
    assert system.startswith("BASE")
    assert "<synap_memory>" in system
    assert "User is an engineer" in system


async def test_empty_recall_injects_nothing(mock_sdk):
    mock_sdk.fetch.return_value.formatted_context = ""
    agent, provider, _ = make_agent(mock_sdk)
    await agent.run("hi")
    assert "synap_memory" not in system_of(provider)


async def test_recall_failure_does_not_end_the_run(failing_sdk):
    agent, provider, _ = make_agent(failing_sdk)
    result = await agent.run("hi")
    assert result.output == "hello back"
    assert "synap_memory" not in system_of(provider)


async def test_memory_prompt_none_fetches_but_injects_nothing(mock_sdk):
    agent, provider, _ = make_agent(mock_sdk, memory_prompt=None)
    await agent.run("hi")
    mock_sdk.fetch.assert_awaited_once()
    assert "synap_memory" not in system_of(provider)


async def test_blank_message_skips_fetch_when_disabled(mock_sdk):
    agent, _, ext = make_agent(mock_sdk, fetch_without_query=False)
    await ext.on_turn_start(agent_context(agent, ext), "   ")
    mock_sdk.fetch.assert_not_awaited()


async def test_block_is_refreshed_each_turn(mock_sdk):
    agent, provider, _ = make_agent(
        mock_sdk, ProviderResponse(text="one"), ProviderResponse(text="two")
    )
    await agent.run("first")
    mock_sdk.fetch.return_value.formatted_context = "second block"
    await agent.run("second")
    assert "second block" in system_of(provider, 1)
    assert "User is an engineer" not in system_of(provider, 1)


def agent_context(agent, ext):
    from harnessx import ExtensionContext

    return ExtensionContext(agent, ext.name)


# -- tools -----------------------------------------------------------------


def test_tools_registered_with_permissions(mock_sdk):
    agent, _, _ = make_agent(mock_sdk)
    assert agent.tools.has_tool("search_memory")
    assert agent.tools.has_tool("store_memory")
    assert agent.tools.get_tool("store_memory").permission_level == PermissionLevel.ASK
    assert agent.tools.get_tool("search_memory").permission_level == PermissionLevel.ALLOW


def test_tools_can_be_disabled(mock_sdk):
    agent, _, _ = make_agent(mock_sdk, register_tools=False)
    assert not agent.tools.has_tool("search_memory")
    assert not agent.tools.has_tool("store_memory")


def test_store_permission_is_configurable(mock_sdk):
    agent, _, _ = make_agent(mock_sdk, store_permission=PermissionLevel.ALLOW)
    assert agent.tools.get_tool("store_memory").permission_level == PermissionLevel.ALLOW


def test_tool_schemas_come_from_docstrings(mock_sdk):
    agent, _, _ = make_agent(mock_sdk)
    schema = agent.tools.get_tool("search_memory").input_schema
    assert "query" in schema["properties"]
    assert schema["properties"]["query"].get("description")


async def test_search_tool_runs_through_engine(mock_sdk):
    call = ToolCall("t1", "search_memory", {"query": "hotels"})
    agent, _, _ = make_agent(
        mock_sdk,
        ProviderResponse(tool_calls=[call], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    )
    await agent.run("find hotels")
    queries = [c.kwargs["search_query"] for c in mock_sdk.fetch.await_args_list]
    assert ["hotels"] in queries
    modes = {c.kwargs["mode"] for c in mock_sdk.fetch.await_args_list}
    assert "accurate" in modes


async def test_store_tool_writes_at_user_scope(mock_sdk):
    agent, _, _ = make_agent(mock_sdk, store_permission=PermissionLevel.ALLOW)
    result = await agent.tools.execute(
        ToolCall("t1", "store_memory", {"content": "likes tea"})
    )
    assert not result.is_error
    assert "ing-001" in result.content
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "likes tea"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] is None


async def test_tool_failure_is_an_error_result_not_a_crash(failing_sdk):
    agent, _, _ = make_agent(failing_sdk, store_permission=PermissionLevel.ALLOW)
    result = await agent.tools.execute(
        ToolCall("t1", "store_memory", {"content": "x"})
    )
    assert result.is_error


# -- scoping ---------------------------------------------------------------


async def test_customer_id_passed_on_b2b(mock_sdk):
    agent, _, _ = make_agent(mock_sdk, customer_id="acme")
    await agent.run("hi")
    assert mock_sdk.fetch.await_args.kwargs["customer_id"] == "acme"


async def test_empty_customer_id_is_dropped_for_b2c(mock_sdk):
    agent, _, _ = make_agent(mock_sdk, customer_id="")
    await agent.run("hi")
    assert mock_sdk.fetch.await_args.kwargs["customer_id"] is None
    kwargs = mock_sdk.conversation.record_message.await_args.kwargs
    assert kwargs["customer_id"] is None


# -- recording -------------------------------------------------------------


async def test_turns_recorded_via_rest_when_no_stream(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=False)
    agent, _, _ = make_agent(mock_sdk)
    await agent.run("hello")

    calls = mock_sdk.conversation.record_message.await_args_list
    assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
        ("user", "hello"),
        ("assistant", "hello back"),
    ]
    assert all(c.kwargs["conversation_id"] == agent.session_id for c in calls)


async def test_stream_used_instead_of_rest_never_both(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=True)
    mock_sdk.instance.send_message = AsyncMock()
    mock_sdk.instance.record_tool_call = AsyncMock()
    mock_sdk.instance.record_tool_result = AsyncMock()
    mock_sdk.instance.end_session = AsyncMock()

    agent, _, _ = make_agent(mock_sdk)
    await agent.run("hello")

    roles = [c.kwargs["event_type"] for c in mock_sdk.instance.send_message.await_args_list]
    assert roles == ["user_message", "assistant_message"]
    mock_sdk.conversation.record_message.assert_not_awaited()


async def test_record_turns_off(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=False)
    agent, _, _ = make_agent(mock_sdk, record_turns=False)
    await agent.run("hello")
    mock_sdk.conversation.record_message.assert_not_awaited()


async def test_record_failure_does_not_end_the_run(failing_sdk):
    failing_sdk.instance = MagicMock(is_listening=False)
    failing_sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=""))
    agent, _, _ = make_agent(failing_sdk)
    result = await agent.run("hello")
    assert result.output == "hello back"


# -- stream events ---------------------------------------------------------


async def test_tool_calls_and_results_reported_on_stream(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=True)
    mock_sdk.instance.send_message = AsyncMock()
    mock_sdk.instance.record_tool_call = AsyncMock()
    mock_sdk.instance.record_tool_result = AsyncMock()
    mock_sdk.instance.end_session = AsyncMock()

    call = ToolCall("t1", "search_memory", {"query": "x"})
    agent, _, _ = make_agent(
        mock_sdk,
        ProviderResponse(tool_calls=[call], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    )
    await agent.run("go")

    started = mock_sdk.instance.record_tool_call.await_args
    assert started.args[0] == "search_memory"
    assert started.kwargs["tool_call_id"] == "t1"
    finished = mock_sdk.instance.record_tool_result.await_args
    assert finished.kwargs["tool_name"] == "search_memory"
    assert finished.kwargs["tool_call_id"] == "t1"


async def test_report_tools_off_registers_no_hooks(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=True)
    mock_sdk.instance.record_tool_call = AsyncMock()
    agent, _, _ = make_agent(mock_sdk, report_tools=False)
    from harnessx.hooks import HookEvent

    assert not agent.hooks._hooks.get(HookEvent.TOOL_CALL_START)


async def test_teardown_ends_session(mock_sdk):
    mock_sdk.instance = MagicMock(is_listening=True)
    mock_sdk.instance.end_session = AsyncMock()
    agent, _, ext = make_agent(mock_sdk)
    await ext.teardown()
    mock_sdk.instance.end_session.assert_awaited_once_with(agent.session_id)


# -- session state ---------------------------------------------------------


async def test_session_state_roundtrip(mock_sdk):
    agent, _, ext = make_agent(mock_sdk)
    ctx = agent_context(agent, ext)
    state = await ext.on_save_session(ctx)
    assert state == {"conversation_id": agent.session_id}

    other = SynapExtension(sdk=mock_sdk, user_id="alice")
    await other.on_load_session(ctx, {"conversation_id": "conv-restored"})
    assert other._conversation_id == "conv-restored"


async def test_pinned_conversation_id_ignores_restored_state(mock_sdk):
    agent, _, ext = make_agent(mock_sdk, conversation_id="conv-fixed")
    await ext.on_load_session(agent_context(agent, ext), {"conversation_id": "other"})
    assert ext._conversation_id == "conv-fixed"


def test_default_prompt_has_slot():
    assert "{synap_memory}" in SYNAP_MEMORY_PROMPT
