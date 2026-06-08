"""Tests for synap_langgraph.short_term — synap_st_prompt + create_synap_st_node.

Covers:
- Construction-time argument validation.
- Cache-hit + cache-miss behaviour against a faked SDK.
- Empty short-term context must not wipe the user's system prompt.
- SDK failure: graceful degrade (default) vs strict raise.
- Async invocation under LangGraph's prompt-callable shape.
- Conversation-id resolution is strictly explicit (no thread_id inference).
- State shape support: both dict and attribute access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from synap_langgraph.short_term import (
    create_synap_st_node,
    synap_st_prompt,
)
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool):
    """Build a stand-in ContextForPromptResponse object (duck-typed).

    The adapter reads only ``available`` + ``formatted_context``, so a
    MagicMock with those attributes is sufficient and avoids importing
    the pydantic model (which would change shape independently).
    """
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "## Summary\nUser likes dark mode.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


def _state_with_messages(*msgs):
    """LangGraph state is typically a dict with a ``messages`` key."""
    return {"messages": list(msgs)}


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestPromptValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_prompt(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_prompt(sdk, "")
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_prompt(sdk, "   ")

    def test_rejects_unknown_style(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_prompt(sdk, "conv_abc", style="poetic")

    def test_accepts_all_documented_styles(self):
        sdk = _fake_sdk()
        for style in ("structured", "narrative", "bullet_points"):
            synap_st_prompt(sdk, "conv_abc", style=style)

    def test_rejects_invalid_on_error(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="on_error"):
            synap_st_prompt(sdk, "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestNodeValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_st_node(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            create_synap_st_node(sdk, "")

    def test_requires_non_empty_state_key(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty state_key"):
            create_synap_st_node(sdk, "conv_abc", state_key="")


# ---------------------------------------------------------------------------
# Prompt callable — cache-hit happy path
# ---------------------------------------------------------------------------


class TestPromptCacheHit:
    @pytest.mark.asyncio
    async def test_prepends_st_block_above_user_system(self):
        sdk = _fake_sdk(formatted="User likes concise answers.")
        prompt = synap_st_prompt(
            sdk,
            "conv_abc",
            system="You are a helpful assistant.",
        )

        result = await prompt(_state_with_messages(HumanMessage(content="hi")))

        # Expect: [SystemMessage(combined), HumanMessage]
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        content = result[0].content
        assert "<synap_short_term_context>" in content
        assert "User likes concise answers." in content
        assert "</synap_short_term_context>" in content
        assert "You are a helpful assistant." in content
        # ST must come first, user system after
        assert content.index("synap_short_term_context") < content.index("helpful assistant")

    @pytest.mark.asyncio
    async def test_calls_sdk_with_explicit_conv_and_style(self):
        sdk = _fake_sdk()
        prompt = synap_st_prompt(sdk, "conv_abc", style="bullet_points", system="X")
        await prompt(_state_with_messages())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble_used(self):
        sdk = _fake_sdk(formatted="hello world")
        prompt = synap_st_prompt(
            sdk,
            "conv_abc",
            system="sys",
            preamble_open="[BEGIN_ST]",
            preamble_close="[END_ST]",
        )
        result = await prompt(_state_with_messages())
        assert "[BEGIN_ST]" in result[0].content
        assert "[END_ST]" in result[0].content
        assert "<synap_short_term_context>" not in result[0].content

    @pytest.mark.asyncio
    async def test_no_preamble_means_raw_concat(self):
        sdk = _fake_sdk(formatted="raw st text")
        prompt = synap_st_prompt(
            sdk,
            "conv_abc",
            system="sys here",
            preamble_open=None,
            preamble_close=None,
        )
        result = await prompt(_state_with_messages())
        assert result[0].content == "raw st text\n\nsys here"


# ---------------------------------------------------------------------------
# Empty ST — the dangerous edge case
# ---------------------------------------------------------------------------


class TestPromptEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_response_does_not_wipe_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        prompt = synap_st_prompt(
            sdk, "conv_abc", system="You are friendly."
        )
        result = await prompt(_state_with_messages(HumanMessage(content="hi")))

        # Must still emit the user's system prompt — never silently drop it.
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are friendly."

    @pytest.mark.asyncio
    async def test_empty_formatted_treated_as_unavailable(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        prompt = synap_st_prompt(sdk, "conv_abc", system="sys")
        result = await prompt(_state_with_messages())
        assert result[0].content == "sys"
        assert "synap_short_term_context" not in result[0].content

    @pytest.mark.asyncio
    async def test_both_empty_returns_messages_only(self):
        sdk = _fake_sdk(formatted=None, available=False)
        prompt = synap_st_prompt(sdk, "conv_abc", system="")
        human = HumanMessage(content="hi")
        result = await prompt(_state_with_messages(human))
        # No SystemMessage prepended when both ST and user system are empty
        assert result == [human]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPromptErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_on_sdk_failure(self, caplog):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("network blip")
        )
        prompt = synap_st_prompt(
            sdk, "conv_abc", system="Stay calm.", on_error="fallback"
        )
        result = await prompt(_state_with_messages(HumanMessage(content="hi")))
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("network blip")
        )
        prompt = synap_st_prompt(
            sdk, "conv_abc", system="X", on_error="raise"
        )
        with pytest.raises(SynapIntegrationError) as ei:
            await prompt(_state_with_messages())
        assert ei.value.operation == "synap_langgraph.synap_st_prompt"
        assert isinstance(ei.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# State shape support
# ---------------------------------------------------------------------------


class TestPromptStateShape:
    @pytest.mark.asyncio
    async def test_supports_attribute_access_state(self):
        sdk = _fake_sdk(formatted="ST")
        prompt = synap_st_prompt(sdk, "conv_abc", system="sys")

        class StateObj:
            def __init__(self, messages):
                self.messages = messages

        s = StateObj([HumanMessage(content="hi")])
        result = await prompt(s)
        assert len(result) == 2
        assert result[1].content == "hi"

    @pytest.mark.asyncio
    async def test_missing_messages_key_treated_as_empty(self):
        sdk = _fake_sdk(formatted="ST")
        prompt = synap_st_prompt(sdk, "conv_abc", system="sys")
        result = await prompt({})  # no messages key at all
        # Should not crash, should still emit SystemMessage with combined content
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)


# ---------------------------------------------------------------------------
# Node — cache-hit + miss + error
# ---------------------------------------------------------------------------


class TestNode:
    @pytest.mark.asyncio
    async def test_writes_st_into_default_state_key(self):
        sdk = _fake_sdk(formatted="User is on the Pro plan.")
        node = create_synap_st_node(sdk, "conv_abc")
        result = await node({})
        assert result == {"synap_st": "User is on the Pro plan."}

    @pytest.mark.asyncio
    async def test_writes_into_custom_state_key(self):
        sdk = _fake_sdk(formatted="hello")
        node = create_synap_st_node(sdk, "conv_abc", state_key="ctx")
        result = await node({})
        assert result == {"ctx": "hello"}

    @pytest.mark.asyncio
    async def test_empty_st_writes_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        node = create_synap_st_node(sdk, "conv_abc")
        result = await node({})
        # Always writes the key so downstream code can `if state["synap_st"]:`
        assert result == {"synap_st": ""}

    @pytest.mark.asyncio
    async def test_fallback_on_sdk_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        node = create_synap_st_node(sdk, "conv_abc")
        result = await node({})
        assert result == {"synap_st": ""}

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        node = create_synap_st_node(sdk, "conv_abc", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await node({})

    @pytest.mark.asyncio
    async def test_passes_style_through(self):
        sdk = _fake_sdk()
        node = create_synap_st_node(sdk, "conv_abc", style="structured")
        await node({})
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )


# ---------------------------------------------------------------------------
# Smoke: imports + public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_langgraph

    assert hasattr(synap_langgraph, "synap_st_prompt")
    assert hasattr(synap_langgraph, "create_synap_st_node")
    assert "synap_st_prompt" in synap_langgraph.__all__
    assert "create_synap_st_node" in synap_langgraph.__all__
