"""Tests for :mod:`synap_deepagents.short_term`."""

import pytest
from synap_integrations_common import SynapIntegrationError

from synap_deepagents.short_term import (
    compose_system_prompt,
    fetch_st_block,
    synap_st_instructions,
)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


async def test_requires_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        await synap_st_instructions(None, "conv-1")


@pytest.mark.parametrize("conversation_id", ["", "   ", None])
async def test_requires_conversation_id(mock_sdk, conversation_id):
    with pytest.raises(ValueError, match="conversation_id"):
        await synap_st_instructions(mock_sdk, conversation_id)


async def test_rejects_unknown_style(mock_sdk):
    with pytest.raises(ValueError, match="unsupported style"):
        await synap_st_instructions(mock_sdk, "conv-1", style="interpretive-dance")


async def test_rejects_unknown_on_error(mock_sdk):
    with pytest.raises(ValueError, match="on_error"):
        await synap_st_instructions(mock_sdk, "conv-1", on_error="explode")


@pytest.mark.parametrize("style", ["structured", "narrative", "bullet_points"])
async def test_accepts_supported_styles(mock_sdk, style):
    await synap_st_instructions(mock_sdk, "conv-1", style=style)
    assert mock_sdk.conversation.context.get_context_for_prompt.await_args.kwargs[
        "style"
    ] == style


# ---------------------------------------------------------------------------
# fetch_st_block
# ---------------------------------------------------------------------------


async def test_fetch_returns_context(mock_sdk):
    assert await fetch_st_block(mock_sdk, "conv-1") == "Recent conversation summary"


async def test_fetch_passes_conversation_id(mock_sdk):
    await fetch_st_block(mock_sdk, "conv-abc")
    assert mock_sdk.conversation.context.get_context_for_prompt.await_args.kwargs[
        "conversation_id"
    ] == "conv-abc"


async def test_fetch_returns_empty_when_unavailable(mock_sdk):
    mock_sdk.conversation.context.get_context_for_prompt.return_value.available = False
    assert await fetch_st_block(mock_sdk, "conv-1") == ""


async def test_fetch_falls_back_on_sdk_failure(failing_sdk):
    assert await fetch_st_block(failing_sdk, "conv-1") == ""


async def test_fetch_raises_when_asked_to(failing_sdk):
    with pytest.raises(SynapIntegrationError):
        await fetch_st_block(failing_sdk, "conv-1", on_error="raise")


# ---------------------------------------------------------------------------
# compose_system_prompt
# ---------------------------------------------------------------------------


def test_compose_wraps_block_in_tags():
    result = compose_system_prompt("history here", "You are helpful.")
    assert result.startswith("<synap_short_term_context>")
    assert "history here" in result
    assert result.endswith("You are helpful.")


def test_compose_without_tags():
    result = compose_system_prompt("history", "system", None, None)
    assert result == "history\n\nsystem"


def test_compose_empty_block_preserves_system():
    """An empty short-term result must never wipe the caller's system prompt."""
    assert compose_system_prompt("", "You are helpful.") == "You are helpful."


def test_compose_empty_system_leaves_no_dangling_tags():
    result = compose_system_prompt("history", "")
    assert result == "<synap_short_term_context>\nhistory\n</synap_short_term_context>"


def test_compose_both_empty():
    assert compose_system_prompt("", "") == ""


def test_compose_strips_whitespace():
    assert compose_system_prompt("  a  ", "  b  ", None, None) == "a\n\nb"


# ---------------------------------------------------------------------------
# synap_st_instructions
# ---------------------------------------------------------------------------


async def test_instructions_combine_block_and_system(mock_sdk):
    result = await synap_st_instructions(
        mock_sdk, "conv-1", system="You are a coding agent."
    )
    assert "Recent conversation summary" in result
    assert result.endswith("You are a coding agent.")


async def test_instructions_fall_back_to_system_on_failure(failing_sdk):
    result = await synap_st_instructions(
        failing_sdk, "conv-1", system="You are a coding agent."
    )
    assert result == "You are a coding agent."


async def test_instructions_raise_when_asked_to(failing_sdk):
    with pytest.raises(SynapIntegrationError):
        await synap_st_instructions(failing_sdk, "conv-1", on_error="raise")


async def test_instructions_custom_tags(mock_sdk):
    result = await synap_st_instructions(
        mock_sdk, "conv-1", preamble_open="<st>", preamble_close="</st>"
    )
    assert result.startswith("<st>")
    assert result.endswith("</st>")
