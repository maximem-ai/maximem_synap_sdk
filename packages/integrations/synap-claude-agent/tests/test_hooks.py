"""Tests for synap_claude_agent.hooks — create_synap_hooks.

Documented contract (hooks.py docstring):
- Returns a ``hooks`` dict with key ``"UserPromptSubmit"`` containing a list
  of HookMatcher instances.
- The registered hook fetches Synap context and injects it via
  ``hookSpecificOutput.additionalContext``.
- The hook NEVER raises — SDK failures degrade gracefully to ``{}`` (no extra
  context, agent run continues uninterrupted).
- If ``record_user_prompts=True`` (default) and a conv_id is available, the
  prompt is recorded to Synap conversation history via
  ``sdk.conversation.record_message``.
- Empty prompt or blank prompt → returns ``{}`` immediately (no SDK call).
- Empty Synap fetch result → returns ``{}`` (no additional context injected).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from synap_claude_agent.hooks import (
    _DEFAULT_CONTEXT_PREAMBLE,
    _field,
    create_synap_hooks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetch_response(formatted: str | None = "User is an engineer"):
    resp = MagicMock()
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User is an engineer"):
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=_make_fetch_response(formatted))
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "msg-1"})
    return sdk


def _hook_callable(hooks_dict: dict) -> object:
    """Extract the single async callable from the hooks dict."""
    matchers = hooks_dict["UserPromptSubmit"]
    assert len(matchers) == 1
    callables = matchers[0].hooks
    assert len(callables) == 1
    return callables[0]


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_hooks(None, user_id="alice")  # type: ignore[arg-type]

    def test_requires_user_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_synap_hooks(sdk, user_id="")

    @pytest.mark.xfail(
        reason=(
            "hooks.py:create_synap_hooks — validation uses `if not user_id` "
            "but a whitespace-only string is truthy; whitespace-only user_id "
            "slips through. Expected ValueError, actual: no error raised."
        ),
        strict=False,
    )
    def test_requires_user_id_not_whitespace(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_synap_hooks(sdk, user_id="   ")

    def test_returns_dict_with_user_prompt_submit_key(self):
        sdk = _fake_sdk()
        result = create_synap_hooks(sdk, user_id="alice")
        assert isinstance(result, dict)
        assert "UserPromptSubmit" in result

    def test_user_prompt_submit_is_list_of_one_hook_matcher(self):
        from claude_agent_sdk import HookMatcher
        sdk = _fake_sdk()
        result = create_synap_hooks(sdk, user_id="alice")
        matchers = result["UserPromptSubmit"]
        assert isinstance(matchers, list)
        assert len(matchers) == 1
        assert isinstance(matchers[0], HookMatcher)

    def test_hook_matcher_has_one_callable(self):
        sdk = _fake_sdk()
        result = create_synap_hooks(sdk, user_id="alice")
        callables = result["UserPromptSubmit"][0].hooks
        assert len(callables) == 1
        assert callable(callables[0])


# ---------------------------------------------------------------------------
# Happy path — context injection
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_injects_additional_context_when_fetch_returns_content(self):
        sdk = _fake_sdk(formatted="User is an engineer.")
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "hello", "session_id": "sess-1"}, None, MagicMock())
        assert "hookSpecificOutput" in out
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "User is an engineer." in ctx

    @pytest.mark.asyncio
    async def test_default_preamble_wraps_context(self):
        sdk = _fake_sdk(formatted="Some context here.")
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "hi", "session_id": "s1"}, None, MagicMock())
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "<synap_memory>" in ctx
        assert "</synap_memory>" in ctx
        assert "Relevant context from the user's long-term memory:" in ctx
        assert "Some context here." in ctx

    @pytest.mark.asyncio
    async def test_custom_preamble_is_used(self):
        sdk = _fake_sdk(formatted="content")
        cb = _hook_callable(
            create_synap_hooks(
                sdk,
                user_id="alice",
                context_preamble="[CTX]{body}[/CTX]",
            )
        )
        out = await cb({"prompt": "hello", "session_id": "s1"}, None, MagicMock())
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "[CTX]" in ctx
        assert "[/CTX]" in ctx
        assert "<synap_memory>" not in ctx

    @pytest.mark.asyncio
    async def test_fetch_called_with_correct_kwargs(self):
        sdk = _fake_sdk()
        cb = _hook_callable(
            create_synap_hooks(
                sdk,
                user_id="alice",
                customer_id="acme",
                mode="fast",
                max_results=5,
            )
        )
        await cb({"prompt": "what do I like?", "session_id": "conv-99"}, None, MagicMock())
        sdk.fetch.assert_awaited_once_with(
            conversation_id="conv-99",
            user_id="alice",
            customer_id="acme",
            search_query=["what do I like?"],
            max_results=5,
            mode="fast",
            include_conversation_context=False,
        )

    @pytest.mark.asyncio
    async def test_static_conversation_id_overrides_session_id(self):
        sdk = _fake_sdk()
        cb = _hook_callable(
            create_synap_hooks(
                sdk, user_id="alice", conversation_id="static-conv-1"
            )
        )
        await cb({"prompt": "hi", "session_id": "session-xyz"}, None, MagicMock())
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] == "static-conv-1"

    @pytest.mark.asyncio
    async def test_session_id_used_when_no_static_conversation_id(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        await cb({"prompt": "hi", "session_id": "sess-from-input"}, None, MagicMock())
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] == "sess-from-input"

    @pytest.mark.asyncio
    async def test_conv_id_none_when_no_session_id_and_no_static(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        await cb({"prompt": "hi"}, None, MagicMock())
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] is None

    @pytest.mark.asyncio
    async def test_prompt_recorded_when_record_user_prompts_true(self):
        sdk = _fake_sdk()
        cb = _hook_callable(
            create_synap_hooks(
                sdk, user_id="alice", customer_id="acme",
                conversation_id="conv-1"
            )
        )
        await cb({"prompt": "remember this"}, None, MagicMock())
        sdk.conversation.record_message.assert_awaited_once_with(
            conversation_id="conv-1",
            role="user",
            content="remember this",
            user_id="alice",
            customer_id="acme",
        )

    @pytest.mark.asyncio
    async def test_prompt_not_recorded_when_record_user_prompts_false(self):
        sdk = _fake_sdk()
        cb = _hook_callable(
            create_synap_hooks(
                sdk, user_id="alice", conversation_id="conv-1",
                record_user_prompts=False
            )
        )
        await cb({"prompt": "hi"}, None, MagicMock())
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_not_recorded_when_no_conv_id(self):
        """record_user_prompts=True but no conv_id — no record call."""
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        # No session_id or static conv_id
        await cb({"prompt": "hi"}, None, MagicMock())
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_customer_id_sent_as_none_to_fetch(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice", customer_id=""))
        await cb({"prompt": "hi", "session_id": "s1"}, None, MagicMock())
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_input_as_attribute_object_instead_of_dict(self):
        """The hook supports TypedDict-as-object (attribute access) as well as dict."""
        sdk = _fake_sdk(formatted="context from object input")
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        input_obj = MagicMock()
        input_obj.prompt = "prompt from object"
        input_obj.session_id = "sess-obj"
        out = await cb(input_obj, None, MagicMock())
        assert "hookSpecificOutput" in out
        assert "context from object input" in out["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# Empty / no-op paths
# ---------------------------------------------------------------------------


class TestEmptyPaths:
    @pytest.mark.asyncio
    async def test_empty_prompt_returns_empty_dict(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": ""}, None, MagicMock())
        assert out == {}
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_prompt_returns_empty_dict(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "   "}, None, MagicMock())
        assert out == {}
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_prompt_key_returns_empty_dict(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({}, None, MagicMock())
        assert out == {}
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_formatted_context_returns_empty_dict(self):
        sdk = _fake_sdk(formatted="")
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "hello", "session_id": "s1"}, None, MagicMock())
        assert out == {}

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_empty_dict(self):
        sdk = _fake_sdk(formatted=None)
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "hello", "session_id": "s1"}, None, MagicMock())
        assert out == {}

    @pytest.mark.asyncio
    async def test_whitespace_only_context_returns_empty_dict(self):
        sdk = _fake_sdk(formatted="   \n  ")
        cb = _hook_callable(create_synap_hooks(sdk, user_id="alice"))
        out = await cb({"prompt": "hello", "session_id": "s1"}, None, MagicMock())
        assert out == {}


# ---------------------------------------------------------------------------
# Failure paths — SDK calls fail
# ---------------------------------------------------------------------------


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_fetch_failure_returns_empty_dict_not_raises(self):
        """SDK fetch failure → {} (hook NEVER raises)."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(return_value={})
        cb = _hook_callable(
            create_synap_hooks(sdk, user_id="alice", conversation_id="conv-1")
        )
        out = await cb({"prompt": "hi"}, None, MagicMock())
        assert out == {}  # must not raise

    @pytest.mark.asyncio
    async def test_fetch_failure_logs_error(self, caplog):
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("boom boom"))
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(return_value={})
        cb = _hook_callable(
            create_synap_hooks(sdk, user_id="alice", conversation_id="conv-1")
        )
        with caplog.at_level(logging.ERROR, logger="synap_claude_agent.hooks"):
            await cb({"prompt": "hi"}, None, MagicMock())
        assert any("sdk.fetch failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_record_message_failure_does_not_raise(self):
        """record_message failure → hook returns context (doesn't crash)."""
        sdk = _fake_sdk(formatted="real context")
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("record fail"))
        cb = _hook_callable(
            create_synap_hooks(sdk, user_id="alice", conversation_id="conv-1")
        )
        out = await cb({"prompt": "hi"}, None, MagicMock())
        # context was fetched successfully before record failed
        assert "hookSpecificOutput" in out

    @pytest.mark.asyncio
    async def test_record_message_failure_logs_error(self, caplog):
        sdk = _fake_sdk(formatted="real context")
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("record explode"))
        cb = _hook_callable(
            create_synap_hooks(sdk, user_id="alice", conversation_id="conv-1")
        )
        with caplog.at_level(logging.ERROR, logger="synap_claude_agent.hooks"):
            await cb({"prompt": "hi"}, None, MagicMock())
        assert any("record_message failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_returns_empty_dict(self, failing_sdk):
        """Using the shared failing_sdk fixture: hook degrades to {}."""
        cb = _hook_callable(
            create_synap_hooks(failing_sdk, user_id="alice", conversation_id="conv-1")
        )
        out = await cb({"prompt": "hello"}, None, MagicMock())
        assert out == {}


# ---------------------------------------------------------------------------
# _field helper unit tests
# ---------------------------------------------------------------------------


class TestFieldHelper:
    def test_dict_access_returns_value(self):
        assert _field({"key": "val"}, "key", "default") == "val"

    def test_dict_missing_key_returns_default(self):
        assert _field({}, "key", "default") == "default"

    def test_attribute_access_returns_value(self):
        obj = MagicMock()
        obj.key = "attrval"
        assert _field(obj, "key", "default") == "attrval"

    def test_attribute_missing_returns_default(self):
        obj = MagicMock(spec=[])  # no attributes
        assert _field(obj, "missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Shared harness fixture smoke-test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_with_shared_mock_sdk_fixture(mock_sdk):
    """Integration smoke: shared mock_sdk works end-to-end with create_synap_hooks."""
    cb = _hook_callable(
        create_synap_hooks(mock_sdk, user_id="alice", conversation_id="conv-1")
    )
    out = await cb({"prompt": "tell me about the user"}, None, MagicMock())
    # shared mock_sdk.fetch returns a UnifiedContextResponse with formatted_context
    assert "hookSpecificOutput" in out
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert len(ctx) > 0
