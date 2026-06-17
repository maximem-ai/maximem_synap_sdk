"""Extended tests for synap_pipecat.short_term — SynapShortTermProcessor.

Complements test_short_term.py (which covers the core happy/failure paths)
with additional coverage for:
- All valid ``style`` values accepted at construction
- Invalid ``on_error`` values rejected
- Whitespace-only conversation_id rejected
- ``preamble_open/close=None`` drops the XML tags but keeps the marker
- ``style`` and ``conversation_id`` forwarded verbatim to sdk
- ``set_context`` runtime swap
- ``_drop_existing_st`` guard when context lacks ``set_messages``
- Shared fixture smoke tests (mock_sdk / failing_sdk)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext

from synap_integrations_common import SynapIntegrationError
from synap_pipecat.short_term import (
    _DEFAULT_CLOSE,
    _DEFAULT_OPEN,
    _ST_MARKER,
    SynapShortTermProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sdk(formatted: str | None = "Recent turns here.", available: bool = True) -> MagicMock:
    sdk = MagicMock()
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    sdk.conversation = MagicMock()
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=resp)
    return sdk


def _failing_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.conversation = MagicMock()
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        side_effect=RuntimeError("sdk boom")
    )
    return sdk


def _st_messages(ctx: LLMContext) -> list:
    return [
        m
        for m in ctx.get_messages()
        if isinstance(m, dict)
        and m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_ST_MARKER)
    ]


# ============================================================
# Additional construction / validation
# ============================================================


class TestSynapShortTermProcessorValidationExtended:
    def test_accepts_style_structured(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", style="structured")
        assert proc.style == "structured"

    def test_accepts_style_narrative(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", style="narrative")
        assert proc.style == "narrative"

    def test_accepts_style_bullet_points(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", style="bullet_points")
        assert proc.style == "bullet_points"

    def test_rejects_invalid_on_error(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="on_error must be"):
            SynapShortTermProcessor(sdk, conversation_id="c1", on_error="ignore")  # type: ignore[arg-type]

    def test_rejects_whitespace_only_conversation_id(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermProcessor(sdk, conversation_id="   ")

    def test_defaults_preamble_open_is_synap_tag(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")
        assert proc.preamble_open == _DEFAULT_OPEN

    def test_defaults_preamble_close_is_synap_tag(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")
        assert proc.preamble_close == _DEFAULT_CLOSE

    def test_accepts_on_error_fallback(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", on_error="fallback")
        assert proc.on_error == "fallback"

    def test_accepts_on_error_raise(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", on_error="raise")
        assert proc.on_error == "raise"

    def test_default_on_error_is_fallback(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")
        assert proc.on_error == "fallback"

    def test_default_style_is_narrative(self):
        sdk = _sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")
        assert proc.style == "narrative"


# ============================================================
# SDK call contract — args forwarded verbatim
# ============================================================


class TestSynapShortTermProcessorSDKCallContract:
    @pytest.mark.asyncio
    async def test_style_forwarded_to_sdk(self):
        """sdk.conversation.context.get_context_for_prompt receives the configured style."""
        sdk = _sdk()
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", context=ctx, style="bullet_points")

        await proc._refresh_st()

        call = sdk.conversation.context.get_context_for_prompt.call_args
        assert call.kwargs["style"] == "bullet_points"

    @pytest.mark.asyncio
    async def test_conversation_id_forwarded_to_sdk(self):
        """sdk call receives the exact conversation_id set at construction."""
        sdk = _sdk()
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="my-conv-99", context=ctx)

        await proc._refresh_st()

        call = sdk.conversation.context.get_context_for_prompt.call_args
        assert call.kwargs["conversation_id"] == "my-conv-99"

    @pytest.mark.asyncio
    async def test_sdk_called_once_per_refresh(self):
        """Each call to _refresh_st results in exactly one sdk call."""
        sdk = _sdk()
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", context=ctx)

        await proc._refresh_st()
        await proc._refresh_st()

        assert sdk.conversation.context.get_context_for_prompt.await_count == 2


# ============================================================
# Preamble tag behaviour
# ============================================================


class TestSynapShortTermProcessorPreamble:
    @pytest.mark.asyncio
    async def test_default_preamble_wraps_context_in_tags(self):
        """Default preamble wraps the ST block in <synap_short_term_context> tags."""
        sdk = _sdk(formatted="User is on trial.")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", context=ctx)

        await proc._refresh_st()

        content = _st_messages(ctx)[0]["content"]
        assert _DEFAULT_OPEN in content
        assert _DEFAULT_CLOSE in content
        assert "User is on trial." in content

    @pytest.mark.asyncio
    async def test_none_preamble_omits_xml_tags(self):
        """preamble_open=None, preamble_close=None → no wrapping tags, just the context."""
        sdk = _sdk(formatted="Plain context.")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(
            sdk, conversation_id="c1", context=ctx,
            preamble_open=None, preamble_close=None
        )

        await proc._refresh_st()

        content = _st_messages(ctx)[0]["content"]
        assert _DEFAULT_OPEN not in content
        assert _DEFAULT_CLOSE not in content
        assert _ST_MARKER in content
        assert "Plain context." in content

    @pytest.mark.asyncio
    async def test_custom_preamble_tags_used(self):
        """Custom open/close preamble strings are used when provided."""
        sdk = _sdk(formatted="ctx text")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(
            sdk, conversation_id="c1", context=ctx,
            preamble_open="[CONTEXT_START]",
            preamble_close="[CONTEXT_END]",
        )

        await proc._refresh_st()

        content = _st_messages(ctx)[0]["content"]
        assert "[CONTEXT_START]" in content
        assert "[CONTEXT_END]" in content
        assert "ctx text" in content


# ============================================================
# set_context runtime swap
# ============================================================


class TestSynapShortTermProcessorSetContext:
    @pytest.mark.asyncio
    async def test_set_context_enables_injection(self):
        """set_context after construction wires up injection on the next call."""
        sdk = _sdk(formatted="Now active")
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")  # no context
        ctx = LLMContext(messages=[])
        proc.set_context(ctx)

        await proc._refresh_st()

        st_msgs = _st_messages(ctx)
        assert len(st_msgs) == 1
        assert "Now active" in st_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_set_context_swaps_target(self):
        """set_context mid-session redirects injection to the new context."""
        sdk = _sdk(formatted="context text")
        ctx_old = LLMContext(messages=[])
        ctx_new = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", context=ctx_old)

        await proc._refresh_st()
        assert len(_st_messages(ctx_old)) == 1

        proc.set_context(ctx_new)
        await proc._refresh_st()

        # Old context frozen; new context now has the message
        assert len(_st_messages(ctx_old)) == 1
        assert len(_st_messages(ctx_new)) == 1


# ============================================================
# _drop_existing_st guard — context without set_messages
# ============================================================


class TestSynapShortTermProcessorDropST:
    @pytest.mark.asyncio
    async def test_drop_existing_st_noop_without_set_messages(self):
        """_drop_existing_st is a no-op when the context object lacks set_messages."""
        sdk = _sdk(formatted=None, available=False)

        class MinimalContext:
            """Minimal duck-typed context that lacks set_messages."""
            def __init__(self):
                self._messages = [
                    {"role": "system", "content": f"{_ST_MARKER}\nstale content"}
                ]

            def get_messages(self):
                return self._messages

            def add_message(self, m):
                self._messages.append(m)

        ctx = MinimalContext()
        proc = SynapShortTermProcessor(sdk, conversation_id="c1")
        proc._context = ctx  # type: ignore[assignment]

        # Should not raise; stale message stays because set_messages is absent
        proc._drop_existing_st()

        assert len(ctx._messages) == 1  # unchanged

    @pytest.mark.asyncio
    async def test_drop_removes_only_st_tagged_messages(self):
        """_drop_existing_st removes only the ST-tagged message, leaving others intact."""
        sdk = _sdk(formatted=None, available=False)
        ctx = LLMContext(messages=[
            {"role": "user", "content": "regular user msg"},
            {"role": "system", "content": f"{_ST_MARKER}\nshort-term context"},
            {"role": "assistant", "content": "assistant reply"},
        ])
        proc = SynapShortTermProcessor(sdk, conversation_id="c1", context=ctx)

        await proc._refresh_st()  # triggers _drop_existing_st (available=False)

        msgs = ctx.get_messages()
        contents = [m.get("content") for m in msgs if isinstance(m, dict)]
        assert "regular user msg" in contents
        assert "assistant reply" in contents
        # ST-tagged message removed
        assert all(not c.startswith(_ST_MARKER) for c in contents if c)


# ============================================================
# Shared fixture smoke tests
# ============================================================


class TestSynapShortTermProcessorSharedFixtures:
    @pytest.mark.asyncio
    async def test_mock_sdk_fixture_injects_context(self, mock_sdk):
        """Shared mock_sdk fixture provides a working get_context_for_prompt response."""
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(mock_sdk, conversation_id="c1", context=ctx)

        await proc._refresh_st()

        st_msgs = _st_messages(ctx)
        assert len(st_msgs) == 1
        assert "Recent conversation summary" in st_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_fallback_no_inject(self, failing_sdk):
        """Shared failing_sdk fixture → fallback mode, no injection, no raise."""
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(failing_sdk, conversation_id="c1", context=ctx)

        await proc._refresh_st()  # must not raise

        assert _st_messages(ctx) == []

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_raise_mode_raises(self, failing_sdk):
        """Shared failing_sdk fixture + on_error='raise' → SynapIntegrationError."""
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(
            failing_sdk, conversation_id="c1", context=ctx, on_error="raise"
        )

        with pytest.raises(SynapIntegrationError):
            await proc._refresh_st()
