"""Tests for synap_pipecat.processors — SynapMemoryProcessor and SynapRecorder.

Error-handling contracts (from processors.py docstring):
- Read failures (SynapMemoryProcessor) degrade GRACEFULLY — log at ERROR and skip
  context injection. A Synap blip must NEVER break a live voice call.
- Write failures (SynapRecorder) surface as SynapIntegrationError via
  wrap_sdk_errors_async. The recorder pushes an ErrorFrame UPSTREAM and swallows
  the exception locally (Pipecat contract: frames-not-raises).
- Non-matching frames pass through untouched.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from synap_integrations_common import SynapIntegrationError
from synap_pipecat.processors import (
    _SYSTEM_MEMORY_PREAMBLE,
    SynapMemoryProcessor,
    SynapRecorder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sdk(formatted: str | None = "User is on Pro plan.") -> MagicMock:
    """Build a minimal SDK mock with a successful sdk.fetch."""
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=formatted))
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    return sdk


def _failing_sdk() -> MagicMock:
    """Build an SDK mock whose every async call raises RuntimeError."""
    sdk = _sdk()
    sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
    sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("sdk boom"))
    return sdk


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u1", timestamp="2026-01-01T00:00:00Z")


def _system_messages(ctx: LLMContext) -> list:
    return [m for m in ctx.get_messages() if isinstance(m, dict) and m.get("role") == "system"]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_exports_memory_processor():
    import synap_pipecat

    assert hasattr(synap_pipecat, "SynapMemoryProcessor")
    assert "SynapMemoryProcessor" in synap_pipecat.__all__


def test_public_exports_recorder():
    import synap_pipecat

    assert hasattr(synap_pipecat, "SynapRecorder")
    assert "SynapRecorder" in synap_pipecat.__all__


# ============================================================
# SynapMemoryProcessor — construction / validation
# ============================================================


class TestSynapMemoryProcessorValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapMemoryProcessor(None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapMemoryProcessor(sdk, user_id="")

    def test_default_customer_id_is_empty_string(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        assert proc.customer_id == ""

    def test_default_mode_is_accurate(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        assert proc.mode == "accurate"

    def test_default_max_results_is_10(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        assert proc.max_results == 10

    def test_default_include_conversation_context_is_false(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        assert proc.include_conversation_context is False

    def test_context_defaults_to_none(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        assert proc._context is None

    def test_set_context_attaches_context(self):
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        ctx = LLMContext(messages=[])
        proc.set_context(ctx)
        assert proc._context is ctx

    def test_set_context_replaces_existing(self):
        sdk = _sdk()
        ctx1 = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx1)
        ctx2 = LLMContext(messages=[])
        proc.set_context(ctx2)
        assert proc._context is ctx2


# ============================================================
# SynapMemoryProcessor — happy paths
# ============================================================


class TestSynapMemoryProcessorHappyPaths:
    @pytest.mark.asyncio
    async def test_inject_context_appends_system_message(self):
        """A transcription with non-empty formatted_context injects a system message."""
        sdk = _sdk(formatted="Customer is VIP.")
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("tell me about my account")

        sys_msgs = _system_messages(ctx)
        assert len(sys_msgs) == 1
        assert "Customer is VIP." in sys_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_injected_message_uses_preamble_template(self):
        """System message content matches the documented preamble format."""
        sdk = _sdk(formatted="Some context body.")
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("query")

        sys_msgs = _system_messages(ctx)
        expected = _SYSTEM_MEMORY_PREAMBLE.format(body="Some context body.")
        assert sys_msgs[0]["content"] == expected

    @pytest.mark.asyncio
    async def test_inject_context_passes_correct_fetch_kwargs(self):
        """sdk.fetch is called with the correct keyword arguments."""
        sdk = _sdk(formatted="ctx")
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(
            sdk,
            user_id="alice",
            customer_id="acme",
            context=ctx,
            mode="fast",
            max_results=5,
            include_conversation_context=True,
        )

        await proc._inject_context("what is my plan?")

        sdk.fetch.assert_awaited_once_with(
            user_id="alice",
            customer_id="acme",
            search_query=["what is my plan?"],
            max_results=5,
            mode="fast",
            include_conversation_context=True,
        )

    @pytest.mark.asyncio
    async def test_empty_customer_id_sends_none_to_sdk(self):
        """Empty customer_id is coerced to None in the sdk.fetch call."""
        sdk = _sdk()
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", customer_id="", context=ctx)

        await proc._inject_context("q")

        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] is None

    @pytest.mark.asyncio
    async def test_non_empty_customer_id_forwarded_as_is(self):
        """Non-empty customer_id is forwarded unchanged to sdk.fetch."""
        sdk = _sdk()
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", customer_id="acme", context=ctx)

        await proc._inject_context("q")

        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_no_injection_when_formatted_context_is_empty(self):
        """Empty string formatted_context → no system message appended."""
        sdk = _sdk(formatted="")
        ctx = LLMContext(messages=[{"role": "user", "content": "hi"}])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")

        assert _system_messages(ctx) == []

    @pytest.mark.asyncio
    async def test_no_injection_when_formatted_context_is_none(self):
        """None formatted_context → no system message appended."""
        sdk = _sdk(formatted=None)
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")

        assert _system_messages(ctx) == []

    @pytest.mark.asyncio
    async def test_no_injection_when_formatted_context_is_whitespace_only(self):
        """Whitespace-only formatted_context → no system message appended."""
        sdk = _sdk(formatted="   \n  ")
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")

        assert _system_messages(ctx) == []

    @pytest.mark.asyncio
    async def test_no_context_means_no_sdk_call(self):
        """When context is None (inert mode), sdk.fetch is never called."""
        sdk = _sdk()
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=None)

        await proc._inject_context("q")

        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_transcriptions_append_multiple_system_messages(self):
        """Long-term memory processor appends on every turn (no dedup)."""
        sdk = _sdk(formatted="ctx")
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("turn 1")
        await proc._inject_context("turn 2")

        assert len(_system_messages(ctx)) == 2

    @pytest.mark.asyncio
    async def test_set_context_after_construction_enables_injection(self):
        """set_context() wires up injection for a processor constructed without context."""
        sdk = _sdk(formatted="wired up")
        proc = SynapMemoryProcessor(sdk, user_id="u1")
        ctx = LLMContext(messages=[])
        proc.set_context(ctx)

        await proc._inject_context("q")

        sys_msgs = _system_messages(ctx)
        assert len(sys_msgs) == 1
        assert "wired up" in sys_msgs[0]["content"]


# ============================================================
# SynapMemoryProcessor — failure path (graceful degrade on read)
# ============================================================


class TestSynapMemoryProcessorFailurePaths:
    @pytest.mark.asyncio
    async def test_sdk_fetch_failure_does_not_raise(self):
        """READ path: sdk.fetch failure must never propagate as an exception."""
        sdk = _failing_sdk()
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")  # must not raise

    @pytest.mark.asyncio
    async def test_sdk_fetch_failure_leaves_context_unchanged(self):
        """READ path: context is unchanged after a fetch failure."""
        sdk = _failing_sdk()
        ctx = LLMContext(messages=[{"role": "user", "content": "existing"}])
        proc = SynapMemoryProcessor(sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")

        # Only the pre-existing user message, no new system message
        assert len(ctx.get_messages()) == 1

    @pytest.mark.asyncio
    async def test_sdk_fetch_failure_logs_at_error(self, caplog):
        """READ path: sdk.fetch failure is logged at ERROR level with user_id."""
        sdk = _failing_sdk()
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(sdk, user_id="alice", context=ctx)

        with caplog.at_level(logging.ERROR, logger="synap_pipecat.processors"):
            await proc._inject_context("q")

        assert len(caplog.records) >= 1
        error_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("alice" in m for m in error_msgs), (
            f"Expected user_id='alice' in ERROR log, got: {error_msgs}"
        )

    @pytest.mark.asyncio
    async def test_sdk_fetch_failure_uses_failing_sdk_fixture(self, failing_sdk):
        """Smoke: shared failing_sdk fixture also degrades gracefully."""
        ctx = LLMContext(messages=[])
        proc = SynapMemoryProcessor(failing_sdk, user_id="u1", context=ctx)

        await proc._inject_context("q")  # must not raise

        assert _system_messages(ctx) == []


# ============================================================
# SynapRecorder — construction / validation
# ============================================================


class TestSynapRecorderValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapRecorder(None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapRecorder(sdk, user_id="")

    def test_default_customer_id_is_empty_string(self):
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1")
        assert rec.customer_id == ""

    def test_conversation_id_auto_generated_when_absent(self):
        """Auto-generated conversation_id starts with 'pipecat-'."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1")
        assert rec.conversation_id.startswith("pipecat-")

    def test_auto_generated_conversation_ids_are_unique(self):
        """Two recorders without explicit conversation_id get different IDs."""
        sdk = _sdk()
        r1 = SynapRecorder(sdk, user_id="u1")
        r2 = SynapRecorder(sdk, user_id="u1")
        assert r1.conversation_id != r2.conversation_id

    def test_explicit_conversation_id_is_preserved(self):
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="my-conv-42")
        assert rec.conversation_id == "my-conv-42"

    def test_initial_user_buffer_is_none(self):
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1")
        assert rec._user_buffer is None

    def test_initial_assistant_parts_is_empty(self):
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1")
        assert rec._assistant_parts == []


# ============================================================
# SynapRecorder — happy paths
# ============================================================


class TestSynapRecorderHappyPaths:
    @pytest.mark.asyncio
    async def test_flush_records_both_user_and_assistant(self):
        """Full turn: both user + assistant messages are sent to sdk."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="alice", customer_id="acme", conversation_id="c1")
        rec._user_buffer = "Tell me a joke"
        rec._assistant_parts = ["Here is one: ", "Why did the chicken cross the road?"]

        await rec._flush()

        assert sdk.conversation.record_message.await_count == 2
        calls = sdk.conversation.record_message.call_args_list
        assert calls[0].kwargs["role"] == "user"
        assert calls[0].kwargs["content"] == "Tell me a joke"
        assert calls[1].kwargs["role"] == "assistant"
        assert calls[1].kwargs["content"] == "Here is one: Why did the chicken cross the road?"

    @pytest.mark.asyncio
    async def test_flush_concatenates_assistant_parts(self):
        """Multiple LLMTextFrame tokens are joined into a single assistant turn."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = None
        rec._assistant_parts = ["Hello", ", ", "world", "!"]

        await rec._flush()

        call = sdk.conversation.record_message.call_args
        assert call.kwargs["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_flush_passes_conversation_user_customer_ids(self):
        """IDs are forwarded on every sdk.conversation.record_message call."""
        sdk = _sdk()
        rec = SynapRecorder(
            sdk, user_id="alice", customer_id="acme", conversation_id="c-test"
        )
        rec._user_buffer = "hello"
        rec._assistant_parts = ["hi"]

        await rec._flush()

        for call in sdk.conversation.record_message.call_args_list:
            assert call.kwargs["conversation_id"] == "c-test"
            assert call.kwargs["user_id"] == "alice"
            assert call.kwargs["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_flush_only_user_text(self):
        """Only user text (no assistant) → exactly one 'user' record call."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = "Only a user message"
        rec._assistant_parts = []

        await rec._flush()

        assert sdk.conversation.record_message.await_count == 1
        assert sdk.conversation.record_message.call_args.kwargs["role"] == "user"

    @pytest.mark.asyncio
    async def test_flush_only_assistant_text(self):
        """Only assistant text (no user) → exactly one 'assistant' record call."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = None
        rec._assistant_parts = ["assistant-only response"]

        await rec._flush()

        assert sdk.conversation.record_message.await_count == 1
        assert sdk.conversation.record_message.call_args.kwargs["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_flush_empty_buffers_is_noop(self):
        """Both buffers empty → no SDK call (guard branch)."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        await rec._flush()

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_whitespace_only_assistant_is_noop(self):
        """Whitespace-only assistant parts (stripped to '') → treated as no assistant text."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = None
        rec._assistant_parts = ["   ", "\n  "]

        await rec._flush()

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_resets_buffers_before_awaiting_sdk(self):
        """Buffers are cleared BEFORE the async sdk calls to prevent re-entrancy issues."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        captured_states: list[tuple] = []

        async def capture_state(*args, **kwargs):
            captured_states.append(
                (rec._user_buffer, list(rec._assistant_parts))
            )

        sdk.conversation.record_message.side_effect = capture_state

        rec._user_buffer = "user text"
        rec._assistant_parts = ["asst"]

        await rec._flush()

        # Both calls should see already-cleared buffers
        for user_buf, asst_parts in captured_states:
            assert user_buf is None
            assert asst_parts == []

    @pytest.mark.asyncio
    async def test_flush_resets_buffers_on_success(self):
        """After a successful flush, buffers are reset to empty."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = "something"
        rec._assistant_parts = ["text"]

        await rec._flush()

        assert rec._user_buffer is None
        assert rec._assistant_parts == []

    @pytest.mark.asyncio
    async def test_default_empty_customer_id_forwarded(self):
        """Default customer_id='' is forwarded to record_message unchanged."""
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u")  # No customer_id
        rec._user_buffer = "hi"
        rec._assistant_parts = ["hello"]

        await rec._flush()

        for call in sdk.conversation.record_message.call_args_list:
            assert call.kwargs["customer_id"] == ""

    @pytest.mark.asyncio
    async def test_uses_shared_mock_sdk_fixture(self, mock_sdk):
        """Smoke: shared mock_sdk fixture round-trips through _flush."""
        rec = SynapRecorder(mock_sdk, user_id="u", conversation_id="c1")
        rec._user_buffer = "hello"
        rec._assistant_parts = ["world"]

        await rec._flush()

        assert mock_sdk.conversation.record_message.await_count == 2


# ============================================================
# SynapRecorder — failure path (write surfaces as ErrorFrame)
# ============================================================


class TestSynapRecorderFailurePaths:
    @pytest.mark.asyncio
    async def test_sdk_failure_does_not_raise(self):
        """WRITE path: sdk failure must not propagate as a Python exception
        (Pipecat contract: frames-not-raises)."""
        sdk = _failing_sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        pushed: list = []

        async def fake_push(frame, direction):
            pushed.append(frame)

        rec.push_frame = fake_push  # type: ignore[method-assign]
        rec._user_buffer = "hello"
        rec._assistant_parts = ["world"]

        await rec._flush()  # must not raise

    @pytest.mark.asyncio
    async def test_sdk_failure_pushes_error_frame(self):
        """WRITE path: sdk failure triggers an ErrorFrame pushed UPSTREAM."""
        sdk = _failing_sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        pushed: list = []
        directions: list = []

        async def fake_push(frame, direction):
            pushed.append(frame)
            directions.append(direction)

        rec.push_frame = fake_push  # type: ignore[method-assign]
        rec._user_buffer = "hello"
        rec._assistant_parts = ["world"]

        await rec._flush()

        error_frames = [f for f in pushed if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 1
        assert directions[pushed.index(error_frames[0])] == FrameDirection.UPSTREAM

    @pytest.mark.asyncio
    async def test_error_frame_contains_synap_integration_error(self):
        """The exception embedded in ErrorFrame is a SynapIntegrationError."""
        sdk = _failing_sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        pushed: list = []

        async def fake_push(frame, direction):
            pushed.append(frame)

        rec.push_frame = fake_push  # type: ignore[method-assign]
        rec._user_buffer = "hi"
        rec._assistant_parts = ["yo"]

        await rec._flush()

        error_frames = [f for f in pushed if isinstance(f, ErrorFrame)]
        assert isinstance(error_frames[0].exception, SynapIntegrationError)

    @pytest.mark.asyncio
    async def test_buffers_reset_even_on_sdk_failure(self):
        """Buffers are cleared even when the sdk call fails."""
        sdk = _failing_sdk()
        rec = SynapRecorder(sdk, user_id="u", conversation_id="c1")

        async def fake_push(frame, direction):
            pass

        rec.push_frame = fake_push  # type: ignore[method-assign]
        rec._user_buffer = "text"
        rec._assistant_parts = ["reply"]

        await rec._flush()

        assert rec._user_buffer is None
        assert rec._assistant_parts == []

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_surfaces_error_frame(self, failing_sdk):
        """Smoke: shared failing_sdk fixture triggers ErrorFrame on write."""
        rec = SynapRecorder(failing_sdk, user_id="u", conversation_id="c1")

        pushed: list = []

        async def fake_push(frame, direction):
            pushed.append(frame)

        rec.push_frame = fake_push  # type: ignore[method-assign]
        rec._user_buffer = "hello"
        rec._assistant_parts = ["hi"]

        await rec._flush()

        assert any(isinstance(f, ErrorFrame) for f in pushed)
