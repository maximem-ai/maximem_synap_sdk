"""The defaults of SynapRecorder: the gRPC stream, and Synap's compaction.

Two things happen with no opt-in from the caller:

- the first turn opens the stream (``sdk.instance.listen()``) and everything
  after rides it;
- once Synap has compacted the conversation, the shared ``LLMContext`` the
  next LLM request is built from carries Synap's compaction in place of the
  earlier turns.

Both are proven through a REAL pipeline: pipecat's own
``LLMAssistantAggregator`` between a stand-in LLM service and the recorder,
because what that aggregator lets through is the whole story. It consumes
``LLMFullResponseEndFrame``, every ``TextFrame`` and the function-call frames,
so a recorder that waited for them behind it recorded nothing at all.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test

from synap_integrations_common import set_auto_listen
from synap_pipecat import SynapRecorder

try:
    from pipecat.frames.frames import LLMContextAssistantTurnFrame
except ImportError:  # pragma: no cover
    LLMContextAssistantTurnFrame = None

needs_turn_frame = pytest.mark.skipif(
    LLMContextAssistantTurnFrame is None,
    reason="this pipecat's assistant aggregator passes LLMFullResponseEndFrame on",
)

PREAMBLE_OPEN = "<synap_conversation_history>"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class UserAndLLM(FrameProcessor):
    """Stands in for the user aggregator and the LLM service.

    On a TranscriptionFrame it does what the user aggregator does (appends the
    user message), then what an LLM service does (reads the context the way the
    OpenAI adapter does, and streams a reply). It waits for the recorder to
    finish the previous turn first, as a real caller's next utterance would.
    """

    def __init__(self, context, recorder):
        super().__init__()
        self.context = context
        self.recorder = recorder
        self.inputs: list = []
        self.turns = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return
        for _ in range(300):
            if self.recorder.done >= self.turns:
                break
            await asyncio.sleep(0.01)
        self.turns += 1
        self.context.add_message({"role": "user", "content": frame.text})
        kwargs = {}
        if "convert_developer_to_user" in inspect.signature(
            OpenAILLMAdapter.get_llm_invocation_params
        ).parameters:
            kwargs["convert_developer_to_user"] = False
        params = OpenAILLMAdapter().get_llm_invocation_params(self.context, **kwargs)
        if inspect.isawaitable(params):
            params = await params
        self.inputs.append([(m["role"], m["content"]) for m in params["messages"]])
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("ok"))
        await self.push_frame(LLMFullResponseEndFrame())


def _sdk(compaction="", turns=(), *, listening=False):
    sdk = MagicMock()
    sdk.instance.is_listening = listening

    async def listen():
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_tool_call", "record_tool_result", "record_thinking"):
        setattr(sdk.instance, name, AsyncMock())
    sdk.conversation.record_message = AsyncMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=SimpleNamespace(
            context_block=compaction,
            history_block="",
            recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
            compaction_id="c1" if compaction else None,
        )
    )
    return sdk


def _context() -> LLMContext:
    ctx = LLMContext()
    ctx.add_message({"role": "system", "content": "be nice"})
    for i in range(1, 7):
        ctx.add_message({"role": "user", "content": f"turn-{i}"})
        ctx.add_message({"role": "assistant", "content": f"reply-{i}"})
    return ctx


def _count_done(recorder):
    """Count finished post-turn passes, so the stand-in knows when the
    recorder has caught up."""
    recorder.done = 0
    original = recorder._compact_context

    async def counted():
        await original()
        recorder.done += 1

    recorder._compact_context = counted


async def _run(sdk, utterances, *, recorder_first=False):
    ctx = _context()
    recorder = SynapRecorder(sdk, user_id="u1", conversation_id="c1", context=ctx)
    _count_done(recorder)
    llm = UserAndLLM(ctx, recorder)
    assistant = LLMAssistantAggregator(ctx)
    chain = [llm, recorder, assistant] if recorder_first else [llm, assistant, recorder]
    frames = [TranscriptionFrame(text=t, user_id="u1", timestamp="2026-01-01T00:00:00Z")
              for t in utterances]
    await run_test(Pipeline(chain), frames_to_send=frames)
    return llm, ctx


def _sent(sdk):
    return [(c.kwargs["event_type"], c.kwargs["content"])
            for c in sdk.instance.send_message.await_args_list]


# ---------------------------------------------------------------------------
# (a) gRPC by default
# ---------------------------------------------------------------------------


@needs_turn_frame
class TestTheStreamOpensByDefault:
    async def test_a_standard_pipeline_streams_every_turn_after_opening_it_once(self):
        sdk = _sdk()
        assert sdk.instance.is_listening is False
        await _run(sdk, ["turn-8", "turn-9"])
        sdk.instance.listen.assert_awaited_once()
        assert _sent(sdk) == [
            ("user_message", "turn-8"), ("assistant_message", "ok"),
            ("user_message", "turn-9"), ("assistant_message", "ok"),
        ]
        sdk.conversation.record_message.assert_not_awaited()

    async def test_the_opt_out_keeps_it_on_rest(self, monkeypatch):
        monkeypatch.setenv("SYNAP_AUTO_LISTEN", "false")
        sdk = _sdk()
        await _run(sdk, ["turn-8"])
        sdk.instance.listen.assert_not_awaited()
        assert sdk.conversation.record_message.await_count == 2

    async def test_set_auto_listen_is_the_per_sdk_switch(self):
        sdk = _sdk()
        set_auto_listen(sdk, False)
        await _run(sdk, ["turn-8"])
        sdk.instance.listen.assert_not_awaited()
        assert sdk.conversation.record_message.await_count == 2


class TestAFirstFrameOtherThanATurnOpensItToo:
    async def test_a_tool_frame_opens_the_stream(self):
        from pipecat.frames.frames import FunctionCallInProgressFrame

        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1")
        rec.push_frame = AsyncMock()
        await rec.process_frame(
            FunctionCallInProgressFrame(function_name="t", tool_call_id="k", arguments={},
                                        cancel_on_interruption=False),
            FrameDirection.DOWNSTREAM)
        sdk.instance.listen.assert_awaited_once()
        sdk.instance.record_tool_call.assert_awaited_once()


# ---------------------------------------------------------------------------
# (b) Synap's compaction in place of the earlier turns
# ---------------------------------------------------------------------------


@needs_turn_frame
class TestTheModelSeesSynapsCompaction:
    COMPACTED = ("SUMMARY: refund requested", [("user", "turn-7"), ("assistant", "reply-7")])

    async def test_once_compacted_the_next_request_gets_the_preamble_not_the_old_turns(self):
        llm, _ = await _run(_sdk(*self.COMPACTED), ["turn-8", "turn-9"])
        first, second = llm.inputs
        assert ("user", "turn-1") in first  # nothing compacted when the call starts
        roles_texts = [t for _r, t in second]
        assert roles_texts[0] == "be nice"
        assert roles_texts[1].startswith(PREAMBLE_OPEN) and "SUMMARY: refund requested" in roles_texts[1]
        assert roles_texts[2:4] == ["turn-7", "reply-7"]
        assert roles_texts[-1] == "turn-9"
        for i in range(1, 7):
            assert f"turn-{i}" not in roles_texts and f"reply-{i}" not in roles_texts

    async def test_the_turn_after_the_splice_is_still_recorded(self):
        """The recorder remembers which context message it recorded by index.
        The splice shortens the list, so an unadjusted index would call every
        later user message 'already recorded' and drop it."""
        sdk = _sdk(*self.COMPACTED)
        await _run(sdk, ["turn-8", "turn-9", "turn-10"])
        users = [t for e, t in _sent(sdk) if e == "user_message"]
        assert users == ["turn-8", "turn-9", "turn-10"]

    async def test_synaps_preamble_is_never_recorded_as_a_turn(self):
        sdk = _sdk(*self.COMPACTED)
        await _run(sdk, ["turn-8", "turn-9", "turn-10"])
        assert not any(t.startswith(PREAMBLE_OPEN) for _e, t in _sent(sdk))

    async def test_with_no_compaction_the_context_is_untouched(self):
        llm, ctx = await _run(_sdk(""), ["turn-8", "turn-9"])
        assert ("user", "turn-1") in llm.inputs[1]
        assert [m["content"] for m in ctx.get_messages()][:5] == [
            "be nice", "turn-1", "reply-1", "turn-2", "reply-2"]
        assert len(ctx.get_messages()) == 1 + 12 + 4

    async def test_when_synap_fails_the_context_is_untouched(self):
        sdk = _sdk("SUMMARY: x")
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
        llm, _ = await _run(sdk, ["turn-8", "turn-9"])
        assert ("user", "turn-1") in llm.inputs[1]

    async def test_a_slow_read_is_not_waited_on_and_not_retried_at_once(self, monkeypatch):
        from synap_pipecat import compaction

        monkeypatch.setattr(compaction, "_FETCH_TIMEOUT_S", 0.05)

        async def slow(**kw):
            await asyncio.sleep(5)

        sdk = _sdk("SUMMARY: x")
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=slow)
        llm, _ = await _run(sdk, ["turn-8", "turn-9", "turn-10"])
        assert ("user", "turn-1") in llm.inputs[2]
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1

    async def test_without_a_context_there_is_nothing_to_splice(self):
        sdk = _sdk(*self.COMPACTED)
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1")
        rec._user_buffer, rec._assistant_parts = "hi", ["yo"]
        rec.push_frame = AsyncMock()
        await rec._flush()
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


# ---------------------------------------------------------------------------
# What reaches a recorder placed behind / before the assistant aggregator
# ---------------------------------------------------------------------------


@needs_turn_frame
class TestWhereTheRecorderSits:
    async def test_before_the_aggregator_a_turn_is_recorded_once_not_twice(self):
        """It then sees LLMFullResponseEndFrame, and the aggregator's
        end-of-turn frame arrives upstream too."""
        sdk = _sdk()
        await _run(sdk, ["turn-8"], recorder_first=True)
        assert _sent(sdk) == [("user_message", "turn-8"), ("assistant_message", "ok")]

    async def test_behind_the_aggregator_the_end_of_turn_frame_carries_the_reply(self):
        sdk = _sdk()
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1")
        rec._user_buffer = "hello"
        rec.push_frame = AsyncMock()
        await rec.process_frame(
            LLMContextAssistantTurnFrame(text="hi there", timestamp="t"),
            FrameDirection.DOWNSTREAM)
        assert _sent(sdk) == [("user_message", "hello"), ("assistant_message", "hi there")]

    async def test_tool_calls_the_aggregator_swallowed_are_read_back_from_the_context(self):
        sdk = _sdk()
        ctx = LLMContext()
        ctx.add_message({"role": "user", "content": "where is my order"})
        ctx.add_message({"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "lookup_order", "arguments": '{"id": 7}'}}]})
        ctx.add_message({"role": "tool", "tool_call_id": "call-1", "content": "shipped"})
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1", context=ctx)
        rec.push_frame = AsyncMock()
        turn = LLMContextAssistantTurnFrame(text="it shipped", timestamp="t")
        await rec.process_frame(turn, FrameDirection.DOWNSTREAM)
        await rec.process_frame(turn, FrameDirection.DOWNSTREAM)
        call = sdk.instance.record_tool_call
        call.assert_awaited_once()
        assert call.await_args.args == ("lookup_order", {"id": 7})
        assert call.await_args.kwargs["tool_call_id"] == "call-1"
        result = sdk.instance.record_tool_result
        result.assert_awaited_once()
        assert result.await_args.args[0] == "shipped"

    async def test_a_tool_result_still_in_progress_is_not_reported(self):
        sdk = _sdk()
        ctx = LLMContext()
        ctx.add_message({"role": "user", "content": "q"})
        ctx.add_message({"role": "assistant", "tool_calls": [{
            "id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        ctx.add_message({"role": "tool", "tool_call_id": "c", "content": "IN_PROGRESS"})
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1", context=ctx)
        rec.push_frame = AsyncMock()
        await rec.process_frame(LLMContextAssistantTurnFrame(text="one sec", timestamp="t"),
                                FrameDirection.DOWNSTREAM)
        sdk.instance.record_tool_result.assert_not_awaited()

    async def test_a_call_seen_as_a_frame_is_not_reported_again_from_the_context(self):
        from pipecat.frames.frames import FunctionCallInProgressFrame

        sdk = _sdk()
        ctx = LLMContext()
        ctx.add_message({"role": "user", "content": "q"})
        ctx.add_message({"role": "assistant", "tool_calls": [{
            "id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        rec = SynapRecorder(sdk, user_id="u1", conversation_id="c1", context=ctx)
        rec.push_frame = AsyncMock()
        await rec.process_frame(
            FunctionCallInProgressFrame(function_name="t", tool_call_id="c", arguments={},
                                        cancel_on_interruption=False),
            FrameDirection.DOWNSTREAM)
        await rec.process_frame(LLMContextAssistantTurnFrame(text="done", timestamp="t"),
                                FrameDirection.DOWNSTREAM)
        assert sdk.instance.record_tool_call.await_count == 1
