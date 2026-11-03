"""``scope_path`` on a deeper scope ladder: reaches every SDK call, and only when set."""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import pytest

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_pipecat import SynapRecorder, SynapMemoryProcessor, SynapShortTermProcessor

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _t(text):
    return TranscriptionFrame(text=text, user_id="u1", timestamp="2026-01-01T00:00:00Z")


async def _read(proc, text="where is my order"):
    await proc.process_frame(_t(text), FrameDirection.DOWNSTREAM)


async def _one_turn(rec):
    await rec.process_frame(_t("hello"), FrameDirection.DOWNSTREAM)
    await rec.process_frame(LLMTextFrame("hi there"), FrameDirection.DOWNSTREAM)
    await rec.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


class TestReader:
    async def test_default_fetch_is_unchanged(self):
        sdk = RecordingSDK()
        await _read(SynapMemoryProcessor(sdk, user_id="u1", customer_id="c1", context=LLMContext()))
        assert sdk.last_kwargs("fetch") == {
            "user_id": "u1", "customer_id": "c1", "search_query": ["where is my order"],
            "max_results": 10, "mode": "accurate", "include_conversation_context": False,
        }

    async def test_path_reaches_fetch(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _read(SynapMemoryProcessor(sdk, user_id="dana", customer_id="acme",
                                         context=LLMContext(), scope_path=PATH))
        assert sdk.last_kwargs("fetch") == {
            "user_id": "dana", "customer_id": "acme", "search_query": ["where is my order"],
            "max_results": 10, "mode": "accurate", "include_conversation_context": False,
            "scope_path": PATH,
        }

    async def test_an_empty_path_is_not_sent(self):
        sdk = RecordingSDK()
        await _read(SynapMemoryProcessor(sdk, user_id="u1", context=LLMContext(), scope_path={}))
        assert "scope_path" not in sdk.last_kwargs("fetch")

    async def test_one_fetch_per_turn_no_extra_calls(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _read(SynapMemoryProcessor(sdk, user_id="dana", context=LLMContext(), scope_path=PATH))
        assert [c.name for c in sdk.calls] == ["fetch"]  # no get_scope_ladder, nothing else

    async def test_the_caller_mutating_its_dict_later_changes_nothing(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        path = dict(PATH)
        proc = SynapMemoryProcessor(sdk, user_id="dana", context=LLMContext(), scope_path=path)
        path["user"] = "mallory"
        await _read(proc)
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH


class TestRecorder:
    async def test_default_turns_on_the_stream_are_unchanged(self):
        sdk = RecordingSDK()
        await _one_turn(SynapRecorder(sdk, user_id="u1", customer_id="c1", conversation_id="conv-1"))
        assert sdk.kwargs_of("instance.send_message") == [
            {"content": "hello", "role": "user", "event_type": "user_message",
             "conversation_id": "conv-1", "user_id": "u1", "customer_id": "c1", "metadata": {}},
            {"content": "hi there", "role": "assistant", "event_type": "assistant_message",
             "conversation_id": "conv-1", "user_id": "u1", "customer_id": "c1", "metadata": {}},
        ]

    async def test_path_reaches_both_turns_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _one_turn(SynapRecorder(sdk, user_id="dana", customer_id="acme",
                                      conversation_id="conv-1", scope_path=PATH))
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
        assert [k["scope_path"] for k in sent] == [PATH, PATH]
        assert sdk.calls_to("conversation.record_message") == []  # never both

    async def test_default_turns_over_rest_are_unchanged(self):
        sdk = RecordingSDK(listening=False)
        set_auto_listen(sdk, False)
        await _one_turn(SynapRecorder(sdk, user_id="u1", conversation_id="conv-1"))
        assert sdk.kwargs_of("conversation.record_message") == [
            {"conversation_id": coerce_conversation_id("conv-1"), "role": "user", "content": "hello", "user_id": "u1"},
            {"conversation_id": coerce_conversation_id("conv-1"), "role": "assistant", "content": "hi there", "user_id": "u1"},
        ]

    async def test_path_reaches_both_turns_over_rest(self):
        sdk = RecordingSDK(listening=False, ladder=make_ladder("deep"))
        set_auto_listen(sdk, False)
        await _one_turn(SynapRecorder(sdk, user_id="dana", conversation_id="conv-1", scope_path=PATH))
        assert [k["scope_path"] for k in sdk.kwargs_of("conversation.record_message")] == [PATH, PATH]
        assert sdk.calls_to("instance.send_message") == []

    async def test_an_older_sdk_without_scope_path_is_never_sent_one_by_default(self):
        sdk = RecordingSDK(accept_scope_path=False)
        await _one_turn(SynapRecorder(sdk, user_id="u1", conversation_id="conv-1"))
        assert len(sdk.calls_to("instance.send_message")) == 2


class TestConversationOnlyReadsStillWork:
    async def test_short_term_processor_next_to_a_scope_path_package(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        ctx = LLMContext()
        st = SynapShortTermProcessor(sdk, conversation_id="conv-1", context=ctx)
        await st.process_frame(_t("hi"), FrameDirection.DOWNSTREAM)
        calls = sdk.calls_to("conversation.context.get_context_for_prompt")
        assert len(calls) == 1
        assert "scope_path" not in calls[0].kwargs


class TestCallsPerTurn:
    async def test_a_full_turn_makes_exactly_one_fetch_and_the_two_turn_writes(self):
        """Voice latency: nothing but these three SDK calls, in this order, with or
        without a scope_path (no ladder lookup, no extra write)."""
        for path, ladder in ((None, make_ladder("b2b")), (PATH, make_ladder("deep"))):
            sdk = RecordingSDK(ladder=ladder)
            ctx = LLMContext()
            mem = SynapMemoryProcessor(sdk, user_id="dana", customer_id="acme",
                                       context=ctx, scope_path=path)
            rec = SynapRecorder(sdk, user_id="dana", customer_id="acme",
                                conversation_id="conv-1", scope_path=path)
            await _read(mem, "hello")
            await _one_turn(rec)
            assert [c.name for c in sdk.calls] == [
                "fetch", "instance.send_message", "instance.send_message"]


class TestIdsMustAgreeWithThePath:
    """The server lets the path's user/customer win over flat ids; refuse a disagreement
    with zero SDK calls. The ids are public attributes, so the per-turn calls check too."""

    BUILD = {
        "reader": lambda sdk, **k: SynapMemoryProcessor(sdk, context=LLMContext(), **k),
        "recorder": lambda sdk, **k: SynapRecorder(sdk, **k),
    }

    @pytest.mark.parametrize("name", ["reader", "recorder"])
    def test_construction_refused_accepted_empty_and_no_path(self, name):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK()
        build = self.BUILD[name]
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="bob", customer_id="acme", scope_path=PATH)
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="dana", customer_id="other", scope_path=PATH)
        build(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
        build(sdk, user_id="dana", customer_id="", scope_path=PATH)
        build(sdk, user_id="bob", customer_id="other")
        assert sdk.calls == []

    async def test_a_reassigned_id_skips_the_read_and_logs_only_fingerprints(self, caplog):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        proc = SynapMemoryProcessor(sdk, user_id="dana", customer_id="acme",
                                    context=LLMContext(), scope_path=PATH)
        proc.user_id = "bob-secret-id"
        with caplog.at_level("ERROR"):
            await _read(proc)
        assert sdk.calls == []
        assert any(r.levelname == "ERROR" for r in caplog.records)
        assert "bob-secret-id" not in caplog.text

    async def test_a_reassigned_id_drops_the_turn_instead_of_filing_it_elsewhere(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        rec = SynapRecorder(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
        rec.user_id = "bob"
        await _one_turn(rec)
        assert sdk.calls_to("instance.send_message") == []
        assert sdk.calls_to("conversation.record_message") == []

    async def test_matching_ids_still_read_and_record(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _read(SynapMemoryProcessor(sdk, user_id="dana", customer_id="acme",
                                         context=LLMContext(), scope_path=PATH))
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH
        rec = SynapRecorder(sdk, user_id="dana", customer_id="", scope_path=PATH)
        await _one_turn(rec)
        assert sdk.calls_to("instance.send_message") or sdk.calls_to("conversation.record_message")


class TestACustomerReassignedAfterConstruction:
    async def test_the_read_is_skipped_and_logged_without_the_id(self, caplog):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        proc = SynapMemoryProcessor(sdk, user_id="dana", customer_id="acme",
                                    context=LLMContext(), scope_path=PATH)
        proc.customer_id = "other-secret-customer"
        with caplog.at_level("ERROR"):
            await _read(proc)
        assert sdk.calls == []
        assert any(r.levelname == "ERROR" for r in caplog.records)
        assert "other-secret-customer" not in caplog.text

    async def test_the_turn_is_dropped(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        rec = SynapRecorder(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
        rec.customer_id = "other"
        await _one_turn(rec)
        assert sdk.calls_to("instance.send_message") == []
        assert sdk.calls_to("conversation.record_message") == []
