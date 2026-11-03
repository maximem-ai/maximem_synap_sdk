"""scope_path on report_turn / record_turn: forwarded only when set, never
sent without it, never raises."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap_integrations_common import record_turn, report_turn, stream_events
from synap_integrations_common.testing import RecordingSDK

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
from synap_integrations_common import coerce_conversation_id  # noqa: E402
CONV_REST = coerce_conversation_id("c1")  # REST wants the UUID the server derives from "c1"
BASE = dict(role="user", content="hi", conversation_id="c1", user_id="dana", customer_id="acme")


@pytest.fixture(autouse=True)
def _fresh_log_state():
    stream_events._scope_path_unsupported_logged.clear()
    yield
    stream_events._scope_path_unsupported_logged.clear()


STREAM_KW = {"content": "hi", "role": "user", "event_type": "user_message",
             "conversation_id": "c1", "user_id": "dana", "customer_id": "acme", "metadata": {}}


# -- forwarded only when set ----------------------------------------------------

@pytest.mark.asyncio
async def test_report_turn_without_scope_path_sends_exactly_todays_kwargs():
    sdk = RecordingSDK()
    assert await report_turn(sdk, **BASE)
    assert sdk.kwargs_of("instance.send_message") == [STREAM_KW]


@pytest.mark.asyncio
async def test_report_turn_with_scope_path_adds_only_that_keyword():
    sdk = RecordingSDK()
    assert await report_turn(sdk, **BASE, scope_path=PATH)
    assert sdk.kwargs_of("instance.send_message") == [{**STREAM_KW, "scope_path": PATH}]


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", [None, {}])
async def test_an_unset_scope_path_is_not_forwarded(unset):
    sdk = RecordingSDK()
    await report_turn(sdk, **BASE, scope_path=unset)
    assert "scope_path" not in sdk.last_kwargs("instance.send_message")


@pytest.mark.asyncio
async def test_record_turn_stream_path_forwards_scope_path_and_skips_rest():
    sdk = RecordingSDK()
    assert await record_turn(sdk, **BASE, scope_path=PATH)
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    assert sdk.calls_to("conversation.record_message") == []


@pytest.mark.asyncio
async def test_record_turn_rest_without_scope_path_is_byte_identical():
    sdk = RecordingSDK(listening=False)
    sdk.instance.listen = AsyncMock(side_effect=RuntimeError("no stream"))
    stream_events.set_auto_listen(sdk, False)
    assert await record_turn(sdk, **BASE)
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": CONV_REST, "role": "user", "content": "hi",
         "user_id": "dana", "customer_id": "acme"}]


@pytest.mark.asyncio
async def test_record_turn_rest_with_scope_path_adds_only_that_keyword():
    sdk = RecordingSDK(listening=False)
    stream_events.set_auto_listen(sdk, False)
    assert await record_turn(sdk, **BASE, metadata={"k": "v"}, scope_path=PATH)
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": CONV_REST, "role": "user", "content": "hi", "user_id": "dana",
         "customer_id": "acme", "metadata": {"k": "v"}, "scope_path": PATH}]
    assert sdk.calls_to("instance.send_message") == []


@pytest.mark.asyncio
async def test_forwarded_path_is_a_copy():
    sdk = RecordingSDK()
    mine = dict(PATH)
    await report_turn(sdk, **BASE, scope_path=mine)
    mine["team"] = "mutated"
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


# -- an SDK that cannot carry the path: never send without it -------------------

@pytest.mark.asyncio
async def test_report_turn_does_not_send_without_the_path_to_an_old_sdk(caplog):
    sdk = RecordingSDK(accept_scope_path=False)
    with caplog.at_level(logging.ERROR, logger=stream_events.logger.name):
        assert await report_turn(sdk, **BASE, scope_path=PATH) is False
    assert sdk.calls == []
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert "scope_path" in caplog.text and "Upgrade" in caplog.text


@pytest.mark.asyncio
async def test_record_turn_sends_nothing_anywhere_to_an_old_sdk():
    sdk = RecordingSDK(accept_scope_path=False)
    assert await record_turn(sdk, **BASE, scope_path=PATH) is False
    assert sdk.calls == []


@pytest.mark.asyncio
async def test_the_error_is_logged_once_per_process_per_method(caplog):
    sdk = RecordingSDK(accept_scope_path=False)
    with caplog.at_level(logging.ERROR, logger=stream_events.logger.name):
        for _ in range(5):
            await record_turn(sdk, **BASE, scope_path=PATH)
    # one for send_message, one for record_message
    assert len(caplog.records) == 2


@pytest.mark.asyncio
async def test_an_old_sdk_still_gets_turns_that_have_no_path():
    sdk = RecordingSDK(accept_scope_path=False)
    assert await report_turn(sdk, **BASE)
    assert sdk.kwargs_of("instance.send_message") == [STREAM_KW]


@pytest.mark.asyncio
async def test_stream_without_path_support_falls_back_to_rest_when_rest_supports_it():
    sdk = RecordingSDK()
    # a stream method from before scope_path, REST method with it
    async def old_send(*, content, role, event_type, conversation_id, user_id, customer_id, metadata):
        raise AssertionError("must not be called")
    sdk.instance.send_message = old_send
    assert await record_turn(sdk, **BASE, scope_path=PATH)
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


@pytest.mark.asyncio
async def test_signature_is_inspected_not_a_typeerror_caught():
    """A method with **kwargs is accepted; one with neither is refused up front
    and never called."""
    called = []

    async def takes_kwargs(**kw):
        called.append(kw)

    async def takes_nothing_extra(*, content, role, event_type, conversation_id,
                                  user_id, customer_id, metadata):
        called.append("old")

    for fn, expect in ((takes_kwargs, True), (takes_nothing_extra, False)):
        called.clear()
        sdk = SimpleNamespace(instance=SimpleNamespace(is_listening=True, send_message=fn))
        stream_events._scope_path_unsupported_logged.clear()
        assert await report_turn(sdk, **BASE, scope_path=PATH) is expect
        assert bool(called) is expect


# -- never raises -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_failures_with_a_scope_path_never_raise():
    sdk = RecordingSDK()
    sdk.errors["instance.send_message"] = RuntimeError("down")
    sdk.errors["conversation.record_message"] = RuntimeError("down")
    assert await record_turn(sdk, **BASE, scope_path=PATH) is False


@pytest.mark.asyncio
async def test_an_unreadable_scope_path_never_raises():
    sdk = RecordingSDK()
    assert await report_turn(sdk, **BASE, scope_path=object()) is False
    assert sdk.calls == []


@pytest.mark.asyncio
async def test_a_sdk_without_an_instance_never_raises():
    assert await report_turn(SimpleNamespace(), **BASE, scope_path=PATH) is False
    assert await record_turn(SimpleNamespace(), **BASE, scope_path=PATH) is False


# -- the other events carry no path (documented, pinned) --------------------------

@pytest.mark.asyncio
async def test_tool_and_reasoning_events_are_unchanged_and_carry_no_path():
    sdk = RecordingSDK()
    se = stream_events
    await se.report_tool_call(sdk, tool_name="t", conversation_id="c1", user_id="dana")
    await se.report_tool_result(sdk, result="r", conversation_id="c1", user_id="dana")
    await se.report_reasoning(sdk, content="hm", conversation_id="c1", user_id="dana")
    for name in ("record_tool_call", "record_tool_result", "record_thinking"):
        assert "scope_path" not in sdk.last_kwargs(f"instance.{name}")


# -- REST-side guard and stream-then-REST fallback -----------------------------

@pytest.mark.asyncio
async def test_rest_side_never_calls_a_record_message_that_cannot_take_scope_path():
    calls = []

    async def old_record_message(conversation_id, role, content, user_id, customer_id=None,
                                 session_id=None, metadata=None):
        calls.append(conversation_id)

    sdk = SimpleNamespace(
        instance=SimpleNamespace(is_listening=False, send_message=AsyncMock()),
        conversation=SimpleNamespace(record_message=old_record_message))
    stream_events.set_auto_listen(sdk, False)
    assert await record_turn(sdk, **BASE, scope_path=PATH) is False
    assert calls == []
    sdk.instance.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stream_that_fails_mid_call_falls_back_to_rest_with_the_path():
    sdk = RecordingSDK()
    sdk.errors["instance.send_message"] = RuntimeError("stream died")
    assert await record_turn(sdk, **BASE, scope_path=PATH) is True
    assert len(sdk.calls_to("instance.send_message")) == 1  # the stream was tried first
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": CONV_REST, "role": "user", "content": "hi", "user_id": "dana",
         "customer_id": "acme", "scope_path": PATH}]
