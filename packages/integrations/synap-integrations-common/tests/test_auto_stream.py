"""The stream is the default path: the first report opens it, REST covers the rest."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap_integrations_common import (
    end_session,
    ensure_stream,
    record_turn,
    report_tool_call,
    set_auto_listen,
    stream_is_active,
)
from synap_integrations_common import stream_events


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self, fail=False, hang=False):
        self.is_listening = False
        self.listen_calls = 0
        self.fail = fail
        self.hang = hang
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.end_session = AsyncMock()
        self._controller = SimpleNamespace(_transport=object())

    async def listen(self):
        self.listen_calls += 1
        if self.hang:
            await asyncio.sleep(3600)
        if self.fail:
            raise ConnectionError("no route to host")
        self.is_listening = True

    async def stop_listening(self):
        self.is_listening = False


def make_sdk(**kw):
    return SimpleNamespace(
        instance=FakeInstance(**kw),
        conversation=SimpleNamespace(record_message=AsyncMock()),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SYNAP_AUTO_LISTEN", raising=False)


@pytest.mark.asyncio
async def test_the_first_turn_opens_the_stream_and_goes_on_it():
    sdk = make_sdk()
    assert await record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1")
    assert sdk.instance.listen_calls == 1
    sdk.instance.send_message.assert_awaited_once()
    sdk.conversation.record_message.assert_not_awaited()
    # The second reuses it.
    assert await record_turn(sdk, role="assistant", content="hello", conversation_id="c1", user_id="u1")
    assert sdk.instance.listen_calls == 1


@pytest.mark.asyncio
async def test_tool_events_open_the_stream_too():
    sdk = make_sdk()
    assert await report_tool_call(sdk, tool_name="search", conversation_id="c1")
    sdk.instance.record_tool_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stream_that_will_not_open_falls_back_to_rest_and_waits_a_minute():
    sdk = make_sdk(fail=True)
    assert await record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1")
    sdk.conversation.record_message.assert_awaited_once()
    assert await record_turn(sdk, role="user", content="again", conversation_id="c1", user_id="u1")
    assert sdk.instance.listen_calls == 1  # not retried on every turn


@pytest.mark.asyncio
async def test_opening_never_waits_forever(monkeypatch):
    monkeypatch.setattr(stream_events, "_OPEN_TIMEOUT_S", 0.05)
    sdk = make_sdk(hang=True)
    assert await record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1")
    sdk.conversation.record_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", "0", "off", "NO"])
async def test_the_env_var_turns_it_off(monkeypatch, value):
    monkeypatch.setenv("SYNAP_AUTO_LISTEN", value)
    sdk = make_sdk()
    assert await record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1")
    assert sdk.instance.listen_calls == 0
    sdk.conversation.record_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_auto_listen_turns_it_off_for_one_sdk(monkeypatch):
    sdk = make_sdk()
    set_auto_listen(sdk, False)
    assert not await ensure_stream(sdk)
    assert sdk.instance.listen_calls == 0
    other = make_sdk()
    assert await ensure_stream(other)


@pytest.mark.asyncio
async def test_a_stream_the_caller_opened_is_used_even_when_auto_is_off():
    sdk = make_sdk()
    sdk.instance.is_listening = True
    set_auto_listen(sdk, False)
    assert await record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1")
    sdk.instance.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_end_session_never_opens_a_stream():
    sdk = make_sdk()
    assert not await end_session(sdk, "c1")
    assert sdk.instance.listen_calls == 0


def run_on_fresh_loop(coro):
    """Like asyncio.run, but immune to a nest_asyncio patch from another test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_a_stream_on_a_closed_loop_is_replaced_by_one_on_the_new_loop():
    sdk = make_sdk()

    async def turn(text):
        return await record_turn(sdk, role="user", content=text, conversation_id="c1", user_id="u1")

    assert run_on_fresh_loop(turn("one"))  # opens on loop 1, which then closes
    assert not stream_is_active(sdk)  # died with its loop
    assert run_on_fresh_loop(turn("two"))
    assert sdk.instance.listen_calls == 2
    assert sdk.instance.send_message.await_count == 2


def test_after_repeated_loop_churn_it_stays_on_rest():
    sdk = make_sdk()

    async def turn():
        return await record_turn(sdk, role="user", content="t", conversation_id="c1", user_id="u1")

    for _ in range(stream_events._MAX_REOPENS + 4):
        assert run_on_fresh_loop(turn())
    assert sdk.instance.listen_calls == stream_events._MAX_REOPENS + 1
    assert sdk.conversation.record_message.await_count >= 1


def test_a_stream_alive_on_another_loop_is_not_written_from_this_one():
    sdk = make_sdk()
    ready = threading.Event()
    stop = threading.Event()
    other = asyncio.new_event_loop()

    def run_other():
        asyncio.set_event_loop(other)
        other.run_until_complete(ensure_stream(sdk))
        ready.set()
        other.run_until_complete(asyncio.sleep(0))
        stop.wait(5)

    t = threading.Thread(target=run_other)
    t.start()
    ready.wait(5)
    try:
        assert stream_is_active(sdk)
        ok = run_on_fresh_loop(record_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u1"))
        assert ok
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_awaited_once()
    finally:
        stop.set()
        t.join(5)
        other.close()


@pytest.mark.asyncio
async def test_never_raises_on_an_sdk_without_an_instance():
    assert not await ensure_stream(SimpleNamespace())
    assert not await ensure_stream(None)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.mark.asyncio
async def test_a_first_failed_open_is_retried_after_seconds_then_a_minute(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(stream_events.time, "monotonic", clock)
    sdk = make_sdk(fail=True)
    assert not await ensure_stream(sdk)              # cold first open fails
    clock.t += 1
    assert not await ensure_stream(sdk)              # inside the quick window: no retry
    assert sdk.instance.listen_calls == 1
    clock.t += stream_events._FIRST_RETRY_AFTER_S
    assert not await ensure_stream(sdk)              # quick retry happens, fails again
    assert sdk.instance.listen_calls == 2
    clock.t += stream_events._FIRST_RETRY_AFTER_S + 1
    assert not await ensure_stream(sdk)              # now the minute cooldown applies
    assert sdk.instance.listen_calls == 2
    clock.t += stream_events._RETRY_AFTER_S
    sdk.instance.fail = False
    assert await ensure_stream(sdk)                  # opens once the minute is up
    assert sdk.instance.listen_calls == 3


@pytest.mark.asyncio
async def test_a_cold_first_open_recovers_within_seconds(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(stream_events.time, "monotonic", clock)
    sdk = make_sdk(fail=True)
    assert await record_turn(sdk, role="user", content="t1", conversation_id="c1", user_id="u1")
    sdk.conversation.record_message.assert_awaited_once()   # turn 1 over REST
    sdk.instance.fail = False                                # the connection is warm now
    clock.t += stream_events._FIRST_RETRY_AFTER_S
    assert await record_turn(sdk, role="user", content="t2", conversation_id="c1", user_id="u1")
    sdk.instance.send_message.assert_awaited_once()          # turn 2 on the stream
    assert sdk.conversation.record_message.await_count == 1
    assert stream_events._state(sdk)["failures"] == 0
