"""Client-side cooldown for ``instance_not_ready`` and ``instance_build_failed``.

JS twin: js/src/__tests__/instance-cooldown.test.ts.

After the server refuses a memory call with 409 ``instance_not_ready`` the SDK
raises the same error locally, with no network call, until the server's wait
(capped at 300 s) has passed. ``instance_build_failed`` is not time based: it
parses to its own error, sets no cooldown and is never retried.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from maximem_synap import ConflictError, InstanceBuildFailedError, InstanceNotReadyError
from maximem_synap.auth.models import AuthContext
from maximem_synap.models.config import RetryPolicy
from maximem_synap.transport import http_client
from maximem_synap.transport.grpc_client import GRPCTransport, StreamState
from maximem_synap.transport.http_client import HTTPTransport, is_gated_path

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")
GATED = ("POST", "/api/v1/memories/create")


def not_ready_body(wait=120):
    flat = {
        "error": "instance_not_ready",
        "message": "This instance's memory architecture is still being built.",
        "state": "building", "retry_after_seconds": wait, "instance_id": "inst_1",
    }
    return {**flat, "detail": {"code": "instance_not_ready", **flat}}


def failed_body():
    flat = {
        "error": "instance_build_failed", "message": "The build failed.",
        "state": "failed", "instance_id": "inst_1", "attempts": 3,
        "hint": "Run: maximem-synap instances rebuild inst_1",
    }
    return {**flat, "detail": {"code": "instance_build_failed", **flat}}


class Fake:
    """Fake network: counts requests and plays back a script of responses."""

    def __init__(self, *script):
        self.script = list(script)
        self.seen = []
        self.default = (200, {"ok": True})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path))
        status, body = self.script.pop(0) if self.script else self.default
        return httpx.Response(status, json=body)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(http_client.time, "monotonic", c)
    return c


def make(fake, retry=None):
    t = HTTPTransport(
        instance_id="inst_1", base_url="http://x", heartbeat_interval=0,
        retry_policy=retry,
    )
    t._client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake.handler), base_url="http://x"
    )
    return t


async def call(t, method="POST", path="/api/v1/memories/create"):
    return await t.request(method, path, AUTH, json={})


class TestCooldown:
    @pytest.mark.asyncio
    async def test_second_call_is_refused_locally_with_no_network(self, clock):
        fake = Fake((409, not_ready_body(120)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError) as first:
            await call(t)
        assert first.value.from_cooldown is False
        assert first.value.retry_after_seconds == 120
        with pytest.raises(InstanceNotReadyError) as second:
            await call(t)
        e = second.value
        assert len(fake.seen) == 1  # the second call never reached the network
        assert e.from_cooldown is True
        assert e.retry_after_seconds == 120
        assert str(e) == str(first.value)  # same message
        assert e.instance_id == "inst_1" and e.state == "building"
        assert isinstance(e, ConflictError)

    @pytest.mark.asyncio
    async def test_remaining_wait_decreases_and_rounds_up(self, clock):
        t = make(Fake((409, not_ready_body(120))))
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        clock.now += 30.4
        with pytest.raises(InstanceNotReadyError) as e:
            await call(t)
        assert e.value.retry_after_seconds == 90  # 89.6 rounded up
        clock.now += 89.0
        with pytest.raises(InstanceNotReadyError) as e:
            await call(t)
        assert e.value.retry_after_seconds == 1  # never below 1

    @pytest.mark.asyncio
    async def test_after_expiry_a_real_request_is_sent(self, clock):
        fake = Fake((409, not_ready_body(60)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        clock.now += 60.0
        assert await call(t) == {"ok": True}
        assert len(fake.seen) == 2

    @pytest.mark.asyncio
    async def test_a_successful_call_clears_the_cooldown(self, clock):
        # Call A is already on the wire when call B is refused and arms the
        # cooldown. A then succeeds: the instance is ready, so the cooldown goes.
        gate = asyncio.Event()
        seen = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["X-Correlation-ID"])
            if len(seen) == 1:  # A
                await gate.wait()
                return httpx.Response(200, json={"ok": True})
            if len(seen) == 2:  # B
                return httpx.Response(409, json=not_ready_body(300))
            return httpx.Response(200, json={"ok": True})

        t = HTTPTransport(instance_id="inst_1", base_url="http://x", heartbeat_interval=0)
        t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
        task_a = asyncio.ensure_future(call(t))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with pytest.raises(InstanceNotReadyError):
            await call(t)  # B arms the cooldown
        assert "inst_1" in t._not_ready_cooldown
        gate.set()
        assert await task_a == {"ok": True}
        assert "inst_1" not in t._not_ready_cooldown
        assert await call(t) == {"ok": True}
        assert len(seen) == 3  # the third call went to the network, not refused locally

    @pytest.mark.asyncio
    async def test_hint_is_capped_at_300_seconds(self, clock):
        t = make(Fake((409, not_ready_body(5000))))
        with pytest.raises(InstanceNotReadyError) as first:
            await call(t)
        assert first.value.retry_after_seconds == 5000  # the real error is untouched
        with pytest.raises(InstanceNotReadyError) as e:
            await call(t)
        assert e.value.retry_after_seconds == 300

    @pytest.mark.asyncio
    async def test_no_hint_means_no_cooldown(self, clock):
        body = not_ready_body()
        body.pop("retry_after_seconds"); body["detail"].pop("retry_after_seconds")
        fake = Fake((409, body))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        await call(t)
        assert len(fake.seen) == 2

    @pytest.mark.asyncio
    async def test_it_is_per_instance_id(self, clock):
        fake = Fake((409, not_ready_body(120)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        other = AuthContext(client_id="c", instance_id="inst_2", api_key="k")
        assert await t.request("POST", "/api/v1/memories/create", other, json={}) == {"ok": True}
        assert len(fake.seen) == 2

    @pytest.mark.asyncio
    async def test_two_transports_do_not_share_it(self, clock):
        a, b = Fake((409, not_ready_body(120))), Fake()
        ta, tb = make(a), make(b)
        with pytest.raises(InstanceNotReadyError):
            await call(ta)
        await call(tb)
        assert len(b.seen) == 1

    @pytest.mark.asyncio
    async def test_concurrent_callers_all_get_the_error_one_request_only(self, clock):
        fake = Fake((409, not_ready_body(120)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        results = await asyncio.gather(*[call(t) for _ in range(50)], return_exceptions=True)
        assert all(isinstance(r, InstanceNotReadyError) and r.from_cooldown for r in results)
        assert len(fake.seen) == 1

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_all_fail_cleanly_and_arm_one_cooldown(self, clock):
        fake = Fake()
        fake.default = (409, not_ready_body(120))
        t = make(fake)
        results = await asyncio.gather(*[call(t) for _ in range(20)], return_exceptions=True)
        assert all(isinstance(r, InstanceNotReadyError) for r in results)
        assert len(t._not_ready_cooldown) == 1

    @pytest.mark.asyncio
    async def test_never_retried_by_the_retry_policy(self, clock):
        fake = Fake((409, not_ready_body(120)))
        t = make(fake, retry=RetryPolicy(max_attempts=3))
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        assert len(fake.seen) == 1


class TestGatedPaths:
    GATED_LIST = [
        ("POST", "/v1/context/conversation/fetch"),
        ("POST", "/v1/context/user/fetch"),
        ("POST", "/v1/context/customer/fetch"),
        ("POST", "/v1/context/client/fetch"),
        ("POST", "/v1/conversations/messages"),
        ("POST", "/v1/conversations/messages/batch"),
        ("POST", "/v1/conversations/ingest"),
        ("POST", "/v1/events/batch"),
        ("POST", "/api/v1/memories/create"),
        ("POST", "/api/v1/memories/batch"),
        ("POST", "/api/v1/memories/upload"),
        ("POST", "/v1/migrations"),
        ("POST", "/v1/migrations/abc/run"),
        ("POST", "/v1/migrations/abc/approve"),
    ]
    NOT_GATED = [
        ("GET", "/api/v1/auth/whoami"),
        ("GET", "/health"),
        ("POST", "/api/v1/auth/init"),
        ("GET", "/api/v1/memories/status/ing_1"),
        ("GET", "/api/v1/memories/m1"),
        ("POST", "/api/v1/memories/forget"),
        ("GET", "/v1/credits/balance"),
        ("POST", "/v1/telemetry/batch"),
        ("POST", "/v1/conversations/compact"),
        ("GET", "/v1/conversations/c1/compaction/status"),
        ("GET", "/v1/conversations/messages"),
        ("GET", "/v1/users/u1/profile"),
        ("POST", "/v1/conversations/c1/messages"),
        ("GET", "/v1/migrations"),
    ]

    def test_the_list(self):
        for m, p in self.GATED_LIST:
            assert is_gated_path(m, p), (m, p)
        for m, p in self.NOT_GATED:
            assert not is_gated_path(m, p), (m, p)

    @pytest.mark.asyncio
    async def test_non_gated_calls_go_to_the_network_during_a_cooldown(self, clock):
        fake = Fake((409, not_ready_body(300)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await call(t)
        for m, p in self.NOT_GATED:
            await t.request(m, p, AUTH, json={} if m == "POST" else None)
        assert len(fake.seen) == 1 + len(self.NOT_GATED)
        # ... and a gated call is still refused locally afterwards
        with pytest.raises(InstanceNotReadyError) as e:
            await call(t)
        assert e.value.from_cooldown is True

    @pytest.mark.asyncio
    async def test_a_409_on_a_non_gated_path_sets_no_cooldown(self, clock):
        fake = Fake((409, not_ready_body(300)))
        t = make(fake)
        with pytest.raises(InstanceNotReadyError):
            await t.request("GET", "/api/v1/auth/whoami", AUTH)
        await call(t)
        assert len(fake.seen) == 2


class TestBuildFailedHttp:
    @pytest.mark.asyncio
    async def test_parsed_with_its_fields(self, clock):
        t = make(Fake((409, failed_body())))
        with pytest.raises(InstanceBuildFailedError) as ei:
            await call(t)
        e = ei.value
        assert e.instance_id == "inst_1"
        assert e.attempts == 3
        assert e.hint == "Run: maximem-synap instances rebuild inst_1"
        assert str(e) == "The build failed."
        assert isinstance(e, ConflictError)
        assert not isinstance(e, InstanceNotReadyError)

    @pytest.mark.asyncio
    async def test_detail_copy_alone_is_enough(self, clock):
        b = failed_body()
        t = make(Fake((409, {"detail": b["detail"]})))
        with pytest.raises(InstanceBuildFailedError) as ei:
            await call(t)
        assert ei.value.attempts == 3

    @pytest.mark.asyncio
    async def test_no_cooldown_and_no_retry(self, clock):
        fake = Fake((409, failed_body()), (409, failed_body()))
        t = make(fake, retry=RetryPolicy(max_attempts=3))
        with pytest.raises(InstanceBuildFailedError):
            await call(t)
        assert len(fake.seen) == 1  # not retried
        with pytest.raises(InstanceBuildFailedError) as e:
            await call(t)
        assert len(fake.seen) == 2  # every call is sent: nothing is cooling down
        assert not t._not_ready_cooldown
        assert e.value is not None
        assert await call(t) == {"ok": True}  # rebuilt: just works


# ---------------------------------------------------------------- the stream

class FakeStream:
    def __init__(self):
        self.written = []

    async def write(self, event):
        self.written.append(event)


class _Signal:
    def __init__(self, metadata, reason="refused"):
        self.signal_type = "throttle"
        self.reason = reason
        self.metadata = metadata


def failed_signal(series="1"):
    return _Signal({
        "code": "instance_build_failed", "state": "failed", "instance_id": "inst_1",
        "attempts": "3", "hint": "Run: maximem-synap instances rebuild inst_1",
        "series": series,
    })


def building_signal():
    return _Signal({"code": "instance_not_ready", "state": "building",
                    "retry_after_seconds": "60", "instance_id": "inst_1"})


@pytest.fixture
def t(tmp_path, monkeypatch):
    monkeypatch.setattr(GRPCTransport, "NOT_READY_MIN_WAIT", 0.02)
    monkeypatch.setattr(GRPCTransport, "NOT_READY_MAX_WAIT", 0.05)
    monkeypatch.setattr(GRPCTransport, "NOT_READY_PROBE_GRACE", 0.05)
    monkeypatch.setattr(GRPCTransport, "BUILD_FAILED_PROBE_INTERVAL", 0.04)
    monkeypatch.setattr(GRPCTransport, "BUILD_FAILED_PROBE_JITTER", 0.02)
    tr = GRPCTransport(instance_id="inst_1", host="localhost", port=1, use_tls=False,
                       storage_path=str(tmp_path))
    tr._stream = FakeStream()
    tr._state = StreamState.CONNECTED
    return tr


def turn(i):
    return {"event_type": "user_message", "conversation_id": "c1", "user_id": "u1",
            "content": f"m{i}", "role": "user", "event_id": f"ev-{i}"}


def warnings_with(caplog, text):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and text in r.getMessage()]


class TestStreamBuildFailed:
    @pytest.mark.asyncio
    async def test_events_kept_one_warning_with_hint_no_reconnect(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        await t.send(turn(2))
        for _ in range(6):
            t._handle_signal(failed_signal())
        assert set(t._unacked) == {"ev-1", "ev-2"}
        w = warnings_with(caplog, "build FAILED")
        assert len(w) == 1
        assert "maximem-synap instances rebuild inst_1" in w[0].getMessage()
        assert t.state == StreamState.CONNECTED
        assert t.instance_not_ready["code"] == "instance_build_failed"
        t._shutdown_event.set()
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_probe_interval_is_the_slow_fixed_one_not_the_hint(self, t):
        t._handle_signal(failed_signal())
        waits = {t._not_ready_wait() for _ in range(50)}
        assert all(0.04 <= w <= 0.06 for w in waits)
        assert len(waits) > 1
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_production_interval_is_60s_plus_jitter(self, tmp_path):
        tr = GRPCTransport(instance_id="inst_1", host="localhost", port=1,
                           use_tls=False, storage_path=str(tmp_path))
        tr._not_ready = {"code": "instance_build_failed", "retry_after_seconds": 0}
        assert all(60.0 <= tr._not_ready_wait() <= 70.0 for _ in range(50))

    @pytest.mark.asyncio
    async def test_probes_repeatedly_without_reconnect_then_recovers(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        t._handle_signal(failed_signal())
        orig = t._replay_unacked
        calls = {"n": 0}

        async def replay():
            calls["n"] += 1
            n = await orig()
            if calls["n"] < 3:
                t._handle_signal(failed_signal())  # still failed
            return n

        t._replay_unacked = replay
        await asyncio.wait_for(t._not_ready_task, 3)
        assert calls["n"] == 3  # probed three times, then the rebuild had worked
        assert len(warnings_with(caplog, "build FAILED")) == 1
        assert t.instance_not_ready is None
        assert t.state == StreamState.CONNECTED
        await t.send(turn(2))
        assert t._stream.written[-1].conversation_event.event_id == "ev-2"

    @pytest.mark.asyncio
    async def test_the_cap_still_applies_and_the_real_value_is_5000(self, t, monkeypatch):
        assert GRPCTransport.NOT_READY_MAX_HELD == 5000
        monkeypatch.setattr(GRPCTransport, "NOT_READY_MAX_HELD", 10)
        await t.send(turn(0))
        t._handle_signal(failed_signal())
        for i in range(1, 25):
            await t.send(turn(i))
        assert len(t._unacked) == 10  # oldest dropped at the cap, newest kept
        assert list(t._unacked) == [f"ev-{i}" for i in range(15, 25)]
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_a_new_series_warns_again(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        t._handle_signal(failed_signal("1"))
        t._handle_signal(failed_signal("1"))
        t._handle_signal(failed_signal("2"))
        assert len(warnings_with(caplog, "build FAILED")) == 2
        t._not_ready_task.cancel()


class TestStreamNotReadyStillWorks:
    @pytest.mark.asyncio
    async def test_instance_not_ready_keeps_its_own_message_and_hint_clamp(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        t._handle_signal(building_signal())
        t._handle_signal(building_signal())
        assert len(warnings_with(caplog, "still building")) == 1
        assert not warnings_with(caplog, "build FAILED")
        assert t.instance_not_ready["code"] == "instance_not_ready"
        await asyncio.wait_for(t._not_ready_task, 3)
        assert t.instance_not_ready is None
