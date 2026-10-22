"""HTTPTransport must work when it is called from more than one event loop.

A pooled httpx connection belongs to the loop that opened it. The SDK is
initialised on the caller's loop, but sync framework bridges call it from other
threads, each with its own loop (often a new one per call). Before the transport
kept one client per loop, every second call from a fresh loop died with "Event
loop is closed" and the write was dropped.
"""

import asyncio
import http.server
import json
import threading
import types

import pytest

from maximem_synap.models.config import RetryPolicy
from maximem_synap.transport.http_client import HTTPTransport


class _Recorder(http.server.BaseHTTPRequestHandler):
    # HTTP/1.1 keeps connections alive, which is what makes pooled reuse (and the
    # bug) possible. An HTTP/1.0 server would hide it.
    protocol_version = "HTTP/1.1"
    seen: list = []

    def _reply(self):
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        type(self).seen.append(
            (self.command, self.path, self.headers.get("X-Instance-ID"))
        )
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    do_POST = _reply
    do_GET = _reply

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    handler = type("Handler", (_Recorder,), {"seen": []})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield types.SimpleNamespace(url=f"http://127.0.0.1:{srv.server_port}", seen=handler.seen)
    srv.shutdown()
    srv.server_close()


def _transport(server, instance_id="inst_aaaaaaaaaaaaaaaa", **kw):
    kw.setdefault("heartbeat_interval", 0)
    return HTTPTransport(
        instance_id, base_url=server.url, retry_policy=RetryPolicy(), **kw
    )


def _auth(instance_id="inst_aaaaaaaaaaaaaaaa"):
    return types.SimpleNamespace(api_key="k", client_id="cli", instance_id=instance_id)


def _post(transport, auth=None):
    return transport.post("/api/v1/memories/create", auth_context=auth or _auth(), json={"a": 1})


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(30)
    assert not t.is_alive(), "worker thread hung"
    if "error" in out:
        raise out["error"]
    return out.get("value")


def test_a_fresh_loop_for_every_call_all_succeed(server):
    """The sync-helper pattern: asyncio.run per call, from a thread."""
    transport = _transport(server)

    def twenty():
        return [asyncio.run(_post(transport)) for _ in range(20)]

    results = _in_thread(twenty)
    assert len(results) == 20
    assert len(server.seen) == 20


def test_calls_alternate_between_a_live_loop_and_fresh_loops(server):
    transport = _transport(server)
    live = asyncio.new_event_loop()
    threading.Thread(target=live.run_forever, daemon=True).start()
    try:
        # What sdk.initialize() does: first use on the caller's own, long-lived loop.
        asyncio.run_coroutine_threadsafe(_post(transport), live).result(10)
        for _ in range(10):
            _in_thread(lambda: asyncio.run(_post(transport)))
            asyncio.run_coroutine_threadsafe(_post(transport), live).result(10)
    finally:
        live.call_soon_threadsafe(live.stop)
    assert len(server.seen) == 21


def test_first_call_after_a_closed_init_loop_is_not_lost(server):
    transport = _transport(server)
    asyncio.run(_post(transport))  # init loop, then closes
    _in_thread(lambda: asyncio.run(_post(transport)))  # would have died here
    assert len(server.seen) == 2


def test_multipart_upload_works_on_a_second_loop(server):
    transport = _transport(server)
    asyncio.run(_post(transport))

    async def upload():
        return await transport.post_multipart(
            "/api/v1/memories/upload", _auth(), data={"x": "1"}, files={"f": ("a.txt", b"hi")}
        )

    assert _in_thread(lambda: asyncio.run(upload())) == {"ok": True}


def test_two_instances_never_share_identity_across_loops(server):
    """Per-loop clients live inside one transport, so instances cannot mix."""
    t1 = _transport(server, "inst_1111111111111111")
    t2 = _transport(server, "inst_2222222222222222")
    a1, a2 = _auth("inst_1111111111111111"), _auth("inst_2222222222222222")

    def work():
        for _ in range(5):
            asyncio.run(_post(t1, a1))
            asyncio.run(_post(t2, a2))

    asyncio.run(_post(t1, a1))
    asyncio.run(_post(t2, a2))
    _in_thread(work)

    by_instance = {}
    for _, _, inst in server.seen:
        by_instance[inst] = by_instance.get(inst, 0) + 1
    assert by_instance == {"inst_1111111111111111": 6, "inst_2222222222222222": 6}
    assert t1._clients is not t2._clients


def test_closed_loop_clients_are_dropped(server):
    transport = _transport(server)
    for _ in range(5):
        asyncio.run(_post(transport))
    # Only the most recent (now closed) loop's client can remain; older ones pruned.
    assert len(transport._clients) <= 2


def test_close_closes_clients_from_every_loop_without_raising(server):
    transport = _transport(server)
    asyncio.run(_post(transport))
    _in_thread(lambda: asyncio.run(_post(transport)))

    async def close():
        await transport.close()

    asyncio.run(close())
    assert transport._clients == {}


def test_heartbeat_restarts_on_a_new_loop(server):
    transport = _transport(server, heartbeat_interval=60)

    async def first():
        await _post(transport)
        return id(asyncio.get_running_loop()), transport._heartbeat_tasks.copy()

    id1, tasks1 = asyncio.run(first())
    assert id1 in tasks1  # started on loop 1; loop 1 is now closed

    async def second():
        await _post(transport)
        loop_id = id(asyncio.get_running_loop())
        task = transport._heartbeat_tasks.get(loop_id)
        assert task is not None and not task.done()
        assert task.get_loop() is asyncio.get_running_loop()
        await transport.close()

    asyncio.run(second())
    assert transport._heartbeat_tasks == {}


def test_a_client_assigned_to_underscore_client_is_used_everywhere(server):
    import httpx

    transport = _transport(server)
    transport._client = httpx.AsyncClient(base_url=server.url)
    asyncio.run(_post(transport))
    assert len(server.seen) == 1


def test_many_threads_on_many_loops_do_not_corrupt_the_bookkeeping(server):
    """Heartbeat and client bookkeeping is shared across threads, so it is locked."""
    transport = _transport(server, heartbeat_interval=60)
    errors = []

    def worker():
        try:
            for _ in range(15):
                asyncio.run(_post(transport))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert errors == []
    assert len(server.seen) == 8 * 15
