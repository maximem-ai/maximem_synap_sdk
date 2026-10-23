"""run_async must not build a new event loop per call.

The Synap SDK's HTTP client pools connections that belong to the loop that opened
them. A helper that runs each call on a brand new loop (asyncio.run per call)
hands the next call a connection from a loop that has already closed, and the
write is lost. These tests pin the behaviour that prevents that.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from synap_integrations_common import async_bridge
from synap_integrations_common.async_bridge import run_async, run_blocking


def _loop_id():
    async def probe():
        return id(asyncio.get_running_loop()), threading.current_thread().name

    return run_async(probe())


def test_every_call_without_a_running_loop_shares_one_loop():
    ids = {_loop_id() for _ in range(20)}
    assert len(ids) == 1
    assert next(iter(ids))[1] == "synap-integrations-loop"


def test_calls_from_different_threads_share_the_same_loop():
    seen, lock = [], threading.Lock()

    def work():
        for _ in range(5):
            with lock:
                seen.append(_loop_id()[0])

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(seen) == 20 and len(set(seen)) == 1


def test_a_running_loop_is_still_used_when_there_is_one():
    async def outer():
        outer_id = id(asyncio.get_running_loop())

        async def inner():
            return id(asyncio.get_running_loop())

        return outer_id, run_async(inner())

    outer_id, inner_id = asyncio.run(outer())
    assert outer_id == inner_id


def test_exceptions_cross_the_thread_boundary_unchanged():
    class Boom(Exception):
        pass

    async def fail():
        raise Boom("x")

    with pytest.raises(Boom, match="x"):
        run_async(fail())


def test_the_background_loop_is_restarted_if_it_stops():
    first = _loop_id()[0]
    loop = async_bridge._bg_loop
    loop.call_soon_threadsafe(loop.stop)
    async_bridge._bg_thread.join(5)
    assert _loop_id()[0] != first  # a new loop took over; the call still worked


# --- run_blocking: never re-enters the caller's loop ------------------------------------


def test_run_blocking_without_a_loop_uses_the_background_loop():
    async def probe():
        return threading.current_thread().name

    assert run_blocking(probe()) == "synap-integrations-loop"


def test_run_blocking_inside_a_running_loop_runs_elsewhere_not_on_that_loop():
    async def outer():
        caller = id(asyncio.get_running_loop())

        async def probe():
            return id(asyncio.get_running_loop())

        return caller, run_blocking(probe())

    caller, ran_on = asyncio.run(outer())
    assert caller != ran_on  # a different loop: the caller's loop was not re-entered


def test_run_blocking_propagates_exceptions():
    class Boom(Exception):
        pass

    async def fail():
        raise Boom("x")

    async def outer():
        run_blocking(fail())

    with pytest.raises(Boom):
        asyncio.run(outer())


def test_many_tasks_calling_run_blocking_on_one_loop_do_not_crash_it():
    """With run_async (nest_asyncio) concurrent tasks like this can kill the loop with
    ``IndexError: pop from an empty deque``; run_blocking never re-enters it."""
    async def work(i):
        await asyncio.sleep(0)

        async def leaf():
            await asyncio.sleep(0.01)
            return i

        return run_blocking(leaf())

    async def main():
        return await asyncio.gather(*[work(i) for i in range(20)])

    assert asyncio.run(main()) == list(range(20))


# --- through the real SDK transport -------------------------------------------------

# Runs in a fresh interpreter on purpose: any earlier test that applied nest_asyncio
# patches asyncio.run for the whole process (it reuses one loop), which hides the very
# bug this test exists to catch.
_SCRIPT = r"""
import asyncio, http.server, sys, threading, types
from maximem_synap.models.config import RetryPolicy
from maximem_synap.transport.http_client import HTTPTransport
from synap_integrations_common import run_async

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    count = 0
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        H.count += 1
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
t = HTTPTransport("inst_aaaaaaaaaaaaaaaa", base_url=f"http://127.0.0.1:{srv.server_port}",
                  retry_policy=RetryPolicy(), heartbeat_interval=0)
auth = types.SimpleNamespace(api_key="k", client_id="c", instance_id="inst_aaaaaaaaaaaaaaaa")
# What sdk.initialize() does: first use of the client on the caller's own loop, which then closes.
asyncio.run(t.post("/api/v1/memories/create", auth_context=auth, json={}))
ok = 0
for i in range(20):
    try:
        run_async(t.post("/api/v1/memories/create", auth_context=auth, json={"i": i}))
        ok += 1
    except Exception as exc:
        print("lost:", type(exc).__name__, exc)
print("OK", ok, "SERVER_SAW", H.count)
"""


def test_twenty_sdk_writes_through_the_helper_all_land():
    import os
    import subprocess
    import sys

    here = os.path.dirname(__file__)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(here, ".."),
            os.path.join(here, "../../../synap/sdk/python"),
            env.get("PYTHONPATH", ""),
        ]
    )
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=60
    )
    assert "OK 20 SERVER_SAW 21" in out.stdout, out.stdout + out.stderr
