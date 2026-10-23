"""Sync-to-async bridge for framework integrations.

Most agent frameworks (CrewAI StorageBackend, Haystack @component.run,
LangChain's sync retriever/tool protocols, Semantic Kernel kernel_function,
Google ADK FunctionTool) expose synchronous entry points. The Synap SDK is
async-first. Integrations therefore need to run a coroutine from sync code.

Two cases:

- **No loop is running in this thread** (a sync framework calling from the
  main thread or a worker thread). The coroutine is handed to ONE persistent
  background loop that lives for the life of the process. We used to call
  ``asyncio.run()`` here, which built a brand new loop for every call and closed
  it afterwards. The SDK's HTTP client keeps pooled connections that belong to the
  loop that opened them, so every second call picked up a connection from a loop
  that no longer existed and the write was lost ("Event loop is closed").
- **A loop is already running** (e.g. Haystack pipelines driven by an async
  runner). We apply ``nest_asyncio`` to that loop (once per loop, tracked by a
  module-level sentinel) and drive the coroutine on it, so the call runs on the
  same loop the caller is on.

If the caller is already async they should ``await`` the SDK directly;
``run_async`` is the last-resort bridge, not the primary API.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")

_patched_loops: "set[int]" = set()

_bg_lock = threading.Lock()
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread] = None
_bg_pid: Optional[int] = None


def _background_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent background loop, starting it on first use.

    Restarted if the thread died or the process forked (a forked child inherits
    the loop object but not the thread that ran it).
    """
    global _bg_loop, _bg_thread, _bg_pid
    with _bg_lock:
        alive = (
            _bg_loop is not None
            and not _bg_loop.is_closed()
            and _bg_thread is not None
            and _bg_thread.is_alive()
            and _bg_pid == os.getpid()
        )
        if not alive:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="synap-integrations-loop",
                daemon=True,
            )
            thread.start()
            _bg_loop, _bg_thread, _bg_pid = loop, thread, os.getpid()
        return _bg_loop  # type: ignore[return-value]


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


def run_async(coro: Awaitable[T]) -> T:
    """Run an awaitable from a synchronous context.

    - When no event loop is running in this thread, runs it on one persistent
      background loop and waits for the result.
    - When a loop is already running, applies ``nest_asyncio`` to that loop
      (once) and drives the coroutine to completion on it.

    Prefer awaiting the underlying async API directly whenever the caller
    can; this helper exists only to satisfy sync framework protocols.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        future = asyncio.run_coroutine_threadsafe(_await(coro), _background_loop())
        try:
            return future.result()
        except BaseException:
            # KeyboardInterrupt while waiting: stop the work we scheduled.
            future.cancel()
            raise

    loop_id = id(loop)
    if loop_id not in _patched_loops:
        import nest_asyncio

        nest_asyncio.apply(loop)
        _patched_loops.add(loop_id)

    return loop.run_until_complete(coro)  # type: ignore[arg-type]


def run_blocking(coro: Awaitable[T]) -> T:
    """Run an awaitable to completion on the background loop and wait, from anywhere.

    Unlike :func:`run_async`, this never re-enters the caller's running loop. Inside a
    running loop ``run_async`` uses ``nest_asyncio`` to run the coroutine on that same loop,
    and when several tasks do that at once the loop can crash (``IndexError: pop from an
    empty deque``). ``run_blocking`` instead hands the coroutine to the persistent
    background loop and blocks the calling thread until it finishes. If the caller is on a
    running loop that loop is blocked for the duration, exactly as it is with ``run_async``,
    but nothing is re-entered. Use it for coroutines that do not touch the caller's loop,
    such as Synap SDK calls. Prefer awaiting directly whenever you can.
    """
    future = asyncio.run_coroutine_threadsafe(_await(coro), _background_loop())
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


def _reset_patched_loops_for_tests() -> None:
    """Test-only hook to clear the patched-loop registry."""
    _patched_loops.clear()


__all__ = ["run_async", "run_blocking"]
