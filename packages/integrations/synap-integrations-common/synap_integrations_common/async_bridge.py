"""Sync-to-async bridge for framework integrations.

Most agent frameworks (CrewAI StorageBackend, Haystack @component.run,
LangChain's sync retriever/tool protocols, Semantic Kernel kernel_function,
Google ADK FunctionTool) expose synchronous entry points. The Synap SDK is
async-first. Integrations therefore need to run a coroutine from sync code.

The naive approach — `asyncio.run()` — fails when the caller is itself
running inside an event loop (e.g. Haystack pipelines driven by an async
runner). We fall back to ``nest_asyncio`` in that case, but:

- ``nest_asyncio.apply()`` patches the running loop class globally. We call
  it at most once per loop via a module-level sentinel so we don't re-patch
  on every call.
- If the caller is already async they should ``await`` the SDK directly;
  ``run_async`` is the last-resort bridge, not the primary API.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")

_patched_loops: "set[int]" = set()


def run_async(coro: Awaitable[T]) -> T:
    """Run an awaitable from a synchronous context.

    - When no event loop is running, uses ``asyncio.run``.
    - When a loop is already running, applies ``nest_asyncio`` to that loop
      (once) and drives the coroutine to completion on it.

    Prefer awaiting the underlying async API directly whenever the caller
    can — this helper exists only to satisfy sync framework protocols.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    loop_id = id(loop)
    if loop_id not in _patched_loops:
        import nest_asyncio

        nest_asyncio.apply(loop)
        _patched_loops.add(loop_id)

    return loop.run_until_complete(coro)  # type: ignore[arg-type]


def _reset_patched_loops_for_tests() -> None:
    """Test-only hook to clear the patched-loop registry."""
    _patched_loops.clear()


__all__ = ["run_async"]
