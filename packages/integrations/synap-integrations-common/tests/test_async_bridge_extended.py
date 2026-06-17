"""Extended tests for synap_integrations_common.async_bridge.

Covers:
- Happy-path: no running loop (asyncio.run path)
- Happy-path: running loop (nest_asyncio path)
- Exception propagation on both paths
- Return-value fidelity (None, complex types, large objects)
- Patched-loop registry deduplication
- Registry reset helper
- Coroutines scheduled after loop entry (multi-step async work)
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from synap_integrations_common.async_bridge import (
    _reset_patched_loops_for_tests,
    run_async,
)

# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure patched-loop registry is clean before and after each test."""
    _reset_patched_loops_for_tests()
    yield
    _reset_patched_loops_for_tests()


# ──────────────────────────────────────────────────────────────────
# 1. No-running-loop path (asyncio.run)
# ──────────────────────────────────────────────────────────────────


def test_no_loop_returns_integer():
    async def work():
        return 99

    assert run_async(work()) == 99


def test_no_loop_returns_none():
    async def work():
        return None

    assert run_async(work()) is None


def test_no_loop_returns_dict():
    async def work():
        return {"key": "value", "num": 42}

    result = run_async(work())
    assert result == {"key": "value", "num": 42}


def test_no_loop_returns_list():
    async def work():
        return [1, 2, 3]

    assert run_async(work()) == [1, 2, 3]


def test_no_loop_returns_tuple():
    async def work():
        return (10, 20)

    assert run_async(work()) == (10, 20)


def test_no_loop_returns_complex_object():
    class Payload:
        def __init__(self, val: int):
            self.val = val

    async def work():
        return Payload(7)

    result = run_async(work())
    assert result.val == 7


def test_no_loop_propagates_value_error():
    async def boom():
        raise ValueError("bad value")

    with pytest.raises(ValueError, match="bad value"):
        run_async(boom())


def test_no_loop_propagates_runtime_error():
    async def boom():
        raise RuntimeError("runtime fail")

    with pytest.raises(RuntimeError, match="runtime fail"):
        run_async(boom())


def test_no_loop_propagates_key_error():
    async def boom():
        d: dict = {}
        return d["missing"]

    with pytest.raises(KeyError):
        run_async(boom())


def test_no_loop_propagates_type_error():
    async def boom():
        return 1 + "x"  # type: ignore[operator]

    with pytest.raises(TypeError):
        run_async(boom())


def test_no_loop_multi_await_steps():
    """Coroutine that yields multiple times returns correct final value."""

    async def multistep():
        a = await asyncio.coroutine_or_future_stub()  # type: ignore[attr-defined]
        return a

    # Use actual awaitables
    async def real_multistep():
        step1 = await asyncio.sleep(0, result=1)
        step2 = await asyncio.sleep(0, result=step1 + 1)
        return step2

    assert run_async(real_multistep()) == 2


def test_no_loop_exception_preserves_traceback():
    """The original exception object is re-raised (not re-wrapped)."""

    original = ValueError("original")

    async def boom():
        raise original

    with pytest.raises(ValueError) as exc_info:
        run_async(boom())

    assert exc_info.value is original


# ──────────────────────────────────────────────────────────────────
# 2. Running-loop path (nest_asyncio)
# ──────────────────────────────────────────────────────────────────


def test_running_loop_returns_value():
    async def outer():
        async def inner():
            return "from-inner"

        return run_async(inner())

    assert asyncio.run(outer()) == "from-inner"


def test_running_loop_returns_none():
    async def outer():
        async def inner():
            return None

        return run_async(inner())

    assert asyncio.run(outer()) is None


def test_running_loop_propagates_exception():
    async def outer():
        async def inner():
            raise ValueError("inner-boom")

        return run_async(inner())

    with pytest.raises(ValueError, match="inner-boom"):
        asyncio.run(outer())


def test_running_loop_propagates_runtime_error():
    async def outer():
        async def inner():
            raise RuntimeError("rt-boom")

        run_async(inner())

    with pytest.raises(RuntimeError, match="rt-boom"):
        asyncio.run(outer())


def test_running_loop_registry_entry_added():
    """After calling run_async inside a loop, the loop id is in _patched_loops."""
    from synap_integrations_common.async_bridge import _patched_loops

    captured_loop_id: list[int] = []

    async def outer():
        loop = asyncio.get_running_loop()
        captured_loop_id.append(id(loop))

        async def inner():
            return 1

        run_async(inner())

    asyncio.run(outer())
    assert len(captured_loop_id) == 1
    assert captured_loop_id[0] in _patched_loops


def test_running_loop_patched_only_once():
    """Calling run_async multiple times on the same loop applies patch once.

    We verify by inspecting the set size — it must be exactly 1 after N calls.
    """
    from synap_integrations_common.async_bridge import _patched_loops

    async def outer():
        async def inner(v):
            return v

        results = [run_async(inner(i)) for i in range(5)]
        return results

    results = asyncio.run(outer())
    assert results == [0, 1, 2, 3, 4]
    # Only one unique loop was patched
    assert len(_patched_loops) == 1


def test_running_loop_nested_values():
    """Deeply nested coroutine chain returns correctly."""

    async def outer():
        async def level2():
            async def level3():
                return "deep"

            return run_async(level3())

        return run_async(level2())

    assert asyncio.run(outer()) == "deep"


def test_running_loop_multi_await_steps():
    async def outer():
        async def inner():
            a = await asyncio.sleep(0, result=10)
            b = await asyncio.sleep(0, result=a * 2)
            return b

        return run_async(inner())

    assert asyncio.run(outer()) == 20


# ──────────────────────────────────────────────────────────────────
# 3. Registry reset helper
# ──────────────────────────────────────────────────────────────────


def test_reset_clears_patched_loops():
    from synap_integrations_common.async_bridge import _patched_loops

    async def outer():
        async def inner():
            return 1

        run_async(inner())

    asyncio.run(outer())
    assert len(_patched_loops) == 1

    _reset_patched_loops_for_tests()
    assert len(_patched_loops) == 0


def test_reset_idempotent_on_empty():
    from synap_integrations_common.async_bridge import _patched_loops

    assert len(_patched_loops) == 0
    _reset_patched_loops_for_tests()
    assert len(_patched_loops) == 0


# ──────────────────────────────────────────────────────────────────
# 4. Thread-isolation: a new event loop in a thread is independent
# ──────────────────────────────────────────────────────────────────


def test_thread_creates_independent_loop():
    """run_async called from a background thread (no running loop there)
    should succeed independently, without interfering with the registry."""
    results: list[Any] = []
    errors: list[BaseException] = []

    def thread_work():
        try:
            async def coro():
                return "thread-result"

            results.append(run_async(coro()))
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=thread_work)
    t.start()
    t.join(timeout=5)

    assert errors == []
    assert results == ["thread-result"]


# ──────────────────────────────────────────────────────────────────
# 5. Public API / __all__ surface
# ──────────────────────────────────────────────────────────────────


def test_run_async_in_all():
    import synap_integrations_common.async_bridge as mod

    assert "run_async" in mod.__all__


def test_reset_not_in_all():
    """_reset_patched_loops_for_tests is a test-only internal; not in __all__."""
    import synap_integrations_common.async_bridge as mod

    assert "_reset_patched_loops_for_tests" not in mod.__all__
