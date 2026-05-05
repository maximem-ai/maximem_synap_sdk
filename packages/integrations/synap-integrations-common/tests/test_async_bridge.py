import asyncio

import pytest

from synap_integrations_common.async_bridge import (
    _reset_patched_loops_for_tests,
    run_async,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    _reset_patched_loops_for_tests()
    yield
    _reset_patched_loops_for_tests()


def test_run_async_no_running_loop_uses_asyncio_run():
    async def work():
        await asyncio.sleep(0)
        return 42

    assert run_async(work()) == 42


def test_run_async_propagates_exceptions():
    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_async(boom())


def test_run_async_from_within_running_loop():
    """Simulate the frameworks-in-async-context path.

    When a running loop is present we apply nest_asyncio and drive the
    coroutine on the existing loop. This exercises the registry branch.
    """

    async def outer():
        async def inner():
            await asyncio.sleep(0)
            return "inner-ok"

        # Call the sync bridge from within an async context — this is the
        # scenario frameworks like Haystack create.
        return run_async(inner())

    assert asyncio.run(outer()) == "inner-ok"


def test_run_async_patches_each_loop_once():
    """Second invocation on the same loop must not re-patch."""

    async def outer():
        async def inner(v):
            return v

        first = run_async(inner("a"))
        second = run_async(inner("b"))
        return first, second

    assert asyncio.run(outer()) == ("a", "b")
