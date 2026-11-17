"""RetryPolicy.execute wraps a user's own call, so it cannot tell whether that call is idempotent.
It retries any SynapTransientError, as before, unless the caller says the call is not idempotent
(``idempotent=False``): then an error whose outcome is unknown (a gateway 504, a read timeout) is
raised at once, because a retry could store and bill the same memory twice."""
import pytest

from maximem_synap.models.errors import (
    NetworkTimeoutError, ServiceUnavailableError, SynapTransientError,
)
from maximem_synap.resilience.retry import RetryPolicy


def _op(error, fail_times=1):
    calls = {"n": 0}

    async def op():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise error
        return "ok"

    return op, calls


def _unknown():
    e = ServiceUnavailableError("gateway")
    e.outcome_unknown = True
    return e


@pytest.mark.asyncio
async def test_default_behaviour_is_unchanged_an_unknown_outcome_is_retried():
    op, calls = _op(_unknown())
    assert await RetryPolicy(base_delay_ms=1).execute(op) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_a_non_idempotent_call_is_not_retried_after_an_unknown_outcome():
    op, calls = _op(_unknown(), fail_times=3)
    with pytest.raises(ServiceUnavailableError):
        await RetryPolicy(base_delay_ms=1).execute(op, idempotent=False)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_a_non_idempotent_call_is_still_retried_after_a_known_outcome():
    op, calls = _op(ServiceUnavailableError("503, the origin answered"))
    assert await RetryPolicy(base_delay_ms=1).execute(op, idempotent=False) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_an_idempotent_call_is_retried_even_after_an_unknown_outcome():
    op, calls = _op(_unknown())
    assert await RetryPolicy(base_delay_ms=1).execute(op, idempotent=True) == "ok"
    assert calls["n"] == 2
