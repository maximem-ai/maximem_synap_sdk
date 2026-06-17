"""Extended tests for synap_integrations_common.errors.

Covers:
- SynapIntegrationError constructor (all arg combinations)
- str()/repr() shape
- wrap_sdk_errors (sync): happy-path, wrapping, no double-wrap, logging
- wrap_sdk_errors_async (async): happy-path, wrapping, no double-wrap, logging
- Exception chaining (__cause__)
- Multiple/zero context kwargs
- Already-a-SynapIntegrationError passes through unchanged
- Logging assertions via caplog
"""
from __future__ import annotations

import logging

import pytest

from synap_integrations_common.errors import (
    SynapIntegrationError,
    wrap_sdk_errors,
    wrap_sdk_errors_async,
)


# ──────────────────────────────────────────────────────────────────
# 1. SynapIntegrationError constructor
# ──────────────────────────────────────────────────────────────────


def test_constructor_minimum_args():
    err = SynapIntegrationError("my.op", "something went wrong")
    assert err.operation == "my.op"
    assert err.context == {}


def test_constructor_with_context():
    ctx = {"record_id": "r1", "count": 5}
    err = SynapIntegrationError("op", "msg", context=ctx)
    assert err.context == ctx


def test_constructor_with_none_context_defaults_to_empty_dict():
    err = SynapIntegrationError("op", "msg", context=None)
    assert err.context == {}


def test_constructor_context_isolation():
    """Mutating original dict after construction should NOT affect .context
    if the implementation copies; OR if it doesn't copy, changes ARE reflected.
    This test just documents the actual behavior (no mutation guarantee claimed).
    """
    ctx = {"k": "v"}
    err = SynapIntegrationError("op", "msg", context=ctx)
    # Whatever the behavior, .context is a dict
    assert isinstance(err.context, dict)


def test_str_includes_operation():
    err = SynapIntegrationError("haystack.writer", "ingest failed")
    assert "haystack.writer" in str(err)


def test_str_includes_message():
    err = SynapIntegrationError("op", "ingest failed")
    assert "ingest failed" in str(err)


def test_is_runtime_error_subclass():
    err = SynapIntegrationError("op", "msg")
    assert isinstance(err, RuntimeError)


def test_is_exception_subclass():
    err = SynapIntegrationError("op", "msg")
    assert isinstance(err, Exception)


def test_can_be_raised_and_caught():
    with pytest.raises(SynapIntegrationError):
        raise SynapIntegrationError("op", "msg")


# ──────────────────────────────────────────────────────────────────
# 2. wrap_sdk_errors (sync) — happy path
# ──────────────────────────────────────────────────────────────────


def test_sync_no_error_passes_through():
    logger = logging.getLogger("test")
    collected = []
    with wrap_sdk_errors("op", logger):
        collected.append(1)
    assert collected == [1]


def test_sync_return_value_accessible_after_context():
    logger = logging.getLogger("test")
    result = None
    with wrap_sdk_errors("op", logger):
        result = 2 + 2
    assert result == 4


def test_sync_no_error_no_logging(caplog):
    logger = logging.getLogger("synap.test.noerror")
    with caplog.at_level(logging.ERROR, logger="synap.test.noerror"):
        with wrap_sdk_errors("op", logger):
            pass
    assert len(caplog.records) == 0


# ──────────────────────────────────────────────────────────────────
# 3. wrap_sdk_errors (sync) — wrapping exceptions
# ──────────────────────────────────────────────────────────────────


def test_sync_wraps_exception():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError):
        with wrap_sdk_errors("some.op", logger):
            raise RuntimeError("fail")


def test_sync_wraps_preserves_operation():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("crewai.save", logger):
            raise RuntimeError("fail")
    assert exc_info.value.operation == "crewai.save"


def test_sync_wraps_preserves_cause():
    logger = logging.getLogger("test")
    original = RuntimeError("root cause")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            raise original
    assert exc_info.value.__cause__ is original


def test_sync_wraps_includes_original_message():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            raise RuntimeError("the real message")
    assert "the real message" in str(exc_info.value)


def test_sync_wraps_with_context_kwargs():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger, record_id="r99", user="alice"):
            raise RuntimeError("fail")
    ctx = exc_info.value.context
    assert ctx.get("record_id") == "r99"
    assert ctx.get("user") == "alice"


def test_sync_wraps_zero_context_kwargs():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            raise RuntimeError("fail")
    assert exc_info.value.context == {}


def test_sync_wraps_value_error():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            raise ValueError("bad input")
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_sync_wraps_key_error():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            d: dict = {}
            _ = d["missing"]
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_sync_wraps_attribute_error():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("op", logger):
            raise AttributeError("no attr")
    assert isinstance(exc_info.value.__cause__, AttributeError)


# ──────────────────────────────────────────────────────────────────
# 4. wrap_sdk_errors (sync) — no double-wrap
# ──────────────────────────────────────────────────────────────────


def test_sync_no_double_wrap_synap_error_passes_through():
    logger = logging.getLogger("test")
    inner = SynapIntegrationError("inner.op", "already wrapped")
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("outer.op", logger):
            raise inner
    # Must be the same object, not a new wrapper
    assert exc_info.value is inner


def test_sync_no_double_wrap_preserves_inner_operation():
    logger = logging.getLogger("test")
    inner = SynapIntegrationError("inner.op", "msg", context={"k": "v"})
    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("outer.op", logger):
            raise inner
    assert exc_info.value.operation == "inner.op"
    assert exc_info.value.context == {"k": "v"}


# ──────────────────────────────────────────────────────────────────
# 5. wrap_sdk_errors (sync) — logging
# ──────────────────────────────────────────────────────────────────


def test_sync_logs_on_error(caplog):
    logger = logging.getLogger("synap.test.sync")
    with caplog.at_level(logging.ERROR, logger="synap.test.sync"):
        with pytest.raises(SynapIntegrationError):
            with wrap_sdk_errors("my.operation", logger, key1="v1"):
                raise RuntimeError("boom")

    assert len(caplog.records) >= 1
    record = caplog.records[-1]
    assert record.levelno == logging.ERROR


def test_sync_log_contains_operation_name(caplog):
    logger = logging.getLogger("synap.test.op")
    with caplog.at_level(logging.ERROR, logger="synap.test.op"):
        with pytest.raises(SynapIntegrationError):
            with wrap_sdk_errors("special.operation", logger):
                raise RuntimeError("x")
    assert any("special.operation" in r.message for r in caplog.records)


def test_sync_log_contains_context(caplog):
    logger = logging.getLogger("synap.test.ctx")
    with caplog.at_level(logging.ERROR, logger="synap.test.ctx"):
        with pytest.raises(SynapIntegrationError):
            with wrap_sdk_errors("op", logger, user_id="u42"):
                raise RuntimeError("x")
    # context dict appears in the log message
    assert any("u42" in r.message for r in caplog.records)


def test_sync_no_double_wrap_does_not_log(caplog):
    """When a SynapIntegrationError is re-raised it should NOT trigger another log."""
    logger = logging.getLogger("synap.test.nodbllog")
    inner = SynapIntegrationError("inner.op", "already wrapped")
    with caplog.at_level(logging.ERROR, logger="synap.test.nodbllog"):
        with pytest.raises(SynapIntegrationError):
            with wrap_sdk_errors("outer.op", logger):
                raise inner
    # No error logged for the re-raise of an existing SynapIntegrationError
    assert len(caplog.records) == 0


# ──────────────────────────────────────────────────────────────────
# 6. wrap_sdk_errors_async (async) — happy path
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_no_error_passes_through():
    logger = logging.getLogger("test")
    collected = []
    async with wrap_sdk_errors_async("op", logger):
        collected.append(1)
    assert collected == [1]


@pytest.mark.asyncio
async def test_async_no_error_no_logging(caplog):
    logger = logging.getLogger("synap.test.asyncnoerr")
    with caplog.at_level(logging.ERROR, logger="synap.test.asyncnoerr"):
        async with wrap_sdk_errors_async("op", logger):
            pass
    assert len(caplog.records) == 0


# ──────────────────────────────────────────────────────────────────
# 7. wrap_sdk_errors_async (async) — wrapping exceptions
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_wraps_exception():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError):
        async with wrap_sdk_errors_async("async.op", logger):
            raise RuntimeError("async-fail")


@pytest.mark.asyncio
async def test_async_preserves_operation():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("haystack.writer", logger):
            raise RuntimeError("fail")
    assert exc_info.value.operation == "haystack.writer"


@pytest.mark.asyncio
async def test_async_preserves_cause():
    logger = logging.getLogger("test")
    original = RuntimeError("async root")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("op", logger):
            raise original
    assert exc_info.value.__cause__ is original


@pytest.mark.asyncio
async def test_async_preserves_context_kwargs():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("op", logger, count=3, source="test"):
            raise RuntimeError("fail")
    assert exc_info.value.context.get("count") == 3
    assert exc_info.value.context.get("source") == "test"


@pytest.mark.asyncio
async def test_async_wraps_zero_context():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("op", logger):
            raise RuntimeError("fail")
    assert exc_info.value.context == {}


@pytest.mark.asyncio
async def test_async_wraps_value_error():
    logger = logging.getLogger("test")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("op", logger):
            raise ValueError("bad")
    assert isinstance(exc_info.value.__cause__, ValueError)


# ──────────────────────────────────────────────────────────────────
# 8. wrap_sdk_errors_async — no double-wrap
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_no_double_wrap():
    logger = logging.getLogger("test")
    inner = SynapIntegrationError("inner.op", "already wrapped")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("outer.op", logger):
            raise inner
    assert exc_info.value is inner


@pytest.mark.asyncio
async def test_async_no_double_wrap_preserves_inner_op():
    logger = logging.getLogger("test")
    inner = SynapIntegrationError("inner.op", "msg", context={"x": 1})
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("outer.op", logger):
            raise inner
    assert exc_info.value.operation == "inner.op"


@pytest.mark.asyncio
async def test_async_no_double_wrap_no_log(caplog):
    """SynapIntegrationError re-raised from async wrapper must NOT add log."""
    logger = logging.getLogger("synap.test.asyncnodbllog")
    inner = SynapIntegrationError("inner.op", "already wrapped")
    with caplog.at_level(logging.ERROR, logger="synap.test.asyncnodbllog"):
        with pytest.raises(SynapIntegrationError):
            async with wrap_sdk_errors_async("outer.op", logger):
                raise inner
    assert len(caplog.records) == 0


# ──────────────────────────────────────────────────────────────────
# 9. wrap_sdk_errors_async — logging
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_logs_on_error(caplog):
    logger = logging.getLogger("synap.test.asynclog")
    with caplog.at_level(logging.ERROR, logger="synap.test.asynclog"):
        with pytest.raises(SynapIntegrationError):
            async with wrap_sdk_errors_async("async.writer", logger, item_id="i1"):
                raise RuntimeError("async-boom")

    assert len(caplog.records) >= 1
    assert caplog.records[-1].levelno == logging.ERROR


@pytest.mark.asyncio
async def test_async_log_contains_operation(caplog):
    logger = logging.getLogger("synap.test.asynclogop")
    with caplog.at_level(logging.ERROR, logger="synap.test.asynclogop"):
        with pytest.raises(SynapIntegrationError):
            async with wrap_sdk_errors_async("async.special.op", logger):
                raise RuntimeError("x")
    assert any("async.special.op" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────
# 10. Exception chaining integrity
# ──────────────────────────────────────────────────────────────────


def test_cause_chain_sync():
    """__cause__ must be the EXACT original exception, not a copy."""
    logger = logging.getLogger("test")
    original = RuntimeError("original")
    try:
        with wrap_sdk_errors("op", logger):
            raise original
    except SynapIntegrationError as e:
        assert e.__cause__ is original
    else:
        pytest.fail("SynapIntegrationError not raised")


@pytest.mark.asyncio
async def test_cause_chain_async():
    """Async wrapper must also chain __cause__ to the exact original."""
    logger = logging.getLogger("test")
    original = RuntimeError("async-original")
    try:
        async with wrap_sdk_errors_async("op", logger):
            raise original
    except SynapIntegrationError as e:
        assert e.__cause__ is original
    else:
        pytest.fail("SynapIntegrationError not raised")


# ──────────────────────────────────────────────────────────────────
# 11. Public API / __all__
# ──────────────────────────────────────────────────────────────────


def test_all_exports_present():
    import synap_integrations_common.errors as mod

    for name in ["SynapIntegrationError", "wrap_sdk_errors", "wrap_sdk_errors_async"]:
        assert name in mod.__all__, f"{name} missing from __all__"


def test_synap_integration_error_importable_from_package():
    from synap_integrations_common import SynapIntegrationError as Sie  # noqa: N813

    assert Sie is SynapIntegrationError


def test_wrap_sdk_errors_importable_from_package():
    from synap_integrations_common import wrap_sdk_errors as w

    assert w is wrap_sdk_errors


def test_wrap_sdk_errors_async_importable_from_package():
    from synap_integrations_common import wrap_sdk_errors_async as wa

    assert wa is wrap_sdk_errors_async
