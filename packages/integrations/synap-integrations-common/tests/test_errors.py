import logging

import pytest

from synap_integrations_common.errors import (
    SynapIntegrationError,
    wrap_sdk_errors,
    wrap_sdk_errors_async,
)


def test_sync_wrapper_passthrough():
    logger = logging.getLogger("t")
    with wrap_sdk_errors("op", logger):
        result = 1 + 1
    assert result == 2


def test_sync_wrapper_wraps_and_preserves_cause():
    logger = logging.getLogger("t")
    original = RuntimeError("underlying")

    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("crewai.asave", logger, record_id="r1"):
            raise original

    err = exc_info.value
    assert err.operation == "crewai.asave"
    assert err.context == {"record_id": "r1"}
    assert err.__cause__ is original
    assert "crewai.asave" in str(err)


def test_sync_wrapper_does_not_double_wrap():
    logger = logging.getLogger("t")
    inner = SynapIntegrationError("inner.op", "already wrapped")

    with pytest.raises(SynapIntegrationError) as exc_info:
        with wrap_sdk_errors("outer.op", logger):
            raise inner

    assert exc_info.value is inner


def test_sync_wrapper_logs_error(caplog):
    logger = logging.getLogger("synap.test")
    with caplog.at_level(logging.ERROR, logger="synap.test"):
        with pytest.raises(SynapIntegrationError):
            with wrap_sdk_errors("op", logger, key="val"):
                raise RuntimeError("boom")

    assert any("op=op" in rec.message for rec in caplog.records)
    assert any("key" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_async_wrapper_wraps():
    logger = logging.getLogger("t")
    with pytest.raises(SynapIntegrationError) as exc_info:
        async with wrap_sdk_errors_async("haystack.writer", logger, count=3):
            raise RuntimeError("ingest failed")

    assert exc_info.value.operation == "haystack.writer"
    assert exc_info.value.context == {"count": 3}
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_async_wrapper_passthrough():
    logger = logging.getLogger("t")
    async with wrap_sdk_errors_async("op", logger):
        pass  # no error -> no raise
