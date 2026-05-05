"""Uniform error handling for Synap integrations.

The pre-existing integrations were inconsistent: some swallowed every SDK
exception as a warning (silent ingestion failures in Haystack, silent
save failures in CrewAI), others let raw SDK exceptions propagate into
framework internals with no context about which operation failed.

This module gives every integration one shape:

    with wrap_sdk_errors("crewai.asave", logger, record_id=record.id):
        await sdk.memories.create(...)

- On success: pass-through.
- On failure: log once with the operation name and structured context,
  then re-raise as ``SynapIntegrationError`` (preserving the original via
  ``__cause__``).

Callers decide whether to catch ``SynapIntegrationError``; the contract is
that integration code NEVER silently discards an SDK failure.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Iterator, AsyncIterator, Optional


class SynapIntegrationError(RuntimeError):
    """Raised when a Synap SDK call fails inside an integration.

    The original exception is available via ``__cause__``. The ``operation``
    and ``context`` attributes are populated so framework-level error
    handlers can log structured data without re-parsing messages.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(f"{operation}: {message}")
        self.operation = operation
        self.context = context or {}


def _log_and_wrap(
    operation: str,
    logger: logging.Logger,
    exc: BaseException,
    context: dict,
) -> SynapIntegrationError:
    logger.error(
        "Synap integration call failed: op=%s error=%s context=%s",
        operation,
        exc,
        context,
        exc_info=True,
    )
    wrapped = SynapIntegrationError(operation, str(exc), context)
    wrapped.__cause__ = exc
    return wrapped


@contextmanager
def wrap_sdk_errors(
    operation: str,
    logger: logging.Logger,
    **context: Any,
) -> Iterator[None]:
    """Sync context manager that wraps SDK exceptions.

    Example::

        with wrap_sdk_errors("haystack.writer.run", logger, count=len(docs)):
            run_async(sdk.memories.create(...))
    """
    try:
        yield
    except SynapIntegrationError:
        raise
    except Exception as exc:  # noqa: BLE001 — intentional broad catch at boundary
        raise _log_and_wrap(operation, logger, exc, context) from exc


@asynccontextmanager
async def wrap_sdk_errors_async(
    operation: str,
    logger: logging.Logger,
    **context: Any,
) -> AsyncIterator[None]:
    """Async variant of :func:`wrap_sdk_errors`.

    Example::

        async with wrap_sdk_errors_async("crewai.asave", logger, id=rec.id):
            await sdk.memories.create(...)
    """
    try:
        yield
    except SynapIntegrationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _log_and_wrap(operation, logger, exc, context) from exc


__all__ = [
    "SynapIntegrationError",
    "wrap_sdk_errors",
    "wrap_sdk_errors_async",
]
