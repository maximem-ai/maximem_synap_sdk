"""Retry logic and circuit breaker implementation."""

import asyncio
import logging
import random
import time
from typing import Callable, Any
from threading import Lock

from ..models.errors import SynapTransientError, SynapError

logger = logging.getLogger("synap.sdk.resilience")


class RetryPolicy:
    """Exponential backoff retry policy."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: int = 100,
        max_delay_ms: int = 5000
    ):
        """Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay_ms: Base delay in milliseconds
            max_delay_ms: Maximum delay cap in milliseconds
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms

    async def execute(self, operation: Callable, *, idempotent: bool = True) -> Any:
        """Execute operation with exponential backoff retry.

        Only retries on SynapTransientError. All other exceptions
        propagate immediately.

        Args:
            operation: Async callable (zero-argument) to execute
            idempotent: Whether running ``operation`` twice is harmless. ``True`` (the
                default, as before) retries any SynapTransientError. Pass ``False`` for a
                create or other write: an error whose outcome is unknown (``outcome_unknown``:
                a gateway 504, a read timeout) is then raised at once, since the first
                attempt may have been processed and a retry could store and bill it twice.

        Returns:
            Result of operation

        Raises:
            SynapTransientError: If all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except SynapTransientError as e:
                last_error = e
                if not idempotent and e.outcome_unknown:
                    break
                if attempt >= self.max_retries:
                    break
                delay_ms = min(
                    self.base_delay_ms * (2 ** attempt),
                    self.max_delay_ms,
                )
                jitter = delay_ms * 0.25 * (random.random() * 2 - 1)
                delay_seconds = max(0, delay_ms + jitter) / 1000.0
                logger.warning(
                    "Retry %d/%d after %.2fs: %s",
                    attempt + 1, self.max_retries, delay_seconds, e,
                )
                await asyncio.sleep(delay_seconds)
            except Exception:
                raise  # Non-transient errors propagate immediately

        raise last_error


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    States:
        closed: Normal operation, tracking failures
        open: Rejecting calls, waiting for recovery timeout
        half-open: Allowing one probe call to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 30000
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout_ms: Time to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self._state = "closed"
        self._lock = Lock()
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    async def call(self, operation: Callable) -> Any:
        """Execute operation through circuit breaker.

        Args:
            operation: Async callable (zero-argument) to execute

        Returns:
            Result of operation

        Raises:
            SynapError: If circuit is open and recovery timeout not elapsed
        """
        with self._lock:
            if self._state == "open":
                elapsed_ms = (
                    time.monotonic() - self._last_failure_time
                ) * 1000
                if elapsed_ms >= self.recovery_timeout_ms:
                    self._state = "half-open"
                else:
                    remaining_s = (
                        self.recovery_timeout_ms - elapsed_ms
                    ) / 1000
                    raise SynapError(
                        f"Circuit breaker is open. "
                        f"Retry after {remaining_s:.1f}s"
                    )

        try:
            result = await operation()
        except Exception:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._state == "half-open":
                    self._state = "open"
                elif self._failure_count >= self.failure_threshold:
                    self._state = "open"
            raise
        else:
            with self._lock:
                self._failure_count = 0
                self._state = "closed"
            return result

    @property
    def state(self) -> str:
        """Get current circuit breaker state.

        Returns:
            One of: "closed", "open", "half-open"
        """
        with self._lock:
            return self._state
