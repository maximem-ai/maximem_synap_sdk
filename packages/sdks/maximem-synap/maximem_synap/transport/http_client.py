"""HTTP/REST transport implementation using httpx."""

import asyncio
import json
import random
import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timezone

import httpx

from ..models.config import TimeoutConfig, RetryPolicy
from .base import BaseTransport
from ..models.errors import (
    SynapError,
    SynapTransientError,
    NetworkTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidInputError,
    ContextNotFoundError,
    InsufficientCreditsError,
    ConflictError,
    TranscriptConflictError,
)
from ..auth.models import AuthContext
from ..utils.correlation import generate_correlation_id


logger = logging.getLogger("synap.sdk.transport.http")


class HTTPTransport:
    """HTTP transport with retries, timeouts, and telemetry.

    Features:
    - Automatic retries with exponential backoff
    - Configurable timeouts
    - Auth context injection
    - Correlation ID propagation
    - Telemetry emission
    """

    # Base URL for Synap API
    DEFAULT_BASE_URL = "https://synap-cloud-prod.maximem.ai"

    # Connection lifecycle. httpx's default keepalive_expiry is 5s, which
    # means real-world call patterns (one fetch per conversation, minutes
    # apart) pay a full fresh-connection setup on every call — measured at
    # ~600-950ms through the CDN to the origin. Instead, idle connections
    # are kept for KEEPALIVE_EXPIRY_SECONDS (closed only after that much
    # inactivity), and an optional background heartbeat pings /health every
    # HEARTBEAT_INTERVAL_SECONDS so the connection stays warm indefinitely
    # while the SDK object lives. The heartbeat request carries no auth and
    # is not metered.
    KEEPALIVE_EXPIRY_SECONDS = 300.0
    HEARTBEAT_INTERVAL_SECONDS = 240.0

    def __init__(
        self,
        instance_id: str,
        base_url: Optional[str] = None,
        timeouts: Optional[TimeoutConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
        telemetry_callback: Optional[Callable[[Dict], None]] = None,
        keepalive_expiry: Optional[float] = None,
        heartbeat_interval: Optional[float] = None,
    ):
        self.instance_id = instance_id
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeouts = timeouts or TimeoutConfig()
        self.retry_policy = retry_policy
        self.telemetry_callback = telemetry_callback

        self._limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=(
                keepalive_expiry
                if keepalive_expiry is not None
                else self.KEEPALIVE_EXPIRY_SECONDS
            ),
        )
        # 0 disables the heartbeat; None means default.
        self._heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else self.HEARTBEAT_INTERVAL_SECONDS
        )
        self._heartbeat_task: Optional["asyncio.Task"] = None

        # Create httpx client with timeout + keepalive config
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=self.timeouts.connect,
                read=self.timeouts.read,
                write=self.timeouts.write,
                pool=self.timeouts.connect,
            ),
            limits=self._limits,
        )

    def _ensure_heartbeat(self) -> None:
        """Start the keepalive heartbeat once a running loop exists.

        Called from request() (not __init__) because the SDK may be
        constructed outside an event loop. Idempotent; a finished/cancelled
        task is restarted.
        """
        if not self._heartbeat_interval:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                # Unauthenticated, unmetered; sole purpose is to keep the
                # pooled connection (and the CDN's origin connection) warm.
                await self._client.get("/health", timeout=10.0)
            except Exception:  # noqa: BLE001 — best-effort; next call reconnects
                pass

    async def close(self) -> None:
        """Close the HTTP client and stop the heartbeat."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        auth_context: AuthContext,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retries and telemetry.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/v1/context/fetch")
            auth_context: Authentication context
            json: JSON body for POST/PUT/PATCH
            params: Query parameters
            correlation_id: Optional correlation ID (generated if not provided)

        Returns:
            Parsed JSON response

        Raises:
            SynapError: On failure after retries exhausted
        """
        correlation_id = correlation_id or generate_correlation_id(self.instance_id)
        start_time = datetime.now(timezone.utc)

        headers = {
            "Authorization": f"Bearer {auth_context.api_key}",
            "X-Correlation-ID": correlation_id,
            "X-Client-ID": auth_context.client_id,
            "X-Instance-ID": auth_context.instance_id,
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        # Whether the most recent failure leaves it unknown if the server
        # processed the request. Gates retries of non-idempotent methods.
        outcome_unknown = False
        attempts = 0
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1

        self._ensure_heartbeat()

        while attempts < max_attempts:
            attempts += 1

            try:
                response = await self._client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json,
                    params=params,
                )

                response_payload = self._build_response_payload_metadata(response)
                telemetry_status = (
                    "success" if response.status_code < 400 else "error"
                )

                self._emit_telemetry(
                    event_type="http_request",
                    correlation_id=correlation_id,
                    status=telemetry_status,
                    latency_ms=self._elapsed_ms(start_time),
                    attempt=attempts,
                    path=path,
                    method=method,
                    status_code=response.status_code,
                    request_payload=self._build_request_payload_metadata(
                        method=method,
                        json_payload=json,
                        query_params=params,
                    ),
                    response_payload=response_payload,
                )

                # Handle response
                return self._handle_response(response, correlation_id)

            except httpx.TimeoutException as e:
                # The request may have been fully received and processed; we just
                # never saw the response. Outcome unknown.
                outcome_unknown = True
                last_error = NetworkTimeoutError(
                    f"Request timed out: {e}",
                    correlation_id=correlation_id,
                )

            except httpx.ConnectError as e:
                # Connection establishment failed, so the application never saw
                # the request. Safe to retry even for non-idempotent methods.
                outcome_unknown = False
                last_error = NetworkTimeoutError(
                    f"Connection failed: {e}",
                    correlation_id=correlation_id,
                )

            except httpx.RemoteProtocolError as e:
                # Typical cause: the server/CDN closed a kept-alive connection
                # right as we reused it. httpx reports that as "Server
                # disconnected without sending a response", and it means the
                # connection was already being torn down, so the request almost
                # never reaches application code. We keep retrying that case even
                # for POST, because stale pooled connections are common enough
                # against the CDN that dropping the retry would be a real
                # reliability regression (see TestStaleConnectionRetry).
                #
                # The residual risk is a server that received the request, began
                # processing, then died before responding. That is rare, and it is
                # only fully solvable with a server-honoured idempotency key.
                # Any other protocol error is treated as ambiguous.
                disconnected_before_response = "disconnected" in str(e).lower()
                outcome_unknown = not disconnected_before_response
                last_error = NetworkTimeoutError(
                    f"Connection closed by peer: {e}",
                    correlation_id=correlation_id,
                )

            except httpx.HTTPStatusError as e:
                # We received a complete response, so the server told us what it
                # did. The retryable statuses here (429, 503) are rejections
                # issued before the request was processed.
                outcome_unknown = False
                last_error = self._map_status_error(e, correlation_id)

            except SynapError:
                raise  # Don't wrap our own errors

            except Exception as e:
                # Unrecognised failure: assume the worst.
                outcome_unknown = True
                last_error = SynapTransientError(
                    f"Unexpected error: {e}",
                    correlation_id=correlation_id,
                )

            # Check if we should retry
            if not self._should_retry(
                last_error,
                attempts,
                max_attempts,
                method=method,
                outcome_unknown=outcome_unknown,
            ):
                break

            # Calculate backoff delay
            delay = self._calculate_backoff(attempts)
            logger.warning(
                f"Request failed (attempt {attempts}/{max_attempts}), "
                f"retrying in {delay:.2f}s: {last_error}"
            )
            await asyncio.sleep(delay)

        # Emit failure telemetry
        self._emit_telemetry(
            event_type="http_request",
            correlation_id=correlation_id,
            status="error",
            latency_ms=self._elapsed_ms(start_time),
            attempt=attempts,
            path=path,
            method=method,
            error_code=type(last_error).__name__,
            request_payload=self._build_request_payload_metadata(
                method=method,
                json_payload=json,
                query_params=params,
            ),
        )

        raise last_error

    def _handle_response(
        self,
        response: httpx.Response,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Handle HTTP response, raising appropriate errors."""
        if response.status_code == 200:
            return response.json()

        if response.status_code == 204:
            return {}

        # Error responses
        error_body: Any = None
        try:
            error_body = response.json()
            error_message = (
                error_body.get("detail", response.text)
                if isinstance(error_body, dict)
                else response.text
            )
        except Exception:
            error_message = response.text

        if response.status_code == 401:
            raise AuthenticationError(error_message, correlation_id=correlation_id)

        if response.status_code == 402:
            # Credit gate rejection. Server returns a structured detail
            # body ({balance_credits, minimum_required_credits, ...})
            # so callers can render a useful recovery prompt.
            payload = {}
            if isinstance(error_body, dict):
                detail = error_body.get("detail")
                payload = detail if isinstance(detail, dict) else error_body
            balance = payload.get("balance_credits")
            min_req = payload.get("minimum_required_credits")
            raise InsufficientCreditsError(
                error_message if isinstance(error_message, str) else "Insufficient credits",
                balance_credits=float(balance) if balance is not None else None,
                minimum_required_credits=float(min_req) if min_req is not None else None,
                recovery_url=payload.get("recovery_url"),
                redeem_url=payload.get("redeem_url"),
                correlation_id=correlation_id,
            )

        if response.status_code == 400:
            raise InvalidInputError(error_message, correlation_id=correlation_id)

        if response.status_code == 404:
            raise ContextNotFoundError(error_message, correlation_id=correlation_id)

        if response.status_code == 409:
            # Conflict is PERMANENT — never retried (unlike the catch-all
            # transient below). Discriminate on the structured error body
            # {"detail": {"code": "transcript_conflict", ...}} so a transcript
            # immutability conflict maps to TranscriptConflictError; any other
            # 409 (e.g. compact()'s "already in progress") maps to the generic
            # permanent ConflictError.
            detail = error_body.get("detail") if isinstance(error_body, dict) else None
            if isinstance(detail, dict):
                conflict_message = detail.get("message") or str(detail)
                if detail.get("code") == "transcript_conflict":
                    raise TranscriptConflictError(
                        conflict_message, correlation_id=correlation_id
                    )
                raise ConflictError(conflict_message, correlation_id=correlation_id)
            raise ConflictError(
                error_message if isinstance(error_message, str) else "Conflict",
                correlation_id=correlation_id,
            )

        if response.status_code == 422:
            # Validation failure (e.g. an out-of-range precision_level or
            # last_n_conversations). Permanent — surface as InvalidInputError
            # rather than falling through to the retryable catch-all.
            raise InvalidInputError(error_message, correlation_id=correlation_id)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                error_message,
                retry_after_seconds=int(retry_after) if retry_after else None,
                correlation_id=correlation_id,
            )

        if response.status_code >= 500:
            raise ServiceUnavailableError(
                error_message,
                correlation_id=correlation_id,
            )

        # Unknown error
        raise SynapTransientError(
            f"HTTP {response.status_code}: {error_message}",
            correlation_id=correlation_id,
        )

    def _map_status_error(
        self,
        error: httpx.HTTPStatusError,
        correlation_id: str,
    ) -> SynapError:
        """Map httpx status error to SDK exception."""
        return self._handle_response(error.response, correlation_id)

    #: Methods that HTTP defines as idempotent, i.e. sending the same request twice
    #: has the same effect as sending it once. POST and PATCH are deliberately
    #: absent: retrying those can duplicate a side effect.
    _IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

    def _should_retry(
        self,
        error: Exception,
        attempts: int,
        max_attempts: int,
        method: str = "GET",
        outcome_unknown: bool = False,
    ) -> bool:
        """Determine if we should retry the request.

        Args:
            error: The error raised by the last attempt.
            attempts: Attempts made so far.
            max_attempts: Configured attempt ceiling.
            method: HTTP method of the request, used for the idempotency check.
            outcome_unknown: True when the failure leaves it genuinely unknown
                whether the server processed the request (a read timeout, or the
                connection dropping mid-flight). False when the request provably
                never reached the application (connection refused) or when the
                server returned a definitive rejection such as 429 or 503.
        """
        if not self.retry_policy:
            return False

        if attempts >= max_attempts:
            return False

        if not isinstance(error, SynapTransientError):
            return False

        # A retry is only safe when either the method is idempotent, or we know
        # the server never processed the first attempt. Retrying a POST whose
        # outcome is unknown can duplicate the side effect: for
        # POST /api/v1/memories/create that means ingesting the same content
        # twice and billing the customer twice for it.
        #
        # NOTE: the durable fix is a server-honoured idempotency key, which would
        # let writes retry safely. Until the server supports one, declining the
        # retry is the only client-side option that cannot double-charge.
        if outcome_unknown and method.upper() not in self._IDEMPOTENT_METHODS:
            logger.warning(
                "Not retrying %s: the outcome is unknown and the method is not "
                "idempotent, so a retry could duplicate the request (for an ingest "
                "that would mean storing and billing it twice). Original error: %s",
                method.upper(),
                error,
            )
            return False

        # Check if error type is in retryable list
        error_name = type(error).__name__
        return error_name in self.retry_policy.retryable_errors

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with optional jitter."""
        if not self.retry_policy:
            return 0

        delay = self.retry_policy.backoff_base * (2 ** (attempt - 1))
        delay = min(delay, self.retry_policy.backoff_max)

        if self.retry_policy.backoff_jitter:
            # Add up to 25% jitter
            jitter = delay * 0.25 * random.random()
            delay += jitter

        return delay

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed milliseconds since start time."""
        elapsed = datetime.now(timezone.utc) - start_time
        return int(elapsed.total_seconds() * 1000)

    @staticmethod
    def _build_request_payload_metadata(
        *,
        method: str,
        json_payload: Optional[Dict[str, Any]],
        query_params: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build PII-safe request metadata for telemetry.

        Captures only shape-level information (keys present, payload size)
        and never the actual values. User data in request bodies must not
        flow into telemetry events.
        """
        normalized_method = str(method or "").strip().upper()
        metadata: Dict[str, Any] = {}

        if query_params:
            metadata["param_keys"] = sorted(query_params.keys())
        if normalized_method != "GET" and json_payload is not None:
            if isinstance(json_payload, dict):
                metadata["body_keys"] = sorted(json_payload.keys())
            try:
                metadata["body_size"] = len(json.dumps(json_payload, default=str))
            except Exception:
                metadata["body_size"] = -1

        return metadata or None

    @staticmethod
    def _build_response_payload_metadata(
        response: httpx.Response,
    ) -> Optional[Dict[str, Any]]:
        """Build PII-safe response metadata for telemetry.

        Captures only status and body size, never the response contents.
        Response bodies may contain user memory data and must not flow
        into telemetry events.
        """
        if response.status_code == 204:
            payload: Dict[str, Any] = {"status_code": 204, "body_size": 0}
        else:
            try:
                body_size = len(response.content)
            except Exception:
                body_size = -1
            payload = {
                "status_code": response.status_code,
                "body_size": body_size,
            }

        # Server-side processing time stamped by the API (middleware header,
        # server >= 2026-08-08). Reported so the dashboard can show server
        # latency next to the SDK's wall-clock; absent on older servers.
        server_ms = response.headers.get("x-synap-server-ms")
        if server_ms is not None:
            try:
                payload["server_latency_ms"] = int(float(server_ms))
            except (TypeError, ValueError):
                pass

        return payload

    def _emit_telemetry(self, **kwargs) -> None:
        """Emit telemetry event if callback configured."""
        if self.telemetry_callback:
            try:
                self.telemetry_callback(kwargs)
            except Exception as e:
                logger.warning(f"Telemetry emission failed: {e}")

    # Convenience methods
    async def get(
        self,
        path: str,
        auth_context: AuthContext,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request("GET", path, auth_context, params=params, **kwargs)

    async def post(
        self,
        path: str,
        auth_context: AuthContext,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request("POST", path, auth_context, json=json, **kwargs)

    async def put(
        self,
        path: str,
        auth_context: AuthContext,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request("PUT", path, auth_context, json=json, **kwargs)

    async def delete(
        self,
        path: str,
        auth_context: AuthContext,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request("DELETE", path, auth_context, **kwargs)

    async def post_multipart(
        self,
        path: str,
        auth_context: AuthContext,
        data: Dict[str, Any],
        files: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a multipart/form-data POST request (for file uploads)."""
        correlation_id = correlation_id or generate_correlation_id(self.instance_id)
        start_time = datetime.now(timezone.utc)

        headers = {
            "Authorization": f"Bearer {auth_context.api_key}",
            "X-Correlation-ID": correlation_id,
            "X-Client-ID": auth_context.client_id,
            "X-Instance-ID": auth_context.instance_id,
            # No Content-Type — httpx sets it with the multipart boundary automatically
        }

        last_error: Optional[Exception] = None
        outcome_unknown = False
        attempts = 0
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1

        self._ensure_heartbeat()

        while attempts < max_attempts:
            attempts += 1
            try:
                response = await self._client.post(
                    url=path,
                    headers=headers,
                    data=data,
                    files=files or {},
                )

                self._emit_telemetry(
                    event_type="http_request",
                    correlation_id=correlation_id,
                    status="success" if response.status_code < 400 else "error",
                    latency_ms=self._elapsed_ms(start_time),
                    attempt=attempts,
                    path=path,
                    method="POST",
                    status_code=response.status_code,
                )

                return self._handle_response(response, correlation_id)

            except httpx.TimeoutException as e:
                # Upload may have been received in full; outcome unknown.
                outcome_unknown = True
                last_error = NetworkTimeoutError(f"Request timed out: {e}", correlation_id=correlation_id)
            except httpx.ConnectError as e:
                # Never reached the application; safe to retry.
                outcome_unknown = False
                last_error = NetworkTimeoutError(f"Connection failed: {e}", correlation_id=correlation_id)
            except SynapError:
                raise
            except Exception as e:
                outcome_unknown = True
                last_error = SynapTransientError(f"Unexpected error: {e}", correlation_id=correlation_id)

            # This path is always a multipart POST, which is not idempotent.
            if not self._should_retry(
                last_error,
                attempts,
                max_attempts,
                method="POST",
                outcome_unknown=outcome_unknown,
            ):
                break

            delay = self._calculate_backoff(attempts)
            await asyncio.sleep(delay)

        raise last_error


class HttpTransport(BaseTransport):
    """Backward-compatible transport shim for legacy tests/imports."""

    def __init__(self, base_url: str, ssl_context: Optional[Any] = None):
        self.base_url = base_url
        self.ssl_context = ssl_context

    async def send(self, request):
        raise NotImplementedError("Legacy HttpTransport shim does not implement send().")

    async def close(self) -> None:
        return None
