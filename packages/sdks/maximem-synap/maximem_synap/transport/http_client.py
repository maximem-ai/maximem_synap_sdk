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

    def __init__(
        self,
        instance_id: str,
        base_url: Optional[str] = None,
        timeouts: Optional[TimeoutConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
        telemetry_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.instance_id = instance_id
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeouts = timeouts or TimeoutConfig()
        self.retry_policy = retry_policy
        self.telemetry_callback = telemetry_callback

        # Create httpx client with timeout config
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=self.timeouts.connect,
                read=self.timeouts.read,
                write=self.timeouts.write,
                pool=self.timeouts.connect,
            ),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
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
        attempts = 0
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1

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
                last_error = NetworkTimeoutError(
                    f"Request timed out: {e}",
                    correlation_id=correlation_id,
                )

            except httpx.ConnectError as e:
                last_error = NetworkTimeoutError(
                    f"Connection failed: {e}",
                    correlation_id=correlation_id,
                )

            except httpx.HTTPStatusError as e:
                last_error = self._map_status_error(e, correlation_id)

            except SynapError:
                raise  # Don't wrap our own errors

            except Exception as e:
                last_error = SynapTransientError(
                    f"Unexpected error: {e}",
                    correlation_id=correlation_id,
                )

            # Check if we should retry
            if not self._should_retry(last_error, attempts, max_attempts):
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

    def _should_retry(
        self,
        error: Exception,
        attempts: int,
        max_attempts: int,
    ) -> bool:
        """Determine if we should retry the request."""
        if not self.retry_policy:
            return False

        if attempts >= max_attempts:
            return False

        if not isinstance(error, SynapTransientError):
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
            return {"status_code": 204, "body_size": 0}

        try:
            body_size = len(response.content)
        except Exception:
            body_size = -1

        return {
            "status_code": response.status_code,
            "body_size": body_size,
        }

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
        attempts = 0
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1

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
                last_error = NetworkTimeoutError(f"Request timed out: {e}", correlation_id=correlation_id)
            except httpx.ConnectError as e:
                last_error = NetworkTimeoutError(f"Connection failed: {e}", correlation_id=correlation_id)
            except SynapError:
                raise
            except Exception as e:
                last_error = SynapTransientError(f"Unexpected error: {e}", correlation_id=correlation_id)

            if not self._should_retry(last_error, attempts, max_attempts):
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
