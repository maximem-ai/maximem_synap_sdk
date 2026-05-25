"""gRPC transport for bidirectional streaming (listening)."""

import asyncio
import logging
import random
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum

import grpc
from grpc import aio

from ..models.config import TimeoutConfig
from .base import BaseTransport
from ..models.errors import (
    NetworkTimeoutError,
    ServiceUnavailableError,
    AuthenticationError,
)
from ..auth.models import AuthContext
from ..utils.correlation import generate_correlation_id


logger = logging.getLogger("synap.sdk.transport.grpc")


class StreamState(str, Enum):
    """gRPC stream states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"


class GRPCTransport:
    """gRPC transport for bidirectional streaming.

    Features:
    - Auto-reconnect with exponential backoff
    - Heartbeat/keepalive
    - Callbacks for connection events
    - Graceful shutdown
    """

    DEFAULT_HOST = "synap-cloud-prod.maximem.ai"
    DEFAULT_PORT = 443

    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 10
    BACKOFF_BASE = 1.0
    BACKOFF_MAX = 30.0

    # Heartbeat settings
    HEARTBEAT_INTERVAL = 30.0  # seconds
    HEARTBEAT_TIMEOUT = 10.0   # seconds
    MAX_MISSED_HEARTBEATS = 3

    def __init__(
        self,
        instance_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_tls: bool = True,
        timeouts: Optional[TimeoutConfig] = None,
        on_reconnect: Optional[Callable[[int], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.instance_id = instance_id
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.use_tls = use_tls
        self.timeouts = timeouts or TimeoutConfig()

        # Callbacks
        self.on_reconnect = on_reconnect
        self.on_disconnect = on_disconnect
        self.on_message = on_message
        self.telemetry_callback = telemetry_callback

        # State
        self._state = StreamState.DISCONNECTED
        self._channel: Optional[aio.Channel] = None
        self._stream = None
        self._auth_context: Optional[AuthContext] = None
        self._reconnect_attempts = 0
        self._last_pong_time: float = 0.0
        self._shutdown_event = asyncio.Event()
        # Serializes ALL writes to the bidi stream. grpc.aio forbids concurrent
        # write() on one call: a 2nd outstanding SEND_MESSAGE terminates the RPC
        # (AioRpcError), after which every write raises InvalidStateError
        # ("RPC already finished"). One shared SDK can fan many users/heartbeats
        # at the stream concurrently, so every _stream.write() MUST hold this.
        self._write_lock = asyncio.Lock()

        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self._state == StreamState.CONNECTED

    async def connect(self, auth_context: AuthContext) -> None:
        """Establish gRPC connection.

        Args:
            auth_context: Authentication context for the connection
        """
        self._auth_context = auth_context
        self._shutdown_event.clear()

        await self._establish_connection()

        # Start background tasks
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"gRPC stream connected for instance {self.instance_id}")

    async def _establish_connection(self) -> None:
        """Establish or re-establish the gRPC connection."""
        self._state = StreamState.CONNECTING

        try:
            channel_options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.min_time_between_pings_ms", 30000),
            ]
            target = f"{self.host}:{self.port}"

            if self.use_tls:
                credentials = grpc.ssl_channel_credentials()
                self._channel = aio.secure_channel(target, credentials, options=channel_options)
            else:
                self._channel = aio.insecure_channel(target, options=channel_options)

            # Wait for channel to be ready
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self.timeouts.connect,
            )

            # Get stub and open bidirectional stream
            self._stub = self._create_stub(self._channel)
            self._stream = await self._open_stream()

            self._state = StreamState.CONNECTED
            self._reconnect_attempts = 0
            self._last_pong_time = asyncio.get_event_loop().time()

            self._emit_telemetry("listen_start", status="success")

        except asyncio.TimeoutError:
            self._state = StreamState.DISCONNECTED
            raise NetworkTimeoutError(
                f"gRPC connection timeout after {self.timeouts.connect}s"
            )
        except grpc.RpcError as e:
            self._state = StreamState.DISCONNECTED
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise AuthenticationError(f"gRPC authentication failed: {e.details()}")
            raise ServiceUnavailableError(f"gRPC connection failed: {e.details()}")

    def _create_stub(self, channel):
        """Create gRPC stub from generated proto code."""
        from .proto import synap_service_pb2_grpc
        return synap_service_pb2_grpc.SynapServiceStub(channel)

    async def _open_stream(self):
        """Open bidirectional Listen stream with auth metadata."""
        metadata = [
            ("authorization", f"Bearer {self._auth_context.api_key}"),
            ("x-client-id", self._auth_context.client_id),
            ("x-instance-id", self._auth_context.instance_id),
        ]
        if self._auth_context.correlation_id:
            metadata.append(("x-correlation-id", self._auth_context.correlation_id))
        return self._stub.Listen(metadata=metadata)

    async def _listen_loop(self) -> None:
        """Main loop for receiving messages from the stream."""
        while not self._shutdown_event.is_set():
            try:
                if not self._stream:
                    await asyncio.sleep(0.1)
                    continue

                message = await self._stream.read()

                if message is aio.EOF or message is None:
                    logger.warning("gRPC stream closed by server")
                    await self._handle_disconnect("server_close")
                    continue

                # Dispatch based on payload type
                payload_type = message.WhichOneof("payload")

                if payload_type == "heartbeat_pong":
                    self._last_pong_time = asyncio.get_event_loop().time()
                    continue

                if payload_type == "signal":
                    self._handle_signal(message.signal)
                    continue

                if payload_type == "context_bundle":
                    bundle_dict = self._proto_to_bundle_dict(message.context_bundle)
                    if self.on_message:
                        try:
                            if asyncio.iscoroutinefunction(self.on_message):
                                await self.on_message(bundle_dict)
                            else:
                                # Call synchronously from the event loop — do NOT
                                # use run_in_executor, as the callback may need
                                # event loop access (e.g. asyncio.create_task).
                                self.on_message(bundle_dict)
                        except Exception as e:
                            logger.error(f"Message handler error: {e}")

            except grpc.RpcError as e:
                logger.warning(f"gRPC stream error: {e}")
                await self._handle_disconnect(f"grpc_error:{e.code()}")

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Unexpected error in listen loop: {e}")
                await self._handle_disconnect(f"error:{e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and detect stale connections via pong timestamps."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)

                if self._state != StreamState.CONNECTED:
                    continue

                # Send heartbeat ping
                try:
                    await self._send_heartbeat()
                except Exception as e:
                    logger.warning(f"Heartbeat send failed: {e}")

                # Check if we've received a pong recently.
                # If the last pong is older than INTERVAL + TIMEOUT, the
                # connection is stale. This avoids the race condition of
                # resetting a counter in both sender and receiver.
                now = asyncio.get_event_loop().time()
                if self._last_pong_time > 0:
                    silence = now - self._last_pong_time
                    if silence > self.HEARTBEAT_INTERVAL + self.HEARTBEAT_TIMEOUT:
                        logger.warning(
                            f"No pong received for {silence:.1f}s — disconnecting"
                        )
                        await self._handle_disconnect("heartbeat_timeout")

            except asyncio.CancelledError:
                break

    async def _send_heartbeat(self) -> None:
        """Send heartbeat ping over the active stream."""
        from .proto import synap_service_pb2

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        ping = synap_service_pb2.StreamEvent(
            heartbeat_ping=synap_service_pb2.HeartbeatPing(timestamp_ms=now_ms)
        )
        async with self._write_lock:
            await self._stream.write(ping)

    async def _handle_disconnect(self, reason: str) -> None:
        """Handle disconnection and attempt reconnect."""
        if self._state in (StreamState.CLOSED, StreamState.RECONNECTING):
            return

        self._state = StreamState.RECONNECTING
        self._emit_telemetry("listen_disconnect", reason=reason)

        # Cleanup current connection
        if self._channel:
            await self._channel.close()
            self._channel = None
        self._stream = None

        # Attempt reconnection
        while (
            self._reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS
            and not self._shutdown_event.is_set()
        ):
            self._reconnect_attempts += 1

            # Calculate backoff
            delay = min(
                self.BACKOFF_BASE * (2 ** (self._reconnect_attempts - 1)),
                self.BACKOFF_MAX,
            )
            # Full jitter (decorrelates reconnection across clients)
            delay = random.uniform(0, delay)

            logger.info(
                f"Reconnecting in {delay:.1f}s "
                f"(attempt {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})"
            )
            await asyncio.sleep(delay)

            try:
                await self._establish_connection()
                self._emit_telemetry("listen_reconnect", attempt=self._reconnect_attempts)

                if self.on_reconnect:
                    self.on_reconnect(self._reconnect_attempts)

                return  # Success

            except Exception as e:
                logger.warning(f"Reconnection failed: {e}")

        # Max retries exceeded
        self._state = StreamState.DISCONNECTED
        logger.error("Max reconnection attempts exceeded")

        if self.on_disconnect:
            self.on_disconnect(reason)

        self._emit_telemetry("listen_disconnect", reason="max_retries_exceeded")

    async def send(self, message: Dict[str, Any]) -> None:
        """Send a conversation message on the stream.

        Args:
            message: Dict with event_type, content, role, conversation_id, etc.
        """
        if self._state != StreamState.CONNECTED:
            raise ServiceUnavailableError("gRPC stream not connected")

        from .proto import synap_service_pb2

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        conv_event = synap_service_pb2.ConversationEvent(
            event_type=message.get("event_type", "user_message"),
            conversation_id=message.get("conversation_id", ""),
            user_id=message.get("user_id", ""),
            role=message.get("role", "user"),
            content=message.get("content", ""),
            customer_id=message.get("customer_id", ""),
            session_id=message.get("session_id", ""),
            metadata=message.get("metadata") or {},
            timestamp_ms=message.get("timestamp_ms", now_ms),
            tool_name=message.get("tool_name", ""),
            tool_args_json=message.get("tool_args_json", ""),
            search_queries=message.get("search_queries") or [],
            context_types=message.get("context_types") or [],
        )
        event = synap_service_pb2.StreamEvent(conversation_event=conv_event)
        async with self._write_lock:
            await self._stream.write(event)

    async def send_context_used(
        self,
        *,
        bundle_id: str,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        served_item_ids: Optional[List[str]] = None,
        scope: str = "",
        source_bundle_ids: Optional[List[str]] = None,
    ) -> None:
        """Emit a ContextUsedEvent over the Listen stream.

        Fire-and-forget telemetry: the SDK calls this after fetch() is served
        from the anticipation cache so the server can attribute outcomes back
        to the originating prefetch and update per-pattern hit rates.

        Privacy: this method MUST NOT be called with raw user prompt content.
        The proto only carries ids and scope.
        """
        if not self._stream:
            raise ServiceUnavailableError("gRPC stream not connected")

        from .proto import synap_service_pb2

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        used = synap_service_pb2.ContextUsedEvent(
            bundle_id=bundle_id,
            conversation_id=conversation_id or "",
            user_id=user_id or "",
            customer_id=customer_id or "",
            served_item_ids=list(served_item_ids or []),
            timestamp_ms=now_ms,
            scope=scope or "",
            source_bundle_ids=list(source_bundle_ids or []),
        )
        event = synap_service_pb2.StreamEvent(context_used=used)
        async with self._write_lock:
            await self._stream.write(event)

    def _handle_signal(self, signal) -> None:
        """Handle a StreamSignal from the server.

        Args:
            signal: StreamSignal proto message
        """
        signal_type = signal.signal_type
        reason = signal.reason

        if signal_type == "throttle":
            logger.warning(f"Server throttle signal: {reason}")
        elif signal_type == "closing":
            logger.info(f"Server closing stream: {reason}")
            # Schedule graceful disconnect — don't block the listen loop
            asyncio.ensure_future(self._handle_disconnect("server_closing"))
        elif signal_type == "error":
            logger.error(f"Server error signal: {reason}")
            asyncio.ensure_future(self._handle_disconnect(f"server_error:{reason}"))
        else:
            logger.warning(f"Unknown signal type '{signal_type}': {reason}")

    def _proto_to_bundle_dict(self, proto) -> Dict[str, Any]:
        """Convert a ContextBundleProto to a plain dict matching ContextBundle.to_dict().

        Args:
            proto: ContextBundleProto message

        Returns:
            Dict suitable for SDK consumption / on_message callback
        """
        items_by_type = {}
        for ctx_type, item_list in proto.items_by_type.items():
            items_by_type[ctx_type] = [
                {
                    "item_id": item.item_id,
                    "content": item.content,
                    "context_type": item.context_type,
                    "source": item.source,
                    "similarity_score": item.similarity_score,
                    "relevance_score": item.relevance_score,
                    "confidence": item.confidence,
                    "scope": item.scope,
                    "entity_id": item.entity_id,
                    "created_at": item.created_at,
                    "event_date": item.event_date or None,
                    "valid_until": item.valid_until or None,
                    "temporal_category": item.temporal_category or None,
                    "temporal_confidence": item.temporal_confidence,
                }
                for item in item_list.items
            ]

        # Deserialize conversation_context if present
        conv_ctx = None
        if proto.HasField("conversation_context"):
            import json as _json
            cc = proto.conversation_context
            conv_ctx = {
                "summary": cc.summary or None,
                "current_state": _json.loads(cc.current_state_json) if cc.current_state_json else {},
                "key_extractions": _json.loads(cc.key_extractions_json) if cc.key_extractions_json else {},
                "recent_turns": [
                    {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                    for t in cc.recent_turns
                ],
                "compaction_id": cc.compaction_id or None,
                "compacted_at": cc.compacted_at or None,
                "conversation_id": cc.conversation_id or None,
            }

        return {
            "bundle_id": proto.bundle_id,
            "decision_id": proto.decision_id,
            "items_by_type": items_by_type,
            "total_tokens": proto.total_tokens,
            "token_budget": proto.token_budget,
            "budget_exceeded": proto.budget_exceeded,
            "retrieval_mode": proto.retrieval_mode,
            "sources_queried": list(proto.sources_queried),
            "degradation_level": proto.degradation_level,
            "warnings": list(proto.warnings),
            "created_at": proto.created_at,
            "retrieval_time_ms": proto.retrieval_time_ms,
            "cache_hit": proto.cache_hit,
            "search_queries": list(proto.search_queries) if proto.search_queries else [],
            "search_keywords": list(proto.search_keywords) if proto.search_keywords else [],
            "_anticipation_user_id": proto.anticipation_user_id or None,
            "_anticipation_customer_id": proto.anticipation_customer_id or None,
            "_anticipation_conversation_id": proto.anticipation_conversation_id or None,
            "_bundle_type": proto.bundle_type or "anticipation",
            "conversation_context": conv_ctx,
            # Section 16 — bundle composition extensions. Defaults preserve
            # backwards compatibility when the server is older than the SDK.
            "_bundle_confidence": float(getattr(proto, "bundle_confidence", 0.0) or 0.0),
            "_origin_pattern_id": getattr(proto, "origin_pattern_id", "") or "",
            "_ttl_hint_seconds": int(getattr(proto, "ttl_hint_seconds", 0) or 0),
        }

    async def close(self) -> None:
        """Gracefully close the connection."""
        logger.info("Closing gRPC stream")
        self._state = StreamState.CLOSED
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._listen_task, self._heartbeat_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close channel
        if self._channel:
            await self._channel.close()
            self._channel = None

        self._stream = None
        logger.info("gRPC stream closed")

    def _emit_telemetry(self, event_type: str, **kwargs) -> None:
        """Emit telemetry event."""
        if self.telemetry_callback:
            try:
                self.telemetry_callback({
                    "event_type": event_type,
                    "instance_id": self.instance_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **kwargs,
                })
            except Exception as e:
                logger.warning(f"Telemetry emission failed: {e}")


class GrpcTransport(BaseTransport):
    """Backward-compatible transport shim for legacy tests/imports."""

    def __init__(
        self,
        host: str,
        port: int,
        ssl_context: Optional[Any] = None,
    ):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context

    async def send(self, request):
        raise NotImplementedError("Legacy GrpcTransport shim does not implement send().")

    async def close(self) -> None:
        return None
