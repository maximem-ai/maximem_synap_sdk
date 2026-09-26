"""gRPC transport for bidirectional streaming (listening)."""

import asyncio
import logging
import random
import uuid
from collections import deque, OrderedDict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

import grpc
from grpc import aio

from ..models.config import TimeoutConfig
from .base import BaseTransport
from ..models.errors import (
    NetworkTimeoutError,
    ServiceUnavailableError,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    SynapError,
)
from ..auth.models import AuthContext
from ..utils.correlation import generate_correlation_id
from .outbox_journal import OutboxJournal


logger = logging.getLogger("synap.sdk.transport.grpc")


# The credit gate refuses a call with RESOURCE_EXHAUSTED, these details, and
# the balance and reason in trailing metadata. See
# synap/cloud/application/credits/grpc_gate.py.
CREDIT_ABORT_DETAILS = "insufficient_credits"

REASON_OVERAGES_DISABLED = "overages_disabled"
REASON_TRIAL_LIMIT_REACHED = "trial_limit_reached"
REASON_SUBSCRIPTION_INACTIVE = "subscription_inactive"

# Reasons that end the stream. A reconnect re-runs the same gate with the same
# balance, so retrying only delays the answer the caller needs.
CREDIT_STOP_REASONS = frozenset(
    {REASON_OVERAGES_DISABLED, REASON_TRIAL_LIMIT_REACHED, REASON_SUBSCRIPTION_INACTIVE}
)


def _trailing_metadata(error: Any) -> Dict[str, str]:
    """Trailing metadata as a lowercased dict, for sync and aio errors alike."""
    getter = getattr(error, "trailing_metadata", None)
    metadata = getter() if callable(getter) else None
    out: Dict[str, str] = {}
    if not metadata:
        return out
    for entry in metadata:
        key = getattr(entry, "key", None)
        value = getattr(entry, "value", None)
        if key is None:
            try:
                key, value = entry
            except (TypeError, ValueError):
                continue
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="replace")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        out[str(key).lower()] = value
    return out


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def credit_error_from_rpc(
    error: Any, correlation_id: Optional[str] = None
) -> Optional[SynapError]:
    """Translate a credit refusal into the same error HTTP would have raised.

    The gate aborts with ``RESOURCE_EXHAUSTED``, which on its own is
    indistinguishable from server overload, so this keys on ``credit-reason``
    in the trailing metadata:

    - ``overages_disabled`` -> :class:`InsufficientCreditsError` (permanent):
      a paid plan at zero that has not allowed usage past zero.
    - ``trial_limit_reached`` / ``subscription_inactive`` ->
      :class:`RateLimitError`: the Trial cap, or a lapsed subscription.

    Returns ``None`` for anything else, including a ``RESOURCE_EXHAUSTED``
    that is not a credit refusal and a credit reason this SDK does not know:
    the caller then keeps its existing behaviour of treating the abort as
    transient.
    """
    code = getattr(error, "code", None)
    if not callable(code) or code() is not grpc.StatusCode.RESOURCE_EXHAUSTED:
        return None

    metadata = _trailing_metadata(error)
    reason = metadata.get("credit-reason")
    if reason not in CREDIT_STOP_REASONS:
        return None

    details = getattr(error, "details", None)
    details = details() if callable(details) else None
    if not isinstance(details, str) or not details:
        details = CREDIT_ABORT_DETAILS

    request_id = metadata.get("request-id") or correlation_id
    manage_url = metadata.get("credit-manage-url")

    if reason == REASON_OVERAGES_DISABLED:
        return InsufficientCreditsError(
            details,
            balance_credits=_float_or_none(metadata.get("credit-balance")),
            minimum_required_credits=_float_or_none(
                metadata.get("credit-minimum-required")
            ),
            recovery_url=metadata.get("credit-recovery-url"),
            redeem_url=metadata.get("credit-redeem-url"),
            correlation_id=request_id,
            reason=reason,
            manage_url=manage_url,
        )

    return RateLimitError(
        details,
        correlation_id=request_id,
        reason=reason,
        upgrade_url=metadata.get("credit-upgrade-url"),
        manage_url=manage_url,
    )


def is_credit_stop(error: Any) -> bool:
    """True for an error this module raised from a credit refusal."""
    return isinstance(
        error, (InsufficientCreditsError, RateLimitError)
    ) and getattr(error, "reason", None) in CREDIT_STOP_REASONS


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

    # Outbound retry queue (D3 in the plan): if send() is called while the
    # stream is disconnected/reconnecting, the ConversationEvent payload is
    # buffered and replayed on reconnect. Bounded so a long outage cannot
    # exhaust SDK memory; the oldest entry is dropped (with a WARN) when
    # either limit is exceeded.
    SEND_QUEUE_MAX_DEPTH = 100
    SEND_QUEUE_MAX_AGE = timedelta(minutes=5)

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
        storage_path: Optional[str] = None,
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
        # Why the stream stopped, when it stopped for a reason the caller can
        # act on (today: a credit refusal). Read via `last_error`.
        self._last_error: Optional[SynapError] = None
        # Serializes ALL writes to the bidi stream. grpc.aio forbids concurrent
        # write() on one call: a 2nd outstanding SEND_MESSAGE terminates the RPC
        # (AioRpcError), after which every write raises InvalidStateError
        # ("RPC already finished"). One shared SDK can fan many users/heartbeats
        # at the stream concurrently, so every _stream.write() MUST hold this.
        self._write_lock = asyncio.Lock()

        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Outbound retry queue for ConversationEvents. Each entry is
        # (timestamp_queued, payload_dict).
        self._send_queue: Deque[Tuple[datetime, Dict[str, Any]]] = deque()
        self._send_queue_lock = asyncio.Lock()
        # Written to a live stream, not yet acknowledged by the server.
        #
        # The send queue above covers "the stream was down when we tried". This
        # covers the other half, which is the one that actually loses sessions:
        # the write succeeded, the stream broke before the server processed it,
        # and nobody knows. Held until an EventAck names the id, replayed on
        # reconnect. The server dedupes on that id, so a replay the server DID
        # process is dropped there rather than doubling the turn.
        self._unacked: "OrderedDict[str, Tuple[datetime, Dict[str, Any]]]" = OrderedDict()

        # Disk backing for the two buffers above. Both survive a dropped
        # connection and neither survives the process, so `kill -9` or a pod
        # eviction between recording a turn and the server acknowledging it
        # loses that turn with nothing to show for it. This is what the next
        # start reads back. See `outbox_journal.py` for why it is a log and
        # not a snapshot.
        from ..config_utils import get_default_storage_path
        _root = Path(storage_path) if storage_path else get_default_storage_path()
        self._journal = OutboxJournal(_root, instance_id)

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self._state == StreamState.CONNECTED

    @property
    def last_error(self) -> Optional[SynapError]:
        """The error that ended the stream, if one did.

        Set when the server refused the stream for credits, so an
        ``on_disconnect`` handler can read the balance, the reason and the
        URL to fix it. ``None`` for an ordinary disconnect.
        """
        return self._last_error

    async def connect(self, auth_context: AuthContext) -> None:
        """Establish gRPC connection.

        Args:
            auth_context: Authentication context for the connection
        """
        self._auth_context = auth_context
        self._shutdown_event.clear()
        self._last_error = None

        # Before the connection, so anything a previous process left behind is
        # already in the queue when the stream comes up and goes out ahead of
        # this process's first event. That is the order it happened in.
        self._restore_from_journal()

        await self._establish_connection()

        # Start background tasks
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # The reconnect path drains on its own; a FIRST connect did not, so a
        # restored buffer sat there until the stream happened to break.
        if self._send_queue:
            try:
                await self._drain_send_queue()
            except Exception as e:  # noqa: BLE001 — never fail listen()
                logger.warning("Restored-event drain failed: %s", e)

        logger.info(f"gRPC stream connected for instance {self.instance_id}")

    def _restore_from_journal(self) -> None:
        """Pull a previous process's unsent events back into the send queue.

        Everything lands in the send queue, including what was unacknowledged
        last time. The stream those events were written to is gone, so "the
        server may already have it" is not a state we can resume; it has to be
        written again. Writing it again is safe because each event carries the
        id it was first sent with and the server drops a second sighting of
        that id before dispatch, so an event it DID process is deduplicated
        there rather than doubling the turn.
        """
        try:
            queued, unacked = self._journal.load()
        except Exception as e:  # noqa: BLE001 — never fail a connect
            logger.warning("Outbox journal restore failed: %s", e)
            return
        if not queued and not unacked:
            return
        now = datetime.now(timezone.utc)
        # Unacknowledged first: an event written to the old stream is older
        # than anything that piled up after it died.
        for payload in unacked + queued:
            self._send_queue.append((now, payload))
        logger.info(
            "Restored %d event(s) from the previous process (%d unacknowledged, "
            "%d queued); they go out on this stream",
            len(queued) + len(unacked), len(unacked), len(queued),
        )
        self._journal.compact([p for _ts, p in self._send_queue], [])

    def _compact_journal(self) -> None:
        self._journal.compact(
            [p for _ts, p in self._send_queue],
            [p for _ts, p in self._unacked.values()],
        )

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

            # ⚠ The stream is NOT usable yet, and this is where a turn used to
            # disappear.
            #
            # `stub.Listen(...)` hands back a call object immediately. It does
            # not wait for the server to accept the RPC: grpc.aio starts the
            # call lazily. Setting CONNECTED here, as this used to, made
            # `send()` believe it had a live stream, so every event went
            # straight into grpc's outgoing buffer instead of the retry queue
            # that exists precisely for "there is no stream yet". Close the SDK
            # before that buffer drains and the events are gone, with every
            # call having returned None.
            #
            # Measured against deployed staging: five events sent immediately
            # after `listen()` returned, server-side `conversation_events=0`
            # and the stream cancelled two seconds after opening. The same five
            # with a three second pause: 5 of 5, twice. How many survived
            # depended only on how long the caller happened to take before
            # exiting, which is why it looked flaky rather than broken.
            #
            # `wait_for_connection()` resolves once the server has accepted the
            # call. Until it does the state stays CONNECTING, so `send()` takes
            # the queue branch and the reconnect drain replays it in order.
            try:
                await asyncio.wait_for(
                    self._stream.wait_for_connection(),
                    timeout=self.timeouts.connect,
                )
            except AttributeError:
                # A stub double in tests will not have it. The real grpc.aio
                # call always does; do not let a test shim change production
                # behaviour by silently skipping the wait.
                pass

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
            credit_error = credit_error_from_rpc(
                e, self._auth_context.correlation_id if self._auth_context else None
            )
            if credit_error is not None:
                raise credit_error
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

                if payload_type == "event_ack":
                    self._handle_event_ack(message.event_ack)
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
                credit_error = credit_error_from_rpc(
                    e, self._auth_context.correlation_id if self._auth_context else None
                )
                if credit_error is not None:
                    await self._handle_credit_stop(credit_error)
                    break
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

    async def _handle_credit_stop(self, error: SynapError) -> None:
        """End the stream because the server refused it for credits.

        Reconnecting cannot clear a credit refusal, so the stream closes and
        the typed error is kept on :attr:`last_error` for the application to
        read. ``on_disconnect`` receives ``credit_stop:<reason>``.
        """
        self._last_error = error
        reason = f"credit_stop:{getattr(error, 'reason', None) or 'unknown'}"

        if self._channel:
            await self._channel.close()
            self._channel = None
        self._stream = None
        self._state = StreamState.DISCONNECTED
        self._shutdown_event.set()

        logger.error("gRPC stream stopped, %s: %s", reason, error)
        self._emit_telemetry("listen_disconnect", reason=reason)

        if self.on_disconnect:
            self.on_disconnect(reason)

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

                # Replay any buffered ConversationEvents before invoking
                # the user-facing reconnect callback, so app-level code
                # never sees a window where the SDK is "connected" but
                # queued turns are still pending.
                try:
                    # Unacknowledged first, then queued: that is the order the
                    # events happened in. An unacked event was written to the
                    # previous stream, so it is older than anything that piled
                    # up while there was no stream at all.
                    await self._replay_unacked()
                    await self._drain_send_queue()
                except Exception as e:
                    logger.warning("Send-queue drain failed: %s", e)

                if self.on_reconnect:
                    self.on_reconnect(self._reconnect_attempts)

                return  # Success

            except Exception as e:
                if is_credit_stop(e):
                    await self._handle_credit_stop(e)
                    return
                logger.warning(f"Reconnection failed: {e}")

        # Max retries exceeded
        self._state = StreamState.DISCONNECTED
        logger.error("Max reconnection attempts exceeded")

        if self.on_disconnect:
            self.on_disconnect(reason)

        self._emit_telemetry("listen_disconnect", reason="max_retries_exceeded")

    async def send(self, message: Dict[str, Any]) -> None:
        """Send a conversation message on the stream.

        When the stream is not currently connected, the message is queued
        for replay on reconnect rather than dropped. The queue is bounded
        in both depth and age (see ``SEND_QUEUE_MAX_DEPTH`` /
        ``SEND_QUEUE_MAX_AGE``) — overflow drops the oldest entry with a
        WARN log.

        Args:
            message: Dict with event_type, content, role, conversation_id, etc.
        """
        # Minted here rather than at the call site so every path gets one, and
        # only once: a replay has to carry the SAME id or the server's dedupe
        # has nothing to match and the retry doubles the turn.
        if not message.get("event_id"):
            message["event_id"] = str(uuid.uuid4())
        if not message.get("sent_at_ms"):
            message["sent_at_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)

        if self._state != StreamState.CONNECTED:
            await self._enqueue_for_retry(message)
            return

        try:
            await self._write_conversation_event(message)
            self._track_unacked(message)
        except Exception as e:
            # Stream broke mid-write; queue the message for the reconnect
            # loop to replay, and let the listen loop handle the error.
            logger.warning("send() failed, queuing for retry: %s", e)
            await self._enqueue_for_retry(message)

    async def _write_conversation_event(self, message: Dict[str, Any]) -> None:
        """Serialize a payload into ConversationEvent and write to the stream."""
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
            tool_result_json=message.get("tool_result_json", ""),
            tool_call_id=message.get("tool_call_id", ""),
            event_id=message.get("event_id", ""),
            sent_at_ms=int(message.get("sent_at_ms", 0) or 0),
            search_queries=message.get("search_queries") or [],
            context_types=message.get("context_types") or [],
        )
        event = synap_service_pb2.StreamEvent(conversation_event=conv_event)
        async with self._write_lock:
            await self._stream.write(event)

    async def send_session_control(
        self,
        *,
        action: str,
        session_id: str = "",
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
    ) -> bool:
        """Open or close a session on the stream. Returns whether it was sent.

        Not queued for replay, unlike a conversation event. A session_start is
        a statement about the stream that is carrying it, and replaying one
        onto a different stream after a reconnect tells the server a session
        began that it has already seen begin. The caller keeps the bookkeeping
        instead: a False here means "not opened", so the next event opens it.

        The server answers a refusal with an error signal rather than closing
        the stream, so a rejected start is visible to the caller through the
        usual disconnect/error path.
        """
        if self._state != StreamState.CONNECTED:
            return False
        from .proto import synap_service_pb2

        control = synap_service_pb2.SessionControl(
            action=action,
            session_id=session_id,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        )
        event = synap_service_pb2.StreamEvent(session_control=control)
        try:
            async with self._write_lock:
                await self._stream.write(event)
            return True
        except Exception as e:  # noqa: BLE001 — never break the caller's turn
            logger.warning("session_control(%s) failed: %s", action, e)
            return False

    def _track_unacked(self, message: Dict[str, Any]) -> None:
        """Hold a written event until the server says it has it.

        Bounded the same way the send queue is, and for the same reason: an
        outage must not be able to grow this without limit. The oldest goes
        first, with a WARN, because losing the oldest event of a long outage is
        the least bad of the available losses and it must not be silent.
        """
        event_id = message.get("event_id")
        if not event_id:
            return
        cutoff = datetime.now(timezone.utc) - self.SEND_QUEUE_MAX_AGE
        while self._unacked:
            oldest_id, (queued_at, payload) = next(iter(self._unacked.items()))
            if queued_at >= cutoff and len(self._unacked) < self.SEND_QUEUE_MAX_DEPTH:
                break
            self._unacked.pop(oldest_id, None)
            self._journal.record_dropped(oldest_id)
            logger.warning(
                "Unacknowledged event dropped (queued_at=%s, event_type=%s, "
                "conversation_id=%s): the server never confirmed it",
                queued_at.isoformat(),
                payload.get("event_type"),
                payload.get("conversation_id"),
            )
        self._unacked[event_id] = (datetime.now(timezone.utc), dict(message))
        self._journal.record_unacked(dict(message))

    def _handle_event_ack(self, ack) -> None:
        """The server has taken responsibility for these events."""
        for event_id in ack.event_ids:
            self._unacked.pop(event_id, None)
            self._journal.record_acked(event_id)
        logger.debug(
            "Acked %d event(s); %d still unacknowledged",
            len(ack.event_ids), len(self._unacked),
        )

    async def _replay_unacked(self) -> int:
        """Re-send everything the server never confirmed. Called on reconnect.

        Safe to repeat: every event carries the id it was first sent with, and
        the server drops a second sighting of that id before dispatch. An event
        it DID process is deduplicated there rather than doubling the turn
        here.
        """
        if not self._unacked:
            return 0
        pending = list(self._unacked.values())
        replayed = 0
        for _queued_at, payload in pending:
            try:
                await self._write_conversation_event(payload)
                replayed += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("Replay write failed (kept for next reconnect): %s", e)
                break
        if replayed:
            logger.info("Replayed %d unacknowledged event(s) after reconnect", replayed)
        return replayed

    # How long close() will wait for the buffer to go out. Short: a shutdown
    # path that blocks is worse than a lost event, and the caller is usually a
    # process that is already on its way out.
    CLOSE_FLUSH_TIMEOUT = 2.0

    async def _flush_before_close(self) -> None:
        """Best effort: write what is buffered before the stream goes away."""
        if self._state != StreamState.CONNECTED:
            if self._send_queue or self._unacked:
                logger.warning(
                    "Closing with %d queued and %d unacknowledged event(s) and "
                    "no stream to send them on; they are lost",
                    len(self._send_queue), len(self._unacked),
                )
            return
        if not self._send_queue:
            return
        try:
            await asyncio.wait_for(
                self._drain_send_queue(), timeout=self.CLOSE_FLUSH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Close flush timed out after %ss with %d event(s) still queued",
                self.CLOSE_FLUSH_TIMEOUT, len(self._send_queue),
            )
        except Exception as e:  # noqa: BLE001 — never fail a teardown
            logger.warning("Close flush failed: %s", e)

    async def _enqueue_for_retry(self, message: Dict[str, Any]) -> None:
        """Buffer a payload for replay after reconnect.

        Drops the oldest entry (with WARN log) when depth or age limits
        would otherwise be exceeded.
        """
        async with self._send_queue_lock:
            self._prune_send_queue_locked()
            if len(self._send_queue) >= self.SEND_QUEUE_MAX_DEPTH:
                evicted_ts, evicted = self._send_queue.popleft()
                self._journal.record_dropped(evicted.get("event_id"))
                logger.warning(
                    "Send queue full (max=%d); dropping oldest event "
                    "(queued_at=%s, event_type=%s, conversation_id=%s)",
                    self.SEND_QUEUE_MAX_DEPTH,
                    evicted_ts.isoformat(),
                    evicted.get("event_type"),
                    evicted.get("conversation_id"),
                )
            self._send_queue.append((datetime.now(timezone.utc), dict(message)))
            self._journal.record_queued(dict(message))

    def _prune_send_queue_locked(self) -> None:
        """Drop send-queue entries older than the max-age window."""
        cutoff = datetime.now(timezone.utc) - self.SEND_QUEUE_MAX_AGE
        while self._send_queue and self._send_queue[0][0] < cutoff:
            ts, msg = self._send_queue.popleft()
            self._journal.record_dropped(msg.get("event_id"))
            logger.warning(
                "Send queue entry expired (queued_at=%s, age>%s); dropping "
                "(event_type=%s, conversation_id=%s)",
                ts.isoformat(),
                self.SEND_QUEUE_MAX_AGE,
                msg.get("event_type"),
                msg.get("conversation_id"),
            )

    async def _drain_send_queue(self) -> int:
        """Replay all queued payloads. Called after a successful reconnect.

        Returns the number of payloads successfully replayed. Entries that
        fail to write are re-queued at the front so the next reconnect
        can retry them.
        """
        if not self._send_queue:
            return 0
        drained = 0
        async with self._send_queue_lock:
            self._prune_send_queue_locked()
            # Snapshot under the lock, then write each event WITHOUT
            # holding the lock so a slow .write() doesn't block other
            # callers from enqueuing new events.
            snapshot = list(self._send_queue)
            self._send_queue.clear()

        for idx, (ts, payload) in enumerate(snapshot):
            if self._state != StreamState.CONNECTED:
                # Lost connection mid-drain; re-queue this item and the
                # rest of the snapshot at the front of the queue,
                # preserving original FIFO order.
                async with self._send_queue_lock:
                    for back_ts, back_payload in reversed(snapshot[idx:]):
                        self._send_queue.appendleft((back_ts, back_payload))
                break
            try:
                await self._write_conversation_event(payload)
                # A drained event has been written and not yet acknowledged,
                # which is exactly what `_unacked` is for. This was missing:
                # the drain wrote and then forgot, so a stream that broke again
                # before the acks arrived lost everything it had just replayed,
                # which is the failure the queue exists to prevent. `send()`
                # has always tracked on its own connected path.
                self._track_unacked(payload)
                drained += 1
            except Exception as e:
                logger.warning("Drain write failed (will re-queue): %s", e)
                async with self._send_queue_lock:
                    for back_ts, back_payload in reversed(snapshot[idx:]):
                        self._send_queue.appendleft((back_ts, back_payload))
                break
        if drained:
            logger.info("Drained %d queued event(s) after reconnect", drained)
        return drained

    async def send_context_assembled(
        self,
        *,
        correlation_id: str,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        final_item_ids: Optional[List[str]] = None,
        final_total_tokens: int = 0,
        compaction_id: str = "",
        recent_turn_count: int = 0,
        compaction_end_timestamp: str = "",
        assembly_source: str = "",
        assembly_duration_ms: int = 0,
        cache_hit: bool = False,
        sdk_version: str = "",
    ) -> None:
        """Emit a ContextAssembledEvent for audit-enrichment.

        Fire-and-forget: silently no-ops when the stream is not connected.
        The server-side finalize-on-timeout watcher will write a synthetic
        ``source='server_snapshot'`` row if no SDK event arrives within
        the configured window.

        Privacy: MUST NOT carry raw user prompt content. Only ids and
        composition metadata.
        """
        if self._state != StreamState.CONNECTED or self._stream is None:
            logger.debug(
                "send_context_assembled: stream not connected, skipping "
                "(correlation_id=%s)",
                correlation_id,
            )
            return

        from .proto import synap_service_pb2

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        assembled = synap_service_pb2.ContextAssembledEvent(
            correlation_id=correlation_id or "",
            conversation_id=conversation_id or "",
            user_id=user_id or "",
            customer_id=customer_id or "",
            final_item_ids=list(final_item_ids or []),
            final_total_tokens=int(final_total_tokens or 0),
            compaction_id=compaction_id or "",
            recent_turn_count=int(recent_turn_count or 0),
            compaction_end_timestamp=compaction_end_timestamp or "",
            assembly_source=assembly_source or "",
            assembly_duration_ms=int(assembly_duration_ms or 0),
            cache_hit=bool(cache_hit),
            timestamp_ms=now_ms,
            sdk_version=sdk_version or "",
        )
        event = synap_service_pb2.StreamEvent(context_assembled=assembled)
        try:
            await self._stream.write(event)
        except Exception as e:
            # Never raise on telemetry emit — the fetch already succeeded
            # and the watcher will backfill if necessary.
            logger.debug(
                "send_context_assembled write failed (non-fatal): %s", e
            )

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
            # The scope rung the server retrieved this bundle at. `getattr`
            # with a default because a server older than this SDK does not
            # send the field, and an absent rung must read as "named no rung"
            # rather than as an AttributeError on the stream-reader loop.
            "_anticipation_scope_rung": (
                getattr(proto, "anticipation_scope_rung", "") or None
            ),
            "_bundle_type": proto.bundle_type or "anticipation",
            "conversation_context": conv_ctx,
            # Section 16 — bundle composition extensions. Defaults preserve
            # backwards compatibility when the server is older than the SDK.
            "_bundle_confidence": float(getattr(proto, "bundle_confidence", 0.0) or 0.0),
            "_origin_pattern_id": getattr(proto, "origin_pattern_id", "") or "",
            "_ttl_hint_seconds": int(getattr(proto, "ttl_hint_seconds", 0) or 0),
        }

    async def close(self) -> None:
        """Gracefully close the connection.

        Anything still buffered goes out first, with a short deadline. A
        process shutting down is the moment a queued turn is most likely to be
        lost forever: the queue lives in memory, so whatever is still in it
        when this returns is gone. The deadline is there because a close that
        hangs on a dead network is its own kind of broken.
        """
        # Stop the heartbeat FIRST. It writes to the stream, and a write after
        # the half-close below raises.
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        await self._flush_before_close()

        # After the flush, so the file describes what is actually still
        # outstanding. An empty pair removes it, which is how a clean
        # shutdown leaves nothing for the next start to replay.
        self._compact_journal()

        # ⚠ Half-close the write side before anything is torn down.
        #
        # Without this the SDK never told the server it had finished writing.
        # It cancelled the background tasks and dropped the channel, which
        # grpc reports to the server as a client CANCEL, and a cancelled
        # bidi stream discards whatever the server has not already read. The
        # staging logs said it plainly every time: "Listen stream cancelled",
        # never a clean end, with `conversation_events=0` on a turn that sent
        # five.
        #
        # `done_writing()` ends the request half cleanly, so the server's read
        # loop drains what is queued and finishes on its own terms. This is the
        # third of the three things that had to be true for a turn to survive,
        # alongside not claiming CONNECTED early and closing with a grace
        # period. Any one of them missing loses the turn.
        if self._stream is not None:
            try:
                async with self._write_lock:
                    await asyncio.wait_for(
                        self._stream.done_writing(), timeout=self.CLOSE_GRACE)
            except AttributeError:
                pass  # a test double without the method
            except Exception as e:  # noqa: BLE001 — never fail a teardown
                logger.debug("done_writing on close failed: %s", e)

        logger.info("Closing gRPC stream")
        self._state = StreamState.CLOSED
        self._shutdown_event.set()

        # ⚠ Do NOT cancel the listen task here. This was the bug.
        #
        # The listen task sits in `await self._stream.read()`. Cancelling a
        # task blocked on a bidi read cancels the whole RPC, and a cancelled
        # RPC throws away every message the server has not yet read. The five
        # events of a turn were written, the client held them as
        # unacknowledged, and the server's read loop counted zero. Measured on
        # staging: `undelivered() == {'queued': 0, 'unacknowledged': 5}` on the
        # client, `conversation_events=0` on the server, every time.
        #
        # After `done_writing()` above, the server drains what is queued and
        # closes its side, which ends the read naturally. So wait for that
        # instead, and only cancel if the server does not finish in time. The
        # wait is bounded for the same reason the grace period is: a shutdown
        # that hangs on a dead network is its own kind of broken.
        if self._listen_task and not self._listen_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._listen_task), timeout=self.CLOSE_GRACE)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:  # noqa: BLE001 — never fail a teardown
                pass
            if not self._listen_task.done():
                logger.warning(
                    "Listen loop did not finish within %ss of the half-close; "
                    "cancelling, which may discard events the server had not "
                    "read yet (%s)", self.CLOSE_GRACE, self.undelivered())
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass

        # Close channel.
        #
        # ⚠ `close()` with no grace period cancels every in-flight RPC at once,
        # including writes grpc has accepted but not yet put on the wire. That
        # is the second half of the lost-turn bug: the events were written, the
        # call returned, and the channel was torn down underneath them. The
        # grace period lets what is already written reach the server. It is
        # short, because a close that hangs on a dead network is its own kind
        # of broken, and whatever misses it is reported below rather than
        # vanishing quietly.
        if self._channel:
            try:
                await self._channel.close(grace=self.CLOSE_GRACE)
            except TypeError:
                # Older grpc.aio, or a test double, without the argument.
                await self._channel.close()
            self._channel = None

        self._stream = None
        logger.info("gRPC stream closed")

    # How long the channel may spend draining writes it has already accepted.
    CLOSE_GRACE = 2.0

    def undelivered(self) -> Dict[str, int]:
        """What this transport could not get to the server.

        Exists because every send path returns None. A caller had no way to
        tell a delivered turn from a lost one, which is how five events could
        go out, one arrive, and the SDK report success five times.
        """
        return {
            "queued": len(self._send_queue),
            "unacknowledged": len(self._unacked),
        }

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
