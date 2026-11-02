"""Report what an agent is doing on the gRPC stream, opening it when needed.

Every integration hooks a different framework, and they all have the same job
once a hook fires: turn it into a Synap stream event. Four copies of that is
four places for the role to be wrong, the ids to be forgotten, or an exception
to escape into somebody's agent loop.

**Three rules, and they are the whole design.**

*The stream is the default path.* The first report on an SDK with no stream
opens one (``sdk.instance.listen()``), so a caller who never heard of
``listen()`` still gets turns, tool calls and reasoning to the server, and
anticipation bundles back. ``SYNAP_AUTO_LISTEN=false`` or
``set_auto_listen(sdk, False)`` turns that off.

*REST when the stream cannot carry it.* A stream that will not open, or one
that lives on a different event loop from the caller, is not used. Turns then
go over REST (``record_turn``); events that only exist on the stream are
dropped. A failed open is retried once after a few seconds (a cold first
connection), then no more than once a minute.

*Never raises.* A hook runs inside the framework's own loop. A telemetry call
that throws there takes down the agent run, and no context is worth that. Every
function swallows everything and logs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

_RETRY_AFTER_S = 60.0
# The first open of a process can lose to a cold connection (seen on India: the
# SDK's 5 s connect timeout, then 0.1 s opens once warm). One quick retry gets
# the stream up a few seconds in instead of a minute; after that, the cooldown.
_FIRST_RETRY_AFTER_S = 3.0
_OPEN_TIMEOUT_S = 10.0
# A caller who runs every turn under its own asyncio.run() closes the loop the
# stream lived on after each one. Reopening per turn costs a connection each
# time, so after this many the SDK stays on REST.
_MAX_REOPENS = 3

# Methods already reported as unable to carry `scope_path` (one error per
# process each, not one per turn).
_scope_path_unsupported_logged: set = set()

# State for an SDK that refuses new attributes, keyed by id().
_fallback_state: Dict[int, Dict[str, Any]] = {}


def _state(sdk: Any) -> Dict[str, Any]:
    try:
        state = vars(sdk).get("_synap_auto_stream")
    except TypeError:  # __slots__: no instance dict
        state = None
    if isinstance(state, dict):
        return state
    state = _fallback_state.get(id(sdk))
    if state is not None:
        return state
    state = {"enabled": None, "loop": None, "failed_at": None, "failures": 0,
             "reopens": 0, "opening": False, "warned": False}
    try:
        sdk._synap_auto_stream = state
    except Exception:  # noqa: BLE001
        _fallback_state[id(sdk)] = state
    return state


def set_auto_listen(sdk: Any, enabled: bool) -> None:
    """Turn opening the stream on first report on or off for one SDK.

    Overrides ``SYNAP_AUTO_LISTEN`` for that SDK. A stream the caller opened
    themselves is used either way.
    """
    _state(sdk)["enabled"] = bool(enabled)


def _auto_listen_enabled(sdk: Any) -> bool:
    enabled = _state(sdk)["enabled"]
    if enabled is not None:
        return enabled
    return os.environ.get("SYNAP_AUTO_LISTEN", "true").strip().lower() not in (
        "0", "false", "no", "off")


async def _within(awaitable: Any, timeout: float) -> Any:
    """``asyncio.wait_for`` without needing a current task.

    ``wait_for`` refuses to run on a loop ``nest_asyncio`` has patched, which is
    every loop ``run_async`` has driven a coroutine on.
    """
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        raise TimeoutError(f"no answer within {timeout:g}s")
    return task.result()


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _listening(sdk: Any) -> bool:
    try:
        return bool(sdk.instance.is_listening)
    except Exception:  # noqa: BLE001
        return False


def stream_is_active(sdk: Any) -> bool:
    """Whether there is a stream to report on.

    A stream this module opened on an event loop that has since closed died
    with it, and answers no. Never raises: an SDK built differently, or
    half-initialised, answers no.
    """
    if not _listening(sdk):
        return False
    loop = _state(sdk)["loop"]
    return loop is None or not loop.is_closed()


def _usable_here(sdk: Any) -> bool:
    """A live stream this call can write to from the loop it is running on.

    The stream's tasks belong to the loop that opened it. A write from any
    other loop would sit in a queue nothing drains, and a turn that was
    accepted and never sent is worse than one sent over REST.
    """
    if not _listening(sdk):
        return False
    loop = _state(sdk)["loop"]
    return loop is None or loop is _running_loop()


async def _discard_dead_stream(sdk: Any, state: Dict[str, Any]) -> None:
    """Forget a stream whose loop has closed, so a new one can open."""
    state["loop"] = None
    state["reopens"] += 1
    try:
        await _within(sdk.instance.stop_listening(), 2.0)
    except Exception:  # noqa: BLE001 — its tasks died with their loop
        pass
    if _listening(sdk):
        controller = getattr(sdk.instance, "_controller", None)
        try:
            controller._transport = None
            sdk._grpc_transport = None
        except Exception:  # noqa: BLE001
            pass


async def ensure_stream(sdk: Any) -> bool:
    """Open the stream if there is none. Returns whether this call can use one.

    Never raises. Never waits more than ten seconds, and after a failure does
    not try again for a minute: until then every turn goes over REST.
    """
    try:
        state = _state(sdk)
        loop = _running_loop()
        if _listening(sdk):
            own = state["loop"]
            if own is None or own is loop:
                return True
            if not own.is_closed():
                return False  # alive on another loop; REST for this call
            await _discard_dead_stream(sdk, state)
            if _listening(sdk):
                return False
        if loop is None or state["opening"] or not _auto_listen_enabled(sdk):
            return False
        if vars(sdk).get("_initialized") is False:
            return False  # not a failure: the next call after initialize() opens it
        if state["reopens"] > _MAX_REOPENS:
            return False
        failed_at = state["failed_at"]
        wait = _FIRST_RETRY_AFTER_S if state.get("failures", 0) <= 1 else _RETRY_AFTER_S
        if failed_at is not None and time.monotonic() - failed_at < wait:
            return False
        state["opening"] = True
        try:
            await _within(sdk.instance.listen(), _OPEN_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — includes the timeout
            if _listening(sdk):
                return True  # somebody else opened it while we waited
            state["failed_at"] = time.monotonic()
            state["failures"] = state.get("failures", 0) + 1
            if not state["warned"]:
                state["warned"] = True
                logger.warning(
                    "synap: could not open the gRPC stream (%s); turns go over "
                    "REST until it opens. SYNAP_AUTO_LISTEN=false stops trying.",
                    exc,
                )
            return False
        finally:
            state["opening"] = False
        if not _listening(sdk):
            state["failed_at"] = time.monotonic()
            state["failures"] = state.get("failures", 0) + 1
            return False
        state["loop"] = loop
        state["failed_at"] = None
        state["failures"] = 0
        return True
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.debug("synap ensure_stream failed", exc_info=True)
        return False


def _accepts_scope_path(method: Any) -> bool:
    """Whether ``method`` can take a ``scope_path`` keyword, by its signature.

    A signature with a ``scope_path`` parameter or ``**kwargs`` accepts it. If
    the signature cannot be read the answer is yes: the call is then tried, and
    a method that does not take it raises ``TypeError``, which the caller
    treats as "not sent" (a turn is never filed without its path).
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True
    return "scope_path" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _can_carry_scope_path(method: Any, name: str) -> bool:
    """True when a turn with a ``scope_path`` may be sent through ``method``.

    When the installed SDK's method has no ``scope_path`` parameter the turn
    must NOT go without it: it would be filed at the wrong, wider scope. Logs
    one clear error per process per method and answers False.
    """
    if _accepts_scope_path(method):
        return True
    if name not in _scope_path_unsupported_logged:
        _scope_path_unsupported_logged.add(name)
        logger.error(
            "synap: a turn carries a scope_path but the installed maximem-synap's "
            "%s() does not accept `scope_path`. The turn is NOT sent, because "
            "without its path it would be filed at the wrong, wider scope. "
            "Upgrade maximem-synap.", name)
    return False


async def report_turn(
    sdk: Any,
    *,
    role: str,
    content: str,
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
    metadata: Optional[Mapping[str, str]] = None,
    scope_path: Optional[Mapping[str, str]] = None,
) -> bool:
    """Report a user or assistant turn. Returns whether it was sent.

    `assistant_message` is the event the anticipation agent acts on: it is the
    moment a turn ends and the next one can be predicted. An integration that
    reports everything else and not this one gets no anticipation at all.

    ``scope_path`` names the conversation's level on an account with a deeper
    scope ladder (``ScopeAddress.scope_path``). It is forwarded to
    ``send_message(scope_path=...)`` only when set; a call without it passes
    exactly the keyword arguments it always did. If it is set and the installed
    SDK's ``send_message`` has no such parameter, nothing is sent (see
    ``_can_carry_scope_path``) and this returns False.
    """
    event_type = "user_message" if role == "user" else "assistant_message"
    extra: Dict[str, Any] = {}
    if scope_path:
        try:
            if not _can_carry_scope_path(sdk.instance.send_message, "send_message"):
                return False
            extra["scope_path"] = dict(scope_path)
        except Exception:  # noqa: BLE001 — a hook must never break the agent run
            logger.debug("synap report_turn scope_path check failed", exc_info=True)
            return False
    return await _send(
        sdk, "send_message",
        content=content,
        role="user" if role == "user" else "assistant",
        event_type=event_type,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
        metadata=dict(metadata or {}),
        **extra,
    )


async def record_turn(
    sdk: Any,
    *,
    role: str,
    content: str,
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
    metadata: Optional[Mapping[str, str]] = None,
    scope_path: Optional[Mapping[str, str]] = None,
) -> bool:
    """Write a user or assistant turn: on the stream (opened if need be), over
    REST when the stream cannot carry it. Returns whether it was written.

    A turn is the conversation itself, not telemetry about it: without it the
    server has nothing to compact and the SDK nothing to search. Stream first,
    REST second, never both: the server persists a streamed turn from its
    event. Never raises.

    ``scope_path`` (the conversation's level on a deeper ladder) is forwarded
    to ``send_message`` and to ``record_message`` only when set, so a call
    without it is unchanged. A transport whose SDK method has no
    ``scope_path`` parameter is not used for a turn that has one (it would be
    filed at the wrong, wider scope): one error is logged per process and, if
    no transport can carry it, nothing is sent and this returns False.

    Not carried on the other events (``report_tool_call``,
    ``report_tool_result``, ``report_reasoning``, ``end_session``): the SDK's
    ``record_tool_call`` / ``record_tool_result`` / ``record_thinking`` take
    no ``scope_path`` (sdk.py), and the server reads the stream event's
    ``scope`` only when it persists a ``user_message`` / ``assistant_message``
    (the conversation row is written from those two event types alone), so a
    path on them would be ignored.
    """
    if not content or not content.strip():
        return False
    if await report_turn(sdk, role=role, content=content, conversation_id=conversation_id,
                         user_id=user_id, customer_id=customer_id, metadata=metadata,
                         scope_path=scope_path):
        return True
    if not conversation_id:
        return False  # the REST write needs one; the stream would have minted it
    kwargs = {"conversation_id": conversation_id, "role": "user" if role == "user" else "assistant",
              "content": content, "user_id": user_id}
    if customer_id:
        kwargs["customer_id"] = customer_id
    if metadata:
        kwargs["metadata"] = dict(metadata)
    try:
        if scope_path:
            if not _can_carry_scope_path(sdk.conversation.record_message, "record_message"):
                return False
            kwargs["scope_path"] = dict(scope_path)
        await sdk.conversation.record_message(**kwargs)
        return True
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.warning("synap turn write failed (conversation_id=%s)", conversation_id, exc_info=True)
        return False


async def report_tool_call(
    sdk: Any,
    *,
    tool_name: str,
    tool_args: Optional[Any] = None,
    tool_call_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report a tool the agent is invoking."""
    return await _send(
        sdk, "record_tool_call",
        tool_name,
        tool_args if isinstance(tool_args, dict) else None,
        tool_call_id=tool_call_id or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def report_tool_result(
    sdk: Any,
    *,
    result: Any,
    tool_name: str = "",
    tool_call_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report what a tool returned.

    ⚠ A tool result is usually the caller's customer data. It is an
    anticipation hint and never becomes a long-term memory, but it does leave
    their process. An integration passes through what the framework gives it
    and does not go looking for more.

    When the stream cannot carry it, the result is still kept in the SDK's
    local turn archive (never sent), so history search can find it after a
    compaction; on the stream, the SDK's ``record_tool_result`` archives it
    itself.
    """
    if conversation_id and not await ensure_stream(sdk):
        archive = getattr(sdk, "archive_tool_result", None)
        if callable(archive):
            try:
                archive(conversation_id, result, tool_name=tool_name or None)
            except Exception:  # noqa: BLE001 — a hook must never break the agent run
                logger.debug("synap tool result archive failed", exc_info=True)
    return await _send(
        sdk, "record_tool_result",
        result,
        tool_name=tool_name or None,
        tool_call_id=tool_call_id or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def report_reasoning(
    sdk: Any,
    *,
    content: str,
    step_index: Optional[int] = None,
    thought_type: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report one reasoning step, when the framework exposes one.

    Most providers return a summary rather than raw reasoning, and some return
    nothing. An integration reports what it is given; a turn with no reasoning
    is an ordinary turn, not a gap to fill.
    """
    if not content:
        return False
    return await _send(
        sdk, "record_thinking",
        content,
        step_index=step_index,
        thought_type=thought_type or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def end_session(sdk: Any, conversation_id: str) -> bool:
    """Close the session for a conversation the integration knows has ended."""
    if not conversation_id:
        return False
    # Never opens a stream: there is no session to end on a new one.
    return await _send(sdk, "end_session", conversation_id, _open=False)


async def _send(sdk: Any, method: str, *args: Any, _open: bool = True, **kwargs: Any) -> bool:
    """One call site for every report, so the rules hold in one place."""
    if not (await ensure_stream(sdk) if _open else _usable_here(sdk)):
        return False
    # `customer_id` is refused on a B2C instance, so an empty one is dropped
    # rather than sent as "". The same applies to any id the caller left out.
    cleaned = {k: v for k, v in kwargs.items() if v not in ("", None)}
    try:
        await getattr(sdk.instance, method)(*args, **cleaned)
        return True
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.debug("synap stream report failed (%s)", method, exc_info=True)
        return False
