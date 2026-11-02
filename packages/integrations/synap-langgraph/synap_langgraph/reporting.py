"""Report a LangGraph run to Synap without LangChain.

:class:`~synap_langgraph.SynapLangGraphCallbackHandler` hears LangChain model
and tool callbacks. A graph that calls its own LLM client (``await
llm.complete(...)``) fires none of them, so it reports nothing. These node
factories report from the graph itself: put a node after the step that holds
the user's message and after the step that produced the answer, and the turn
reaches Synap whatever called the model.

⚠ Use these nodes OR :class:`~synap_langgraph.SynapLangGraphCallbackHandler`,
not both: on a LangChain model each reports the user and assistant turns, so
together every turn is sent twice.

They are built on the framework-neutral helpers in
``synap_integrations_common`` (``record_turn``, ``report_tool_call``,
``report_tool_result``, ``report_reasoning``), the same ones the LangChain
handler uses, so the events are the same.

Identity and conversation come from the run's config, per call, so a graph
compiled once serves many users and conversations::

    config = {"configurable": {
        "user_id": "alice",            # required
        "conversation_id": "conv-1",   # required
        "customer_id": "acme",         # B2B
        "scope_path": {...},           # optional, deeper scope ladder
    }}

An argument given to the factory wins over the config. A missing ``user_id``
or ``conversation_id`` is an error, never a guess: with
``on_error="fallback"`` (default) it is logged at ERROR and the node does
nothing; with ``"raise"`` it raises :class:`SynapIntegrationError`. A turn that
neither the gRPC stream nor REST accepted also raises under ``"raise"``. Tool
and reasoning events are silent when no stream is running, by design.

The content comes from the state, by key or by callable::

    graph.add_node("emit_user", create_synap_turn_node(sdk, role="user", content="question"))
    graph.add_node("emit_assistant", create_synap_turn_node(
        sdk, role="assistant", content=lambda state: state["answer"]))

A state value may be a string, a message (anything with ``.content``), a dict
with ``content`` or ``text``, or a list of those (the last one is used).

``scope_path`` is carried on the user and assistant turns only: the SDK's tool
and reasoning calls take no path.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Union

from langchain_core.runnables import RunnableConfig
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    ensure_ids_match_path,
    SynapIntegrationError,
    record_turn_via,
    report_reasoning,
    set_auto_listen,
    report_tool_call,
    report_tool_result,
)

logger = logging.getLogger(__name__)

#: A state key (``str``) or a callable ``(state) -> value`` (sync or async).
Source = Union[str, Callable[[Any], Any]]


async def _read(source: Optional[Source], state: Any) -> Any:
    if source is None:
        return None
    if callable(source):
        value = source(state)
        return await value if inspect.isawaitable(value) else value
    if isinstance(state, Mapping):
        return state.get(source)
    return getattr(state, source, None)


def _text_of(value: Any) -> str:
    """The words in a state value (string, message, dict, or a list of them)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return _text_of(value[-1]) if value else ""
    if isinstance(value, Mapping):
        return _text_of(value.get("content", value.get("text")))
    content = getattr(value, "content", None)
    if content is not None:
        if isinstance(content, list):  # content blocks: join the text ones
            return "".join(
                b if isinstance(b, str) else str(b.get("text", ""))
                for b in content if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
            )
        return _text_of(content)
    return str(value)


class _Identity:
    __slots__ = ("conversation_id", "user_id", "customer_id", "scope_path")

    def __init__(self, conversation_id: str, user_id: str, customer_id: str,
                 scope_path: Optional[Dict[str, str]]) -> None:
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        self.scope_path = scope_path


def _config_or_current(config: Optional[RunnableConfig]) -> Optional[Mapping[str, Any]]:
    if config is not None:
        return config
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:  # noqa: BLE001 — called outside a graph run
        return None


def _identity(
    config: Optional[RunnableConfig],
    conversation_id: Optional[str],
    user_id: Optional[str],
    customer_id: Optional[str],
    scope_path: Optional[Mapping[str, str]],
    op: str,
) -> _Identity:
    cfg = _config_or_current(config)
    conf = cfg.get("configurable") if isinstance(cfg, Mapping) else None
    conf = conf if isinstance(conf, Mapping) else {}

    def pick(explicit: Optional[str], key: str) -> str:
        value = explicit or conf.get(key) or ""
        return value if isinstance(value, str) else ""

    cid, uid = pick(conversation_id, "conversation_id"), pick(user_id, "user_id")
    if not uid or not cid:
        missing = [n for n, v in (("user_id", uid), ("conversation_id", cid)) if not v]
        raise SynapIntegrationError(
            op,
            f"missing {' and '.join(missing)}: pass it to the factory or put it in "
            "config[\"configurable\"] for the run. Nothing was reported.",
        )
    path = scope_path or conf.get("scope_path") or None
    if path is not None and not (
        isinstance(path, Mapping) and all(isinstance(k, str) and k and isinstance(v, str) and v
                                          for k, v in path.items())
    ):
        raise SynapIntegrationError(
            op, f"scope_path must map level keys to non-empty strings, got {path!r}.")
    cust = pick(customer_id, "customer_id")
    # The server lets the path's user/customer win over the flat ids; a run for
    # another user would be filed as the path's user. Raised here, so the node
    # skips the run (logged at ERROR, nothing sent) unless on_error="raise".
    ensure_ids_match_path(path, user_id=uid, customer_id=cust or None, where=op)
    return _Identity(cid, uid, cust, dict(path) if path else None)


def _validate(sdk: Any, on_error: str, site: str, sdk_provider: Any = None) -> None:
    if sdk is None and sdk_provider is None:
        raise ValueError(f"{site} requires a non-None sdk (or an sdk_provider)")
    if sdk is not None and sdk_provider is not None:
        raise ValueError(f"{site} takes sdk OR sdk_provider, not both")
    if on_error not in ("fallback", "raise"):
        raise ValueError(f"{site}: on_error must be 'fallback' or 'raise', got {on_error!r}")


def _make_node(
    sdk: MaximemSynapSDK,
    site: str,
    on_error: str,
    conversation_id: Optional[str],
    user_id: Optional[str],
    customer_id: Optional[str],
    scope_path: Optional[Mapping[str, str]],
    action: Callable[[Any, _Identity, Any], Awaitable[Optional[str]]],
    name: str,
    sdk_provider: Optional[Callable[[], Any]] = None,
    auto_listen: Optional[bool] = None,
    transport_key: Optional[str] = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    async def _node(state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            ident = _identity(config, conversation_id, user_id, customer_id, scope_path, site)
            the_sdk = sdk
            if sdk_provider is not None:
                the_sdk = sdk_provider()
                if inspect.isawaitable(the_sdk):
                    the_sdk = await the_sdk
                if the_sdk is None:
                    raise SynapIntegrationError(
                        site, "sdk_provider() returned no SDK. Nothing was reported.")
            if auto_listen is not None:
                set_auto_listen(the_sdk, auto_listen)
            transport = await action(state, ident, the_sdk)
            if transport_key and transport is not None:
                out[transport_key] = transport
        except SynapIntegrationError as exc:
            if on_error == "raise":
                raise
            logger.error("%s", exc)
            if transport_key:
                out[transport_key] = "failed"
        except Exception as exc:  # noqa: BLE001 — reporting must not break the graph
            if on_error == "raise":
                raise SynapIntegrationError(site, str(exc)) from exc
            logger.error("%s failed: %s", site, exc, exc_info=True)
            if transport_key:
                out[transport_key] = "failed"
        return out

    _node.__name__ = name
    return _node


def create_synap_turn_node(
    sdk: Optional[MaximemSynapSDK] = None,
    *,
    role: str,
    content: Source,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    scope_path: Optional[Mapping[str, str]] = None,
    on_error: str = "fallback",
    sdk_provider: Optional[Callable[[], Any]] = None,
    auto_listen: Optional[bool] = None,
    transport_key: Optional[str] = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """A node that reports one user or assistant turn.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        role: ``"user"`` or ``"assistant"``.
        content: A state key or a ``(state) -> text`` callable.
        conversation_id, user_id, customer_id, scope_path: Override the run's
            ``config["configurable"]`` values of the same name.
        on_error: ``"fallback"`` (default) logs; ``"raise"`` raises
            :class:`SynapIntegrationError`, also when the turn was not recorded.

    The turn goes on the gRPC stream (opened by the first one) and over REST
    when the stream cannot carry it, never both. The node returns ``{}``: it
    changes nothing in the state. Add one after the step holding the user's
    message and one after the step that answered.

    ``transport_key``: a state key to receive how the turn travelled,
    ``"stream"``, ``"rest"``, ``"failed"`` (under ``on_error="fallback"``) or
    ``"skipped"`` (nothing was said). Declare it in your state. Tool and
    reasoning nodes take the same option and write ``"stream"`` or ``"none"``
    (they exist only on the stream, so no stream means nothing was sent).

    ``auto_listen``: ``None`` (default) follows the SDK's behaviour (open the
    stream on the first report unless ``SYNAP_AUTO_LISTEN=false``). ``False``
    never opens one for this SDK, so a turn goes over REST when no stream was
    opened by the caller; ``True`` forces opening. It is set on the SDK each
    time the node runs (``set_auto_listen``), so it applies to every reporter
    that shares the SDK object.

    ``sdk_provider``: a callable (sync or async) returning the SDK, called each
    time the node runs, instead of ``sdk``. The package keeps its stream state
    on the SDK object, so a node bound to a stand-in would share it between
    agents; a provider returns the real one.
    """
    _validate(sdk, on_error, "create_synap_turn_node", sdk_provider)
    if role not in ("user", "assistant"):
        raise ValueError(f"create_synap_turn_node: role must be 'user' or 'assistant', got {role!r}")
    site = f"synap_langgraph.report_{role}_turn"
    ensure_ids_match_path(
        scope_path, user_id=user_id or None, customer_id=customer_id or None, where=site)

    async def action(state: Any, ident: _Identity, sdk: Any) -> Optional[str]:
        text = _text_of(await _read(content, state))
        if not text.strip():
            return "skipped"  # nothing was said
        extra: Dict[str, Any] = {"scope_path": ident.scope_path} if ident.scope_path else {}
        how = await record_turn_via(
            sdk, role=role, content=text, conversation_id=ident.conversation_id,
            user_id=ident.user_id, customer_id=ident.customer_id, **extra,
        )
        if how == "failed":
            raise SynapIntegrationError(
                site, "the turn was not recorded: neither the gRPC stream nor REST accepted it",
                {"role": role, "conversation_id": ident.conversation_id},
            )
        return how

    return _make_node(sdk, site, on_error, conversation_id, user_id, customer_id,
                      scope_path, action, f"synap_report_{role}_turn", sdk_provider,
                      auto_listen, transport_key)


def create_synap_tool_call_node(
    sdk: Optional[MaximemSynapSDK] = None,
    *,
    tool_name: Union[str, Callable[[Any], str]],
    tool_args: Optional[Source] = None,
    tool_call_id: Optional[Source] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    on_error: str = "fallback",
    sdk_provider: Optional[Callable[[], Any]] = None,
    auto_listen: Optional[bool] = None,
    transport_key: Optional[str] = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """A node that reports a tool the agent is about to call.

    ``tool_name`` is the name, or a callable ``(state) -> name``. ``tool_args``
    (a dict) and ``tool_call_id`` (a string that ties this call to its result)
    are state keys or callables. Silent when no stream is running.
    """
    _validate(sdk, on_error, "create_synap_tool_call_node", sdk_provider)
    site = "synap_langgraph.report_tool_call"

    async def action(state: Any, ident: _Identity, sdk: Any) -> Optional[str]:
        name = tool_name(state) if callable(tool_name) else tool_name
        args = await _read(tool_args, state)
        call_id = await _read(tool_call_id, state)
        sent = await report_tool_call(
            sdk, tool_name=str(name or "tool"),
            tool_args=args if isinstance(args, dict) else None,
            tool_call_id=str(call_id) if call_id else "",
            conversation_id=ident.conversation_id, user_id=ident.user_id,
            customer_id=ident.customer_id,
        )
        return "stream" if sent else "none"

    return _make_node(sdk, site, on_error, conversation_id, user_id, customer_id,
                      None, action, "synap_report_tool_call", sdk_provider, auto_listen,
                      transport_key)


def create_synap_tool_result_node(
    sdk: Optional[MaximemSynapSDK] = None,
    *,
    result: Source,
    tool_name: Optional[Union[str, Callable[[Any], str]]] = None,
    tool_call_id: Optional[Source] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    on_error: str = "fallback",
    sdk_provider: Optional[Callable[[], Any]] = None,
    auto_listen: Optional[bool] = None,
    transport_key: Optional[str] = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """A node that reports what a tool returned.

    ``result`` is a state key or callable; pass the same ``tool_call_id`` source
    as the call node so the two are paired. A message object is unwrapped to its
    ``.content`` (the SDK JSON-encodes the result). ⚠ A tool result is usually
    the caller's customer data: it leaves your process as an anticipation hint.
    """
    _validate(sdk, on_error, "create_synap_tool_result_node", sdk_provider)
    site = "synap_langgraph.report_tool_result"

    async def action(state: Any, ident: _Identity, sdk: Any) -> Optional[str]:
        value = await _read(result, state)
        content = getattr(value, "content", None)
        if content is not None and hasattr(value, "tool_call_id"):
            value = content  # a ToolMessage is not JSON serialisable
        name = tool_name(state) if callable(tool_name) else tool_name
        call_id = await _read(tool_call_id, state)
        sent = await report_tool_result(
            sdk, result=value, tool_name=str(name or ""),
            tool_call_id=str(call_id) if call_id else "",
            conversation_id=ident.conversation_id, user_id=ident.user_id,
            customer_id=ident.customer_id,
        )
        return "stream" if sent else "none"

    return _make_node(sdk, site, on_error, conversation_id, user_id, customer_id,
                      None, action, "synap_report_tool_result", sdk_provider, auto_listen,
                      transport_key)


def create_synap_reasoning_node(
    sdk: Optional[MaximemSynapSDK] = None,
    *,
    content: Source,
    step_index: Union[int, None, Callable[[Any], Optional[int]]] = None,
    thought_type: Union[str, Callable[[Any], str]] = "",
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    on_error: str = "fallback",
    sdk_provider: Optional[Callable[[], Any]] = None,
    auto_listen: Optional[bool] = None,
    transport_key: Optional[str] = None,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """A node that reports one reasoning step, when the graph has one.

    ``content`` is a state key or callable. An empty value reports nothing.
    Silent when no stream is running.
    """
    _validate(sdk, on_error, "create_synap_reasoning_node", sdk_provider)
    site = "synap_langgraph.report_reasoning"

    async def action(state: Any, ident: _Identity, sdk: Any) -> Optional[str]:
        text = _text_of(await _read(content, state))
        if not text.strip():
            return "none"
        step = step_index(state) if callable(step_index) else step_index
        kind = thought_type(state) if callable(thought_type) else thought_type
        sent = await report_reasoning(
            sdk, content=text.strip(), step_index=step, thought_type=kind or "",
            conversation_id=ident.conversation_id, user_id=ident.user_id,
            customer_id=ident.customer_id,
        )
        return "stream" if sent else "none"

    return _make_node(sdk, site, on_error, conversation_id, user_id, customer_id,
                      None, action, "synap_report_reasoning", sdk_provider, auto_listen,
                      transport_key)


__all__ = [
    "create_synap_turn_node",
    "create_synap_tool_call_node",
    "create_synap_tool_result_node",
    "create_synap_reasoning_node",
]
