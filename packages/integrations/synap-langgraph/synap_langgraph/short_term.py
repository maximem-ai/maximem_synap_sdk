"""Synap short-term context for LangGraph.

Two composable surfaces, both wrapping
``sdk.conversation.context.get_context_for_prompt`` (which is cache-first
behind the ``SYNAP_SDK_ST_AUTHORITATIVE`` flag — see
``docs/internal/sdk_authoritative_short_term_context_plan.md``):

- :func:`synap_st_prompt` — returns an async callable suitable for
  ``create_react_agent(prompt=...)``. ``prompt`` is the hook that decides
  what the model is sent without touching what the graph stores, so this is
  where Synap's compaction replaces the turns before the current one: once the
  conversation has been compacted the model gets the compaction above your
  system prompt, then the turns after it, then the current turn. Before that
  the messages go through as they are. Use for **prebuilt** agents where the
  one-line drop-in matters.

- :func:`create_synap_st_node` — returns an async node suitable for
  ``StateGraph.add_node(...)``. Writes the compaction string into
  ``state[state_key]`` for downstream LLM nodes to assemble however they
  want. Use for **custom graphs** where you control the prompt template.

Both honour the project-wide policy:

- The conversation id is the caller's, never inferred from the LangGraph
  ``thread_id`` (the two namespaces can diverge). Pass ``conversation_id=`` to
  pin one conversation, or leave it out and the helper reads
  ``config["configurable"]["conversation_id"]`` **when it runs**, so a graph
  compiled once serves many conversations. An explicit argument always wins.
  With neither, the helper raises (``on_error="raise"``) or logs and degrades
  (``on_error="fallback"``).
- SDK failures **never** crash the graph by default
  (``on_error="fallback"``): we log via :class:`SynapIntegrationError`'s
  log path, then degrade to the user's bare system prompt /
  empty state slot. Pass ``on_error="raise"`` for strict environments.
- No compaction yet is a no-op — it must not wipe the user's system
  prompt, and your own message list is the record until then. Synap's
  ``formatted_context`` carries the recent turns too, which a graph that keeps
  its own message list already sends, so neither helper uses it. A graph that
  does NOT keep its own message list (it sends every turn to Synap and reads
  them back) passes ``include_recent_turns=True`` to get Synap's recent turns
  as well.

The wire shape (cache-first, server fallback, SDK-authoritative when the
flag is on) is owned entirely by the SDK helper. These adapters are thin
shape converters — they intentionally do NOT format, truncate, or
re-summarise.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    CompactedHistory,
    SynapIntegrationError,
    splice_history,
    wrap_sdk_errors_async,
)
from langchain_core.runnables import RunnableConfig
from synap_integrations_common.history import HISTORY_CLOSE, HISTORY_OPEN
from synap_langchain._messages import message_text, role_of

logger = logging.getLogger(__name__)

# Mirror the SDK formatter's supported styles. We re-validate here so a
# construction-time typo fails fast instead of surfacing on first agent
# turn. Kept in sync with maximem_synap.formatter.context_for_prompt —
# importing the constant directly would couple this package to an
# internal SDK module path that may move; the trade-off is a single
# tuple to maintain.
_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")

# Default tags wrap the ST block so a model can recognise it as a
# distinct section. Overridable; setting to ``None`` strips the wrapper
# entirely (the block then prepends as raw text).
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]

# Wrapper for Synap's recent turns when ``include_recent_turns=True``.
_RECENT_OPEN = "<synap_recent_turns>"
_RECENT_CLOSE = "</synap_recent_turns>"


def _get_state_messages(state: Any) -> List[BaseMessage]:
    """Read ``messages`` from a state object that may be a dict OR an
    object with attribute access (LangGraph supports both shapes).
    """
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", None) or [])


def _validate_args(
    sdk: Optional[MaximemSynapSDK],
    conversation_id: Optional[str],
    style: str,
    on_error: str,
    site: str,
) -> None:
    if sdk is None:
        raise ValueError(f"{site} requires a non-None sdk")
    if conversation_id is not None and not str(conversation_id).strip():
        raise ValueError(
            f"{site} requires a non-empty conversation_id when one is given "
            f"(or omit it to read config[\"configurable\"][\"conversation_id\"] "
            f"at call time)"
        )
    if style not in _SUPPORTED_STYLES:
        raise ValueError(
            f"{site}: unsupported style={style!r}; "
            f"expected one of {_SUPPORTED_STYLES}"
        )
    if on_error not in ("fallback", "raise"):
        raise ValueError(
            f"{site}: on_error must be 'fallback' or 'raise', got {on_error!r}"
        )


def _config_or_current(config: Optional[RunnableConfig]) -> Optional[RunnableConfig]:
    """The run's config: the one LangGraph injected, else the running one."""
    if config is not None:
        return config
    try:
        from langgraph.config import get_config

        return get_config()
    except Exception:  # noqa: BLE001 — called outside a graph run
        return None


def _resolve_conversation_id(
    explicit: Optional[str],
    config: Optional[RunnableConfig],
    on_error: _OnError,
    site: str,
) -> Optional[str]:
    """The conversation to read: the explicit argument, else
    ``config["configurable"]["conversation_id"]`` read now.

    ``None`` (after logging) under ``on_error="fallback"`` when there is
    neither; :class:`SynapIntegrationError` under ``"raise"``.
    """
    if explicit:
        return explicit
    cfg = _config_or_current(config)
    configurable = (cfg or {}).get("configurable") if isinstance(cfg, dict) else None
    found = configurable.get("conversation_id") if isinstance(configurable, dict) else None
    if isinstance(found, str) and found.strip():
        return found
    err = SynapIntegrationError(
        site,
        "no conversation id: pass conversation_id=... when building the helper, "
        "or put \"conversation_id\" in config[\"configurable\"] for the run.",
    )
    if on_error == "raise":
        raise err
    logger.error("%s", err)
    return None


def _recent_turns(response: Any) -> List[Dict[str, str]]:
    """Synap's recent turns as ``{"role", "content"}`` dicts, oldest first."""
    turns: List[Dict[str, str]] = []
    for rm in getattr(response, "recent_messages", None) or []:
        role, content = getattr(rm, "role", None), getattr(rm, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append({"role": role, "content": content})
    return turns


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _history_from_context(response: Any) -> Optional[CompactedHistory]:
    """Synap's compaction of the conversation out of an SDK response, or None."""
    if getattr(response, "available", True) is False:
        return None
    compaction = getattr(response, "context_block", None)
    if not isinstance(compaction, str) or not compaction.strip():
        return None  # not compacted yet: the graph's own messages are the record
    parts = [compaction.strip()]
    found = getattr(response, "history_block", None)
    if isinstance(found, str) and found.strip():
        parts.append(found.strip())
    turns = []
    for rm in getattr(response, "recent_messages", None) or []:
        role, content = getattr(rm, "role", None), getattr(rm, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append((role, content))
    return CompactedHistory(
        preamble=f"{HISTORY_OPEN}\n" + "\n\n".join(parts) + f"\n{HISTORY_CLOSE}",
        turns=tuple(turns),
    )


async def _fetch_context(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> Any:
    """The SDK's ``get_context_for_prompt`` answer, or None.

    None on an SDK failure under ``on_error="fallback"``; under ``"raise"``
    the failure propagates as :class:`SynapIntegrationError`.
    """
    try:
        async with wrap_sdk_errors_async(
            site,
            logger,
            conversation_id=conversation_id,
            style=style,
        ):
            response = await sdk.conversation.context.get_context_for_prompt(
                conversation_id=conversation_id,
                style=style,
            )
    except SynapIntegrationError:
        if on_error == "raise":
            raise
        # wrap_sdk_errors_async already logged at ERROR with full context.
        return None
    return response


async def _fetch_history(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> Optional[CompactedHistory]:
    """The compacted conversation, or None when there is nothing to replace.

    None also on an SDK failure under ``on_error="fallback"``; under
    ``"raise"`` the failure propagates as :class:`SynapIntegrationError`.
    """
    response = await _fetch_context(sdk, conversation_id, style, on_error, site)
    return None if response is None else _history_from_context(response)


def _inner(preamble: str) -> str:
    """The compaction text without the tags that mark it as Synap's."""
    text = preamble.strip()
    if text.startswith(HISTORY_OPEN) and text.endswith(HISTORY_CLOSE):
        text = text[len(HISTORY_OPEN):-len(HISTORY_CLOSE)]
    return text.strip()


async def _fetch_st_block(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> str:
    """Synap's compaction as text, or ``""`` when there is none.

    Never the recent turns: the graph sends those, and ``formatted_context``
    carries them too, so a block built from it showed each one twice.
    """
    history = await _fetch_history(sdk, conversation_id, style, on_error, site)
    return _inner(history.preamble) if history is not None else ""


def _make_message(role: str, text: str) -> BaseMessage:
    if role == "system":
        return SystemMessage(content=text)
    return AIMessage(content=text) if role == "assistant" else HumanMessage(content=text)


def _splice(messages: List[BaseMessage], history: CompactedHistory) -> List[BaseMessage]:
    """``messages`` with the turns before the current one replaced by Synap's.

    The compaction message splice adds is left out: the caller puts it in the
    system prompt, where a provider that wants one system message first
    (Anthropic) accepts it.
    """
    if not any(role_of(m) == "user" for m in messages):
        return messages  # no current turn to anchor on: leave the list alone
    try:
        spliced = splice_history(
            messages, history,
            role_of=role_of, text_of=message_text, make_message=_make_message,
        )
    except Exception:  # noqa: BLE001 — the graph's own messages still stand
        logger.warning("synap: could not splice the compacted history", exc_info=True)
        return messages
    return [m for m in spliced
            if not (role_of(m) == "system" and message_text(m) == history.preamble)]


def _compose_system_prompt(
    st_block: str,
    user_system: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
    recent_block: str = "",
) -> str:
    """Combine ST block (+ recent turns) + user system prompt into one string.

    Empty inputs are dropped silently; the result is always a clean
    string (no double blank lines, no orphan tags, no leading/trailing
    whitespace). When both inputs are empty we return ``""`` — the
    caller decides whether to emit a SystemMessage or skip it.
    """
    parts: List[str] = []
    st_block = (st_block or "").strip()
    user_system = (user_system or "").strip()

    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if recent_block:
        parts.append(recent_block)
    if user_system:
        parts.append(user_system)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public surface — Option 1: prompt callable for create_react_agent
# ---------------------------------------------------------------------------


def synap_st_prompt(
    sdk: MaximemSynapSDK,
    conversation_id: Optional[str] = None,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
    include_recent_turns: bool = False,
) -> Callable[..., Awaitable[List[BaseMessage]]]:
    """Return an async ``prompt`` callable for ``create_react_agent``.

    On every LLM step the callable:

    1. Resolves the conversation: ``conversation_id`` if you passed one, else
       ``config["configurable"]["conversation_id"]`` of the run (LangGraph
       passes ``config`` to a ``prompt`` callable that declares it; this one
       does). One agent compiled once therefore serves many conversations.
    2. Awaits :func:`MaximemSynapSDK.conversation.context.get_context_for_prompt`
       (cache-first; near-zero overhead when the SDK store is warm).
    3. If the conversation has been compacted, replaces every turn before the
       current one with Synap's turns after the compaction, and puts the
       compaction above ``system`` inside the configured preamble tags. Tool
       calls and results of the current turn are kept, paired.
    4. Returns ``[SystemMessage(content=<combined>), *messages]``
       — the message list LangGraph passes straight to the model.

    Until Synap has compacted the conversation, ``state.messages`` go through
    as they are (after your ``system``). The graph's own state is never
    changed: this only decides what the model sees.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. Optional: omit it to read
            ``config["configurable"]["conversation_id"]`` at call time. An
            explicit value wins over the config.
        system: The user's own system prompt. Stays authoritative for
            behaviour; the ST block is contextual framing prepended
            above it.
        style: Deprecated, kept for compatibility. Validated and sent to the
            SDK as before, but it has no effect on this helper's output: the
            helper reads the compaction block and the recent turns, which have
            one rendering.
        preamble_open / preamble_close: Wrapping tags for the ST block.
            Pass ``None`` for both to drop the tags and prepend raw
            text. Defaults to a ``<synap_short_term_context>`` XML pair.
        on_error: ``"fallback"`` (default) logs and returns the bare
            ``system`` prompt on SDK failure (or a missing conversation id);
            ``"raise"`` propagates :class:`SynapIntegrationError`.
        include_recent_turns: For a graph that does NOT keep its own message
            list (it sends every turn to Synap and reads them back). When
            ``True``, Synap's recent turns (``recent_messages``) are added to
            the system message in a ``<synap_recent_turns>`` block, with or
            without a compaction, and the graph's messages go through
            unchanged instead of being spliced. A trailing user turn that is
            the same as the graph's last user message is left out so the
            current message is not sent twice. Default ``False``.

    Returns:
        Async callable accepted by ``create_react_agent(prompt=...)``.

    Raises:
        ValueError: On invalid construction args (caught at agent build
        time, not at first LLM call).

    Example::

        from langgraph.prebuilt import create_react_agent
        from maximem_synap import MaximemSynapSDK
        from synap_langgraph import synap_st_prompt

        sdk = MaximemSynapSDK()
        agent = create_react_agent(
            model="anthropic:claude-3-5-sonnet-20241022",
            tools=[...],
            prompt=synap_st_prompt(sdk, system="You are a support agent."),
        )
        await agent.ainvoke(
            {"messages": [("user", "hi")]},
            config={"configurable": {"conversation_id": "conv_abc123"}},
        )
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_prompt"
    )
    site = "synap_langgraph.synap_st_prompt"

    async def _prompt(
        state: Any, config: Optional[RunnableConfig] = None,
    ) -> List[BaseMessage]:
        messages = _get_state_messages(state)
        cid = _resolve_conversation_id(conversation_id, config, on_error, site)
        response = (
            await _fetch_context(sdk, cid, style, on_error, site) if cid else None
        )
        history = None if response is None else _history_from_context(response)
        recent_block = ""
        st_block = ""
        if include_recent_turns:
            turns = _recent_turns(response) if response is not None else []
            last_user = next((message_text(m) for m in reversed(messages)
                              if role_of(m) == "user"), None)
            if (turns and turns[-1]["role"] == "user" and last_user is not None
                    and _norm(turns[-1]["content"]) == _norm(last_user)):
                turns = turns[:-1]  # the graph already holds the current message
            if turns:
                lines = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
                recent_block = f"{_RECENT_OPEN}\n{lines}\n{_RECENT_CLOSE}"
            if history is not None:
                st_block = _inner(history.preamble)
        elif history is not None:
            messages = _splice(messages, history)
            st_block = _inner(history.preamble)
        combined = _compose_system_prompt(
            st_block, system, preamble_open, preamble_close, recent_block
        )
        if combined:
            return [SystemMessage(content=combined), *messages]
        # No ST, no user system — pass messages through unchanged so the
        # graph's default behaviour is preserved.
        return messages

    _prompt.__name__ = "synap_st_prompt"
    return _prompt


# ---------------------------------------------------------------------------
# Public surface — Option 2: state-mutating node for custom StateGraphs
# ---------------------------------------------------------------------------


def create_synap_st_node(
    sdk: MaximemSynapSDK,
    conversation_id: Optional[str] = None,
    *,
    state_key: str = "synap_st",
    style: str = "narrative",
    on_error: _OnError = "fallback",
    include_recent_turns: bool = False,
    recent_key: str = "synap_recent_turns",
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Return an async node that writes Synap's short-term context into state.

    Place this node before your LLM node in a custom ``StateGraph``.
    The LLM node then reads ``state[state_key]`` and composes its own
    prompt — useful when you need to do something more than "prepend ST
    above one static system prompt" (e.g. multi-LLM flows, conditional
    routing, per-step system prompts, structured-output flows that need
    ST inline rather than in the system slot).

    The node always writes ``state[state_key]``. The value is:

    - Synap's compaction of the conversation (its summary of the turns so
      far, without the tags) once the conversation has been compacted;
    - ``""`` before the first compaction (the recent turns are NOT in this
      string; ask for them with ``include_recent_turns``), so downstream code
      can ``if state["synap_st"]:`` uniformly;
    - ``""`` on SDK failure with the default ``on_error="fallback"``;
      otherwise :class:`SynapIntegrationError` is raised.

    With ``include_recent_turns=True`` the node also writes
    ``state[recent_key]``: Synap's recent turns (the turns after the
    compaction, or all turns before the first one), oldest first, as a list of
    ``{"role": "user" | "assistant", "content": str}``. ``[]`` when there are
    none or on a failure under ``on_error="fallback"``. Declare that key in
    your state schema.

    The conversation is ``conversation_id`` when given, else
    ``config["configurable"]["conversation_id"]`` read when the node runs, so
    a graph compiled once serves many conversations. With neither, the node
    raises (``on_error="raise"``) or writes the empty values.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. Optional; an explicit value
            wins over the config.
        state_key: State dict key to write the ST string into.
            Defaults to ``"synap_st"``.
        style: Deprecated, kept for compatibility. Validated and sent to the
            SDK as before, but it has no effect on the value written.
        on_error: ``"fallback"`` (default) | ``"raise"``.
        include_recent_turns: Also write the recent turns to ``recent_key``.
        recent_key: State key for the recent turns. Defaults to
            ``"synap_recent_turns"``.

    Returns:
        Async node compatible with ``StateGraph.add_node``.

    Example::

        from langgraph.graph import StateGraph, START, END
        from synap_langgraph import create_synap_st_node

        graph = StateGraph(MyState)
        graph.add_node("st", create_synap_st_node(sdk))
        graph.add_node("llm", my_llm_node)  # reads state["synap_st"]
        graph.add_edge(START, "st")
        graph.add_edge("st", "llm")
        graph.add_edge("llm", END)
        app = graph.compile()
        await app.ainvoke(
            {...}, config={"configurable": {"conversation_id": "conv_abc"}})
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "create_synap_st_node"
    )
    if not state_key or not str(state_key).strip():
        raise ValueError(
            "create_synap_st_node requires a non-empty state_key"
        )
    if include_recent_turns and (not recent_key or not str(recent_key).strip()):
        raise ValueError(
            "create_synap_st_node requires a non-empty recent_key"
        )
    site = "synap_langgraph.create_synap_st_node"

    async def _node(
        state: Dict[str, Any], config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        cid = _resolve_conversation_id(conversation_id, config, on_error, site)
        response = (
            await _fetch_context(sdk, cid, style, on_error, site) if cid else None
        )
        history = None if response is None else _history_from_context(response)
        update: Dict[str, Any] = {
            state_key: _inner(history.preamble) if history is not None else ""
        }
        if include_recent_turns:
            update[recent_key] = _recent_turns(response) if response is not None else []
        return update

    _node.__name__ = "synap_st_node"
    return _node


__all__ = [
    "synap_st_prompt",
    "create_synap_st_node",
]
