"""Synap short-term context for LangGraph.

Two composable surfaces, both wrapping
``sdk.conversation.context.get_context_for_prompt`` (which is cache-first
behind the ``SYNAP_SDK_ST_AUTHORITATIVE`` flag — see
``docs/internal/sdk_authoritative_short_term_context_plan.md``):

- :func:`synap_st_prompt` — returns an async callable suitable for
  ``create_react_agent(prompt=...)``. Prepends the Synap short-term
  context block above the user's system prompt and returns the full
  message list LangGraph expects. Use for **prebuilt** agents where the
  one-line drop-in matters.

- :func:`create_synap_st_node` — returns an async node suitable for
  ``StateGraph.add_node(...)``. Writes the ST string into
  ``state[state_key]`` for downstream LLM nodes to assemble however they
  want. Use for **custom graphs** where you control the prompt template.

Both honour the project-wide policy:

- The ``conversation_id`` is **required** at construction time — explicit
  caller-supplied. We deliberately do not infer from the LangGraph
  ``thread_id`` because the two namespaces can diverge.
- SDK failures **never** crash the graph by default
  (``on_error="fallback"``): we log via :class:`SynapIntegrationError`'s
  log path, then degrade to the user's bare system prompt /
  empty state slot. Pass ``on_error="raise"`` for strict environments.
- An empty short-term result (no compaction yet **and** no recent turns
  in the SDK cache) is treated as a no-op — it must not wipe the user's
  system prompt.

The wire shape (cache-first, server fallback, SDK-authoritative when the
flag is on) is owned entirely by the SDK helper. These adapters are thin
shape converters — they intentionally do NOT format, truncate, or
re-summarise.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

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


def _get_state_messages(state: Any) -> List[BaseMessage]:
    """Read ``messages`` from a state object that may be a dict OR an
    object with attribute access (LangGraph supports both shapes).
    """
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", None) or [])


def _validate_args(
    sdk: Optional[MaximemSynapSDK],
    conversation_id: str,
    style: str,
    on_error: str,
    site: str,
) -> None:
    if sdk is None:
        raise ValueError(f"{site} requires a non-None sdk")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError(
            f"{site} requires a non-empty conversation_id "
            f"(pass it explicitly per-run for multi-conversation agents)"
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


async def _fetch_st_block(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> str:
    """Fetch the formatted short-term context string.

    Returns an **empty string** when no short-term context is available
    OR when the SDK call fails and ``on_error="fallback"``. Raises
    :class:`SynapIntegrationError` when the SDK call fails and
    ``on_error="raise"``.
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
        return ""

    # The SDK marks ``available=False`` when there is no compaction yet
    # AND no recent turns — that's a legitimate empty result, not a
    # failure. Return empty so the caller can keep the user's system
    # prompt intact.
    if not getattr(response, "available", False):
        return ""
    formatted = getattr(response, "formatted_context", None)
    return (formatted or "").strip()


def _compose_system_prompt(
    st_block: str,
    user_system: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    """Combine ST block + user system prompt into a single string.

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
    if user_system:
        parts.append(user_system)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public surface — Option 1: prompt callable for create_react_agent
# ---------------------------------------------------------------------------


def synap_st_prompt(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Callable[[Any], Awaitable[List[BaseMessage]]]:
    """Return an async ``prompt`` callable for ``create_react_agent``.

    On every LLM step the callable:

    1. Awaits :func:`MaximemSynapSDK.conversation.context.get_context_for_prompt`
       (cache-first; near-zero overhead when the SDK store is warm).
    2. Prepends the ST block above ``system`` inside the configured
       preamble tags.
    3. Returns ``[SystemMessage(content=<combined>), *state.messages]``
       — the message list LangGraph passes straight to the model.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.** For
            multi-conversation agents construct one prompt callable per
            conversation (typically inside your per-run setup).
        system: The user's own system prompt. Stays authoritative for
            behaviour; the ST block is contextual framing prepended
            above it.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
            Defaults to ``"narrative"`` because that reads most cleanly
            inside a system prompt.
        preamble_open / preamble_close: Wrapping tags for the ST block.
            Pass ``None`` for both to drop the tags and prepend raw
            text. Defaults to a ``<synap_short_term_context>`` XML pair.
        on_error: ``"fallback"`` (default) logs and returns the bare
            ``system`` prompt on SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

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
            prompt=synap_st_prompt(
                sdk,
                conversation_id="conv_abc123",
                system="You are a helpful customer support agent.",
            ),
        )
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_prompt"
    )

    async def _prompt(state: Any) -> List[BaseMessage]:
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_langgraph.synap_st_prompt",
        )
        combined = _compose_system_prompt(
            st_block, system, preamble_open, preamble_close
        )
        messages = _get_state_messages(state)
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
    conversation_id: str,
    *,
    state_key: str = "synap_st",
    style: str = "narrative",
    on_error: _OnError = "fallback",
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Return an async node that writes the Synap ST string into state.

    Place this node before your LLM node in a custom ``StateGraph``.
    The LLM node then reads ``state[state_key]`` and composes its own
    prompt — useful when you need to do something more than "prepend ST
    above one static system prompt" (e.g. multi-LLM flows, conditional
    routing, per-step system prompts, structured-output flows that need
    ST inline rather than in the system slot).

    The node always writes ``state[state_key]``. The value is:

    - the formatted ST string when short-term context exists;
    - ``""`` when no ST is available yet (no compaction **and** no
      recent turns), so downstream code can ``if state["synap_st"]:``
      uniformly;
    - ``""`` on SDK failure with the default ``on_error="fallback"``;
      otherwise :class:`SynapIntegrationError` is raised.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        state_key: State dict key to write the ST string into.
            Defaults to ``"synap_st"``.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        on_error: ``"fallback"`` (default) | ``"raise"``.

    Returns:
        Async node compatible with ``StateGraph.add_node``.

    Example::

        from langgraph.graph import StateGraph, START, END
        from synap_langgraph import create_synap_st_node

        graph = StateGraph(MyState)
        graph.add_node("st", create_synap_st_node(sdk, "conv_abc"))
        graph.add_node("llm", my_llm_node)  # reads state["synap_st"]
        graph.add_edge(START, "st")
        graph.add_edge("st", "llm")
        graph.add_edge("llm", END)
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "create_synap_st_node"
    )
    if not state_key or not str(state_key).strip():
        raise ValueError(
            "create_synap_st_node requires a non-empty state_key"
        )

    async def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_langgraph.create_synap_st_node",
        )
        return {state_key: st_block}

    _node.__name__ = "synap_st_node"
    return _node


__all__ = [
    "synap_st_prompt",
    "create_synap_st_node",
]
