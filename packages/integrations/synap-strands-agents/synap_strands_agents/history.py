"""Put Synap's compaction where Strands builds the model's input.

Strands has no hook that may edit the messages a model call is sent
(``BeforeModelCallEvent`` can only cancel) and a ``ConversationManager`` is
synchronous and rewrites ``agent.messages`` for good. What it does have is
middleware on the model call: an ``InvokeModelStage`` *input* handler receives
the messages about to be sent, as a copy, and returns the ones to send instead.
That is the seam used here: once Synap has compacted the conversation, every
turn before the current one is replaced, for that call only, by the compaction
block and the turns after it
(:func:`synap_integrations_common.splice_history`).

``agent.messages`` is not touched, so Strands' own conversation manager keeps
working on it exactly as before. It now has a short history to look at and does
not fire; it is also what the model is sent whenever Synap has nothing.

The middleware is installed on first use, from the ``BeforeInvocationEvent``
callback of :class:`SynapShortTermHook` and :class:`SynapStreamHook`, because a
hook is given the registry and not the agent, and the agent is what takes
middleware. Whichever hook runs first installs it; the second finds it there.

Strands moved its middleware out of a private module in 1.62. Both spellings
are tried; an install on a version with neither logs once and changes nothing.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from synap_integrations_common import fetch_compacted_history, splice_history

from synap_strands_agents._util import message_text

logger = logging.getLogger(__name__)

_MARK = "_synap_history_installed"
_STATE_KEY = "synap_compacted_history"
_warned = False


def _role(message: Dict[str, Any]) -> str:
    if message.get("role") == "assistant":
        return "assistant"
    content = message.get("content") or []
    if content and all(isinstance(b, dict) and "toolResult" in b for b in content):
        return "other"  # a tool result travels as a user message but is not a turn
    return "user"


def _make(role: str, text: str) -> Dict[str, Any]:
    # Strands has no system role: the compaction block rides in a user message,
    # which also keeps the system prompt, and anything cached on it, as it was.
    return {"role": "assistant" if role == "assistant" else "user", "content": [{"text": text}]}


def _is_text_only(message: Dict[str, Any]) -> bool:
    return all(isinstance(b, dict) and "text" in b for b in message.get("content") or [])


def _alternate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge neighbours of the same role, which several providers refuse.

    Only text-only messages are merged. The compaction block is a user message
    and Synap's first turn after it usually is too.
    """
    out: List[Dict[str, Any]] = []
    for message in messages:
        prev = out[-1] if out else None
        if (prev is not None and prev["role"] == message["role"]
                and _is_text_only(prev) and _is_text_only(message)):
            out[-1] = {**prev, "content": [*prev["content"], *message["content"]]}
        else:
            out.append(message)
    return out


async def _compacted(context: Any, sdk: Any, conversation_id: str) -> Any:
    """One fetch per invocation: a tool loop makes several model calls."""
    state = getattr(context, "invocation_state", None)
    if isinstance(state, dict) and _STATE_KEY in state:
        return state[_STATE_KEY]
    history = await fetch_compacted_history(sdk, conversation_id)
    if isinstance(state, dict):
        state[_STATE_KEY] = history
    return history


def _stage() -> Optional[Any]:
    try:
        from strands.middleware import InvokeModelStage
    except ImportError:
        try:
            from strands._middleware import InvokeModelStage  # type: ignore[no-redef]
        except ImportError:
            return None
    return InvokeModelStage


def install_synap_history(agent: Any, sdk: Any, conversation_id: str) -> bool:
    """Make the agent's model calls carry Synap's compaction. Idempotent per
    agent; returns whether it is installed afterwards. Never raises."""
    global _warned
    try:
        if vars(agent).get(_MARK):
            return True
        stage = _stage()
        add = getattr(agent, "add_middleware", None)
        if add is None:
            registry = getattr(agent, "_middleware_registry", None)
            add = getattr(registry, "add_middleware", None)
        if stage is None or add is None:
            if not _warned:
                _warned = True
                logger.warning(
                    "synap strands: this version of strands-agents has no model-call "
                    "middleware, so Synap's compaction is not applied.")
            return False

        async def synap_history(context: Any) -> Any:
            history = await _compacted(context, sdk, conversation_id)
            if history is None:
                return context
            try:
                spliced = splice_history(
                    list(context.messages), history,
                    role_of=_role, text_of=message_text, make_message=_make,
                )
                return dataclasses.replace(context, messages=_alternate(spliced))
            except Exception:  # noqa: BLE001 — the framework's own history still stands
                logger.warning("synap strands: splice failed (conversation_id=%s)",
                               conversation_id, exc_info=True)
                return context

        add(stage.Input, synap_history)
        agent._synap_history_installed = True
        return True
    except Exception:  # noqa: BLE001 — never break the caller's agent
        logger.warning("synap strands: could not install the history middleware", exc_info=True)
        return False


__all__ = ["install_synap_history"]
