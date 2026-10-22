"""Synap's compaction in place of an agent's earlier steps.

Smolagents builds the model's input in one place: ``MultiStepAgent.write_memory_to_messages``
turns the system prompt and every memory step into messages, and each model call
(a step, a planning step, the final answer after max steps) goes through it.
There is no hook around that call and no compaction of its own: with
``agent.run(task, reset=False)`` the list only grows, and with the default
``reset=True`` each run starts from nothing.

:func:`attach_synap_history` wraps that method on one agent. Once Synap has
compacted the conversation, the model sees the compaction block, Synap's turns
after it, and the current task with its steps, instead of the earlier steps in
``agent.memory``. Until then (and whenever Synap cannot answer) the framework's
messages pass through untouched. ``agent.memory`` itself is never edited.

:func:`create_synap_recorder` attaches it on the first step it sees, by default.
That is after the first step of the first run, because smolagents has no hook
before a model call: call :func:`attach_synap_history` right after building the
agent to cover a resumed conversation's first step too.

The conversation is read once per run, not once per step.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from synap_integrations_common import fetch_compacted_history, run_async, splice_history
from synap_integrations_common.history import CompactedHistory

logger = logging.getLogger(__name__)

_ATTACHED = "_synap_history_attached"
# What TaskStep.to_messages puts in front of the task.
_TASK_PREFIX = "New task:\n"


def _role(message: Any) -> str:
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role))


def _text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _role_of(message: Any) -> str:
    """Smolagents sends several kinds of message as ``user``: the task, the
    "Now proceed and carry out this plan." that follows a plan, and tool
    observations with images. Only the task starts a turn, and the splice takes
    the last ``user`` message to be where the current turn begins."""
    role = _role(message)
    if role == "user":
        return "user" if _text(message).startswith(_TASK_PREFIX) else "other"
    return role if role in ("system", "assistant") else "other"


def _text_of(message: Any) -> str:
    text = _text(message)
    if _role(message) == "user" and text.startswith(_TASK_PREFIX):
        return text[len(_TASK_PREFIX):]
    return text


def _make_message(role: str, text: str) -> Any:
    from smolagents.models import ChatMessage, MessageRole

    return ChatMessage(
        role=MessageRole(role), content=[{"type": "text", "text": text}]
    )


def _current_task_step(agent: Any) -> Optional[Any]:
    """The TaskStep the current run appended; a new object for every run."""
    from smolagents.memory import TaskStep

    for step in reversed(list(getattr(getattr(agent, "memory", None), "steps", None) or [])):
        if isinstance(step, TaskStep):
            return step
    return None


def attach_synap_history(agent: Any, sdk: Any, conversation_id: str) -> bool:
    """Make ``agent`` send Synap's compaction instead of its own earlier steps.

    Idempotent. Returns whether the agent is now wrapped. Safe to call before the
    first ``run()``: nothing is read until the model is about to be called.
    """
    if getattr(agent, _ATTACHED, False):
        return True
    original = getattr(agent, "write_memory_to_messages", None)
    if not callable(original) or not conversation_id:
        return False

    # One read per run. Holding the TaskStep (not its id) keeps it from being
    # collected and its id reused by the next run's.
    cache: dict = {"task_step": None, "history": None}

    def history_for_run() -> Optional[CompactedHistory]:
        task_step = _current_task_step(agent)
        if task_step is not None and cache["task_step"] is task_step:
            return cache["history"]
        history = run_async(fetch_compacted_history(sdk, conversation_id))
        if task_step is not None:
            cache["task_step"], cache["history"] = task_step, history
        return history

    def write_memory_to_messages(*args: Any, **kwargs: Any) -> List[Any]:
        messages = original(*args, **kwargs)
        try:
            history = history_for_run()
            if history is None:
                return messages
            return splice_history(
                messages, history,
                role_of=_role_of, text_of=_text_of, make_message=_make_message,
            )
        except Exception:  # noqa: BLE001: the agent's own messages still stand
            logger.warning("synap: history splice failed (conversation_id=%s)",
                           conversation_id, exc_info=True)
            return messages

    try:
        agent.write_memory_to_messages = write_memory_to_messages
        setattr(agent, _ATTACHED, True)
    except Exception:  # noqa: BLE001
        logger.warning("synap: could not wrap write_memory_to_messages", exc_info=True)
        return False
    return True
