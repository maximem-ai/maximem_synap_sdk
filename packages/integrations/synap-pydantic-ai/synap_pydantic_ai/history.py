"""Put Synap's compaction where Pydantic AI's message history goes.

Pydantic AI sends the model whatever ``message_history`` the caller passes plus
the new prompt, and its one seam over that list is a *history processor*: a
callable run before every model request that returns the messages the model
will see. This module's processor asks Synap for the conversation as Synap
holds it and, once a compaction exists, replaces every turn before the current
one with the compaction block and the turns after it
(:func:`synap_integrations_common.splice_history`).

The processor is installed on the agent by :func:`register_synap_tools`,
:func:`register_synap_st_system_prompt` and :func:`synap_run`, so it is on by
default. Pydantic AI's own processors are left alone: they now see a short
history and do not fire, and they are still the answer when Synap has nothing.

Two things about how Pydantic AI treats the result are worth knowing. The
processed list replaces the run's history, so ``result.all_messages()`` carries
the compaction block; a caller who feeds it back as ``message_history`` gets it
replaced again by the next splice, never stacked. And the processor runs on
the first model request of a run only: a run that calls tools makes several,
and the answer cannot change between them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, List, Optional

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from synap_integrations_common import fetch_compacted_history, splice_history

from synap_pydantic_ai.deps import SynapDeps

logger = logging.getLogger(__name__)

_MARK = "_synap_history_processor"

# Once per process: an agent without the seam is a property of the install.
_warned_no_seam = False


def _text(value: Any) -> str:
    """The text of a prompt, which is a string or a sequence of user content."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(item for item in value if isinstance(item, str))
    return ""


def _role(message: ModelMessage) -> str:
    if isinstance(message, ModelResponse):
        return "assistant"
    if isinstance(message, ModelRequest):
        if any(isinstance(p, UserPromptPart) for p in message.parts):
            return "user"
        if message.parts and all(isinstance(p, SystemPromptPart) for p in message.parts):
            return "system"
    return "other"


def _message_text(message: ModelMessage) -> str:
    if isinstance(message, ModelResponse):
        return "\n".join(p.content for p in message.parts if isinstance(p, TextPart))
    if isinstance(message, ModelRequest):
        kept = (UserPromptPart, SystemPromptPart)
        return "\n".join(_text(p.content) for p in message.parts if isinstance(p, kept))
    return ""


def _make_message(role: str, text: str) -> ModelMessage:
    if role == "assistant":
        return ModelResponse(parts=[TextPart(content=text)])
    if role == "system":
        return ModelRequest(parts=[SystemPromptPart(content=text)])
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _split_system_parts(messages: List[ModelMessage]) -> List[ModelMessage]:
    """One system prompt per message, apart from the user prompt.

    The first request carries the system prompt and the first user prompt
    together, and Pydantic AI merges the compaction block into that request on
    the way back out. The splice treats a message as one role and tells its own
    block by what a message starts with, so pull them apart.
    """
    out: List[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest) and any(
            isinstance(p, SystemPromptPart) for p in message.parts
        ):
            rest = [p for p in message.parts if not isinstance(p, SystemPromptPart)]
            out.extend(ModelRequest(parts=[p]) for p in message.parts
                       if isinstance(p, SystemPromptPart))
            if rest:
                out.append(replace(message, parts=rest))
            continue
        out.append(message)
    return out


def synap_history_processor() -> Callable[..., Any]:
    """A Pydantic AI history processor that swaps in Synap's compaction.

    It reads ``sdk`` and ``conversation_id`` from the run's :class:`SynapDeps`.
    With no ``conversation_id``, with no compaction yet, and on any error, the
    messages are returned as they came.

    Example::

        agent = Agent("openai:gpt-4o", deps_type=SynapDeps,
                      history_processors=[synap_history_processor()])

    On Pydantic AI versions that replaced ``history_processors`` with
    capabilities, wrap it: ``capabilities=[ProcessHistory(synap_history_processor())]``.
    The register functions and :func:`synap_run` do this for you.
    """

    async def _synap_history(
        ctx: RunContext[SynapDeps], messages: List[ModelMessage],
    ) -> List[ModelMessage]:
        deps = ctx.deps
        sdk = getattr(deps, "sdk", None)
        conversation_id = getattr(deps, "conversation_id", None)
        if sdk is None or not conversation_id or ctx.run_step > 1:
            return messages
        history = await fetch_compacted_history(sdk, conversation_id)
        if history is None:
            return messages
        try:
            return splice_history(
                _split_system_parts(messages), history,
                role_of=_role, text_of=_message_text, make_message=_make_message,
            )
        except Exception:  # noqa: BLE001 — the framework's own history still stands
            logger.warning("synap pydantic-ai: splice failed (conversation_id=%s)",
                           conversation_id, exc_info=True)
            return messages

    setattr(_synap_history, _MARK, True)
    return _synap_history


def _is_ours(item: Any) -> bool:
    if getattr(item, _MARK, False):
        return True
    return getattr(getattr(item, "processor", None), _MARK, False)


def install_synap_history(agent: Any) -> bool:
    """Put :func:`synap_history_processor` on the agent. Idempotent; returns
    whether it is installed afterwards.

    Pydantic AI 1.x keeps a ``history_processors`` list on the agent. 2.x
    replaced it with capabilities and exposes the agent's as
    ``agent.root_capability.capabilities``, a list a ``ProcessHistory``
    capability can be appended to after construction.
    """
    try:
        processors: Optional[list] = getattr(agent, "history_processors", None)
        if isinstance(processors, list):
            if not any(_is_ours(p) for p in processors):
                processors.append(synap_history_processor())
            return True
        capabilities = getattr(getattr(agent, "root_capability", None), "capabilities", None)
        if isinstance(capabilities, list):
            if not any(_is_ours(c) for c in capabilities):
                from pydantic_ai.capabilities import ProcessHistory

                capabilities.append(ProcessHistory(synap_history_processor()))
            return True
    except Exception:  # noqa: BLE001 — never break the caller's agent set-up
        logger.warning("synap pydantic-ai: could not install the history processor",
                       exc_info=True)
        return False
    global _warned_no_seam
    if not _warned_no_seam:
        _warned_no_seam = True
        logger.warning(
            "synap pydantic-ai: this agent has no history_processors or capabilities, "
            "so Synap's compaction is not applied. Pass synap_history_processor() yourself.")
    return False


__all__ = ["synap_history_processor", "install_synap_history"]
