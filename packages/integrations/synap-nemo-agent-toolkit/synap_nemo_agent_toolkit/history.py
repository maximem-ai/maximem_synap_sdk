"""Put Synap's compaction where a NAT agent reads its conversation.

NAT agents are handed the conversation as a ``ChatRequest`` (or a
``ChatRequestOrMessage``) and keep the last ``max_history`` messages of it
(``trim_messages`` in ``react_agent`` and ``tool_calling_agent``). Nothing in
the agent picks the history from anywhere else, so the one place to change what
it sees is its input. NAT's function middleware wraps a function call and may
return a different input, which is what :class:`SynapHistoryMiddleware` does:
once Synap has compacted the conversation, every message before the current one
is replaced by the compaction block and the turns after it
(:func:`synap_integrations_common.splice_history`).

The agent's own trimming stays in place. It now sees a short list and does not
cut anything, and it is what the agent has when Synap has nothing.

Wire it in YAML, on the agent::

    middleware:
      synap_history:
        _type: synap_history

    functions:
      my_agent:
        _type: tool_calling_agent
        middleware: [synap_history]

The conversation id is NAT's own for the run (set by the HTTP front end or by
``session.run``), else the configured ``conversation_id``. With neither, the
input is left as it is.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.cli.register_workflow import register_middleware
from nat.data_models.api_server import Message, UserMessageContentRoleType
from nat.data_models.common import OptionalSecretStr
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import InvocationContext

from synap_integrations_common import fetch_compacted_history, splice_history

from synap_nemo_agent_toolkit._sdk import shared_sdk

logger = logging.getLogger(__name__)

_Role = UserMessageContentRoleType


def _role(message: Message) -> str:
    if message.role == _Role.SYSTEM:
        return "system"
    if message.role == _Role.USER:
        return "user"
    if message.role == _Role.ASSISTANT:
        return "assistant"
    return "other"


def _text(message: Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI-style content parts
        return "\n".join(
            str(getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else "") or "")
            for part in content
        ).strip()
    return ""


def _make(role: str, text: str) -> Message:
    return Message(role=_Role(role), content=text)


class SynapHistoryMiddlewareConfig(FunctionMiddlewareBaseConfig, name="synap_history"):
    """YAML-wired config for replacing an agent's earlier turns with Synap's compaction."""

    conversation_id: str = Field(
        default="",
        description="Fallback conversation id, used only when NAT's own is unset for the run.",
    )
    api_key: OptionalSecretStr = Field(
        default=None,
        description="Synap API key. When omitted, falls back to SYNAP_API_KEY env var.",
    )
    instance_id: str = Field(
        default="",
        description=(
            "Optional Synap instance ID. When empty the SDK resolves it from the API "
            "key via /auth/whoami. Falls back to SYNAP_INSTANCE_ID env."
        ),
    )


class SynapHistoryMiddleware(FunctionMiddleware):
    """Swap an agent's earlier turns for Synap's compaction, when Synap has one."""

    def __init__(self, sdk: Any, *, conversation_id: str = "") -> None:
        super().__init__()
        self._sdk = sdk
        self._conversation_id = conversation_id

    def _conversation(self) -> str:
        try:
            return ContextState.get().conversation_id.get() or self._conversation_id
        except Exception:  # noqa: BLE001 — no NAT context: use the configured one
            return self._conversation_id

    async def pre_invoke(self, context: InvocationContext) -> Optional[InvocationContext]:
        try:
            args = context.modified_args
            request = args[0] if args else None
            messages: Optional[List[Message]] = getattr(request, "messages", None)
            conversation_id = self._conversation()
            if not messages or not conversation_id:
                return None
            history = await fetch_compacted_history(self._sdk, conversation_id)
            if history is None:
                return None
            spliced = splice_history(
                messages, history, role_of=_role, text_of=_text, make_message=_make,
            )
            context.modified_args = (request.model_copy(update={"messages": spliced}), *args[1:])
            return context
        except Exception:  # noqa: BLE001 — the agent's own history still stands
            logger.warning("synap_history: splice failed", exc_info=True)
            return None

    async def post_invoke(self, context: InvocationContext) -> Optional[InvocationContext]:
        return None


@register_middleware(config_type=SynapHistoryMiddlewareConfig)
async def synap_history_middleware(config: SynapHistoryMiddlewareConfig, builder: Builder):
    """Yield the middleware on the process's shared Synap SDK."""
    async with shared_sdk(config) as sdk:
        yield SynapHistoryMiddleware(sdk, conversation_id=config.conversation_id)


__all__ = [
    "SynapHistoryMiddleware",
    "SynapHistoryMiddlewareConfig",
    "synap_history_middleware",
]
