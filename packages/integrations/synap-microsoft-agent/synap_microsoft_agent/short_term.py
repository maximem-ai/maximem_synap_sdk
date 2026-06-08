"""Synap short-term context for Microsoft Agent Framework (MAF).

Adds :class:`SynapShortTermContextProvider`, a MAF ``ContextProvider``
that injects Synap **short-term** context (compacted summary + recent
turns, per conversation) on every ``before_run``. Composes alongside
the existing :class:`SynapContextProvider` (long-term semantic memory)
— register both providers on the agent to get both kinds of context.

Wraps ``sdk.conversation.context.get_context_for_prompt`` (cache-first
behind ``SYNAP_SDK_ST_AUTHORITATIVE``).

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- ``before_run`` never raises by default (``on_error="fallback"``):
  logged via :class:`SynapIntegrationError`, no instructions extended.
  ``on_error="raise"`` propagates.
- Empty ST is a no-op — never extends instructions with blank content.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal, Optional

from agent_framework import ContextProvider
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]


class SynapShortTermContextProvider(ContextProvider):
    """Inject Synap short-term context on every agent turn."""

    DEFAULT_SOURCE_ID: ClassVar[str] = "synap_short_term"

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
    ) -> None:
        if sdk is None:
            raise ValueError(
                "SynapShortTermContextProvider requires a non-None sdk"
            )
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapShortTermContextProvider requires a non-empty conversation_id"
            )
        if style not in _SUPPORTED_STYLES:
            raise ValueError(
                f"SynapShortTermContextProvider: unsupported style={style!r}; "
                f"expected one of {_SUPPORTED_STYLES}"
            )
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                "SynapShortTermContextProvider: on_error must be 'fallback' "
                f"or 'raise', got {on_error!r}"
            )

        super().__init__(source_id)
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.style = style
        self.preamble_open = preamble_open
        self.preamble_close = preamble_close
        self.on_error: _OnError = on_error

    async def before_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Fetch ST from Synap and append to the agent's instructions."""
        st_block = ""
        try:
            async with wrap_sdk_errors_async(
                "synap_microsoft_agent.SynapShortTermContextProvider.before_run",
                logger,
                conversation_id=self.conversation_id,
                style=self.style,
            ):
                response = await self.sdk.conversation.context.get_context_for_prompt(
                    conversation_id=self.conversation_id,
                    style=self.style,
                )
            if getattr(response, "available", False):
                st_block = (getattr(response, "formatted_context", None) or "").strip()
        except SynapIntegrationError:
            if self.on_error == "raise":
                raise
            return

        if not st_block:
            return

        if self.preamble_open and self.preamble_close:
            wrapped = f"{self.preamble_open}\n{st_block}\n{self.preamble_close}"
        else:
            wrapped = st_block

        context.extend_instructions(self.source_id, wrapped)


__all__ = ["SynapShortTermContextProvider"]
