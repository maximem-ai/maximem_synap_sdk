"""Synap short-term context for Haystack pipelines.

Mirrors the LangGraph template, adapted to Haystack's
``@component``/``Pipeline`` model. Wraps
``sdk.conversation.context.get_context_for_prompt`` (cache-first behind
``SYNAP_SDK_ST_AUTHORITATIVE``).

Exposes :class:`SynapShortTermContext` — a component that emits the
formatted Synap short-term context string on its ``synap_st`` output.
Wire it before your prompt builder, then reference ``{synap_st}`` (or
your prompt builder's variable name) in the template. The component
re-fetches on every pipeline run, so ST stays fresh.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- Pipeline failures by default are swallowed — Haystack components that
  raise abort the whole pipeline. The default ``on_error="fallback"``
  emits the bare user system text (or empty string) on SDK failure,
  with an ERROR log. ``on_error="raise"`` surfaces failures.
- Empty ST is a no-op — never wipes the user's system text.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

from haystack import component
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]


@component
class SynapShortTermContext:
    """Haystack component emitting Synap short-term context as a string.

    Each pipeline run calls the SDK helper
    ``sdk.conversation.context.get_context_for_prompt`` (cache-first)
    and emits the combined ``<ST block>\\n\\n<system>`` string on the
    ``synap_st`` output. Pipe into a prompt builder to inject ST at
    every step.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        system: Static framing prepended below the ST block.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) emits the bare ``system`` on
            SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Example::

        from haystack import Pipeline
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import OpenAIGenerator
        from synap_haystack import SynapShortTermContext

        st = SynapShortTermContext(sdk, conversation_id="conv_abc",
                                   system="You are a helpful agent.")
        prompt = PromptBuilder(
            template="{{ synap_st }}\\n\\nQ: {{ question }}\\nA:"
        )
        gen = OpenAIGenerator(model="gpt-4o")

        pipe = Pipeline()
        pipe.add_component("st", st)
        pipe.add_component("prompt", prompt)
        pipe.add_component("gen", gen)
        pipe.connect("st.synap_st", "prompt.synap_st")
        pipe.connect("prompt.prompt", "gen.prompt")

        result = pipe.run({"question": "What's my next step?"})
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        *,
        system: str = "",
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapShortTermContext requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapShortTermContext requires a non-empty conversation_id"
            )
        if style not in _SUPPORTED_STYLES:
            raise ValueError(
                f"SynapShortTermContext: unsupported style={style!r}; "
                f"expected one of {_SUPPORTED_STYLES}"
            )
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                "SynapShortTermContext: on_error must be 'fallback' or "
                f"'raise', got {on_error!r}"
            )

        self.sdk = sdk
        self.conversation_id = conversation_id
        self.system = system
        self.style = style
        self.preamble_open = preamble_open
        self.preamble_close = preamble_close
        self.on_error: _OnError = on_error

    @component.output_types(synap_st=str)
    def run(self) -> Dict[str, str]:
        return run_async(self._arun())

    async def _arun(self) -> Dict[str, str]:
        st_block = ""
        try:
            async with wrap_sdk_errors_async(
                "synap_haystack.SynapShortTermContext.run",
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
            st_block = ""

        combined = _compose(
            st_block, self.system, self.preamble_open, self.preamble_close
        )
        return {"synap_st": combined}


def _compose(
    st_block: str,
    user_system: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
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


__all__ = ["SynapShortTermContext"]
