"""Synap short-term context for NVIDIA NeMo Agent Toolkit (NAT).

NAT is YAML-declarative: agents and pipelines are wired via workflow
config, not Python factory callables. The hook closest to "system-prompt
prep at every LLM step" is a registered **Function** — a YAML-callable
component whose output can be templated into a prompt or piped into
downstream nodes.

This module exposes :func:`synap_short_term_function`, registered via
``@register_function`` with config type
:class:`SynapShortTermConfig`. Drop ``_type: synap_short_term`` into a
NAT workflow's ``functions:`` block; the function fetches Synap
short-term context (cache-first via the SDK helper) and returns the
formatted string. Reference its output in your LLM's prompt template,
or chain it through other nodes.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` is required + explicit (configured per-call via
  the function's input, or pinned via the YAML config).
- Read failures degrade gracefully (``on_error="fallback"`` is the
  default for SDK fetch); the function returns an empty string on
  failure so downstream prompt templates render cleanly. Workflow
  authors can opt into hard-fail by setting ``on_error: raise``.
- Empty ST is a no-op — returns an empty string.

For programmatic use (no YAML), instantiate
:class:`SynapShortTermFunction` directly with a pre-constructed SDK and
call ``await func(conversation_id=...)``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import OptionalSecretStr, get_secret_value
from nat.data_models.function import FunctionBaseConfig

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


# ---------------------------------------------------------------------------
# Programmatic class (no NAT required at runtime — useful for direct usage
# and for unit tests)
# ---------------------------------------------------------------------------


class SynapShortTermFunction:
    """Async callable that returns Synap short-term context for a conversation.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers. Pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) returns ``""`` on SDK
            failure; ``"raise"`` propagates :class:`SynapIntegrationError`.
        default_conversation_id: Optional fallback used when
            ``__call__`` receives ``conversation_id=""``. Useful when
            the NAT workflow pins a single conversation; leave unset to
            require an explicit per-call ID.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
        default_conversation_id: str = "",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapShortTermFunction requires a non-None sdk")
        if style not in _SUPPORTED_STYLES:
            raise ValueError(
                f"SynapShortTermFunction: unsupported style={style!r}; "
                f"expected one of {_SUPPORTED_STYLES}"
            )
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                "SynapShortTermFunction: on_error must be 'fallback' or "
                f"'raise', got {on_error!r}"
            )

        self.sdk = sdk
        self.style = style
        self.preamble_open = preamble_open
        self.preamble_close = preamble_close
        self.on_error: _OnError = on_error
        self.default_conversation_id = default_conversation_id

    async def __call__(self, conversation_id: str = "") -> str:
        conv_id = conversation_id or self.default_conversation_id
        if not conv_id or not str(conv_id).strip():
            raise ValueError(
                "SynapShortTermFunction: conversation_id is required "
                "(pass it per-call or set default_conversation_id at construction)"
            )

        try:
            async with wrap_sdk_errors_async(
                "synap_nemo_agent_toolkit.SynapShortTermFunction",
                logger,
                conversation_id=conv_id,
                style=self.style,
            ):
                response = await self.sdk.conversation.context.get_context_for_prompt(
                    conversation_id=conv_id,
                    style=self.style,
                )
        except SynapIntegrationError:
            if self.on_error == "raise":
                raise
            return ""

        if not getattr(response, "available", False):
            return ""
        formatted = (getattr(response, "formatted_context", None) or "").strip()
        if not formatted:
            return ""
        if self.preamble_open and self.preamble_close:
            return f"{self.preamble_open}\n{formatted}\n{self.preamble_close}"
        return formatted


# ---------------------------------------------------------------------------
# NAT YAML-wired plugin
# ---------------------------------------------------------------------------


class SynapShortTermConfig(FunctionBaseConfig, name="synap_short_term"):
    """YAML-wired config for the Synap short-term context function.

    Example YAML::

        functions:
          synap_st:
            _type: synap_short_term
            conversation_id: "conv_abc123"   # pin per workflow OR pass per call
            style: "narrative"
            on_error: "fallback"
    """

    conversation_id: str = Field(
        default="",
        description=(
            "Default conversation_id for this function. When non-empty, the "
            "function uses this if no conversation_id input is supplied at "
            "call time. Pass the empty string to require per-call input."
        ),
    )
    style: str = Field(
        default="narrative",
        description="ST formatting style: structured | narrative | bullet_points.",
    )
    preamble_open: str = Field(
        default=_DEFAULT_OPEN,
        description=(
            "Opening tag wrapping the ST block. Set to empty string to drop tags."
        ),
    )
    preamble_close: str = Field(
        default=_DEFAULT_CLOSE,
        description=(
            "Closing tag wrapping the ST block. Set to empty string to drop tags."
        ),
    )
    on_error: str = Field(
        default="fallback",
        description=(
            "What to do on SDK failure: 'fallback' returns empty string; "
            "'raise' propagates SynapIntegrationError up the workflow."
        ),
    )
    api_key: OptionalSecretStr = Field(
        default=None,
        description=(
            "Synap API key. When omitted, falls back to SYNAP_API_KEY env var."
        ),
    )
    instance_id: str = Field(
        default="",
        description=(
            "Optional Synap instance ID. When empty the SDK resolves it from "
            "the API key via /auth/whoami. Falls back to SYNAP_INSTANCE_ID env."
        ),
    )


@register_function(config_type=SynapShortTermConfig)
async def synap_short_term_function(config: SynapShortTermConfig, builder: Builder):
    """Construct + initialize a Synap SDK, yield a Synap-ST async function."""
    api_key: Optional[str] = get_secret_value(config.api_key) or os.environ.get(
        "SYNAP_API_KEY"
    )
    if api_key is None:
        raise RuntimeError(
            "Synap API key is not set. Provide it via SynapShortTermConfig.api_key "
            "or the SYNAP_API_KEY environment variable."
        )

    sdk = MaximemSynapSDK(
        instance_id=config.instance_id or os.environ.get("SYNAP_INSTANCE_ID", ""),
        api_key=api_key,
        _force_new=True,
    )
    await sdk.initialize()

    if config.style not in _SUPPORTED_STYLES:
        raise ValueError(
            f"synap_short_term: unsupported style={config.style!r}; "
            f"expected one of {_SUPPORTED_STYLES}"
        )
    if config.on_error not in ("fallback", "raise"):
        raise ValueError(
            "synap_short_term: on_error must be 'fallback' or 'raise', "
            f"got {config.on_error!r}"
        )

    fn = SynapShortTermFunction(
        sdk=sdk,
        style=config.style,
        preamble_open=config.preamble_open or None,
        preamble_close=config.preamble_close or None,
        on_error=config.on_error,  # type: ignore[arg-type]
        default_conversation_id=config.conversation_id,
    )

    async def _invoke(conversation_id: str = "") -> str:
        """Return Synap short-term context for ``conversation_id``.

        When ``conversation_id`` is empty, the function falls back to
        the ``conversation_id`` set on the YAML config (if any).
        """
        return await fn(conversation_id=conversation_id)

    info = FunctionInfo.from_fn(
        _invoke,
        description=(
            "Fetch Synap short-term conversation context (compacted summary + "
            "recent turns) for a conversation_id. Returns the formatted string "
            "ready to splice into a prompt template, or empty string on miss."
        ),
    )

    try:
        yield info
    finally:
        close = getattr(sdk, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 — teardown must not mask workflow errors
                logger.exception(
                    "synap_short_term_function: SDK teardown raised (suppressing)"
                )


__all__ = [
    "SynapShortTermFunction",
    "SynapShortTermConfig",
    "synap_short_term_function",
]
