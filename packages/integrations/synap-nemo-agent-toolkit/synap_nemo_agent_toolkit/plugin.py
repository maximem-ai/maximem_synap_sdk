"""NAT ``register_memory`` factory for YAML-wired Synap memory.

Drop ``_type: synap_memory`` into a NAT workflow's ``memory:`` block and
the toolkit will construct a :class:`SynapMemoryEditor` from env vars
(``SYNAP_API_KEY`` / optional ``SYNAP_INSTANCE_ID``). Config fields let
operators override the scope, fetch mode, and document type from YAML.

Example YAML::

    memory:
      synap:
        _type: synap_memory
        customer_id: "acme"
        mode: "accurate"

Programmatic callers that want to share a pre-constructed SDK should
instantiate :class:`SynapMemoryEditor` directly instead.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.common import OptionalSecretStr, get_secret_value
from nat.data_models.memory import MemoryBaseConfig

from maximem_synap import MaximemSynapSDK

from synap_nemo_agent_toolkit.editor import SynapMemoryEditor

logger = logging.getLogger(__name__)


class SynapMemoryClientConfig(MemoryBaseConfig, name="synap_memory"):
    """YAML-wired config for the Synap memory provider."""

    customer_id: str = Field(
        default="",
        description="Optional customer/org scope. Empty means customer-less.",
    )
    mode: str = Field(
        default="accurate",
        description='Synap fetch mode: "accurate" or "fast".',
    )
    document_type: str = Field(
        default="ai-chat-conversation",
        description="document_type stamped on every Synap memories.create write.",
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


@register_memory(config_type=SynapMemoryClientConfig)
async def synap_memory_client(config: SynapMemoryClientConfig, builder: Builder):
    """Construct + initialize a Synap SDK, yield a :class:`SynapMemoryEditor`."""
    api_key: Optional[str] = get_secret_value(config.api_key) or os.environ.get(
        "SYNAP_API_KEY"
    )
    if api_key is None:
        raise RuntimeError(
            "Synap API key is not set. Provide it via SynapMemoryClientConfig.api_key "
            "or the SYNAP_API_KEY environment variable."
        )

    sdk = MaximemSynapSDK(
        instance_id=config.instance_id or os.environ.get("SYNAP_INSTANCE_ID", ""),
        api_key=api_key,
        _force_new=True,
    )
    await sdk.initialize()

    editor = SynapMemoryEditor(
        sdk=sdk,
        customer_id=config.customer_id,
        mode=config.mode,
        document_type=config.document_type,
    )
    try:
        yield editor
    finally:
        close = getattr(sdk, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 — teardown must not mask workflow errors
                logger.exception(
                    "synap_memory_client: SDK teardown raised (suppressing)"
                )


__all__ = ["SynapMemoryClientConfig", "synap_memory_client"]
