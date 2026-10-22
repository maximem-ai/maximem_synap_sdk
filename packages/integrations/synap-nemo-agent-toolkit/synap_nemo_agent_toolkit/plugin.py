"""NAT ``register_memory`` factory for YAML-wired Synap memory.

Drop ``_type: synap_memory`` into a NAT workflow's ``memory:`` block and
the toolkit will construct a :class:`SynapMemoryEditor` from env vars
(``SYNAP_API_KEY`` / optional ``SYNAP_INSTANCE_ID``). Config fields let
operators override the scope, fetch mode, and document type from YAML. The SDK
is the process's shared one for those credentials, so the stream the
``synap_stream`` exporter opens serves this editor's reads too.

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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.common import OptionalSecretStr
from nat.data_models.memory import MemoryBaseConfig

from synap_nemo_agent_toolkit._sdk import shared_sdk
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
    """Yield a :class:`SynapMemoryEditor` on the process's shared Synap SDK."""
    async with shared_sdk(config) as sdk:
        yield SynapMemoryEditor(
            sdk=sdk,
            customer_id=config.customer_id,
            mode=config.mode,
            document_type=config.document_type,
        )


__all__ = ["SynapMemoryClientConfig", "synap_memory_client"]
