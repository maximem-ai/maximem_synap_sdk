"""Synap memory plugin for NVIDIA NeMo Agent Toolkit (NAT).

Three integration paths:

1. **Programmatic** — construct a :class:`MaximemSynapSDK` yourself and
   wrap it in :class:`SynapMemoryEditor` to plug into any NAT surface
   that expects a :class:`nat.memory.interfaces.MemoryEditor`.

2. **YAML-wired** — declare ``_type: synap_memory`` in a NAT workflow's
   ``memory:`` block. NAT's plugin loader imports
   :mod:`synap_nemo_agent_toolkit.register` via the ``nat.components``
   entry-point, which pulls in :class:`SynapMemoryClientConfig` and the
   ``@register_memory`` factory.

3. **Stream reporting** — declare ``_type: synap_stream`` under
   ``general.telemetry.tracing`` and every event of every run (user
   message, tool call, tool result, reasoning, assistant message) is
   reported on Synap's live gRPC stream. See
   :mod:`synap_nemo_agent_toolkit.stream`.

Error policy (matches every other Synap integration):

- Reads (``search``) degrade gracefully — a Synap blip returns ``[]``
  rather than crashing the agent turn.
- Writes (``add_items``) surface as
  :class:`synap_integrations_common.SynapIntegrationError`.
- Deletes (``remove_items``) warn once and no-op — Synap has no public
  delete API.
"""

from synap_nemo_agent_toolkit.editor import SynapMemoryEditor
from synap_nemo_agent_toolkit.short_term import (
    SynapShortTermConfig,
    SynapShortTermFunction,
)
from synap_nemo_agent_toolkit.stream import (
    SynapStreamExporter,
    SynapStreamExporterConfig,
)

__all__ = [
    "SynapMemoryEditor",
    "SynapShortTermFunction",
    "SynapShortTermConfig",
    "SynapStreamExporter",
    "SynapStreamExporterConfig",
]
