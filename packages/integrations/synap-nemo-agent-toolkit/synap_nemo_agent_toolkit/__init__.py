"""Synap memory plugin for NVIDIA NeMo Agent Toolkit (NAT).

Two integration paths:

1. **Programmatic** — construct a :class:`MaximemSynapSDK` yourself and
   wrap it in :class:`SynapMemoryEditor` to plug into any NAT surface
   that expects a :class:`nat.memory.interfaces.MemoryEditor`.

2. **YAML-wired** — declare ``_type: synap_memory`` in a NAT workflow's
   ``memory:`` block. NAT's plugin loader imports
   :mod:`synap_nemo_agent_toolkit.register` via the ``nat.components``
   entry-point, which pulls in :class:`SynapMemoryClientConfig` and the
   ``@register_memory`` factory.

Error policy (matches every other Synap integration):

- Reads (``search``) degrade gracefully — a Synap blip returns ``[]``
  rather than crashing the agent turn.
- Writes (``add_items``) surface as
  :class:`synap_integrations_common.SynapIntegrationError`.
- Deletes (``remove_items``) warn once and no-op — Synap has no public
  delete API.
"""

from synap_nemo_agent_toolkit.editor import SynapMemoryEditor

__all__ = ["SynapMemoryEditor"]
