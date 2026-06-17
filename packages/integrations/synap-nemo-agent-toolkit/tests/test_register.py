"""Tests for the YAML-wired entry point: register.py + SynapMemoryClientConfig.

``synap_nemo_agent_toolkit.register`` is the ``nat.components`` entry point;
importing it must register the memory provider + short-term function without
error. ``SynapMemoryClientConfig`` is the operator-facing YAML config.

Imports are at module top so they bind against the real ``nat`` toolkit at
collection time (this file sorts before ``test_short_term.py``, which stubs
``nat`` in sys.modules).
"""

from __future__ import annotations

import synap_nemo_agent_toolkit.register  # noqa: F401  — side effect: registers
from synap_nemo_agent_toolkit.plugin import SynapMemoryClientConfig


def test_register_entrypoint_imports_cleanly():
    # The import above triggers @register_memory + @register_function against
    # the real NAT registry; reaching here means registration did not raise.
    import synap_nemo_agent_toolkit.register as reg

    assert reg is not None


class TestClientConfig:
    def test_defaults(self):
        cfg = SynapMemoryClientConfig()
        assert cfg.customer_id == ""
        assert cfg.mode == "accurate"
        assert cfg.document_type == "ai-chat-conversation"
        assert cfg.instance_id == ""

    def test_overrides(self):
        cfg = SynapMemoryClientConfig(
            customer_id="acme", mode="fast", document_type="note"
        )
        assert cfg.customer_id == "acme"
        assert cfg.mode == "fast"
        assert cfg.document_type == "note"
