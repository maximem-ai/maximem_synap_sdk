"""Tests for the synap_microsoft_agent public surface.

Verifies that the package's __all__ exports are consistent and that
each public class is importable, correctly named, and is a proper subclass
of the expected MAF base classes.
"""

from __future__ import annotations

import synap_microsoft_agent


class TestPublicSurface:
    def test_all_defined(self):
        assert hasattr(synap_microsoft_agent, "__all__")
        assert isinstance(synap_microsoft_agent.__all__, list)

    def test_context_provider_exported(self):
        assert hasattr(synap_microsoft_agent, "SynapContextProvider")
        assert "SynapContextProvider" in synap_microsoft_agent.__all__

    def test_history_provider_exported(self):
        assert hasattr(synap_microsoft_agent, "SynapHistoryProvider")
        assert "SynapHistoryProvider" in synap_microsoft_agent.__all__

    def test_short_term_context_provider_exported(self):
        assert hasattr(synap_microsoft_agent, "SynapShortTermContextProvider")
        assert "SynapShortTermContextProvider" in synap_microsoft_agent.__all__

    def test_no_extra_undocumented_exports(self):
        """__all__ must have exactly the 3 documented exports and no extras."""
        expected = {
            "SynapContextProvider",
            "SynapHistoryProvider",
            "SynapShortTermContextProvider",
        }
        assert set(synap_microsoft_agent.__all__) == expected

    def test_context_provider_is_context_provider_subclass(self):
        from agent_framework import ContextProvider
        assert issubclass(synap_microsoft_agent.SynapContextProvider, ContextProvider)

    def test_history_provider_is_history_provider_subclass(self):
        from agent_framework import HistoryProvider
        assert issubclass(synap_microsoft_agent.SynapHistoryProvider, HistoryProvider)

    def test_short_term_context_provider_is_context_provider_subclass(self):
        from agent_framework import ContextProvider
        assert issubclass(synap_microsoft_agent.SynapShortTermContextProvider, ContextProvider)

    def test_context_provider_importable_directly(self):
        from synap_microsoft_agent import SynapContextProvider
        assert SynapContextProvider is not None

    def test_history_provider_importable_directly(self):
        from synap_microsoft_agent import SynapHistoryProvider
        assert SynapHistoryProvider is not None

    def test_short_term_importable_directly(self):
        from synap_microsoft_agent import SynapShortTermContextProvider
        assert SynapShortTermContextProvider is not None

    def test_context_provider_has_default_source_id(self):
        from synap_microsoft_agent import SynapContextProvider
        assert hasattr(SynapContextProvider, "DEFAULT_SOURCE_ID")
        assert SynapContextProvider.DEFAULT_SOURCE_ID == "synap"

    def test_history_provider_has_default_source_id(self):
        from synap_microsoft_agent import SynapHistoryProvider
        assert hasattr(SynapHistoryProvider, "DEFAULT_SOURCE_ID")
        assert SynapHistoryProvider.DEFAULT_SOURCE_ID == "synap_history"

    def test_short_term_has_default_source_id(self):
        from synap_microsoft_agent import SynapShortTermContextProvider
        assert hasattr(SynapShortTermContextProvider, "DEFAULT_SOURCE_ID")
        assert SynapShortTermContextProvider.DEFAULT_SOURCE_ID == "synap_short_term"
