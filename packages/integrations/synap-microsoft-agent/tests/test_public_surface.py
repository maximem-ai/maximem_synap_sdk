"""Tests for the synap_microsoft_agent public surface.

Verifies that the package's __all__ exports are consistent and that
each public class is importable, correctly named, and is a proper subclass
of the expected MAF base classes.
"""

from __future__ import annotations

import pytest

import synap_microsoft_agent
from tests._harness import requires_harness


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
        """__all__ must have exactly the documented exports and no extras."""
        expected = {
            # SDK surfaces — importable on agent-framework>=1.0
            "SynapContextProvider",
            "SynapHistoryProvider",
            "SynapShortTermContextProvider",
            # Harness surfaces — resolved lazily, need agent-framework>=1.13
            "SynapMemoryStore",
            "SynapMemoryContextProvider",
            "SynapAgentFileStore",
            "create_synap_harness_memory",
            "session_conversation_id",
            "TopicRecordStore",
            "InMemoryTopicRecordStore",
        }
        assert set(synap_microsoft_agent.__all__) == expected

    def test_unknown_attribute_still_raises_attribute_error(self):
        """The lazy ``__getattr__`` must not swallow genuine typos."""
        with pytest.raises(AttributeError):
            synap_microsoft_agent.SynapNotAThing

    def test_dir_lists_every_export(self):
        assert set(dir(synap_microsoft_agent)) == set(synap_microsoft_agent.__all__)

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


class TestPrivateImports:
    """D5: import harness names from ``agent_framework``, never ``_harness``.

    The underscore paths are private and MAF's own feature-stage docstring
    warns its members may move. A grep is the only thing that catches a
    regression here, because a private import works fine right up until the
    release that reorganises it.
    """

    def test_no_private_agent_framework_imports(self):
        import pathlib

        package = pathlib.Path(synap_microsoft_agent.__file__).parent
        # Match import statements only. The name appears in prose in
        # `_harness_compat.py`, explaining why we do not do this — a substring
        # check would flag its own documentation.
        offenders = [
            f"{path.name}:{number}"
            for path in sorted(package.glob("*.py"))
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if line.lstrip().startswith(("from agent_framework._", "import agent_framework._"))
        ]
        assert offenders == [], f"private agent_framework imports: {offenders}"


@requires_harness
class TestHarnessSurface:
    """The harness exports resolve, and are what they claim to be."""

    def test_memory_store_is_a_memory_store(self):
        from agent_framework import MemoryStore

        assert issubclass(synap_microsoft_agent.SynapMemoryStore, MemoryStore)

    def test_file_store_is_an_agent_file_store(self):
        from agent_framework import AgentFileStore

        assert issubclass(synap_microsoft_agent.SynapAgentFileStore, AgentFileStore)

    def test_memory_context_provider_is_a_history_provider(self):
        from agent_framework import HistoryProvider, MemoryContextProvider

        assert issubclass(synap_microsoft_agent.SynapMemoryContextProvider, HistoryProvider)
        assert issubclass(
            synap_microsoft_agent.SynapMemoryContextProvider, MemoryContextProvider
        )

    def test_factory_is_callable(self):
        assert callable(synap_microsoft_agent.create_synap_harness_memory)

    def test_every_abstract_method_is_implemented(self):
        assert not synap_microsoft_agent.SynapMemoryStore.__abstractmethods__
        assert not synap_microsoft_agent.SynapAgentFileStore.__abstractmethods__

    def test_experimental_warning_is_not_suppressed(self):
        """If MAF wants callers to know the surface is unstable, we must not
        hide it on their behalf (D6).

        MAF raises ``ExperimentalWarning`` when the class is *subclassed*, so
        it fires once at import and cannot be caught around a constructor
        call. The invariant worth pinning is therefore that this package never
        installs a warning filter — a suppression added later would be silent
        and permanent.
        """
        import pathlib

        package = pathlib.Path(synap_microsoft_agent.__file__).parent
        offenders = [
            f"{path.name}:{number}"
            for path in sorted(package.glob("*.py"))
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if "filterwarnings" in line or "simplefilter" in line
        ]
        assert offenders == [], f"warning filters found: {offenders}"
