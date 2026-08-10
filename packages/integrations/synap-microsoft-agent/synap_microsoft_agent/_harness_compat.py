"""Import the MAF harness names, or fail with a message that says why.

The harness (``MemoryStore``, ``AgentFileStore``, ``MemoryContextProvider``,
``FileMemoryProvider``) arrived long after ``agent-framework`` 1.0. This package
keeps its dependency floor at ``>=1.0`` because the SDK surfaces
(``SynapContextProvider``, ``SynapHistoryProvider``, ``SynapShortTermContextProvider``)
work there and raising the floor would break existing users for a feature they
are not using.

So the harness modules import through here. On an older install the
``ImportError`` becomes a message naming the version to upgrade to, in the
style MAF uses for its own optional connectors.

Every name is taken from the top-level ``agent_framework`` package, never from
``agent_framework._harness``. The underscore path is private and MAF's own
feature-stage docstring warns that these members may move.
"""

from __future__ import annotations

MINIMUM_HARNESS_VERSION = "1.13"

_MESSAGE = (
    "The Synap harness surfaces need the Microsoft Agent Framework harness API, "
    f"which requires agent-framework>={MINIMUM_HARNESS_VERSION}. "
    "Install it with `pip install 'agent-framework>=" + MINIMUM_HARNESS_VERSION + "'`. "
    "The SDK surfaces (SynapContextProvider, SynapHistoryProvider, "
    "SynapShortTermContextProvider) work on agent-framework>=1.0 and need no upgrade."
)


def require_harness() -> None:
    """Raise ``ImportError`` with an actionable message if the harness is absent."""
    try:
        import agent_framework  # noqa: F401

        getattr(agent_framework, "MemoryStore")
        getattr(agent_framework, "AgentFileStore")
    except (ImportError, AttributeError) as exc:
        raise ImportError(_MESSAGE) from exc


__all__ = ["require_harness", "MINIMUM_HARNESS_VERSION"]
