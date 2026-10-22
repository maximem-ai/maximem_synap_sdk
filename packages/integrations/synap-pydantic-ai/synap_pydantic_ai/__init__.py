"""Synap memory integration for Pydantic AI."""

from synap_pydantic_ai.deps import SynapDeps, register_synap_tools
from synap_pydantic_ai.history import install_synap_history, synap_history_processor
from synap_pydantic_ai.short_term import register_synap_st_system_prompt
from synap_pydantic_ai.stream import (
    SynapEventStreamHandler,
    synap_run,
    synap_run_sync,
)

__all__ = [
    "SynapDeps",
    "register_synap_tools",
    "register_synap_st_system_prompt",
    "synap_history_processor",
    "install_synap_history",
    "SynapEventStreamHandler",
    "synap_run",
    "synap_run_sync",
]
