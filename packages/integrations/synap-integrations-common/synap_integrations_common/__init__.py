"""Shared utilities for Synap framework integrations."""

from synap_integrations_common.async_bridge import run_async, run_blocking
from synap_integrations_common.errors import (
    SynapIntegrationError,
    wrap_sdk_errors,
    wrap_sdk_errors_async,
)
from synap_integrations_common.history import (
    COMPACTION_PREFIX,
    CompactedHistory,
    fetch_compacted_history,
    is_synap_history,
    splice_history,
)
from synap_integrations_common.scope import default_scope
from synap_integrations_common.stream_events import (
    end_session,
    ensure_stream,
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    set_auto_listen,
    stream_is_active,
)

__all__ = [
    "run_async",
    "run_blocking",
    "SynapIntegrationError",
    "wrap_sdk_errors",
    "wrap_sdk_errors_async",
    "default_scope",
    "stream_is_active",
    "ensure_stream",
    "set_auto_listen",
    "report_turn",
    "record_turn",
    "report_tool_call",
    "report_tool_result",
    "report_reasoning",
    "end_session",
    "COMPACTION_PREFIX",
    "CompactedHistory",
    "fetch_compacted_history",
    "is_synap_history",
    "splice_history",
]
