"""Shared utilities for Synap framework integrations."""

from synap_integrations_common.async_bridge import run_async
from synap_integrations_common.errors import (
    SynapIntegrationError,
    wrap_sdk_errors,
    wrap_sdk_errors_async,
)
from synap_integrations_common.scope import default_scope
from synap_integrations_common.stream_events import (
    end_session,
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    stream_is_active,
)

__all__ = [
    "run_async",
    "SynapIntegrationError",
    "wrap_sdk_errors",
    "wrap_sdk_errors_async",
    "default_scope",
    "stream_is_active",
    "report_turn",
    "report_tool_call",
    "report_tool_result",
    "report_reasoning",
    "end_session",
]
