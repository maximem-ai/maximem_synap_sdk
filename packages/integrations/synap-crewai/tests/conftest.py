"""Pytest conftest — re-exports shared fixtures from the common harness.

The shared harness (synap_integrations_common.testing) is the canonical
source of truth for mock_sdk, failing_sdk, and make_* factories.
The local _helpers.py is kept for backward compatibility.
"""

from synap_integrations_common.testing import (  # noqa: F401
    mock_sdk,
    failing_sdk,
    make_fact,
    make_preference,
    make_episode,
    make_emotion,
    make_temporal_event,
    make_unified_response,
)
