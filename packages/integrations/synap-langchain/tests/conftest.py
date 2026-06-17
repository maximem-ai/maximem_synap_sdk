"""Pytest conftest — imports fixtures from the shared harness and legacy helpers.

The shared harness (synap_integrations_common.testing) is the canonical source
of truth. The local _helpers.py is kept for backward compatibility with existing
tests that use its local mock_sdk fixture.
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
