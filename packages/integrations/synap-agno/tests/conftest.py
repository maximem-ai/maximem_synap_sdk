"""Pytest conftest for synap-agno tests.

Re-exports shared fixtures from the canonical harness so tests can request
``mock_sdk`` and ``failing_sdk`` without local duplication — same pattern as
synap-langchain.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"),
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"),
)

from synap_integrations_common.testing import (  # noqa: F401, E402
    mock_sdk,
    failing_sdk,
    make_fact,
    make_preference,
    make_episode,
    make_emotion,
    make_temporal_event,
    make_unified_response,
)
