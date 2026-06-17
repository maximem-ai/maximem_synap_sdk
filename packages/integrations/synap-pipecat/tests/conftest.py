"""Pytest conftest for synap-pipecat tests.

Re-exports shared fixtures from synap_integrations_common.testing so that
any test file in this package can declare ``mock_sdk`` / ``failing_sdk``
as a fixture parameter and receive a pre-wired SDK mock without importing
the shared harness directly.
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
