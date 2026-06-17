"""Pytest conftest for synap-claude-agent tests.

Re-exports the shared test harness fixtures so every test module in this
package can use ``mock_sdk`` / ``failing_sdk`` and the ``make_*`` factories
without importing them individually.
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
