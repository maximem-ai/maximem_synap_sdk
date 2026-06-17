"""Pytest conftest for synap-microsoft-agent tests.

Re-exports shared fixtures from synap_integrations_common.testing so that
tests in this package can declare ``mock_sdk`` / ``failing_sdk`` as fixture
parameters without duplicating the wiring logic.
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
    make_unified_response,
)
