"""Pytest conftest for synap-nemo-agent-toolkit tests."""

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

# Shared harness (per the distributor bar — no local re-implementation).
from synap_integrations_common.testing import (  # noqa: E402,F401
    failing_sdk,
    make_fact,
    make_unified_response,
    mock_sdk,
)
