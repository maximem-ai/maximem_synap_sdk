"""Pytest conftest for synap-harnessx tests.

Re-exports shared fixtures from the canonical harness so tests can request
``mock_sdk`` and ``failing_sdk`` without local duplication, the same pattern as
synap-deepagents and synap-langchain.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"),
)

from synap_integrations_common.testing import (  # noqa: F401, E402
    mock_sdk,
    failing_sdk,
    make_unified_response,
)
