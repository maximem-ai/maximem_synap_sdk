"""Pytest conftest — imports fixtures from the shared harness.

The shared harness (synap_integrations_common.testing) is the canonical
source of truth for mock_sdk / failing_sdk fixtures and make_* factories.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"))

from synap_integrations_common.testing import (  # noqa: F401
    mock_sdk,
    failing_sdk,
    make_fact,
    make_preference,
    make_unified_response,
)
