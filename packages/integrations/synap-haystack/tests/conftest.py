"""Pytest conftest — imports shared harness fixtures.

The shared harness (synap_integrations_common.testing) is the canonical
source of truth for mock_sdk and failing_sdk fixtures plus make_* factories.
"""

import sys
import os

# Ensure paths are visible (mirrors the pytest invocation's PYTHONPATH)
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "../../synap-integrations-common"))

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
