"""Pytest conftest for synap-openai-agents tests.

Re-exports fixtures and factories from the shared harness so every test file
can rely on a single, canonical source of truth.
"""

import sys, os
# Ensure local packages are importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"))

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
