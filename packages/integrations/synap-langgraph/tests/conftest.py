"""Pytest conftest for synap-langgraph tests.

Adds the in-repo SDK + integration package paths so tests can run
without installing wheels. Re-exports shared fixtures from
synap_integrations_common.testing so every test module can use
``mock_sdk`` / ``failing_sdk`` and the ``make_*`` factories directly.
"""

import os
import sys

# In-repo SDK (private monorepo layout: synap/sdk/python)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"),
)
# This integration package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Shared integrations utilities
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"),
)
# Sister LangChain package (re-exported by __init__)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../synap-langchain"),
)

# Re-export shared fixtures so every test module can use them without
# explicitly importing from synap_integrations_common.testing.
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
