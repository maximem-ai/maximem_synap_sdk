"""Pytest conftest for synap-langgraph tests.

Adds the in-repo SDK + integration package paths so tests can run
without installing wheels. Mirrors the pattern in synap-langchain's
tests/_helpers.py.
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
