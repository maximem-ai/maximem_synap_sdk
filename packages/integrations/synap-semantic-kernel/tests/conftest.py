"""Pytest fixtures for synap-semantic-kernel.

Re-exports the SHARED harness (``mock_sdk`` / ``failing_sdk`` / response
factories) from ``synap_integrations_common.testing`` instead of
re-implementing it locally — per the distributor test bar
(`testing-suite/pass-criteria.md` → "Shared harness used").

The ``sys.path`` inserts make the sibling source packages importable when
the suite is run from a checkout without editable installs.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../../synap-integrations-common"))

from synap_integrations_common.testing import (  # noqa: E402,F401
    failing_sdk,
    make_emotion,
    make_episode,
    make_fact,
    make_preference,
    make_temporal_event,
    make_unified_response,
    mock_sdk,
)
