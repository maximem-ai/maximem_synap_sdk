"""pytest conftest — re-exports shared harness fixtures.

The canonical source of truth for mock_sdk / failing_sdk and all
make_* factories is synap_integrations_common.testing.  Tests in this
package import from there via conftest so individual test modules don't
need to duplicate fixture wiring.
"""

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
