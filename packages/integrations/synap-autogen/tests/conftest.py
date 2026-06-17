"""Pytest conftest — re-exports shared fixtures and adds AutoGen-specific helpers.

The shared harness (synap_integrations_common.testing) is the canonical
source of truth for mock_sdk / failing_sdk and make_* factories.
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

import pytest
from autogen_core import CancellationToken


@pytest.fixture
def ct():
    """A fresh, non-cancelled :class:`CancellationToken`.

    Use this instead of ``MagicMock()`` when the tool under test calls
    ``cancellation_token.is_cancelled()`` — a MagicMock returns a truthy
    MagicMock for every attribute access, which makes every call look
    like the token was already cancelled.
    """
    return CancellationToken()


@pytest.fixture
def cancelled_ct():
    """A :class:`CancellationToken` that is already cancelled at creation."""
    token = CancellationToken()
    token.cancel()
    return token
