"""A user_id / customer_id that disagrees with ``scope_path`` is refused.

The server lets the path's ``user`` and ``customer`` win over flat ids, so a
mismatch would read and write as the path's user.
"""

from __future__ import annotations

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import RecordingSDK, make_ladder

from synap_google_adk import create_synap_callbacks, create_synap_tools, report_user_turn

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
B2C = {"customer": "dana", "user": "dana"}


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


SITES = {
    "tools": lambda sdk, **k: create_synap_tools(sdk, **k),
    "callbacks": lambda sdk, **k: create_synap_callbacks(sdk, conversation_id="c", **k),
}


@pytest.mark.parametrize("site", SITES)
def test_other_user_refused_with_no_sdk_call(sdk, site):
    with pytest.raises(SynapIntegrationError):
        SITES[site](sdk, user_id="bob", scope_path=PATH)
    assert list(sdk.calls) == []


@pytest.mark.parametrize("site", SITES)
def test_other_customer_refused_with_no_sdk_call(sdk, site):
    with pytest.raises(SynapIntegrationError):
        SITES[site](sdk, user_id="dana", customer_id="evil", scope_path=PATH)
    assert list(sdk.calls) == []


@pytest.mark.parametrize("site", SITES)
def test_match_empty_and_b2c_accepted(sdk, site):
    SITES[site](sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    SITES[site](sdk, user_id="dana", customer_id="", scope_path=PATH)
    SITES[site](sdk, user_id="dana", scope_path=B2C)
    SITES[site](sdk, user_id="dana", customer_id="dana", scope_path=B2C)


@pytest.mark.parametrize("site", SITES)
def test_no_path_never_refused(sdk, site):
    SITES[site](sdk, user_id="bob", customer_id="whatever")


async def test_report_user_turn_other_user_refused_with_no_sdk_call(sdk):
    with pytest.raises(SynapIntegrationError):
        await report_user_turn(sdk, "hi", conversation_id="c", user_id="bob", scope_path=PATH)
    with pytest.raises(SynapIntegrationError):
        await report_user_turn(
            sdk, "hi", conversation_id="c", user_id="dana", customer_id="evil", scope_path=PATH)
    assert list(sdk.calls) == []


async def test_report_user_turn_match_b2c_and_no_path_accepted(sdk):
    assert await report_user_turn(
        sdk, "hi", conversation_id="c", user_id="dana", customer_id="acme", scope_path=PATH)
    assert await report_user_turn(
        sdk, "hi", conversation_id="c", user_id="dana", scope_path=B2C)
    n = len(sdk.calls)
    assert await report_user_turn(sdk, "hi", conversation_id="c", user_id="bob")
    assert len(sdk.calls) == n + 1


async def test_no_path_tool_calls_byte_identical(sdk):
    search, store = create_synap_tools(sdk, user_id="alice", customer_id="acme")
    await search.func("q")
    await store.func("c")
    assert "scope_path" not in sdk.last_kwargs("fetch")
    assert "scope" not in sdk.last_kwargs("memories.create")
