"""A user_id / customer_id that disagrees with ``scope_path`` is refused.

The server lets the path's ``user`` and ``customer`` win over flat ids, so a
mismatch would read and write as the path's user. Every construction point
refuses it before any SDK call.
"""

from unittest.mock import MagicMock

import pytest
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import RecordingSDK, make_ladder

from synap_deepagents import (
    SynapBackend,
    SynapMemoryMiddleware,
    SynapSearchTool,
    SynapStoreTool,
    SynapStreamMiddleware,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
B2C = {"customer": "dana", "user": "dana"}


def as_sdk(rec):
    m = MagicMock(spec=MaximemSynapSDK)
    m.fetch = rec.fetch
    m.memories = rec.memories
    return m


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


def _calls(sdk):
    return list(sdk.calls)


SITES = {
    "backend": lambda sdk, **k: SynapBackend(sdk, **k),
    "memory_mw": lambda sdk, **k: SynapMemoryMiddleware(sdk=sdk, **k),
    "stream_mw": lambda sdk, **k: SynapStreamMiddleware(
        sdk=sdk, conversation_id="c", user_id=k.pop("user_id", "dana"), **k),
    "search_tool": lambda sdk, **k: SynapSearchTool(sdk=as_sdk(sdk), **k),
    "store_tool": lambda sdk, **k: SynapStoreTool(sdk=as_sdk(sdk), **k),
}


@pytest.mark.parametrize("site", SITES)
def test_other_user_refused_with_no_sdk_call(sdk, site):
    with pytest.raises(SynapIntegrationError):
        SITES[site](sdk, user_id="bob", scope_path=PATH)
    assert _calls(sdk) == []


@pytest.mark.parametrize("site", SITES)
def test_other_customer_refused_with_no_sdk_call(sdk, site):
    with pytest.raises(SynapIntegrationError):
        SITES[site](sdk, user_id="dana", customer_id="evil", scope_path=PATH)
    assert _calls(sdk) == []


@pytest.mark.parametrize("site", SITES)
def test_matching_ids_accepted(sdk, site):
    SITES[site](sdk, user_id="dana", customer_id="acme", scope_path=PATH)


@pytest.mark.parametrize("site", SITES)
def test_empty_customer_accepted(sdk, site):
    SITES[site](sdk, user_id="dana", customer_id="", scope_path=PATH)
    SITES[site](sdk, user_id="dana", scope_path=PATH)


@pytest.mark.parametrize("site", SITES)
def test_b2c_collapse_accepted(sdk, site):
    SITES[site](sdk, user_id="dana", scope_path=B2C)
    SITES[site](sdk, user_id="dana", customer_id="dana", scope_path=B2C)


@pytest.mark.parametrize("site", ["backend", "memory_mw", "search_tool", "store_tool"])
def test_path_only_no_ids_accepted(sdk, site):
    SITES[site](sdk, scope_path=PATH)


@pytest.mark.parametrize("site", SITES)
def test_no_path_never_refused(sdk, site):
    SITES[site](sdk, user_id="bob", customer_id="whatever")


async def test_no_path_calls_byte_identical(sdk):
    b = SynapBackend(sdk, user_id="alice", customer_id="acme")
    await b.aread("/AGENTS.md")
    await b.awrite("/n.md", "x")
    assert "scope_path" not in sdk.last_kwargs("fetch")
    assert "scope" not in sdk.last_kwargs("memories.create")


@pytest.mark.parametrize("site", ["search_tool", "store_tool"])
@pytest.mark.parametrize("field,value", [("user_id", "bob"), ("customer_id", "globex")])
async def test_ids_reassigned_after_construction_refused_per_call(sdk, site, field, value):
    tool = SITES[site](sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    setattr(tool, field, value)
    with pytest.raises(SynapIntegrationError):
        if site == "search_tool":
            await tool._arun("q")
        else:
            await tool._arun("note")
    assert _calls(sdk) == []


@pytest.mark.parametrize("site", ["search_tool", "store_tool"])
def test_ids_reassigned_sync_run_refused(sdk, site):
    tool = SITES[site](sdk, user_id="dana", scope_path=PATH)
    tool.user_id = "bob"
    with pytest.raises(SynapIntegrationError):
        tool._run("x")
    assert _calls(sdk) == []
