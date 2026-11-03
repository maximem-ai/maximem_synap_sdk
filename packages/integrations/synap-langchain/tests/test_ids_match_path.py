"""A configured user_id / customer_id must agree with a fixed ``scope_path``.

The server lets the path's ``user`` and ``customer`` win over flat ids, so a
mismatch would read and write as the path's user. Every construction point
refuses it before any SDK call.
"""

import pytest

from synap_integrations_common import SynapIntegrationError

from synap_langchain import (
    SynapCallbackHandler,
    SynapChatMessageHistory,
    SynapRetriever,
    SynapSearchTool,
    SynapStoreTool,
    create_synap_node,
)
from tests.test_scope_path import PATH, RecordingSDK, run  # noqa: F401

# name -> factory(sdk, **ids) building the object with PATH
SITES = {
    "history": lambda sdk, **ids: SynapChatMessageHistory(
        sdk, "c1", scope_path=PATH, **({"user_id": "dana"} | ids)),
    "handler": lambda sdk, **ids: SynapCallbackHandler(
        sdk, "c1", scope_path=PATH, **({"user_id": "dana"} | ids)),
    "node": lambda sdk, **ids: create_synap_node(
        sdk, scope_path=PATH, **({"user_id": "dana"} | ids)),
    "retriever": lambda sdk, **ids: SynapRetriever(
        sdk=sdk, scope_path=PATH, **({"user_id": "dana"} | ids)),
    "search_tool": lambda sdk, **ids: SynapSearchTool(
        sdk=sdk, scope_path=PATH, **({"user_id": "dana"} | ids)),
    "store_tool": lambda sdk, **ids: SynapStoreTool(
        sdk=sdk, scope_path=PATH, **({"user_id": "dana"} | ids)),
}
NO_PATH = {
    "history": lambda sdk, **ids: SynapChatMessageHistory(sdk, "c1", **ids),
    "handler": lambda sdk, **ids: SynapCallbackHandler(sdk, "c1", **ids),
    "node": lambda sdk, **ids: create_synap_node(sdk, **ids),
    "retriever": lambda sdk, **ids: SynapRetriever(sdk=sdk, **ids),
    "search_tool": lambda sdk, **ids: SynapSearchTool(sdk=sdk, **ids),
    "store_tool": lambda sdk, **ids: SynapStoreTool(sdk=sdk, **ids),
}


@pytest.mark.parametrize("site", SITES)
def test_other_user_refused_with_zero_sdk_calls(site):
    sdk = RecordingSDK()
    with pytest.raises(SynapIntegrationError) as ei:
        SITES[site](sdk, user_id="bob")
    assert "bob" not in str(ei.value)
    assert sdk.calls == []


@pytest.mark.parametrize("site", SITES)
def test_other_customer_refused_with_zero_sdk_calls(site):
    sdk = RecordingSDK()
    with pytest.raises(SynapIntegrationError):
        SITES[site](sdk, customer_id="globex")
    assert sdk.calls == []


@pytest.mark.parametrize("site", SITES)
def test_matching_ids_accepted(site):
    SITES[site](RecordingSDK(), user_id="dana", customer_id="acme")


@pytest.mark.parametrize("site", ["history", "handler", "node"])
def test_empty_customer_id_accepted(site):
    SITES[site](RecordingSDK(), customer_id="")


@pytest.mark.parametrize("site", SITES)
def test_unset_customer_id_accepted(site):
    SITES[site](RecordingSDK())


@pytest.mark.parametrize("site", SITES)
def test_b2c_collapse_accepted(site):
    b2c = {"customer": "dana", "user": "dana"}
    sdk = RecordingSDK()
    # customer == user in the path, customer_id left out: passes
    if site == "history":
        SynapChatMessageHistory(sdk, "c1", "dana", scope_path=b2c)
    elif site == "handler":
        SynapCallbackHandler(sdk, "c1", "dana", scope_path=b2c)
    elif site == "node":
        create_synap_node(sdk, "dana", scope_path=b2c)
    elif site == "retriever":
        SynapRetriever(sdk=sdk, user_id="dana", scope_path=b2c)
    elif site == "search_tool":
        SynapSearchTool(sdk=sdk, user_id="dana", scope_path=b2c)
    else:
        SynapStoreTool(sdk=sdk, user_id="dana", scope_path=b2c)


@pytest.mark.parametrize("site", NO_PATH)
def test_no_path_any_ids_unchanged(site):
    NO_PATH[site](RecordingSDK(), user_id="bob", customer_id="globex")


def test_no_path_call_is_byte_identical_for_a_mismatched_pair():
    sdk = RecordingSDK()
    run(SynapSearchTool(sdk=sdk, user_id="bob", customer_id="globex")._arun("q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "bob" and kw["customer_id"] == "globex" and "scope_path" not in kw


# ids reassigned AFTER construction are re-checked before every SDK call
def _call(site, obj):
    if site == "retriever":
        return obj.ainvoke("q")
    return obj.ainvoke({"query": "q"} if site == "search_tool" else {"content": "c"})


@pytest.mark.parametrize("site", ["retriever", "search_tool", "store_tool"])
@pytest.mark.parametrize("field,value", [("user_id", "bob"), ("customer_id", "globex")])
def test_ids_changed_after_construction_refused_with_zero_sdk_calls(site, field, value):
    sdk = RecordingSDK()
    obj = SITES[site](sdk)
    setattr(obj, field, value)
    with pytest.raises(SynapIntegrationError):
        run(_call(site, obj))
    assert sdk.calls == []


@pytest.mark.parametrize("site", ["retriever", "search_tool", "store_tool"])
def test_unchanged_ids_still_call_after_recheck(site):
    sdk = RecordingSDK()
    obj = SITES[site](sdk)
    try:
        run(_call(site, obj))
    except SynapIntegrationError:
        raise
    except Exception:
        pass  # fake responses may not satisfy the formatter; the call must still happen
    assert len(sdk.calls) == 1
