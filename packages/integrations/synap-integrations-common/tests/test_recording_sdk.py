"""The shared RecordingSDK, make_ladder and the recording_sdk fixture."""
import inspect
from unittest.mock import AsyncMock

import pytest

from maximem_synap import ScopeLadder

from synap_integrations_common.testing import (
    RecordingSDK,
    failing_sdk,  # noqa: F401 — the existing fixtures must still import
    make_ladder,
    mock_sdk,  # noqa: F401
    recording_sdk,  # noqa: F401
)


@pytest.mark.asyncio
async def test_every_listed_call_records_its_exact_keyword_arguments(recording_sdk):
    s = recording_sdk
    await s.fetch(search_query=["q"], user_id="u", customer_id="c", scope_path={"customer": "c", "user": "u"})
    await s.memories.create("doc", user_id="u", customer_id="c", scope={"customer": "c", "user": "u"})
    await s.memories.forget("that", user_id="u", customer_id="c", scope={"customer": "c", "user": "u"})
    await s.conversation.record_message(conversation_id="k", role="user", content="x", user_id="u",
                                        scope_path={"user": "u"})
    await s.instance.send_message(content="x", user_id="u", scope_path={"user": "u"})
    await s.get_scope_ladder(refresh=True)
    assert [c.name for c in s.calls] == [
        "fetch", "memories.create", "memories.forget", "conversation.record_message",
        "instance.send_message", "get_scope_ladder"]
    assert s.last_kwargs("fetch") == {
        "search_query": ["q"], "user_id": "u", "customer_id": "c",
        "scope_path": {"customer": "c", "user": "u"}}
    assert s.calls_to("memories.create")[0].args == ("doc",)
    assert s.last_kwargs("memories.forget")["scope"] == {"customer": "c", "user": "u"}
    assert s.last_kwargs("get_scope_ladder") == {"refresh": True}


@pytest.mark.asyncio
async def test_a_call_without_a_keyword_records_no_such_key(recording_sdk):
    await recording_sdk.fetch(search_query=["q"], user_id="u")
    assert recording_sdk.last_kwargs("fetch") == {"search_query": ["q"], "user_id": "u"}


@pytest.mark.asyncio
async def test_an_unknown_keyword_raises_like_the_real_sdk_and_is_not_recorded(recording_sdk):
    with pytest.raises(TypeError):
        await recording_sdk.fetch(search_query=["q"], not_a_parameter=1)
    assert recording_sdk.calls == []


@pytest.mark.asyncio
async def test_accept_scope_path_false_behaves_like_an_sdk_from_before_the_ladder():
    old = RecordingSDK(accept_scope_path=False)
    for fn, kw in ((old.fetch, "scope_path"), (old.instance.send_message, "scope_path"),
                   (old.conversation.record_message, "scope_path")):
        assert kw not in inspect.signature(fn).parameters
    for fn in (old.memories.create, old.memories.forget):
        assert "scope" not in inspect.signature(fn).parameters
    with pytest.raises(TypeError):
        await old.fetch(search_query=["q"], scope_path={"user": "u"})
    assert old.calls == []


@pytest.mark.asyncio
async def test_ladder_is_configurable_and_can_be_none_or_an_error():
    assert await RecordingSDK().get_scope_ladder() is not None
    assert (await RecordingSDK().get_scope_ladder()).is_default
    deep = make_ladder("deep")
    assert await RecordingSDK(ladder=deep).get_scope_ladder() is deep
    assert await RecordingSDK(ladder=None).get_scope_ladder() is None
    with pytest.raises(RuntimeError):
        await RecordingSDK(ladder=RuntimeError("x")).get_scope_ladder()


@pytest.mark.asyncio
async def test_results_and_errors_can_be_configured_after_recording():
    s = RecordingSDK()
    s.results["fetch"] = "canned"
    s.errors["memories.create"] = RuntimeError("nope")
    assert await s.fetch(search_query=["q"]) == "canned"
    with pytest.raises(RuntimeError):
        await s.memories.create("d")
    assert [c.name for c in s.calls] == ["fetch", "memories.create"]


def test_reset_clears_calls_only():
    s = RecordingSDK()
    s.calls.append(object())
    s.reset()
    assert s.calls == [] and s.ladder is not None


def test_last_kwargs_of_a_call_that_never_happened_fails_loudly():
    with pytest.raises(AssertionError, match="never called"):
        RecordingSDK().last_kwargs("fetch")


def test_make_ladder_shapes():
    b2b, b2c, deep = make_ladder("b2b"), make_ladder("b2c"), make_ladder("deep")
    assert all(isinstance(x, ScopeLadder) for x in (b2b, b2c, deep))
    assert b2b.keys == ("customer", "user") and b2b.is_default and not b2b.b2c
    assert b2b.ladder_ratified is False and b2b.nested_scoping_enabled is False
    assert b2c.keys == ("user",) and b2c.is_default and b2c.b2c
    assert deep.keys == ("customer", "team", "user") and not deep.is_default
    assert deep.ladder_ratified and deep.nested_scoping_enabled
    assert [lv.role for lv in deep.levels] == ["tenant_of_record", None, "identity"]
    assert [lv.position for lv in deep.levels] == [1, 2, 3]


def test_make_ladder_overrides_and_bad_kind():
    four = make_ladder(keys=("customer", "region", "team", "user"))
    assert four.keys == ("customer", "region", "team", "user") and not four.is_default
    assert make_ladder("deep", ratified=False).ladder_ratified is False
    assert make_ladder(keys=("customer", "team", "user"), b2c=True).b2c is True
    with pytest.raises(ValueError):
        make_ladder("nope")


def test_existing_fixtures_are_unchanged(mock_sdk, failing_sdk):  # noqa: F811
    assert mock_sdk.instance_id == "test-instance"
    assert isinstance(mock_sdk.fetch, AsyncMock)  # still the AsyncMock one
    assert isinstance(failing_sdk.fetch, AsyncMock)
