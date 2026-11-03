"""resolve_address / scope_string: priority, fail-closed cases, label depth."""
import pytest

from maximem_synap import ScopeAddress, ScopeLadderError

from synap_integrations_common import SynapIntegrationError, resolve_address, scope_string
from synap_integrations_common.testing import RecordingSDK, make_ladder

DEEP_PATH = {"customer": "acme", "team": "payments", "user": "dana"}


# -- priority -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_scope_path_wins_ids_as_given_and_no_ladder_read():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    addr = await resolve_address(sdk, user_id="dana", customer_id="acme", scope_path=DEEP_PATH)
    assert addr == ScopeAddress(user_id="dana", customer_id="acme", scope_path=DEEP_PATH)
    assert sdk.calls_to("get_scope_ladder") == []


@pytest.mark.asyncio
async def test_scope_path_is_copied_not_aliased():
    path = dict(DEEP_PATH)
    addr = await resolve_address(RecordingSDK(), scope_path=path)
    path["team"] = "mutated"
    assert addr.scope_path == DEEP_PATH


@pytest.mark.asyncio
async def test_values_go_through_the_ladder_on_a_deeper_ladder():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    addr = await resolve_address(sdk, values=["acme", "payments", "dana"])
    assert addr.read_kwargs() == {"user_id": "dana", "customer_id": "acme", "scope_path": DEEP_PATH}
    assert addr.write_kwargs() == {"user_id": "dana", "customer_id": "acme", "scope": DEEP_PATH}
    assert len(sdk.calls_to("get_scope_ladder")) == 1


@pytest.mark.asyncio
async def test_values_on_the_default_b2b_ladder_give_ids_and_no_path():
    sdk = RecordingSDK(ladder=make_ladder("b2b"))
    addr = await resolve_address(sdk, values=["acme", "dana"])
    assert addr == ScopeAddress(user_id="dana", customer_id="acme", scope_path=None)


@pytest.mark.asyncio
async def test_values_on_the_default_b2c_ladder_give_only_user_id():
    sdk = RecordingSDK(ladder=make_ladder("b2c"))
    addr = await resolve_address(sdk, values=["dana"])
    assert addr == ScopeAddress(user_id="dana", customer_id=None, scope_path=None)


@pytest.mark.asyncio
async def test_a_shorter_values_list_addresses_a_higher_level():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    addr = await resolve_address(sdk, values=["acme", "payments"])
    assert addr.customer_id == "acme" and addr.user_id is None
    assert addr.scope_path == {"customer": "acme", "team": "payments"}


@pytest.mark.asyncio
async def test_plain_ids_make_no_network_call_and_no_path():
    sdk = RecordingSDK()
    addr = await resolve_address(sdk, user_id="dana", customer_id="acme")
    assert addr == ScopeAddress(user_id="dana", customer_id="acme", scope_path=None)
    assert addr.read_kwargs() == {"user_id": "dana", "customer_id": "acme"}
    assert sdk.calls == []


@pytest.mark.asyncio
async def test_plain_ids_work_with_an_sdk_that_has_no_ladder_method():
    class Bare:  # no get_scope_ladder at all
        pass

    addr = await resolve_address(Bare(), user_id="dana")
    assert addr.read_kwargs() == {"user_id": "dana"}


@pytest.mark.asyncio
async def test_empty_path_counts_as_unset_like_the_sdk():
    addr = await resolve_address(RecordingSDK(), user_id="dana", scope_path={})
    assert addr.scope_path is None


# -- fail closed ----------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"values": ["acme", "dana"], "scope_path": DEEP_PATH},
    {"values": ["acme", "dana"], "user_id": "dana"},
    {"values": ["acme", "dana"], "customer_id": "acme"},
])
async def test_conflicting_inputs_are_an_error_and_nothing_is_read(kwargs):
    sdk = RecordingSDK()
    with pytest.raises(SynapIntegrationError):
        await resolve_address(sdk, **kwargs)
    assert sdk.calls == []


@pytest.mark.asyncio
async def test_values_with_no_ladder_fail_closed_never_guess():
    sdk = RecordingSDK(ladder=None)
    with pytest.raises(SynapIntegrationError, match="could not be determined"):
        await resolve_address(sdk, values=["acme", "dana"])


@pytest.mark.asyncio
async def test_a_failed_ladder_read_is_wrapped_with_the_cause():
    boom = RuntimeError("whoami down")
    sdk = RecordingSDK(ladder=boom)
    with pytest.raises(SynapIntegrationError) as ei:
        await resolve_address(sdk, values=["acme", "dana"])
    assert ei.value.__cause__ is boom


@pytest.mark.asyncio
async def test_ladder_errors_from_the_sdk_propagate_unchanged():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    with pytest.raises(ScopeLadderError):
        await resolve_address(sdk, values=["a", "b", "c", "d"])  # too many
    with pytest.raises(ScopeLadderError):
        await resolve_address(sdk, values=[])


@pytest.mark.asyncio
async def test_a_deeper_ladder_the_server_would_refuse_raises_before_any_call():
    sdk = RecordingSDK(ladder=make_ladder("deep", ratified=False))
    with pytest.raises(ScopeLadderError, match="refresh=True"):
        await resolve_address(sdk, values=["acme", "payments", "dana"])


@pytest.mark.asyncio
async def test_nothing_given_is_an_error():
    with pytest.raises(SynapIntegrationError, match="no identity"):
        await resolve_address(RecordingSDK())


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["team"], {"team": ""}, {"": "x"}, {"team": 3}, "acme/dana"])
async def test_a_malformed_scope_path_is_refused(bad):
    with pytest.raises(SynapIntegrationError, match="scope_path"):
        await resolve_address(RecordingSDK(), scope_path=bad)


# -- scope_string -----------------------------------------------------------------

@pytest.mark.parametrize("values,expected", [
    (["a"], "/a"),
    (["a", "b"], "/a/b"),
    (["a", "b", "c"], "/a/b/c"),
    (["a", "b", "c", "d"], "/a/b/c/d"),
    (("acme", "dana"), "/acme/dana"),
])
def test_scope_string_at_every_depth(values, expected):
    assert scope_string(values) == expected


@pytest.mark.parametrize("bad", [[], "acme", ["a", ""], ["a", None], ["a", 3], None])
def test_scope_string_refuses_what_would_name_the_wrong_node(bad):
    with pytest.raises(ValueError):
        scope_string(bad)


# -- empty scope_path ---------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"user_id": "dana"}, {"customer_id": "acme"}, {"values": ["acme", "dana"]}])
async def test_an_empty_scope_path_with_ids_or_values_warns(kwargs, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="synap_integrations_common.scope"):
        addr = await resolve_address(RecordingSDK(), scope_path={}, **kwargs)
    assert addr.scope_path is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "empty scope_path" in caplog.text


@pytest.mark.asyncio
async def test_no_warning_when_scope_path_is_simply_absent(caplog):
    await resolve_address(RecordingSDK(), user_id="dana")
    assert caplog.records == []
