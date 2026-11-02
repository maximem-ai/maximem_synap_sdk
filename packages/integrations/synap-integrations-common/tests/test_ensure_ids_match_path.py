"""ensure_ids_match_path: per-call ids must agree with a fixed scope path."""
import pytest

from synap_integrations_common import SynapIntegrationError, ensure_ids_match_path

PATH = {"customer": "acme", "user": "alice"}


def test_matching_ids_pass():
    ensure_ids_match_path(PATH, user_id="alice", customer_id="acme", where="t")


def test_mismatch_on_user_raises_and_names_the_key_and_where():
    with pytest.raises(SynapIntegrationError, match="user_id") as ei:
        ensure_ids_match_path(PATH, user_id="bob", customer_id="acme", where="Store.search")
    assert "Store.search" in str(ei.value)
    assert ei.value.context["key"] == "user"


def test_mismatch_on_customer_raises():
    with pytest.raises(SynapIntegrationError, match="customer_id") as ei:
        ensure_ids_match_path(PATH, user_id="alice", customer_id="rival")
    assert ei.value.context["key"] == "customer"


def test_the_exposure_case_bob_and_rival_is_refused():
    with pytest.raises(SynapIntegrationError):
        ensure_ids_match_path(PATH, user_id="bob", customer_id="rival", where="nemo")


def test_only_the_keys_present_in_the_path_are_checked():
    ensure_ids_match_path({"user": "alice"}, user_id="alice", customer_id="anything")
    with pytest.raises(SynapIntegrationError):
        ensure_ids_match_path({"user": "alice"}, user_id="bob")
    ensure_ids_match_path({"team": "payments"}, user_id="bob", customer_id="rival")


@pytest.mark.parametrize("empty", [None, ""])
def test_an_empty_id_is_fine_the_path_supplies_it(empty):
    ensure_ids_match_path(PATH, user_id=empty, customer_id=empty)
    ensure_ids_match_path(PATH, user_id="alice", customer_id=empty)


def test_b2c_collapse_passes_with_no_customer_id_or_an_equal_one():
    b2c = {"customer": "alice", "user": "alice"}
    ensure_ids_match_path(b2c, user_id="alice", customer_id=None)
    ensure_ids_match_path(b2c, user_id="alice", customer_id="alice")
    with pytest.raises(SynapIntegrationError):
        ensure_ids_match_path(b2c, user_id="alice", customer_id="rival")


@pytest.mark.parametrize("falsy", [None, {}])
def test_a_falsy_path_is_a_no_op(falsy):
    ensure_ids_match_path(falsy, user_id="anyone", customer_id="anyone")


@pytest.mark.parametrize("bad", [
    {"user": "a/b"}, {"user": ""}, {"user": 3}, {"": "x"}, {"user": None}, {"team": "x/"},
])
def test_odd_path_values_are_refused_even_when_no_ids_are_given(bad):
    with pytest.raises(SynapIntegrationError, match="scope_path"):
        ensure_ids_match_path(bad)


def test_a_non_mapping_path_is_refused():
    with pytest.raises(SynapIntegrationError, match="mapping"):
        ensure_ids_match_path(["acme", "alice"])


def test_error_messages_and_context_do_not_contain_the_raw_ids():
    with pytest.raises(SynapIntegrationError) as ei:
        ensure_ids_match_path({"customer": "acme-secret", "user": "alice-secret"},
                              user_id="bob-secret", customer_id="rival-secret", where="w")
    blob = str(ei.value) + repr(ei.value.context)
    for raw in ("acme-secret", "alice-secret", "bob-secret", "rival-secret"):
        assert raw not in blob
    assert ei.value.context["id_fp"] != ei.value.context["path_fp"]
    assert len(ei.value.context["id_fp"]) == 8


def test_bad_path_error_does_not_leak_the_value():
    with pytest.raises(SynapIntegrationError) as ei:
        ensure_ids_match_path({"user": "secret/value"})
    assert "secret" not in str(ei.value)
