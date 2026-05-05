import pytest

from synap_integrations_common.scope import default_scope


def test_user_only():
    assert default_scope("alice") == "/alice"


def test_user_and_customer():
    assert default_scope("alice", "acme") == "/acme/alice"


def test_empty_customer_treated_as_absent():
    assert default_scope("alice", "") == "/alice"
    assert default_scope("alice", None) == "/alice"


def test_empty_user_id_raises():
    with pytest.raises(ValueError):
        default_scope("")
    with pytest.raises(ValueError):
        default_scope("", "acme")
