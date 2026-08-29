"""The SDK refuses a customer_id on a B2C instance, at the call site.

The server rejects independently. This layer exists because the old failure was
silent: sending a customer_id on a B2C instance produced no exception anywhere,
the write was filed under the customer, the read asked for the user, and both
returned success. One client ran 4,634 consecutive empty fetches over seven days
with nothing to look at.

The guard at the bottom matters more than any single case here. Wiring ten entry
points by hand is exactly the kind of job where the eleventh gets forgotten, and
a forgotten one is silent by construction.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from maximem_synap.scoping import (
    B2C_ISOLATION,
    CustomerIdNotAcceptedError,
    check_customer_id,
)

WHERE = "probe"


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------

def test_b2c_with_a_customer_id_raises():
    with pytest.raises(CustomerIdNotAcceptedError) as exc:
        check_customer_id(B2C_ISOLATION, "acme", where=WHERE, instance_id="inst_1")
    msg = str(exc.value)
    assert "Send user_id only" in msg, "the error must say what to do instead"
    assert "acme" in msg, "the rejected value must be echoed or it cannot be traced"
    assert "inst_1" in msg


def test_b2c_without_a_customer_id_is_the_correct_shape():
    check_customer_id(B2C_ISOLATION, None, where=WHERE)
    check_customer_id(B2C_ISOLATION, "", where=WHERE)


def test_b2b_is_never_refused():
    """customer_id is REQUIRED on B2B. Refusing it there would break every B2B
    caller, which is a worse failure than the one this prevents."""
    check_customer_id("strict", "acme", where=WHERE)
    check_customer_id("maca_suggests", "acme", where=WHERE)


def test_an_unknown_mode_never_refuses():
    """None is what an older server returns, because it has no
    `user_context_isolation` field in whoami yet. This SDK must work against
    those servers, so not knowing has to mean not enforcing."""
    check_customer_id(None, "acme", where=WHERE)


def test_the_error_is_a_valueerror():
    """Callers already catching ValueError around id handling keep working."""
    assert issubclass(CustomerIdNotAcceptedError, ValueError)


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

class _FakeSDK:
    """Just enough SDK to exercise `_check_customer_id`."""

    instance_id = "inst_test"

    def __init__(self, isolation):
        self._user_context_isolation = isolation

    _check_customer_id = None  # replaced below


def _sdk_with(isolation):
    from maximem_synap.sdk import MaximemSynapSDK
    sdk = _FakeSDK(isolation)
    sdk._check_customer_id = MaximemSynapSDK._check_customer_id.__get__(sdk, _FakeSDK)
    return sdk


@pytest.mark.parametrize("isolation,should_raise", [
    (B2C_ISOLATION, True),
    ("strict", False),
    (None, False),
])
def test_sdk_check_respects_the_mode(isolation, should_raise):
    sdk = _sdk_with(isolation)
    if should_raise:
        with pytest.raises(CustomerIdNotAcceptedError):
            sdk._check_customer_id("acme", where="probe")
    else:
        sdk._check_customer_id("acme", where="probe")


def test_whoami_absent_field_leaves_the_mode_unknown():
    """A server that does not report the mode must leave it None, not default
    to something. `.get()` on a dict without the key returns None, and this
    pins that the SDK stores exactly that."""
    from maximem_synap.sdk import MaximemSynapSDK
    assert MaximemSynapSDK._user_context_isolation is None


# ---------------------------------------------------------------------------
# Call-site guard
# ---------------------------------------------------------------------------

_SDK_DIR = Path(__file__).resolve().parents[1] / "maximem_synap"

# Internal plumbing that forwards ids it was handed; the public entry point
# above each of these has already applied the check, and re-checking inside the
# transport would fire on our own retries.
_EXEMPT_FILES = {
    "scoping.py",          # the check itself
    "grpc_client.py",      # transport, forwards what it was given
    "anticipation_cache.py",  # cache key construction, never a caller's entry point
    "manager.py",          # cache manager, called by CacheInterface which checks
    "controllers.py",      # facade internals behind the checked interfaces
    "conversation.py",     # ditto
    "models.py",
}


def _public_methods_taking_customer_id():
    out = []
    for path in sorted(_SDK_DIR.rglob("*.py")):
        if path.name in _EXEMPT_FILES or path.name.startswith("_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if fn.name.startswith("_"):
                    continue
                args = [a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)]
                if "customer_id" not in args:
                    continue
                out.append((path, cls.name, fn))
    return out


def _calls_the_check(fn) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name in ("_check_customer_id", "check_customer_id"):
                return True
    return False


def test_the_guard_is_not_vacuous():
    assert _public_methods_taking_customer_id(), "found no methods; the rule below would pass for the wrong reason"


def test_every_public_method_taking_a_customer_id_checks_it():
    missing = [
        f"{p.relative_to(_SDK_DIR)}:{fn.lineno} {cls}.{fn.name}()"
        for p, cls, fn in _public_methods_taking_customer_id()
        if not _calls_the_check(fn)
    ]
    assert not missing, (
        "these public SDK methods accept a customer_id but never check it, so a "
        "B2C caller gets no error from the SDK:\n  " + "\n  ".join(missing)
    )


def test_the_two_signatures_that_forced_the_wrong_shape_are_now_optional():
    """`record_message` and `create_from_file` used to require a customer_id,
    which meant a B2C caller could not call them correctly at all: they were
    forced to send the field the server files their data under, then read it
    back under a different one."""
    from maximem_synap.sdk import ConversationInterface
    from maximem_synap.memories.interface import MemoriesInterface

    for fn in (ConversationInterface.record_message, MemoriesInterface.create_from_file):
        param = inspect.signature(fn).parameters["customer_id"]
        assert param.default is None, (
            f"{fn.__qualname__} still requires customer_id, so a B2C caller "
            f"cannot use it"
        )
