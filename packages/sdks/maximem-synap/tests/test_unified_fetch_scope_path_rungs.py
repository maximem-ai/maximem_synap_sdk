"""A unified fetch sends each scope the part of the path that scope accepts.

Found live on production (maximem-synap 0.7.0): on a ladder customer -> team ->
user, ``sdk.fetch(user_id='dana', customer_id='acme', scope_path={customer,
team, user})`` failed with HTTP 400. The SDK forwarded the SAME full path to
every sub-fetch, and ``POST /v1/context/customer/fetch`` refuses a path that
names the identity rung (``customer_scope_reaches_an_identity`` in
``synap/cloud/application/admin_api/routes/sdk_context.py``): a customer-scope
read is shared material, never one person's.

The fake server below encodes that rule by ROLE, as the server does, so the
test fails for the real reason and cannot pass by matching the SDK's own
output. When the server code is importable, a second test pins the fake to the
real function.
"""
import asyncio

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK
from maximem_synap.scope_ladder import ScopeLadder, ScopeLevel

REFUSAL = (
    "The scope path names the identity rung, which addresses one person "
    "rather than the tenant they belong to."
)


def _ladder(identity_key="user", extra=(), below=()):
    levels = [ScopeLevel("customer", "Customer", 1, "tenant_of_record", True)]
    pos = 2
    for k in extra:
        levels.append(ScopeLevel(k, k.title(), pos, None, True))
        pos += 1
    levels.append(ScopeLevel(identity_key, "User", pos, "identity", True))
    pos += 1
    for k in below:
        levels.append(ScopeLevel(k, k.title(), pos, None, True))
        pos += 1
    return ScopeLadder(
        levels=tuple(levels), is_default=False, ladder_ratified=True,
        nested_scoping_enabled=True, source="server",
    )


class _Server:
    """The scope rules of the real routes, by role, over a recording transport."""

    def __init__(self, ladder):
        self.ladder = ladder
        self.sent = []
        self.whoami_calls = 0
        self.whoami_fails = False
        self.whoami_delay = 0.0

    def _roles(self, path):
        by_key = {lv.key: lv for lv in self.ladder.levels}
        return [by_key[k].role for k in path if k in by_key]

    async def get(self, path, **kw):
        assert path == "/api/v1/auth/whoami", path
        self.whoami_calls += 1
        if self.whoami_delay:
            await asyncio.sleep(self.whoami_delay)
        if self.whoami_fails:
            raise RuntimeError("whoami is down")
        return {
            "user_context_isolation": "strict",
            "scope_ladder": {
                "levels": [
                    {"key": lv.key, "label": lv.label, "position": lv.position,
                     "role": lv.role, "required": lv.required}
                    for lv in self.ladder.levels
                ],
                "is_default": False, "ladder_ratified": True,
                "nested_scoping_enabled": True,
            },
        }

    async def post(self, path, **kw):
        body = kw.get("json") or {}
        self.sent.append((path, body))
        scope = body.get("scope")
        if path.endswith("/customer/fetch") and scope and "identity" in self._roles(scope):
            raise InvalidInputError(REFUSAL + " A customer-scope fetch reads shared material only.")
        return {"context": {}}


class _NoCache:
    def get(self, **_):
        return None

    def set(self, **_):
        pass


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk(ladder=None, load_ladder=True):
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = _Server(ladder or _ladder(extra=("team",)))
    sdk._cache_manager = _NoCache()
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    sdk._client_id = "cli"
    if load_ladder:
        sdk._scope_ladder = sdk._http_transport.ladder
        sdk._scope_ladder_loaded = True
        sdk._scope_ladder_expires_at = float("inf")
    return sdk


FULL = {"customer": "acme", "team": "payments", "user": "dana"}


def _by_route(sdk):
    return {p.rsplit("/", 2)[-2]: b for p, b in sdk._http_transport.sent}


def test_the_default_scopes_fetch_does_not_send_the_user_rung_to_the_customer_route():
    sdk = _sdk()
    ctx = asyncio.run(sdk.fetch(
        user_id="dana", customer_id="acme", scope_path=FULL, search_query=["x"],
    ))
    sent = _by_route(sdk)
    assert sent["user"]["scope"] == FULL
    assert sent["customer"]["scope"] == {"customer": "acme", "team": "payments"}
    assert "customer" in ctx.scopes_queried and "user" in ctx.scopes_queried


def test_the_customer_route_refuses_the_full_path_as_the_fake_encodes_it():
    """Control: the fake really does refuse, so the test above can fail."""
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="identity rung"):
        asyncio.run(sdk.customer.context.fetch(
            customer_id="acme", search_query=["x"], scope_path=FULL))


def test_a_customer_only_caller_with_a_user_key_in_the_path_is_trimmed_too():
    sdk = _sdk()
    asyncio.run(sdk.fetch(customer_id="acme", scope_path=FULL, search_query=["x"]))
    sent = _by_route(sdk)
    assert list(sent) == ["customer"]
    assert sent["customer"]["scope"] == {"customer": "acme", "team": "payments"}


def test_scopes_user_only_keeps_the_full_path():
    sdk = _sdk()
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=FULL,
                          scopes=["user"], search_query=["x"]))
    sent = _by_route(sdk)
    assert list(sent) == ["user"] and sent["user"]["scope"] == FULL


def test_a_path_without_the_identity_rung_is_unchanged_for_every_route():
    sdk = _sdk()
    path = {"customer": "acme", "team": "payments"}
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=path,
                          scopes=["user", "customer"], search_query=["x"]))
    sent = _by_route(sdk)
    assert sent["user"]["scope"] == path and sent["customer"]["scope"] == path


def test_a_customer_only_path_is_unchanged():
    sdk = _sdk()
    asyncio.run(sdk.fetch(customer_id="acme", scope_path={"customer": "acme"},
                          search_query=["x"]))
    assert _by_route(sdk)["customer"]["scope"] == {"customer": "acme"}


def test_the_identity_rung_is_found_by_role_when_the_client_renamed_it():
    ladder = _ladder(identity_key="member", extra=("team",))
    sdk = _sdk(ladder)
    path = {"customer": "acme", "team": "payments", "member": "dana"}
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=path,
                          search_query=["x"]))
    sent = _by_route(sdk)
    assert sent["user"]["scope"] == path
    assert sent["customer"]["scope"] == {"customer": "acme", "team": "payments"}


def test_with_no_cached_ladder_the_reserved_user_key_is_removed():
    sdk = _sdk(load_ladder=False)
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=FULL,
                          search_query=["x"]))
    assert _by_route(sdk)["customer"]["scope"] == {"customer": "acme", "team": "payments"}


def test_the_client_scope_gets_no_path():
    """The client read is meant to return account-wide shared memory; the user
    and customer reads already cover the person's and the team's levels.
    POST /v1/context/client/fetch applies a path (it narrows the read to that
    chain's own nodes) and validates it strictly (400 when nested scoping is
    off, the ladder is unratified, a key is unknown or a rung is skipped). So
    a path there would narrow the shared read, duplicate results and create
    registry nodes on a read. 0.7.0 sent it."""
    sdk = _sdk()
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=FULL,
                          scopes=["client"], search_query=["x"]))
    assert "scope" not in _by_route(sdk)["client"]


def test_the_default_user_customer_conversation_and_client_scopes_each_get_their_path():
    sdk = _sdk()
    asyncio.run(sdk.fetch(
        conversation_id="11111111-2222-3333-4444-555555555555", user_id="dana",
        customer_id="acme", scope_path=FULL, scopes=["conversation", "user", "customer", "client"],
        include_conversation_context=False, search_query=["x"]))
    sent = _by_route(sdk)
    assert sent["conversation"]["scope"] == FULL
    assert sent["user"]["scope"] == FULL
    assert sent["customer"]["scope"] == {"customer": "acme", "team": "payments"}
    assert "scope" not in sent["client"]


def test_a_call_without_a_path_sends_no_scope_anywhere():
    sdk = _sdk()
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", search_query=["x"]))
    assert all("scope" not in b for _, b in sdk._http_transport.sent)


def test_the_default_two_level_ladder_with_a_user_path_is_trimmed_to_the_customer():
    ladder = _ladder()  # customer -> user
    sdk = _sdk(ladder)
    path = {"customer": "acme", "user": "dana"}
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=path,
                          search_query=["x"]))
    sent = _by_route(sdk)
    assert sent["customer"]["scope"] == {"customer": "acme"}
    assert sent["user"]["scope"] == path


def test_as_tool_unified_sends_the_trimmed_path_and_the_model_cannot_choose_it():
    sdk = _sdk()
    tool = sdk.as_tool(scope="unified", user_id="dana", customer_id="acme",
                       scope_path=FULL, style="openai")
    props = tool["function"]["parameters"]["properties"]
    assert "scope_path" not in props and "scope" not in props
    asyncio.run(tool["handler"](search_query=["x"], scope_path={"customer": "evil"}))
    sent = _by_route(sdk)
    assert sent["user"]["scope"] == FULL
    assert sent["customer"]["scope"] == {"customer": "acme", "team": "payments"}


def test_the_fake_agrees_with_the_real_server_rule():
    rule = pytest.importorskip(
        "synap.cloud.application.admin_api.routes.sdk_context"
    ).customer_scope_reaches_an_identity
    rungs = lambda roles: [{"role": r, "roles": [r]} for r in roles]  # noqa: E731
    assert rule(rungs(["root", "tenant_of_record", "team", "identity"]))
    assert rule(rungs(["root", "tenant_of_record", "team"])) is None


# --- the identity rung is found on the account's ladder, read first ---------

ABOVE = {"customer": "acme", "team": "payments"}


def test_an_uncached_ladder_is_read_once_so_a_renamed_identity_key_is_found():
    sdk = _sdk(_ladder(identity_key="member", extra=("team",)), load_ladder=False)
    path = {"customer": "acme", "team": "payments", "member": "dana"}
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=path,
                          search_query=["x"]))
    sent = _by_route(sdk)
    assert sdk._http_transport.whoami_calls == 1
    assert sent["customer"]["scope"] == ABOVE
    assert sent["user"]["scope"] == path


def test_levels_below_the_identity_rung_are_removed_with_it_uncached_and_cached():
    ladder = _ladder(extra=("team",), below=("device",))
    path = {"customer": "acme", "team": "payments", "user": "dana", "device": "d1"}
    for load in (False, True):
        sdk = _sdk(ladder, load_ladder=load)
        asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=path,
                              search_query=["x"]))
        sent = _by_route(sdk)
        assert sent["customer"]["scope"] == ABOVE, load
        assert sent["user"]["scope"] == path, load


def test_a_failed_ladder_read_keeps_the_reserved_key_fallback_and_logs(caplog):
    sdk = _sdk(load_ladder=False)
    sdk._http_transport.whoami_fails = True
    with caplog.at_level("WARNING"):
        asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=FULL,
                              search_query=["x"]))
    assert sdk._http_transport.whoami_calls == 1
    assert _by_route(sdk)["customer"]["scope"] == ABOVE
    assert any("scope ladder" in r.getMessage() for r in caplog.records)


def test_a_failed_ladder_read_only_removes_the_reserved_key():
    """The fallback cannot know about a level below it or a renamed identity
    key: the server's clear 400 then reaches the caller, as it did before."""
    ladder = _ladder(identity_key="member", extra=("team",))
    sdk = _sdk(ladder, load_ladder=False)
    sdk._http_transport.whoami_fails = True
    path = {"customer": "acme", "team": "payments", "member": "dana"}
    with pytest.raises(InvalidInputError, match="identity rung"):
        asyncio.run(sdk.fetch(customer_id="acme", scope_path=path, search_query=["x"]))


def test_concurrent_fetches_share_one_ladder_read():
    sdk = _sdk(load_ladder=False)
    sdk._http_transport.whoami_delay = 0.05

    async def go():
        await asyncio.gather(*[
            sdk.fetch(user_id=f"u{i}", customer_id="acme",
                      scope_path={**FULL, "user": f"u{i}"}, search_query=["x"])
            for i in range(4)
        ])
    asyncio.run(go())
    assert sdk._http_transport.whoami_calls == 1
    assert len([1 for p, _ in sdk._http_transport.sent if p.endswith("/customer/fetch")]) == 4


def test_a_cached_ladder_makes_no_request():
    sdk = _sdk()
    asyncio.run(sdk.fetch(user_id="dana", customer_id="acme", scope_path=FULL,
                          search_query=["x"]))
    assert sdk._http_transport.whoami_calls == 0


@pytest.mark.parametrize("kw", [
    dict(user_id="dana", customer_id="acme"),                                   # no path
    dict(user_id="dana", customer_id="acme", scope_scopes=["user"]),            # customer does not run
    dict(user_id="dana", scope_path=FULL),                                      # no customer id
])
def test_calls_that_do_not_need_the_ladder_make_zero_whoami_calls(kw):
    kw = dict(kw)
    if "scope_scopes" in kw:
        kw["scopes"] = kw.pop("scope_scopes")
        kw["scope_path"] = FULL
    sdk = _sdk(load_ladder=False)
    asyncio.run(sdk.fetch(search_query=["x"], **kw))
    assert sdk._http_transport.whoami_calls == 0
