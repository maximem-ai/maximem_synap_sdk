"""The package must import and mostly work against an SDK that has no scope
ladder (no ScopeAddress, no get_scope_ladder)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PKG = str(Path(__file__).resolve().parents[1])

_SCRIPT = textwrap.dedent('''
    import asyncio, sys, types
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    # An older maximem_synap: importable, but without the scope ladder (and
    # without the models the testing helpers use).
    stub = types.ModuleType("maximem_synap")
    stub.MaximemSynapSDK = type("MaximemSynapSDK", (), {})
    sys.modules["maximem_synap"] = stub

    import synap_integrations_common as c
    from synap_integrations_common import scope, testing
    assert scope._SDK_HAS_LADDER is False
    assert testing.ScopeLadder is None

    assert c.default_scope("alice", "acme") == "/acme/alice"
    assert c.scope_string(["a", "b", "c"]) == "/a/b/c"

    async def main():
        sdk = SimpleNamespace(
            instance=SimpleNamespace(is_listening=True, send_message=AsyncMock()),
            conversation=SimpleNamespace(record_message=AsyncMock()))
        assert await c.report_turn(sdk, role="user", content="hi", conversation_id="c1", user_id="u")
        sdk.instance.send_message.assert_awaited_once_with(
            content="hi", role="user", event_type="user_message",
            conversation_id="c1", user_id="u", metadata={})
        assert await c.record_turn(sdk, role="assistant", content="yo", conversation_id="c1", user_id="u")

        # ids and an explicit path still resolve, with the local ScopeAddress
        a = await c.resolve_address(sdk, user_id="u", customer_id="acme")
        assert a.read_kwargs() == {"user_id": "u", "customer_id": "acme"}
        a = await c.resolve_address(sdk, user_id="u", scope_path={"user": "u"})
        assert a.read_kwargs()["scope_path"] == {"user": "u"} and a.write_kwargs()["scope"] == {"user": "u"}

        # values need the ladder: clear error with an upgrade hint
        try:
            await c.resolve_address(sdk, values=["acme", "u"])
        except c.SynapIntegrationError as e:
            assert "Upgrade" in str(e) and "get_scope_ladder" in str(e), str(e)
        else:
            raise AssertionError("values must fail on an SDK without a ladder")
    asyncio.run(main())
    print("OK")
''')


def test_imports_and_degrades_against_an_sdk_without_the_ladder():
    env = dict(os.environ)
    env["PYTHONPATH"] = _PKG
    out = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("OK")


@pytest.mark.asyncio
async def test_values_on_an_sdk_object_without_get_scope_ladder_raise_an_upgrade_hint():
    from synap_integrations_common import SynapIntegrationError, resolve_address

    with pytest.raises(SynapIntegrationError, match="Upgrade"):
        await resolve_address(object(), values=["acme", "dana"])
