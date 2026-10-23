"""One Synap SDK per set of credentials, shared by every component.

``synap_memory``, ``synap_short_term``, ``synap_stream`` and ``synap_history``
each built an SDK of their own, so the stream the exporter opened never reached
the SDK the others read from. These bind the real NAT toolkit (the stubbed
``nat`` modules of ``test_short_term.py`` are dropped first, as in
``test_stream.py``).
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

for _name in [n for n in list(sys.modules) if n == "nat" or n.startswith("nat.")]:
    if getattr(sys.modules[_name], "__file__", None) is None:
        del sys.modules[_name]

import synap_nemo_agent_toolkit._sdk as sdk_module  # noqa: E402
from synap_nemo_agent_toolkit._sdk import shared_sdk  # noqa: E402
from synap_nemo_agent_toolkit.history import synap_history_middleware, SynapHistoryMiddlewareConfig  # noqa: E402
from synap_nemo_agent_toolkit.plugin import SynapMemoryClientConfig, synap_memory_client  # noqa: E402
from synap_nemo_agent_toolkit.short_term import SynapShortTermConfig, synap_short_term_function  # noqa: E402
from synap_nemo_agent_toolkit.stream import SynapStreamExporterConfig, synap_stream_exporter  # noqa: E402


def _sdk():
    sdk = MagicMock()
    sdk.initialize = AsyncMock()
    sdk.shutdown = AsyncMock()
    sdk.instance.is_listening = False
    sdk.instance.stop_listening = AsyncMock()
    del sdk.close  # MagicMock would otherwise auto-create close()
    return sdk


@pytest.fixture
def built(monkeypatch):
    """Every MaximemSynapSDK(...) the toolkit builds, in order."""
    made = []

    def factory(**kwargs):
        sdk = _sdk()
        sdk.kwargs = kwargs
        made.append(sdk)
        return sdk

    monkeypatch.setattr(sdk_module, "MaximemSynapSDK", factory)
    monkeypatch.setenv("SYNAP_API_KEY", "sk-test")
    monkeypatch.delenv("SYNAP_AUTO_LISTEN", raising=False)
    sdk_module._registry.clear()
    yield made
    sdk_module._registry.clear()


def _cfg(**kw):
    return SimpleNamespace(**{"api_key": None, "instance_id": "", **kw})


class TestSharing:
    @pytest.mark.asyncio
    async def test_two_components_get_the_same_sdk_built_once(self, built):
        async with shared_sdk(_cfg()) as a, shared_sdk(_cfg()) as b:
            assert a is b
        assert len(built) == 1
        built[0].initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_different_credentials_do_not_share(self, built):
        async with shared_sdk(_cfg()) as a, shared_sdk(_cfg(instance_id="other")) as b:
            assert a is not b

    @pytest.mark.asyncio
    async def test_the_last_one_out_shuts_it_down_once(self, built):
        async with shared_sdk(_cfg()):
            async with shared_sdk(_cfg()):
                pass
            built[0].shutdown.assert_not_awaited()
        built[0].shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_open_stream_is_stopped_before_shutdown(self, built):
        async with shared_sdk(_cfg()) as sdk:
            sdk.instance.is_listening = True
        sdk.instance.stop_listening.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_teardown_failure_is_suppressed(self, built):
        async with shared_sdk(_cfg()) as sdk:
            sdk.shutdown = AsyncMock(side_effect=RuntimeError("transport already gone"))
        sdk.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failed_initialize_is_not_cached(self, built, monkeypatch):
        calls = {"n": 0}
        real = sdk_module.MaximemSynapSDK

        def flaky(**kw):
            sdk = real(**kw)
            calls["n"] += 1
            if calls["n"] == 1:
                sdk.initialize = AsyncMock(side_effect=RuntimeError("auth"))
            return sdk

        monkeypatch.setattr(sdk_module, "MaximemSynapSDK", flaky)
        with pytest.raises(RuntimeError, match="auth"):
            async with shared_sdk(_cfg()):
                pass
        async with shared_sdk(_cfg()) as sdk:
            assert sdk is not None

    @pytest.mark.asyncio
    async def test_a_missing_api_key_names_the_config(self, built, monkeypatch):
        monkeypatch.delenv("SYNAP_API_KEY")
        with pytest.raises(RuntimeError, match="SimpleNamespace.api_key"):
            async with shared_sdk(_cfg()):
                pass


class TestEveryFactoryUsesTheSharedSdk:
    @pytest.mark.asyncio
    async def test_all_four_components_share_one_sdk_and_it_is_shut_down_once(self, built):
        async with AsyncExitStack() as stack:
            editor = await stack.enter_async_context(synap_memory_client(SynapMemoryClientConfig(), builder=None))
            await stack.enter_async_context(synap_short_term_function(SynapShortTermConfig(), builder=None))
            exporter = await stack.enter_async_context(synap_stream_exporter(SynapStreamExporterConfig(), builder=None))
            middleware = await stack.enter_async_context(
                synap_history_middleware(SynapHistoryMiddlewareConfig(), builder=None))

            assert len(built) == 1
            assert editor.sdk is exporter.sdk is middleware._sdk is built[0]
            built[0].shutdown.assert_not_awaited()
        built[0].shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_exporter_opens_the_stream_at_start_up_without_listen_being_called_by_hand(self, built, monkeypatch):
        real = sdk_module.MaximemSynapSDK

        def with_listen(**kw):
            sdk = real(**kw)
            sdk.instance.listen = AsyncMock(side_effect=lambda: setattr(sdk.instance, "is_listening", True))
            return sdk

        monkeypatch.setattr(sdk_module, "MaximemSynapSDK", with_listen)
        async with synap_stream_exporter(SynapStreamExporterConfig(), builder=None):
            built[0].instance.listen.assert_awaited_once()
            assert built[0].instance.is_listening is True

    @pytest.mark.asyncio
    async def test_opting_out_means_the_exporter_never_opens_a_stream(self, built, monkeypatch):
        monkeypatch.setenv("SYNAP_AUTO_LISTEN", "false")
        async with synap_stream_exporter(SynapStreamExporterConfig(), builder=None):
            built[0].instance.listen.assert_not_called()
