"""Tests for synap_nemo_agent_toolkit.short_term.

Exercises the programmatic :class:`SynapShortTermFunction` directly so
the test suite doesn't require ``nvidia-nat-core`` to be installed
locally (the @register_function decorator is exercised in CI where the
NAT runtime is present).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Load short_term.py directly to avoid triggering the package __init__
# (which imports editor.py, which needs nat.memory.interfaces).
_HERE = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "synap_nemo_agent_toolkit"))
# Need to make 'maximem_synap' and 'synap_integrations_common' resolvable —
# conftest already inserted SDK + common into sys.path. We also have to
# defer the NAT-dependent decorator import; we do that by loading just the
# class via importlib spec rather than `import synap_nemo_agent_toolkit`.
_spec = importlib.util.spec_from_file_location(
    "_synap_nat_short_term", os.path.join(_PKG_ROOT, "short_term.py")
)
_short_term_module = importlib.util.module_from_spec(_spec)

# Stub out NAT-only imports so module load succeeds without nvidia-nat-core.
# Tests exercise SynapShortTermFunction (which does NOT depend on NAT).
class _NATStub:
    pass


_stub_nat = type(sys)("nat")
_stub_builder = type(sys)("nat.builder")
_stub_builder_builder = type(sys)("nat.builder.builder")
_stub_builder_builder.Builder = _NATStub
_stub_builder_fn_info = type(sys)("nat.builder.function_info")


class _FnInfoStub:
    @classmethod
    def from_fn(cls, fn, description=""):
        return ("FunctionInfo", fn, description)


_stub_builder_fn_info.FunctionInfo = _FnInfoStub
_stub_cli = type(sys)("nat.cli")
_stub_register = type(sys)("nat.cli.register_workflow")


def _register_function(*, config_type=None):
    def _decorator(fn):
        return fn
    return _decorator


_stub_register.register_function = _register_function
_stub_data_models = type(sys)("nat.data_models")
_stub_data_models_common = type(sys)("nat.data_models.common")


class _OptionalSecretStrStub(str):
    pass


def _get_secret_value(v):
    return v if isinstance(v, str) else None


_stub_data_models_common.OptionalSecretStr = _OptionalSecretStrStub
_stub_data_models_common.get_secret_value = _get_secret_value
_stub_data_models_function = type(sys)("nat.data_models.function")


class _FunctionBaseConfigStub:
    """Stand-in for nat.data_models.function.FunctionBaseConfig."""

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._yaml_name = name


_stub_data_models_function.FunctionBaseConfig = _FunctionBaseConfigStub

# Install all stubs before module exec
for mod_name, mod in {
    "nat": _stub_nat,
    "nat.builder": _stub_builder,
    "nat.builder.builder": _stub_builder_builder,
    "nat.builder.function_info": _stub_builder_fn_info,
    "nat.cli": _stub_cli,
    "nat.cli.register_workflow": _stub_register,
    "nat.data_models": _stub_data_models,
    "nat.data_models.common": _stub_data_models_common,
    "nat.data_models.function": _stub_data_models_function,
}.items():
    sys.modules[mod_name] = mod

_spec.loader.exec_module(_short_term_module)
SynapShortTermFunction = _short_term_module.SynapShortTermFunction

from synap_integrations_common import SynapIntegrationError  # noqa: E402


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User asked about pricing.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermFunction(None)  # type: ignore[arg-type]

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermFunction(_fake_sdk(), style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermFunction(_fake_sdk(), on_error="ignore")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_requires_conversation_id(self):
        fn = SynapShortTermFunction(_fake_sdk())
        with pytest.raises(ValueError, match="conversation_id"):
            await fn(conversation_id="")


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_wrapped_st(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        fn = SynapShortTermFunction(sdk)
        out = await fn(conversation_id="conv_abc")
        assert "<synap_short_term_context>" in out
        assert "User is VIP." in out
        assert "</synap_short_term_context>" in out

    @pytest.mark.asyncio
    async def test_passes_style(self):
        sdk = _fake_sdk()
        fn = SynapShortTermFunction(sdk, style="bullet_points")
        await fn(conversation_id="conv_abc")
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_default_conversation_id_used_when_omitted(self):
        sdk = _fake_sdk()
        fn = SynapShortTermFunction(sdk, default_conversation_id="conv_default")
        await fn()  # no explicit conv id
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_default",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_explicit_overrides_default(self):
        sdk = _fake_sdk()
        fn = SynapShortTermFunction(sdk, default_conversation_id="conv_default")
        await fn(conversation_id="conv_override")
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_override",
            style="narrative",
        )


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_empty(self):
        sdk = _fake_sdk(formatted=None, available=False)
        fn = SynapShortTermFunction(sdk)
        assert (await fn(conversation_id="conv_abc")) == ""

    @pytest.mark.asyncio
    async def test_blank_formatted_returns_empty(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        fn = SynapShortTermFunction(sdk)
        assert (await fn(conversation_id="conv_abc")) == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_empty_on_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        fn = SynapShortTermFunction(sdk)
        assert (await fn(conversation_id="conv_abc")) == ""

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        fn = SynapShortTermFunction(sdk, on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await fn(conversation_id="conv_abc")


def _fake_config(**overrides):
    """Stand-in for SynapShortTermConfig.

    The real config is a pydantic FunctionBaseConfig, and this module's NAT
    stubs replace that base with a plain class — so the declared Fields never
    become constructor arguments. Supply the attributes the factory reads.
    """
    from types import SimpleNamespace

    defaults = dict(
        api_key="sk-test",
        instance_id="",
        style="structured",
        preamble_open="",
        preamble_close="",
        on_error="fallback",
        conversation_id="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestTeardown:
    """The factory's ``finally`` block probed ``getattr(sdk, "close", None)``,
    but the SDK's teardown is ``shutdown()`` — there is no ``close()``. The
    guard therefore resolved to ``None`` on every run and teardown never
    happened, leaking the HTTP transport, the gRPC stream and the telemetry
    collector once per workflow.
    """

    def test_the_sdk_really_has_no_close(self):
        """Pins the premise. If the SDK ever grows close(), this test says so
        rather than letting the old probe look correct by accident."""
        from maximem_synap import MaximemSynapSDK

        assert hasattr(MaximemSynapSDK, "shutdown")
        assert not hasattr(MaximemSynapSDK, "close")

    @pytest.mark.asyncio
    async def test_factory_shuts_the_sdk_down_on_exit(self, monkeypatch):
        sdk = _fake_sdk()
        sdk.initialize = AsyncMock()
        sdk.shutdown = AsyncMock()
        del sdk.close  # MagicMock would otherwise auto-create close()

        monkeypatch.setattr(
            _short_term_module, "MaximemSynapSDK", MagicMock(return_value=sdk)
        )
        config = _fake_config()

        agen = _short_term_module.synap_short_term_function(config, builder=None)
        await agen.__anext__()
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

        sdk.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_teardown_failure_is_suppressed(self, monkeypatch):
        sdk = _fake_sdk()
        sdk.initialize = AsyncMock()
        sdk.shutdown = AsyncMock(side_effect=RuntimeError("transport already gone"))
        del sdk.close

        monkeypatch.setattr(
            _short_term_module, "MaximemSynapSDK", MagicMock(return_value=sdk)
        )
        config = _fake_config()

        agen = _short_term_module.synap_short_term_function(config, builder=None)
        await agen.__anext__()
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()  # teardown raised, but must not surface

        sdk.shutdown.assert_awaited_once()
