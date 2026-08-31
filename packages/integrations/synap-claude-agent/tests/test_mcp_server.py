"""Tests for synap_claude_agent.mcp_server — create_synap_mcp_server and _build_synap_tools.

Documented contract (mcp_server.py docstring):

synap_search tool:
- Returns formatted context as plain text.
- Missing ``query`` → isError=True response (no SDK call).
- SDK failure → isError=False "no context available" response (graceful degrade,
  loop doesn't wedge).
- Empty formatted_context → "synap_search: no relevant context." sentinel.

synap_remember tool:
- Returns a text message containing the ingestion_id on success.
- Missing/blank ``content`` → isError=True response (no SDK call).
- Non-dict ``metadata`` is coerced to {} and source="claude_agent_sdk" is set.
- SDK failure → isError=True response (write failures are OBSERVABLE; the tool
  surfaces them so ingestion outages are noticed).

create_synap_mcp_server:
- Validates sdk + user_id at construction time (ValueError).
- Returns a McpSdkServerConfig (dict with type="sdk" key).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_claude_agent.mcp_server import _build_synap_tools, create_synap_mcp_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_sdk(formatted: str | None = "User context here"):
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=formatted))
    sdk.memories = MagicMock()
    create_result = MagicMock()
    create_result.ingestion_id = "ing-abc"
    sdk.memories.create = AsyncMock(return_value=create_result)
    return sdk


def _get_tools(sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"):
    """Build tool list and return (search_tool_fn, remember_tool_fn)."""
    tools = _build_synap_tools(
        sdk=sdk,
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        mode=mode,
    )
    assert len(tools) == 2
    # tools[0] is synap_search, tools[1] is synap_remember
    # Each SdkMcpTool is callable (its __call__ invokes the wrapped async fn)
    return tools[0], tools[1]


async def _call_tool(sdk_tool, args: dict):
    """Invoke a SdkMcpTool via its .handler attribute."""
    return await sdk_tool.handler(args)


# ---------------------------------------------------------------------------
# Understand SdkMcpTool structure (smoke)
# ---------------------------------------------------------------------------


def test_build_synap_tools_returns_two_tools():
    sdk = _fake_sdk()
    tools = _build_synap_tools(
        sdk=sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"
    )
    assert len(tools) == 2


def test_build_synap_tools_first_is_synap_search():
    sdk = _fake_sdk()
    tools = _build_synap_tools(
        sdk=sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"
    )
    # SdkMcpTool should carry a name attr
    assert getattr(tools[0], "name", None) == "synap_search"


def test_build_synap_tools_second_is_synap_remember():
    sdk = _fake_sdk()
    tools = _build_synap_tools(
        sdk=sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"
    )
    assert getattr(tools[1], "name", None) == "synap_remember"


# ---------------------------------------------------------------------------
# create_synap_mcp_server — construction & validation
# ---------------------------------------------------------------------------


class TestCreateSynapMcpServer:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_mcp_server(None, user_id="alice")  # type: ignore[arg-type]

    def test_requires_user_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_synap_mcp_server(sdk, user_id="")

    @pytest.mark.xfail(
        reason=(
            "mcp_server.py:create_synap_mcp_server — validation uses `if not user_id` "
            "but a whitespace-only string is truthy; whitespace-only user_id slips "
            "through. Expected ValueError, actual: no error raised."
        ),
        strict=False,
    )
    def test_requires_user_id_not_just_spaces(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_synap_mcp_server(sdk, user_id="   ")

    def test_returns_mcp_sdk_server_config(self):
        """Returns a McpSdkServerConfig (TypedDict with type='sdk')."""
        sdk = _fake_sdk()
        result = create_synap_mcp_server(sdk, user_id="alice")
        assert isinstance(result, dict)
        assert result.get("type") == "sdk"

    def test_returned_config_has_name(self):
        sdk = _fake_sdk()
        result = create_synap_mcp_server(sdk, user_id="alice", name="my_synap")
        assert result.get("name") == "my_synap"

    def test_returned_config_has_default_name(self):
        sdk = _fake_sdk()
        result = create_synap_mcp_server(sdk, user_id="alice")
        assert result.get("name") == "synap"

    def test_returned_config_has_instance(self):
        sdk = _fake_sdk()
        result = create_synap_mcp_server(sdk, user_id="alice")
        assert "instance" in result

    def test_custom_name_and_version_propagated(self):
        sdk = _fake_sdk()
        result = create_synap_mcp_server(
            sdk, user_id="alice", name="my_synap", version="2.0.0"
        )
        assert result.get("name") == "my_synap"


# ---------------------------------------------------------------------------
# synap_search tool — happy path
# ---------------------------------------------------------------------------


class TestSynapSearchHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self):
        sdk = _fake_sdk(formatted="User loves Python.")
        search, _ = _get_tools(sdk)
        result = await search.handler({"query": "programming languages"})
        assert result["content"][0]["text"] == "User loves Python."
        assert result.get("isError") is not True

    @pytest.mark.asyncio
    async def test_calls_sdk_fetch_with_correct_args(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(
            sdk, user_id="bob", customer_id="acme", conversation_id="conv-x", mode="fast"
        )
        await search.handler({"query": "coffee", "max_results": 3})
        sdk.fetch.assert_awaited_once_with(
            conversation_id="conv-x",
            user_id="bob",
            customer_id="acme",
            search_query=["coffee"],
            max_results=3,
            mode="fast",
            include_conversation_context=False,
        )

    @pytest.mark.asyncio
    async def test_default_max_results_is_10(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk)
        await search.handler({"query": "something"})
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["max_results"] == 10

    @pytest.mark.asyncio
    async def test_custom_max_results_passed_through(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk)
        await search.handler({"query": "q", "max_results": 7})
        assert sdk.fetch.call_args.kwargs["max_results"] == 7

    @pytest.mark.asyncio
    async def test_empty_formatted_context_returns_sentinel(self):
        sdk = _fake_sdk(formatted="")
        search, _ = _get_tools(sdk)
        result = await search.handler({"query": "anything"})
        assert result["content"][0]["text"] == "synap_search: no relevant context."

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_sentinel(self):
        sdk = _fake_sdk(formatted=None)
        search, _ = _get_tools(sdk)
        result = await search.handler({"query": "anything"})
        assert result["content"][0]["text"] == "synap_search: no relevant context."

    @pytest.mark.asyncio
    async def test_empty_customer_id_sent_as_none_to_fetch(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk, customer_id="")
        await search.handler({"query": "q"})
        assert sdk.fetch.call_args.kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_with_shared_mock_sdk(self, mock_sdk):
        """Smoke test using the shared mock_sdk fixture."""
        tools = _build_synap_tools(
            sdk=mock_sdk, user_id="alice", customer_id="",
            conversation_id=None, mode="accurate"
        )
        search = tools[0]
        result = await search.handler({"query": "test"})
        assert "content" in result
        assert result["content"][0]["type"] == "text"


# ---------------------------------------------------------------------------
# synap_search tool — missing/invalid arguments
# ---------------------------------------------------------------------------


class TestSynapSearchInvalidArgs:
    @pytest.mark.asyncio
    async def test_missing_query_returns_is_error(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk)
        result = await search.handler({})
        assert result["isError"] is True
        assert "missing `query`" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_query_does_not_call_sdk(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk)
        await search.handler({})
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_query_returns_is_error(self):
        sdk = _fake_sdk()
        search, _ = _get_tools(sdk)
        result = await search.handler({"query": ""})
        assert result["isError"] is True
        sdk.fetch.assert_not_awaited()


# ---------------------------------------------------------------------------
# synap_search tool — SDK failure (graceful degrade)
# ---------------------------------------------------------------------------


class TestSynapSearchFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_returns_no_context_message(self):
        """SDK fetch failure → isError=False, explanatory message (agent loop continues)."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        tools = _build_synap_tools(
            sdk=sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"
        )
        result = await tools[0].handler({"query": "anything"})
        assert result.get("isError") is False
        assert "no context available" in result["content"][0]["text"]
        assert "RuntimeError" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_sdk_failure_logs_error(self, caplog):
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("fetch gone"))
        tools = _build_synap_tools(
            sdk=sdk, user_id="alice", customer_id="", conversation_id=None, mode="accurate"
        )
        with caplog.at_level(logging.ERROR, logger="synap_claude_agent.mcp_server"):
            await tools[0].handler({"query": "q"})
        assert any("sdk.fetch failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_search_degrades(self, failing_sdk):
        """Shared failing_sdk fixture: synap_search gracefully degrades."""
        tools = _build_synap_tools(
            sdk=failing_sdk, user_id="alice", customer_id="",
            conversation_id=None, mode="accurate"
        )
        result = await tools[0].handler({"query": "q"})
        assert result.get("isError") is False
        assert "no context available" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# synap_remember tool — happy path
# ---------------------------------------------------------------------------


class TestSynapRememberHappyPath:
    @pytest.mark.asyncio
    async def test_returns_ingestion_id_in_message(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "User prefers dark mode."})
        assert result["content"][0]["text"] == "synap_remember: recorded (ingestion_id=ing-abc)."

    @pytest.mark.asyncio
    async def test_calls_memories_create_with_correct_args(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk, user_id="bob", customer_id="acme")
        await remember.handler({"content": "User is 30 years old."})
        sdk.memories.create.assert_awaited_once_with(
            document="User is 30 years old.",
            user_id="bob",
            customer_id="acme",
            metadata={"source": "claude_agent_sdk"},
        )

    @pytest.mark.asyncio
    async def test_empty_customer_id_sent_as_none(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk, user_id="alice", customer_id="")
        await remember.handler({"content": "Something to remember."})
        kw = sdk.memories.create.call_args.kwargs
        assert kw["customer_id"] is None

    @pytest.mark.asyncio
    async def test_custom_metadata_merged_with_source(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        await remember.handler(
            {"content": "User likes cats.", "metadata": {"tag": "animal"}}
        )
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"]["tag"] == "animal"
        assert kw["metadata"]["source"] == "claude_agent_sdk"

    @pytest.mark.asyncio
    async def test_metadata_source_not_overwritten_if_already_set(self):
        """If caller provides metadata.source, setdefault won't override it."""
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        await remember.handler(
            {"content": "fact", "metadata": {"source": "user_explicit"}}
        )
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"]["source"] == "user_explicit"

    @pytest.mark.asyncio
    async def test_non_dict_metadata_coerced_to_dict(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        # pass a string as metadata — should be silently coerced to {}
        await remember.handler({"content": "fact", "metadata": "not-a-dict"})
        kw = sdk.memories.create.call_args.kwargs
        assert isinstance(kw["metadata"], dict)
        assert kw["metadata"]["source"] == "claude_agent_sdk"

    @pytest.mark.asyncio
    async def test_missing_metadata_defaults_to_source_only(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        await remember.handler({"content": "remember me"})
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"] == {"source": "claude_agent_sdk"}

    @pytest.mark.asyncio
    async def test_result_contains_content_list(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "test content"})
        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_with_shared_mock_sdk(self, mock_sdk):
        tools = _build_synap_tools(
            sdk=mock_sdk, user_id="alice", customer_id="",
            conversation_id=None, mode="accurate"
        )
        result = await tools[1].handler({"content": "remember this"})
        assert "content" in result
        assert "ingestion_id" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# synap_remember tool — missing/invalid arguments
# ---------------------------------------------------------------------------


class TestSynapRememberInvalidArgs:
    @pytest.mark.asyncio
    async def test_missing_content_returns_is_error(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        result = await remember.handler({})
        assert result["isError"] is True
        assert "missing `content`" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_content_does_not_call_sdk(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        await remember.handler({})
        sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_content_returns_is_error(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": ""})
        assert result["isError"] is True
        sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_returns_is_error(self):
        sdk = _fake_sdk()
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "   "})
        assert result["isError"] is True
        sdk.memories.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# synap_remember tool — SDK failure (OBSERVABLE error)
# ---------------------------------------------------------------------------


class TestSynapRememberFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_returns_is_error_true(self):
        """Write failures are OBSERVABLE (isError=True) so ingestion outages are noticed."""
        sdk = _fake_sdk()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("write failed"))
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "important fact"})
        assert result["isError"] is True
        assert "ingestion failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_sdk_failure_message_contains_exception_text(self):
        sdk = _fake_sdk()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("disk full"))
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "fact"})
        assert "disk full" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_sdk_failure_logs_error(self, caplog):
        sdk = _fake_sdk()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("write explode"))
        _, remember = _get_tools(sdk)
        with caplog.at_level(logging.ERROR, logger="synap_claude_agent.mcp_server"):
            await remember.handler({"content": "fact"})
        assert any("sdk.memories.create failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_remember_is_error(self, failing_sdk):
        """Shared failing_sdk: synap_remember surfaces the failure as isError=True."""
        tools = _build_synap_tools(
            sdk=failing_sdk, user_id="alice", customer_id="",
            conversation_id=None, mode="accurate"
        )
        result = await tools[1].handler({"content": "remember this"})
        assert result["isError"] is True


# ---------------------------------------------------------------------------
# Ingestion_id edge cases
# ---------------------------------------------------------------------------


class TestIngestionIdEdgeCases:
    @pytest.mark.asyncio
    async def test_none_ingestion_id_gives_empty_string(self):
        sdk = _fake_sdk()
        result_obj = MagicMock()
        result_obj.ingestion_id = None
        sdk.memories.create = AsyncMock(return_value=result_obj)
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "no id fact"})
        # When ingestion_id is None, the product uses empty string:
        # "synap_remember: recorded (ingestion_id=)."
        assert "ingestion_id=)." in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_ingestion_id_attr_gives_empty_string(self):
        sdk = _fake_sdk()
        result_obj = MagicMock(spec=[])  # no ingestion_id attribute
        sdk.memories.create = AsyncMock(return_value=result_obj)
        _, remember = _get_tools(sdk)
        result = await remember.handler({"content": "no attr fact"})
        assert "ingestion_id=" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# The identifier contract: customer_id is B2B only
# ---------------------------------------------------------------------------
#
# customer_id is bound at construction and injected into every call, so the model
# cannot leave it out. On a B2C instance the API refuses it, which used to mean
# every search and every write failed forever for a reason no tool argument could
# fix. These pin the conditional injection, and pin just as hard that B2B still
# gets the id it REQUIRES.

from synap_claude_agent.mcp_server import (  # noqa: E402
    B2C_ISOLATION,
    _CustomerScope,
    _is_b2c_rejection,
    _sdk_isolation,
)

B2B_ISOLATION = "strict"


def _sdk_in_mode(isolation, formatted="ctx"):
    """A fake SDK whose whoami-derived mode is `isolation` (None = not known)."""
    sdk = _fake_sdk(formatted)
    if isolation is not None:
        sdk._user_context_isolation = isolation
    return sdk


class _Refused(Exception):
    """Shaped like the rejection the SDK raises and the API returns."""

    def __init__(self):
        super().__init__(
            "customer_id is not accepted on this instance, which is B2C "
            f"(user_context_isolation='{B2C_ISOLATION}'). Send user_id only."
        )


class TestModeDetection:
    def test_reads_the_mode_the_sdk_learned_from_whoami(self):
        assert _sdk_isolation(_sdk_in_mode(B2C_ISOLATION)) == B2C_ISOLATION

    def test_a_mock_attribute_is_not_an_answer(self):
        """A MagicMock answers every attribute. Treating that as a mode would let
        a test double decide scoping, so only a real non-empty string counts."""
        assert _sdk_isolation(MagicMock()) is None

    def test_a_property_that_raises_is_not_an_answer(self):
        class Exploding:
            @property
            def user_context_isolation(self):
                raise RuntimeError("boom")

        assert _sdk_isolation(Exploding()) is None

    def test_recognises_the_rejection_by_text(self):
        assert _is_b2c_rejection(_Refused())

    def test_recognises_the_api_error_code(self):
        assert _is_b2c_rejection(Exception("customer_id_not_accepted_on_b2c"))

    def test_an_unrelated_failure_is_not_a_rejection(self):
        """Otherwise a timeout would silently drop the id on a B2B instance,
        turning a transient outage into permanently wrong scoping."""
        assert not _is_b2c_rejection(TimeoutError("connection timed out"))


class TestCustomerScope:
    def test_b2b_keeps_the_id(self):
        """The half that must not move: customer_id is REQUIRED on B2B."""
        scope = _CustomerScope(_sdk_in_mode(B2B_ISOLATION), "acme")
        assert scope.value() == "acme"
        assert not scope.suppressed

    def test_b2c_drops_the_id_before_the_first_call(self):
        scope = _CustomerScope(_sdk_in_mode(B2C_ISOLATION), "acme")
        assert scope.value() is None
        assert scope.suppressed

    def test_unknown_mode_keeps_the_id(self):
        """Not knowing must mean not acting. Guessing B2C here would strip a B2B
        caller's mandatory field against any server that has not shipped the
        whoami field, or any SDK not yet initialised."""
        scope = _CustomerScope(_sdk_in_mode(None), "acme")
        assert scope.value() == "acme"

    def test_learns_from_the_first_rejection(self):
        scope = _CustomerScope(_sdk_in_mode(None), "acme")
        assert scope.note_rejection(_Refused()) is True
        assert scope.value() is None

    def test_does_not_learn_from_an_unrelated_failure(self):
        scope = _CustomerScope(_sdk_in_mode(None), "acme")
        assert scope.note_rejection(TimeoutError("boom")) is False
        assert scope.value() == "acme"

    def test_no_configured_id_means_nothing_to_learn(self):
        scope = _CustomerScope(_sdk_in_mode(None), "")
        assert scope.note_rejection(_Refused()) is False
        assert scope.value() is None

    def test_the_warning_names_the_argument_to_remove(self, caplog):
        with caplog.at_level(logging.WARNING):
            _CustomerScope(_sdk_in_mode(B2C_ISOLATION), "acme")
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "acme" in text
        assert "create_synap_mcp_server" in text

    def test_the_warning_is_logged_once_not_per_call(self, caplog):
        scope = _CustomerScope(_sdk_in_mode(None), "acme")
        with caplog.at_level(logging.WARNING):
            scope.note_rejection(_Refused())
            scope.note_rejection(_Refused())
        assert len([r for r in caplog.records if "customer_id" in r.getMessage()]) == 1


@pytest.mark.asyncio
class TestToolsHonourTheContract:
    async def test_b2c_search_omits_the_customer_id(self):
        sdk = _sdk_in_mode(B2C_ISOLATION)
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        await _call_tool(search, {"query": "q"})
        assert sdk.fetch.call_args.kwargs["customer_id"] is None

    async def test_b2b_search_still_sends_the_customer_id(self):
        sdk = _sdk_in_mode(B2B_ISOLATION)
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        await _call_tool(search, {"query": "q"})
        assert sdk.fetch.call_args.kwargs["customer_id"] == "acme"

    async def test_b2c_remember_omits_the_customer_id(self):
        sdk = _sdk_in_mode(B2C_ISOLATION)
        _, remember = _get_tools(sdk, user_id="alice", customer_id="acme")
        await _call_tool(remember, {"content": "x"})
        assert sdk.memories.create.call_args.kwargs["customer_id"] is None

    async def test_a_rejected_search_is_retried_without_the_id_and_succeeds(self):
        """The SDK may not have been initialised, so the first call is where the
        instance says no. Nothing was read, so retrying is safe, and it is what
        turns "the first call always fails" into "it works"."""
        sdk = _sdk_in_mode(None)
        sdk.fetch = AsyncMock(
            side_effect=[_Refused(), MagicMock(formatted_context="recovered")]
        )
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        result = await _call_tool(search, {"query": "q"})
        assert sdk.fetch.await_count == 2
        assert sdk.fetch.await_args_list[0].kwargs["customer_id"] == "acme"
        assert sdk.fetch.await_args_list[1].kwargs["customer_id"] is None
        assert "recovered" in result["content"][0]["text"]
        assert not result.get("isError")

    async def test_a_rejected_write_is_retried_without_the_id_and_succeeds(self):
        sdk = _sdk_in_mode(None)
        ok = MagicMock()
        ok.ingestion_id = "ing-retry"
        sdk.memories.create = AsyncMock(side_effect=[_Refused(), ok])
        _, remember = _get_tools(sdk, user_id="alice", customer_id="acme")
        result = await _call_tool(remember, {"content": "x"})
        assert sdk.memories.create.await_count == 2
        assert "ing-retry" in result["content"][0]["text"]
        assert not result.get("isError")

    async def test_the_id_is_dropped_for_every_later_call_too(self):
        """Learning once and forgetting would pay the rejection on every call."""
        sdk = _sdk_in_mode(None)
        sdk.fetch = AsyncMock(
            side_effect=[_Refused(), MagicMock(formatted_context="a"),
                         MagicMock(formatted_context="b")]
        )
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        await _call_tool(search, {"query": "one"})
        await _call_tool(search, {"query": "two"})
        assert sdk.fetch.await_count == 3, "the second call must not be retried"
        assert sdk.fetch.await_args_list[2].kwargs["customer_id"] is None

    async def test_a_persistent_rejection_is_never_reported_as_no_context(self):
        """The read path degrades quietly on outages, and must NOT degrade quietly
        on a refused shape: "no relevant context" for a call the instance rejects
        is the silent failure this whole contract exists to remove."""
        sdk = _sdk_in_mode(None)
        sdk.fetch = AsyncMock(side_effect=_Refused())
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        result = await _call_tool(search, {"query": "q"})
        text = result["content"][0]["text"]
        assert result["isError"] is True
        assert "no context available" not in text
        assert B2C_ISOLATION in text
        assert "Remove customer_id" in text

    async def test_a_persistent_rejection_on_write_names_the_fix(self):
        sdk = _sdk_in_mode(None)
        sdk.memories.create = AsyncMock(side_effect=_Refused())
        _, remember = _get_tools(sdk, user_id="alice", customer_id="acme")
        result = await _call_tool(remember, {"content": "x"})
        assert result["isError"] is True
        assert "Remove customer_id" in result["content"][0]["text"]

    async def test_an_ordinary_read_failure_still_degrades_quietly(self):
        """The non-contract path is unchanged: a timeout must not wedge the loop."""
        sdk = _sdk_in_mode(None)
        sdk.fetch = AsyncMock(side_effect=TimeoutError("boom"))
        search, _ = _get_tools(sdk, user_id="alice", customer_id="acme")
        result = await _call_tool(search, {"query": "q"})
        assert result["isError"] is False
        assert "no context available" in result["content"][0]["text"]
