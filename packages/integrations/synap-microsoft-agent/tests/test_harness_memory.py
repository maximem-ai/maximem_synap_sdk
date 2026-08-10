"""Tests for the harness memory wiring — the provider and the factory.

The test that matters most is ``test_factory_returns_a_history_provider``.
``create_harness_agent`` has exactly one ``history_provider`` slot, and the
whole D2 footgun rests on ``MemoryContextProvider`` being a ``HistoryProvider``.
If an upstream release changes that base class, this fails in CI instead of
silently breaking a user's agent.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from tests._harness import requires_harness

pytestmark = requires_harness


def build_provider(sdk, **kwargs):
    from synap_microsoft_agent import create_synap_harness_memory

    kwargs.setdefault("user_id", "alice")
    kwargs.setdefault("customer_id", "acme")
    return create_synap_harness_memory(sdk, **kwargs)


class TestFactory:
    def test_factory_returns_a_history_provider(self, mock_sdk):
        """The base-class assumption D2 rests on. Breaks loudly if it changes."""
        from agent_framework import HistoryProvider

        assert isinstance(build_provider(mock_sdk), HistoryProvider)

    def test_factory_builds_a_synap_store(self, mock_sdk):
        from synap_microsoft_agent import SynapMemoryStore

        provider = build_provider(mock_sdk)
        assert isinstance(provider.store, SynapMemoryStore)
        assert provider.store.user_id == "alice"

    def test_factory_passes_provider_kwargs_through(self, mock_sdk):
        provider = build_provider(mock_sdk, recent_turns=4, selection_limit=7)
        assert provider.recent_turns == 4
        assert provider.selection_limit == 7

    def test_factory_forwards_include_recall(self, mock_sdk):
        assert build_provider(mock_sdk, include_recall=False).store.include_recall is False

    def test_prebuilt_store_is_accepted(self, mock_sdk):
        from synap_microsoft_agent import SynapMemoryStore, create_synap_harness_memory

        store = SynapMemoryStore(mock_sdk, user_id="bob")
        provider = create_synap_harness_memory(mock_sdk, store=store)
        assert provider.store is store

    def test_prebuilt_store_plus_scope_arguments_is_rejected(self, mock_sdk):
        from synap_microsoft_agent import SynapMemoryStore, create_synap_harness_memory

        store = SynapMemoryStore(mock_sdk, user_id="bob")
        with pytest.raises(ValueError, match="pass one or the other"):
            create_synap_harness_memory(mock_sdk, store=store, user_id="alice")


class TestQueryConditionedRecall:
    """The recall block must be scoped to the turn's question.

    Without this the block is an unqueried scope fetch, which returns what the
    scope finds generally salient rather than what was asked. Measured on the
    bench: "when do we send invoices" came back with code-review preferences.
    """

    async def test_before_run_passes_the_question_to_the_fetch(self, mock_sdk):
        from unittest.mock import MagicMock

        from agent_framework import AgentSession, Message

        provider = build_provider(mock_sdk)
        context = _FakeContext("when do we send invoices to customers?")

        await provider.before_run(
            agent=MagicMock(client=None),
            session=AgentSession(session_id="s1"),
            context=context,
            state={},
        )

        queries = [
            call.kwargs.get("search_query")
            for call in mock_sdk.fetch.await_args_list
            if call.kwargs.get("search_query")
        ]
        assert queries, "no queried fetch was made"
        assert "invoices" in queries[0][0]

    async def test_the_query_does_not_outlive_the_turn(self, mock_sdk):
        from unittest.mock import MagicMock

        from agent_framework import AgentSession

        provider = build_provider(mock_sdk)
        await provider.before_run(
            agent=MagicMock(client=None),
            session=AgentSession(session_id="s1"),
            context=_FakeContext("a question"),
            state={},
        )

        mock_sdk.fetch.reset_mock()
        provider.store.get_index_text(
            AgentSession(session_id="s2"), source_id="memory",
            line_limit=200, line_length=150,
        )
        assert mock_sdk.fetch.await_args.kwargs["search_query"] is None

    async def test_a_failing_turn_still_clears_the_query(self, mock_sdk):
        from unittest.mock import MagicMock

        from agent_framework import AgentSession

        provider = build_provider(mock_sdk)
        provider.store.rebuild_index = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError):
            await provider.before_run(
                agent=MagicMock(client=None),
                session=AgentSession(session_id="s1"),
                context=_FakeContext("a question"),
                state={},
            )

        mock_sdk.fetch.reset_mock()
        provider.store.rebuild_index = lambda *a, **k: []
        provider.store.get_index_text(
            AgentSession(session_id="s2"), source_id="memory",
            line_limit=200, line_length=150,
        )
        assert mock_sdk.fetch.await_args.kwargs["search_query"] is None


class _FakeContext:
    """The slice of ``SessionContext`` ``before_run`` reaches for."""

    def __init__(self, question: str) -> None:
        from agent_framework import Message

        self.session_id = "s1"
        self.input_messages = [Message(role="user", contents=[question])]
        self.response = None
        self.instructions: list = []
        self.messages: list = []
        self.tools: list = []

    def extend_instructions(self, source_id, value):
        self.instructions.extend(value if isinstance(value, list) else [value])

    def extend_messages(self, source, messages, *, origin_session_ids=None):
        self.messages.extend(messages if isinstance(messages, list) else [messages])

    def extend_tools(self, source_id, value):
        self.tools.extend(value if isinstance(value, list) else [value])


class TestConversationIds:
    def test_ids_are_valid_uuids(self, mock_sdk):
        """Synap validates ``conversation_id`` as a UUID client-side, and a MAF
        session id is an arbitrary string."""
        import uuid

        from synap_microsoft_agent import session_conversation_id

        uuid.UUID(session_conversation_id("session-not-a-uuid"))

    def test_ids_are_deterministic(self):
        from synap_microsoft_agent import session_conversation_id

        assert session_conversation_id("s1", owner="/acme/alice") == (
            session_conversation_id("s1", owner="/acme/alice")
        )

    def test_owners_do_not_collide_on_the_same_session_id(self):
        from synap_microsoft_agent import session_conversation_id

        assert session_conversation_id("s1", owner="/acme/alice") != (
            session_conversation_id("s1", owner="/acme/bob")
        )

    def test_a_missing_session_id_still_resolves(self):
        import uuid

        from synap_microsoft_agent import session_conversation_id

        uuid.UUID(session_conversation_id(None))


class TestTranscripts:
    async def test_save_records_messages_to_synap(self, mock_sdk):
        from agent_framework import Message

        provider = build_provider(mock_sdk)
        await provider.save_messages(
            "s1",
            [Message(role="user", contents=["hello"])],
            state={},
        )

        mock_sdk.conversation.record_message.assert_awaited_once()
        kwargs = mock_sdk.conversation.record_message.await_args.kwargs
        assert kwargs["content"] == "hello"
        assert kwargs["role"] == "user"
        assert kwargs["user_id"] == "alice"

    async def test_save_touches_no_filesystem(self, mock_sdk, tmp_path):
        """The point of D4: transcripts go to the cloud, not to disk."""
        from agent_framework import Message

        from synap_microsoft_agent import SynapMemoryStore, create_synap_harness_memory

        root = tmp_path / "transcripts"
        store = SynapMemoryStore(mock_sdk, user_id="alice", transcripts_root=str(root))
        provider = create_synap_harness_memory(mock_sdk, store=store)
        provider.user_id = "alice"

        await provider.save_messages("s1", [Message(role="user", contents=["hi"])], state={})
        assert not root.exists()

    async def test_system_and_tool_messages_are_dropped_not_relabelled(self, mock_sdk):
        """Synap accepts user/assistant only. Relabelling a tool result as user
        speech would poison the extracted memory with words nobody said."""
        from agent_framework import Message

        provider = build_provider(mock_sdk)
        await provider.save_messages(
            "s1",
            [
                Message(role="system", contents=["you are helpful"]),
                Message(role="user", contents=["hello"]),
            ],
            state={},
        )

        assert mock_sdk.conversation.record_message.await_count == 1
        assert mock_sdk.conversation.record_message.await_args.kwargs["content"] == "hello"

    async def test_blank_messages_are_skipped(self, mock_sdk):
        from agent_framework import Message

        provider = build_provider(mock_sdk)
        await provider.save_messages("s1", [Message(role="user", contents=["   "])], state={})
        mock_sdk.conversation.record_message.assert_not_awaited()

    async def test_no_messages_is_a_no_op(self, mock_sdk):
        await build_provider(mock_sdk).save_messages("s1", [], state={})
        mock_sdk.conversation.record_message.assert_not_awaited()

    async def test_history_message_filter_is_honoured(self, mock_sdk):
        from agent_framework import Message

        provider = build_provider(mock_sdk, history_message_filter=lambda m: None)
        await provider.save_messages("s1", [Message(role="user", contents=["secret"])], state={})
        mock_sdk.conversation.record_message.assert_not_awaited()

    async def test_scopeless_provider_skips_rather_than_misfiling(self, mock_sdk):
        from agent_framework import Message

        from synap_microsoft_agent import SynapMemoryStore, create_synap_harness_memory

        store = SynapMemoryStore(mock_sdk, customer_id="acme")
        provider = create_synap_harness_memory(mock_sdk, store=store)

        await provider.save_messages("s1", [Message(role="user", contents=["hi"])], state={})
        mock_sdk.conversation.record_message.assert_not_awaited()

    async def test_save_failure_raises(self, failing_sdk):
        from agent_framework import Message

        from synap_integrations_common import SynapIntegrationError

        provider = build_provider(failing_sdk)
        with pytest.raises(SynapIntegrationError):
            await provider.save_messages("s1", [Message(role="user", contents=["hi"])], state={})

    async def test_get_loads_messages_back(self, mock_sdk):
        recorded = type("M", (), {"role": "user", "content": "hello"})()
        mock_sdk.conversation.context.get_context_for_prompt = AsyncMock(
            return_value=type("R", (), {"recent_messages": [recorded]})()
        )

        messages = await build_provider(mock_sdk).get_messages("s1", state={})
        assert len(messages) == 1
        assert messages[0].text == "hello"

    async def test_get_degrades_to_empty_on_failure(self, failing_sdk):
        """A transcript read sits inside ``before_run``; raising would end the turn."""
        assert await build_provider(failing_sdk).get_messages("s1", state={}) == []

    async def test_get_and_save_agree_on_the_conversation_id(self, mock_sdk):
        from agent_framework import Message

        provider = build_provider(mock_sdk)
        state = provider.store.export_provider_state(
            __import__("agent_framework").AgentSession(session_id="s1")
        )

        await provider.save_messages("s1", [Message(role="user", contents=["hi"])], state=state)
        await provider.get_messages("s1", state=state)

        written = mock_sdk.conversation.record_message.await_args.kwargs["conversation_id"]
        read = mock_sdk.conversation.context.get_context_for_prompt.await_args.kwargs[
            "conversation_id"
        ]
        assert written == read
