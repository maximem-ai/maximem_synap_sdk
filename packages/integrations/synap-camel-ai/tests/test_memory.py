"""Tests for SynapAgentMemory — the CAMEL-AI native memory backend.

Covers the architect-reviewed contract: construction does no SDK I/O, completed
turns persist the accumulated transcript to a stable doc id, retrieve augments the
real local conversation with prepended Synap recall, reads degrade, and the in-loop
persistence write is best-effort (log-not-raise) by default.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from camel.memories import AgentMemory, MemoryRecord
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole

from synap_camel_ai import SynapAgentMemory
from synap_integrations_common import SynapIntegrationError

_MEM_TAG = "<synap_long_term_memory>"


# ── record factories ─────────────────────────────────────────────────────────


def user_rec(text):
    return MemoryRecord(
        message=BaseMessage.make_user_message("user", text),
        role_at_backend=OpenAIBackendRole.USER,
    )


def asst_rec(text):
    return MemoryRecord(
        message=BaseMessage.make_assistant_message("assistant", text),
        role_at_backend=OpenAIBackendRole.ASSISTANT,
    )


def sys_rec(text):
    return MemoryRecord(
        message=BaseMessage.make_system_message("system", text),
        role_at_backend=OpenAIBackendRole.SYSTEM,
    )


def contents(records):
    return [r.memory_record.message.content for r in records]


def roles(records):
    return [r.memory_record.role_at_backend for r in records]


# ── construction / validation ────────────────────────────────────────────────


def test_is_agent_memory(mock_sdk):
    assert isinstance(SynapAgentMemory(mock_sdk, "alice"), AgentMemory)


def test_construction_does_no_sdk_io(failing_sdk):
    SynapAgentMemory(failing_sdk, "alice", customer_id="acme")
    failing_sdk.memories.create.assert_not_called()
    failing_sdk.fetch.assert_not_called()


def test_requires_sdk():
    with pytest.raises(ValueError):
        SynapAgentMemory(None, "alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        SynapAgentMemory(mock_sdk, "")


def test_rejects_bad_on_error(mock_sdk):
    with pytest.raises(ValueError):
        SynapAgentMemory(mock_sdk, "alice", on_error="explode")


# ── writes: persistence semantics ────────────────────────────────────────────


def test_user_only_write_does_not_persist(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([user_rec("What's the plan?")])
    mock_sdk.memories.create.assert_not_called()


def test_system_write_is_not_persisted_or_tracked(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([sys_rec("You are helpful")])
    mock_sdk.memories.create.assert_not_called()
    assert mem._turns == []


def test_completed_turn_persists_accumulated_transcript(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice", customer_id="acme")
    mem.write_records([user_rec("What's the plan?")])
    mock_sdk.memories.create.assert_not_called()
    mem.write_records([asst_rec("The plan is X")])
    mock_sdk.memories.create.assert_called_once()
    kwargs = mock_sdk.memories.create.call_args.kwargs
    assert kwargs["document_type"] == "ai-chat-conversation"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert "User: What's the plan?" in kwargs["document"]
    assert "Assistant: The plan is X" in kwargs["document"]


def test_stable_doc_id_and_growing_transcript_across_turns(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([user_rec("q1")])
    mem.write_records([asst_rec("a1")])
    first_doc = mock_sdk.memories.create.call_args.kwargs["document_id"]
    mem.write_records([user_rec("q2")])
    mem.write_records([asst_rec("a2")])
    second = mock_sdk.memories.create.call_args.kwargs
    assert second["document_id"] == first_doc
    assert all(x in second["document"] for x in ("q1", "a1", "q2", "a2"))
    assert mock_sdk.memories.create.call_count == 2


def test_doc_id_tracks_agent_id(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    fallback = mem._doc_id
    mem.agent_id = "agent-42"
    assert mem._doc_id == "camel-agent-42"
    assert mem._doc_id != fallback


def test_persist_failure_is_swallowed_by_default(failing_sdk):
    mem = SynapAgentMemory(failing_sdk, "alice")
    mem.write_records([user_rec("hi")])
    mem.write_records([asst_rec("hello")])  # persist raises internally → swallowed


def test_persist_failure_raises_when_strict(failing_sdk):
    mem = SynapAgentMemory(failing_sdk, "alice", on_error="raise")
    mem.write_records([user_rec("hi")])
    with pytest.raises(SynapIntegrationError):
        mem.write_records([asst_rec("hello")])


# ── reads: recall augmentation ───────────────────────────────────────────────


def test_retrieve_prepends_recall_after_system(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([sys_rec("You are helpful")])
    mem.write_records([user_rec("What's my plan?")])
    recs = mem.retrieve()
    assert len(recs) == 3
    assert roles(recs)[0] == OpenAIBackendRole.SYSTEM  # original system first
    assert _MEM_TAG in contents(recs)[1]  # recall injected second
    assert roles(recs)[1] == OpenAIBackendRole.SYSTEM  # as a system block
    assert roles(recs)[2] == OpenAIBackendRole.USER  # conversation preserved
    assert "What's my plan?" in contents(recs)[2]
    mock_sdk.fetch.assert_called_once()
    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["What's my plan?"]


def test_retrieve_no_user_message_skips_fetch(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([sys_rec("You are helpful")])
    recs = mem.retrieve()
    mock_sdk.fetch.assert_not_called()
    assert not any(_MEM_TAG in (c or "") for c in contents(recs))


def test_retrieve_degrades_to_local_on_fetch_failure(failing_sdk):
    mem = SynapAgentMemory(failing_sdk, "alice")
    mem.write_records([user_rec("hi there")])  # user-only: no persist, no raise
    recs = mem.retrieve()  # fetch raises → degrade
    assert [r.memory_record.message.content for r in recs] == ["hi there"]
    assert not any(_MEM_TAG in (c or "") for c in contents(recs))


def test_retrieve_no_recall_when_context_empty(mock_sdk):
    resp = MagicMock()
    resp.formatted_context = ""
    mock_sdk.fetch = AsyncMock(return_value=resp)
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([user_rec("anything")])
    recs = mem.retrieve()
    assert not any(_MEM_TAG in (c or "") for c in contents(recs))


def test_retrieve_uses_last_user_message_as_query(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([user_rec("first question")])
    mem.write_records([asst_rec("an answer")])
    mem.write_records([user_rec("second question")])
    mem.retrieve()
    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["second question"]


def test_max_results_flows_to_fetch(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice", max_results=17)
    mem.write_records([user_rec("q")])
    mem.retrieve()
    assert mock_sdk.fetch.call_args.kwargs["max_results"] == 17


# ── clear ────────────────────────────────────────────────────────────────────


def test_clear_is_silent_and_resets_turns(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    mem.write_records([user_rec("hi")])
    mem.clear()
    assert mem._turns == []
    assert mem.retrieve() == []


# ── structural: real ChatAgent accepts it ────────────────────────────────────


def test_chatagent_accepts_memory(mock_sdk, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # construction validates presence only
    from camel.agents import ChatAgent

    mem = SynapAgentMemory(mock_sdk, "alice")
    agent = ChatAgent(system_message="You are helpful", memory=mem)
    assert agent.memory is mem
    mock_sdk.memories.create.assert_not_called()  # no I/O during construction
