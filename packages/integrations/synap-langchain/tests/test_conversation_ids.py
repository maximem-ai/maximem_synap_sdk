"""Any string is a conversation id; every REST-backed read sends the UUID the server derives.

The SDK's ``fetch(conversation_id=...)`` refuses a non-UUID with
``InvalidConversationIdError``. The tools, the retriever and the graph node used to
pass the caller's id on unchanged, so a ``pgv2_...`` or ``chat-42`` id failed.
"""
import asyncio
import uuid

import pytest
from langchain_core.messages import HumanMessage
from maximem_synap.models.errors import InvalidConversationIdError
from synap_integrations_common.testing import RecordingSDK

from synap_langchain.graph import create_synap_node
from synap_langchain.retriever import SynapRetriever
from synap_langchain.tools import SynapSearchTool, _SearchInput

PG = "pgv2_0123456789abcdef0123456789abcdef"
PG_UUID = str(uuid.uuid5(uuid.NAMESPACE_URL, PG))


def run(coro):
    return asyncio.run(coro)


def test_premise_the_strict_sdk_refuses_the_raw_id():
    with pytest.raises(InvalidConversationIdError):
        run(RecordingSDK(strict_conversation_ids=True).fetch(conversation_id=PG, user_id="u"))


def test_search_tool_sends_the_derived_uuid():
    sdk = RecordingSDK(strict_conversation_ids=True)
    tool = SynapSearchTool.model_construct(
        sdk=sdk, user_id="u1", customer_id="c1", conversation_id=PG, mode="accurate",
        max_results=5, name="search_memory", description="d", args_schema=_SearchInput)
    run(tool._arun("tea"))
    assert sdk.last_kwargs("fetch")["conversation_id"] == PG_UUID


def test_retriever_sends_the_derived_uuid():
    sdk = RecordingSDK(strict_conversation_ids=True)
    r = SynapRetriever.model_construct(
        sdk=sdk, user_id="u1", customer_id="c1", conversation_id=PG, max_results=5,
        mode="accurate", types=None, scope_path=None)
    run(r._aget_relevant_documents("tea", run_manager=None))
    assert sdk.last_kwargs("fetch")["conversation_id"] == PG_UUID


def test_graph_node_sends_the_derived_uuid_from_the_factory_and_from_the_state():
    sdk = RecordingSDK(strict_conversation_ids=True)
    run(create_synap_node(sdk, user_id="u1", conversation_id=PG)({"messages": [HumanMessage("tea?")]}))
    assert sdk.last_kwargs("fetch")["conversation_id"] == PG_UUID
    sdk.reset()
    run(create_synap_node(sdk, user_id="u1")({"messages": [HumanMessage("tea?")], "conversation_id": PG}))
    assert sdk.last_kwargs("fetch")["conversation_id"] == PG_UUID


def test_a_uuid_and_no_id_are_sent_as_they_were():
    sdk = RecordingSDK(strict_conversation_ids=True)
    u = "123e4567-e89b-12d3-a456-426614174000"
    run(create_synap_node(sdk, user_id="u1", conversation_id=u)({"messages": [HumanMessage("q")]}))
    assert sdk.last_kwargs("fetch")["conversation_id"] == u
    sdk.reset()
    run(create_synap_node(sdk, user_id="u1")({"messages": [HumanMessage("q")]}))
    assert not sdk.last_kwargs("fetch").get("conversation_id")
