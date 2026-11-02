"""``scope_path`` (a deeper scope ladder) reaches every SDK call in this package.

Built on ``RecordingSDK``: each fake method has the installed SDK's real
signature and records its exact keyword arguments.
"""

from unittest.mock import MagicMock

import pytest
from maximem_synap import MaximemSynapSDK
from maximem_synap.models.context import ContextForPromptResponse
from langchain_core.messages import AIMessage, HumanMessage

from synap_integrations_common.testing import RecordingSDK, make_ladder

from synap_deepagents import (
    SynapBackend,
    SynapMemoryMiddleware,
    SynapSearchTool,
    SynapShortTermMiddleware,
    SynapStoreTool,
    SynapStreamMiddleware,
)
from synap_deepagents.history import fetch_history

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "x", "user": "y"}


def as_sdk(rec):
    """The tools type ``sdk`` as ``MaximemSynapSDK`` (pydantic), so hand them a
    spec'd mock whose calls are the RecordingSDK's."""
    m = MagicMock(spec=MaximemSynapSDK)
    m.fetch = rec.fetch
    m.memories = rec.memories
    return m


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


# -- backend ---------------------------------------------------------------


async def test_backend_default_fetch_and_write_kwargs_unchanged(sdk):
    b = SynapBackend(sdk, user_id="alice", customer_id="acme")
    await b.aread("/AGENTS.md")
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "alice", "customer_id": "acme",
        "search_query": [b.recall_query], "max_results": 20, "mode": "fast",
        "precision_level": "high", "include_conversation_context": False,
    }
    await b.awrite("/n.md", "hello")
    assert sdk.last_kwargs("memories.create") == {
        "document": "hello", "user_id": "alice", "customer_id": "acme",
        "document_type": "ai-chat-conversation", "mode": "long-range",
    }


async def test_backend_read_carries_path(sdk):
    b = SynapBackend(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await b.aread("/AGENTS.md")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_backend_grep_carries_path(sdk):
    b = SynapBackend(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    sdk.results["fetch"] = type("R", (), {"formatted_context": "a line"})()
    await b.agrep("what")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_backend_download_and_arbitrary_path_read_carry_path(sdk):
    b = SynapBackend(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await b.adownload_files(["/some-topic.md", "/AGENTS.md"])
    assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH, PATH]


async def test_backend_write_carries_path_as_scope(sdk):
    b = SynapBackend(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await b.awrite("/n.md", "hello")
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


async def test_backend_edit_carries_path_as_scope(sdk):
    b = SynapBackend(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await b.aedit("/n.md", "", "fact")
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


async def test_backend_upload_carries_path_as_scope(sdk):
    b = SynapBackend(sdk, user_id="dana", scope_path=PATH)
    await b.aupload_files([("/u.md", b"text")])
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


def test_backend_path_alone_is_a_scope(sdk):
    assert SynapBackend(sdk, scope_path=PATH).scope_path == PATH
    with pytest.raises(ValueError):
        SynapBackend(sdk)


async def test_backend_copies_the_path(sdk):
    p = dict(PATH)
    b = SynapBackend(sdk, user_id="dana", scope_path=p)
    p["user"] = "mallory"
    await b.aread("/AGENTS.md")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


# -- tools: closed over, never model-visible ------------------------------


@pytest.mark.parametrize("cls", [SynapSearchTool, SynapStoreTool])
def test_tool_schema_has_no_scope_path(sdk, cls):
    tool = cls(sdk=as_sdk(sdk), user_id="dana", scope_path=PATH)
    assert "scope_path" not in tool.args_schema.model_json_schema()["properties"]
    assert "scope_path" not in str(tool.tool_call_schema.model_json_schema())


async def test_search_tool_default_unchanged(sdk):
    await SynapSearchTool(sdk=as_sdk(sdk), user_id="dana", customer_id="acme").ainvoke({"query": "q"})
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "dana", "customer_id": "acme",
        "search_query": ["q"], "mode": "accurate", "max_results": 10,
        "include_conversation_context": False,
    }


async def test_search_tool_carries_closed_over_path_and_ignores_model_path(sdk):
    tool = SynapSearchTool(sdk=as_sdk(sdk), user_id="dana", scope_path=PATH)
    await tool.ainvoke({"query": "q", "scope_path": OTHER})
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_search_tool_model_path_ignored_when_none_configured(sdk):
    tool = SynapSearchTool(sdk=as_sdk(sdk), user_id="dana")
    await tool.ainvoke({"query": "q", "scope_path": OTHER})
    assert "scope_path" not in sdk.last_kwargs("fetch")


async def test_store_tool_default_unchanged(sdk):
    await SynapStoreTool(sdk=as_sdk(sdk), user_id="dana").ainvoke({"content": "c"})
    assert sdk.last_kwargs("memories.create") == {
        "document": "c", "user_id": "dana", "customer_id": None,
        "document_type": "ai-chat-conversation", "mode": "long-range",
    }


async def test_store_tool_carries_closed_over_path_and_ignores_model_path(sdk):
    tool = SynapStoreTool(sdk=as_sdk(sdk), user_id="dana", scope_path=PATH)
    await tool.ainvoke({"content": "c", "scope_path": OTHER, "scope": OTHER})
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


async def test_store_tool_model_path_ignored_when_none_configured(sdk):
    await SynapStoreTool(sdk=as_sdk(sdk), user_id="dana").ainvoke(
        {"content": "c", "scope_path": OTHER, "scope": OTHER})
    kw = sdk.last_kwargs("memories.create")
    assert "scope" not in kw and "scope_path" not in kw


# -- middleware -----------------------------------------------------------


STATE = {"messages": [HumanMessage("what about payments?")]}


async def test_memory_middleware_default_fetch_unchanged(sdk):
    mw = SynapMemoryMiddleware(sdk=sdk, user_id="alice", customer_id="acme")
    await mw.abefore_agent(STATE, None)
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "alice", "customer_id": "acme",
        "search_query": ["what about payments?"], "max_results": 20, "mode": "fast",
        "precision_level": "high", "include_conversation_context": False,
    }


async def test_memory_middleware_fetch_carries_path(sdk):
    mw = SynapMemoryMiddleware(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await mw.abefore_agent(STATE, None)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_memory_middleware_path_alone_is_a_scope(sdk):
    SynapMemoryMiddleware(sdk=sdk, scope_path=PATH)
    with pytest.raises(ValueError):
        SynapMemoryMiddleware(sdk=sdk)


async def test_memory_middleware_conversation_half_carries_path(sdk):
    mw = SynapMemoryMiddleware(sdk=sdk, user_id="dana", customer_id="acme",
                               conversation_id="c1", scope_path=PATH)
    await mw.abefore_agent(STATE, None)
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_stream_middleware_user_and_assistant_turns_carry_path(sdk):
    mw = SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="dana",
                               customer_id="acme", scope_path=PATH)
    await mw.abefore_agent(STATE, None)
    await mw.aafter_agent({"messages": [AIMessage("answer", id="a1")]}, None)
    sends = sdk.kwargs_of("instance.send_message")
    assert [k["scope_path"] for k in sends] == [PATH, PATH]


async def test_stream_middleware_default_turn_unchanged(sdk):
    mw = SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="dana",
                               customer_id="acme")
    await mw.abefore_agent(STATE, None)
    kw = sdk.last_kwargs("instance.send_message")
    assert "scope_path" not in kw
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"


async def test_stream_middleware_rest_fallback_carries_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    sdk.instance.listen = _fail
    mw = SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="dana",
                               customer_id="acme", scope_path=PATH)
    await mw.abefore_agent(STATE, None)
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


async def _fail():
    raise RuntimeError("no stream")


async def test_older_sdk_never_sent_the_keyword_when_unset():
    sdk = RecordingSDK(accept_scope_path=False)
    b = SynapBackend(sdk, user_id="alice")
    await b.aread("/AGENTS.md")
    await b.awrite("/n.md", "x")
    await SynapSearchTool(sdk=as_sdk(sdk), user_id="alice").ainvoke({"query": "q"})
    await SynapStoreTool(sdk=as_sdk(sdk), user_id="alice").ainvoke({"content": "c"})
    await SynapMemoryMiddleware(sdk=sdk, user_id="alice").abefore_agent(STATE, None)
    assert len(sdk.calls_to("fetch")) == 3 and len(sdk.calls_to("memories.create")) == 2


# -- conversation-only reads: no path, still work -------------------------


async def test_short_term_reads_work_with_a_path_configured(sdk):
    sdk.results["conversation.context.get_context_for_prompt"] = ContextForPromptResponse(
        formatted_context="x", context_block="## Conversation so far (compacted 1x)\nS",
        available=True)
    history = await fetch_history(sdk, "c1")
    assert history is not None and "compacted" in history.preamble
    kw = sdk.last_kwargs("conversation.context.get_context_for_prompt")
    assert "scope_path" not in kw and "scope" not in kw
    mw = SynapShortTermMiddleware(sdk=sdk, conversation_id="c1")
    assert mw.conversation_id == "c1"
