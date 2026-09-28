"""Regression: the factories must survive real Agent wiring.

The published docs once showed ``FunctionTool(search_fn, name_override=...)``
and ``tools=[search_fn]``. Both construct without complaint and then fail —
the first with TypeError at wiring time, the second with UserError at the
first model call. The unit tests never caught it because none of them built
an Agent. These do.
"""

import pytest
from agents import Agent, FunctionTool, function_tool

from synap_openai_agents import create_search_tool, create_store_tool


class _FakeResponse:
    formatted_context = "Aditya prefers concise answers."


class _FakeIngestion:
    ingestion_id = "ing_test"


class _FakeMemories:
    async def create(self, **kwargs):
        return _FakeIngestion()


class _FakeSDK:
    memories = _FakeMemories()

    async def fetch(self, **kwargs):
        return _FakeResponse()


@pytest.fixture
def sdk():
    return _FakeSDK()


def _converter():
    try:
        from agents.models.chatcmpl_converter import Converter
    except ImportError:  # older layout
        from agents.models.openai_chatcompletions import Converter
    return Converter


def test_function_tool_helper_wraps_the_factories(sdk):
    """The documented pattern builds an Agent whose tools convert for the API."""
    tools = [
        function_tool(create_search_tool(sdk=sdk, user_id="u1"), name_override="synap_search"),
        function_tool(create_store_tool(sdk=sdk, user_id="u1"), name_override="synap_store"),
    ]
    agent = Agent(name="memory-agent", instructions="test", tools=tools)

    assert [t.name for t in agent.tools] == ["synap_search", "synap_store"]
    assert all(isinstance(t, FunctionTool) for t in agent.tools)

    # The step that actually rejected a bare callable.
    convert = _converter().tool_to_openai
    assert convert(agent.tools[0])["function"]["name"] == "synap_search"
    assert convert(agent.tools[1])["function"]["name"] == "synap_store"


def test_argument_schema_is_derived_from_the_signature(sdk):
    search = function_tool(create_search_tool(sdk=sdk, user_id="u1"), name_override="synap_search")
    store = function_tool(create_store_tool(sdk=sdk, user_id="u1"), name_override="synap_store")

    assert search.params_json_schema["required"] == ["query"]
    assert store.params_json_schema["required"] == ["content"]


def test_bare_callable_in_tools_is_rejected_at_conversion(sdk):
    """Guards the old platform-doc snippet: tools=[search_fn] must not be advertised."""
    search_fn = create_search_tool(sdk=sdk, user_id="u1")
    agent = Agent(name="memory-agent", instructions="test", tools=[search_fn])

    with pytest.raises(Exception) as excinfo:
        _converter().tool_to_openai(agent.tools[0])
    assert "tool type" in str(excinfo.value).lower()


def test_functiontool_dataclass_rejects_a_bare_function(sdk):
    """Guards the old skill snippet: FunctionTool(fn, name_override=...) is not valid."""
    search_fn = create_search_tool(sdk=sdk, user_id="u1")

    with pytest.raises(TypeError):
        FunctionTool(search_fn, name_override="synap_search")
