"""Synap LangGraph integration.

Provides a node factory for injecting Synap memory context into
LangGraph state before the LLM node processes it.
"""

import logging
from typing import Any, Callable, Dict, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


def create_synap_node(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    state_key: str = "synap_context",
    messages_key: str = "messages",
    mode: str = "fast",
    max_results: int = 20,
    include_scope_labels: bool = False,
) -> Callable:
    """Create a LangGraph node that injects Synap context into state.

    Place this node before the LLM node in your graph. It reads the
    latest user message from state, fetches relevant memory context
    from Synap, and writes it into ``state[state_key]``.

    Example::

        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(MyState)
        graph.add_node("memory", create_synap_node(sdk, user_id="u1"))
        graph.add_node("llm", llm_node)
        graph.add_edge(START, "memory")
        graph.add_edge("memory", "llm")
        graph.add_edge("llm", END)
        app = graph.compile()
    """
    if sdk is None:
        raise ValueError("create_synap_node requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_node requires a non-empty user_id")

    async def synap_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
        conv_id = conversation_id or state.get("conversation_id")

        query = None
        messages = state.get(messages_key, [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                query = [str(msg.content)]
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = [str(msg.get("content", ""))]
                break

        async with wrap_sdk_errors_async(
            "langchain.synap_memory_node", logger, user_id=user_id,
        ):
            response = await sdk.fetch(
                conversation_id=conv_id,
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=query,
                max_results=max_results,
                mode=mode,
                include_conversation_context=False,
                include_scope_labels=include_scope_labels,
            )

        return {state_key: response.formatted_context or ""}

    synap_memory_node.__name__ = "synap_memory"
    return synap_memory_node
