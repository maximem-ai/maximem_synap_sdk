"""Synap integration for Anthropic's Claude Agent SDK.

Composable plug points:

- :func:`create_synap_hooks` — returns a ``hooks={...}`` dict suitable for
  ``ClaudeAgentOptions(hooks=...)``. Injects Synap context into the agent's
  prompt as ``additionalContext``, records the user's prompt and the agent's
  reply, reports tool calls, and hands Synap's compaction to Claude Code's own
  summarizer when it compacts.

- :func:`create_synap_session_store` — ``ClaudeAgentOptions(session_store=...)``:
  a resumed session starts from Synap's compaction instead of the stored
  transcript once the conversation has been compacted.

- :func:`create_synap_mcp_server` — returns an in-process MCP server with
  ``synap_search`` and ``synap_remember`` tools, ready to splat into
  ``ClaudeAgentOptions(mcp_servers={"synap": server})``. Lets the agent
  read/write Synap memory explicitly via tool calls.

Use either or both. Typical wiring::

    from claude_agent_sdk import query, ClaudeAgentOptions
    from maximem_synap import MaximemSynapSDK
    from synap_claude_agent import create_synap_hooks, create_synap_mcp_server

    sdk = MaximemSynapSDK(api_key="sk-...")

    options = ClaudeAgentOptions(
        hooks=create_synap_hooks(sdk, user_id="alice"),
        mcp_servers={"synap": create_synap_mcp_server(sdk, user_id="alice")},
        allowed_tools=["mcp__synap__synap_search", "mcp__synap__synap_remember"],
    )

    async for message in query(prompt="What did I tell you about?", options=options):
        print(message)
"""

from synap_claude_agent.hooks import create_synap_hooks, report_assistant_turn
from synap_claude_agent.mcp_server import create_synap_mcp_server
from synap_claude_agent.session_store import create_synap_session_store
from synap_claude_agent.short_term import create_synap_st_hook

__all__ = [
    "create_synap_hooks",
    "report_assistant_turn",
    "create_synap_mcp_server",
    "create_synap_session_store",
    "create_synap_st_hook",
]
