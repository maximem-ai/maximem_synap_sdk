# synap-claude-agent

Synap memory integration for Anthropic's [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) (Python).

A TypeScript sibling package with the same API is published as `@maximem/synap-claude-agent`.

## Install

```bash
pip install synap-claude-agent
```

Requires `claude-agent-sdk>=0.1`, `maximem-synap>=0.2.0`.

## Two plug points

### 1. Hooks — automatic context injection

`create_synap_hooks(...)` installs a `UserPromptSubmit` hook that fetches Synap context for each prompt and injects it via `additionalContext`. Optionally records the user prompt to conversation history for future recall.

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from maximem_synap import MaximemSynapSDK
from synap_claude_agent import create_synap_hooks

sdk = MaximemSynapSDK(api_key="sk-...")

async for message in query(
    prompt="What did I tell you about my trial?",
    options=ClaudeAgentOptions(
        hooks=create_synap_hooks(sdk, user_id="alice", customer_id="acme"),
    ),
):
    print(message)
```

### 2. MCP tools — explicit read/write

`create_synap_mcp_server(...)` returns an in-process MCP server with two tools:

- `synap_search(query, max_results?)` — semantic search over the user's memory
- `synap_remember(content, metadata?)` — persist an explicit fact

```python
options = ClaudeAgentOptions(
    mcp_servers={"synap": create_synap_mcp_server(sdk, user_id="alice")},
    allowed_tools=["mcp__synap__synap_search", "mcp__synap__synap_remember"],
)
```

Use both together for the full experience: automatic context injection plus agent-initiated read/write.

## Error policy

- **Hooks** never raise — SDK failures log at `ERROR` and fall through to no-op (no context injected, no prompt recorded). Context-provider style.
- **`synap_search` tool** returns a "no context available" message on SDK failure — the tool call succeeds so the agent loop keeps going.
- **`synap_remember` tool** returns `isError=true` on ingestion failure — silent drops would hide ingestion outages, so we surface them to the agent explicitly.
