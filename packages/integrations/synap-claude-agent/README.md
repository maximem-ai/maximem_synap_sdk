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

`create_synap_hooks(...)` installs hooks that fetch Synap context for each prompt and inject it via `additionalContext`, record the user's prompt and the agent's reply (the `Stop` hook's `last_assistant_message`), and report tool calls and results.

The first hook to fire opens the gRPC stream (`sdk.instance.listen()`) if you have not, and everything goes on it. `SYNAP_AUTO_LISTEN=false` turns that off and the two turns go over REST.

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

### 3. Session store: resume from Synap's compaction

Claude Code owns its transcript, and a hook can add context to a prompt but not remove what the transcript holds. The one place the SDK lets you decide what a *resumed* session starts from is `session_store`:

```python
from synap_claude_agent import create_synap_session_store

options = ClaudeAgentOptions(
    resume=session_id,
    session_store=create_synap_session_store(sdk),
    hooks=create_synap_hooks(sdk, user_id="alice"),
)
```

Once Synap has compacted the conversation, the resumed session starts from Synap's compaction and the turns after it instead of the stored transcript. Before that, or if Synap cannot answer, it resumes from the transcript unchanged. Inside a running session Claude Code still compacts its own transcript; the `PreCompact` hook gives its summarizer Synap's compaction to build on.

Use both together for the full experience: automatic context injection plus agent-initiated read/write.

## Error policy

- **Hooks** never raise — SDK failures log at `ERROR` and fall through to no-op (no context injected, no prompt recorded). Context-provider style.
- **`synap_search` tool** returns a "no context available" message on SDK failure — the tool call succeeds so the agent loop keeps going.
- **`synap_remember` tool** returns `isError=true` on ingestion failure — silent drops would hide ingestion outages, so we surface them to the agent explicitly.
