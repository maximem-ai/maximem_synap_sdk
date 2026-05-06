# Framework integrations — index

One file per supported framework. Read only the one matching the user's stack. Every file follows the same shape: install, what's included, copy-pasteable quick start, scoping notes, and a link to the canonical doc.

## Routing — pick by what the user mentioned

| If the user mentions… | Read |
| --- | --- |
| LangChain, `RunnableWithMessageHistory`, `ConversationalRetrievalChain`, callbacks | `langchain.md` |
| LangGraph, checkpointer, `BaseStore`, state graph, threads | `langgraph.md` |
| LlamaIndex, `BaseMemory`, `CondensePlusContextChatEngine`, NodeWithScore | `llamaindex.md` |
| OpenAI Agents SDK, `Agent`, `Runner`, `FunctionTool` (the `agents` package) | `openai-agents.md` |
| Pydantic AI, `Agent[Deps, ...]` | `pydantic-ai.md` |
| CrewAI, Crew, Task, Agent (the crewai package) | `crewai.md` |
| AutoGen, `AssistantAgent`, `BaseTool`, `CancellationToken` | `autogen.md` |
| Google ADK, `gemini-2.0-flash`, the `google.adk` package | `google-adk.md` |
| Haystack, Pipeline, `Document`, components | `haystack.md` |
| Agno, `InMemoryDb`, `enable_user_memories` | `agno.md` |
| Semantic Kernel, Kernel, plugins, kernel functions | `semantic-kernel.md` |
| Microsoft Agent Framework, MAF, `as_agent`, context providers | `microsoft-agent.md` |
| NVIDIA NeMo, NAT, `MemoryEditor`, `MemoryItem` | `nemo-agent-toolkit.md` |
| LiveKit voice agent, `AgentSession`, `JobContext`, `ChatContext` | `livekit-agents.md` |
| Pipecat, frame processors, voice pipeline | `pipecat.md` |
| Claude Agent SDK, `query()`, hooks, `ClaudeAgentOptions`, MCP | `claude-agent.md` |
| Mastra, `@mastra/core`, MastraMemory, TypeScript agent | `mastra.md` |
| Vercel AI SDK, `ai` package, `generateText`, model wrapping | `vercel-adk.md` |

## Common shape across all packages

Every integration:

1. **Takes a constructed, initialized `MaximemSynapSDK`** — never creates one for you. The user wires `sdk` once at app startup.
2. **Accepts `user_id`, optional `customer_id`, optional `conversation_id`** as scoping parameters.
3. **Degrades reads gracefully, surfaces writes explicitly.** Read failures return empty results + log; write failures raise `SynapIntegrationError` (or framework-equivalent).
4. **Defaults `mode="fast"` for retrieval, `mode="long-range"` for ingestion** — change only if the situation demands it.

If the user has a custom or unsupported framework, fall back to `reference/ingestion.md` and `reference/context-fetch.md` — every integration is a thin wrapper over those two primitives.
