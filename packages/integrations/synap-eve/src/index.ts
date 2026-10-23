/**
 * Synap memory integration for Vercel eve agents.
 *
 * Four surfaces, authored under an eve `agent/` directory (the last in `agent.ts`):
 *
 *   - `createSynapSearchTool` / `createSynapStoreTool` — explicit recall/store
 *     tools for `agent/tools/*.ts` (the filename becomes the tool name).
 *   - `createSynapInstructions` — a per-turn short-term-context resolver for
 *     `agent/instructions/*.ts`, injecting Synap's compacted summary into the
 *     system prompt.
 *   - `createSynapStreamHooks` — an `agent/hooks/*.ts` subscriber that reports
 *     the run on Synap's live stream (the user's turn, the reply, the
 *     reasoning, the tools and their results) so the anticipation agent can
 *     predict the next turn. The stream opens on the first event; if it cannot,
 *     the user and assistant turns are written over REST
 *     (`conversation.record_message`). `SYNAP_AUTO_LISTEN=false` keeps it closed.
 *   - `createSynapCompactionModel` — wraps the agent's model for
 *     `defineAgent({ model })` (not `compaction.model`, which eve never calls):
 *     when eve compacts, its checkpoint is Synap's compaction instead of an
 *     LLM-written summary.
 *
 * Identity auto-scopes from the eve session (`ctx.session.auth.current.principalId`,
 * `ctx.session.id`), with explicit overrides for unauthenticated channels.
 */

export {
  createSynapSearchTool,
  createSynapStoreTool,
} from "./tools.js";
export type { SynapToolOptions } from "./tools.js";

export { createSynapCompactionModel } from "./compaction.js";
export type { SynapCompactionModelOptions } from "./compaction.js";

export { createSynapStreamHooks } from "./stream.js";
export type { SynapStreamHooksOptions, SynapTurnSdkLike } from "./stream.js";
export { setAutoListen } from "./stream-events.js";
export type { SynapStreamSdkLike } from "./stream-events.js";

export {
  createSynapInstructions,
  fetchSynapShortTerm,
  buildSynapShortTermMarkdown,
} from "./short_term.js";
export type {
  SynapShortTermStyle,
  SynapShortTermOnError,
  SynapShortTermResult,
  FetchSynapShortTermOptions,
  BuildSynapShortTermMarkdownOptions,
  SynapInstructionsOptions,
} from "./short_term.js";

export {
  SynapIntegrationError,
  resolveUserId,
  resolveConversationId,
} from "./types.js";
export type {
  SynapSdkLike,
  SynapFetchArgs,
  SynapFetchResponseLike,
  SynapMemoryCreateArgs,
  SynapMemoryCreateResult,
  SynapPromptContext,
  SynapRecentMessage,
  EveSessionLike,
  EveAuthContextLike,
} from "./types.js";
