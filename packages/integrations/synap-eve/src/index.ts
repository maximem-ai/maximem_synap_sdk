/**
 * Synap memory integration for Vercel eve agents.
 *
 * Two surfaces, both authored under an eve `agent/` directory:
 *
 *   - `createSynapSearchTool` / `createSynapStoreTool` — explicit recall/store
 *     tools for `agent/tools/*.ts` (the filename becomes the tool name).
 *   - `createSynapInstructions` — a per-turn short-term-context resolver for
 *     `agent/instructions/*.ts`, injecting Synap's compacted summary into the
 *     system prompt.
 *
 * Identity auto-scopes from the eve session (`ctx.session.auth.principalId`,
 * `ctx.session.id`), with explicit overrides for unauthenticated channels.
 */

export {
  createSynapSearchTool,
  createSynapStoreTool,
} from "./tools.js";
export type { SynapToolOptions } from "./tools.js";

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
