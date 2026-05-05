// ─── Primary API ──────────────────────────────────────────────────────────────
export { createSynap, SynapProvider } from './provider.js';

// ─── Low-level building blocks (advanced use) ─────────────────────────────────
export { createSynapMiddleware } from './middleware.js';
export { AnticipationCache } from './context/anticipation-cache.js';
export { fetchContext } from './context/http-fetcher.js';
export { writeMemory } from './memory/writer.js';
export { CredentialManager } from './auth/credential-manager.js';
export {
  buildContextSystemBlock,
  injectContextIntoPrompt,
  extractSearchQuery,
  promptToTranscript,
} from './transform/messages.js';

// ─── Types ────────────────────────────────────────────────────────────────────
export type {
  SynapProviderOptions,
  SynapModelOptions,
  FetchedContext,
  ContextItem,
  ConversationContext,
  Credentials,
  CachedBundle,
} from './types.js';
