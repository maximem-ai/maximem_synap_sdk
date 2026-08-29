/**
 * @maximem/synap-js-sdk
 *
 * Native TypeScript SDK for Synap. No Python runtime required.
 */

export {
  SynapClient,
  type SynapClientOptions, type SynapLogger, type SynapLogLevel, type ConfigureOptions, type UnifiedFetchOptions,
} from './client.js';
export { SDK_VERSION } from './version.js';

export * from './errors.js';

export type {
  ConversationContext, ContextMetadata, Emotion, Episode, Fact, FetchOptions,
  Json, NormalisedContext, Preference, RawContext, RawContextItem,
  RawConversationContext, TemporalEvent, TemporalFields,
} from './context/types.js';

export type { Scope } from './context/fetch.js';
export { flattenContextItems, type FlatMemory } from './context/flatten.js';

export {
  DOCUMENT_TYPES, INGEST_MODES, DEFAULT_DOCUMENT_TYPE, DEFAULT_INGEST_MODE,
  type CreateMemoryOptions, type CreateMemoryResult, type MemoriesInterface,
  type DocumentType, type IngestMode, type UpdateMemoryOptions, type WaitOptions,
  type CreateFromFileOptions, type BatchCreateResult,
} from './memories/interface.js';
export type {
  ConversationNamespace, RecordMessageOptions, IngestTranscriptOptions,
  CompactOptions, TranscriptTurn, CompactionCallback, WriteInvalidator,
  TranscriptIngestResult, CompactionLevel,
} from './conversation/interface.js';
export { COMPACTION_LEVELS } from './conversation/interface.js';
export type { UserNamespace } from './user/interface.js';
export type {
  CreditsNamespace, EstimateOptions, LedgerOptions, CreditBalance, CreditBucket,
  CreditLedgerEntry, CreditLedgerPage, CreditEstimate, RedeemResult,
} from './credits/interface.js';
export type { CacheNamespace, CacheStats } from './cache/interface.js';
export type {
  InstanceNamespace, ListenOptions, SendMessageOptions, RecordThinkingOptions,
} from './instance/interface.js';

export {
  AnticipationCache, ANY_SCOPE,
  type AnticipationCacheOptions, type ContextBundle, type LookupResult,
  type LookupParams, type LookupTelemetry, type ExitReason,
  type AnticipationCacheSnapshot, type SnapshotBundle, type SnapshotItemRecord,
  type SnapshotItemPreview, type BundleItem,
} from './context/anticipation-cache.js';

// The return type of client.fetch() and the shape its formatter renders. A
// caller who cannot name these cannot write a function that takes or returns
// one, which is most of the value of a typed SDK.
export {
  mergeScopeResults, formatForPrompt, ITEM_COLLECTIONS,
  type UnifiedContext, type ItemCollection, type FormatOptions,
} from './context/unified.js';

// as_tool's return type. Without it a caller cannot hold the tool definition
// in a typed variable, which is the entire point of the helper.
export type {
  ToolDefinition, AsToolOptions, ToolScope, ToolStyle, ToolHandler,
  OpenAiToolDefinition, AnthropicToolDefinition,
} from './tool/as-tool.js';

export { BM25, tokenize, stem } from './cache/bm25.js';
export { isRecallQuery, THRESHOLDS, CONTRACT } from './behavior/contract.js';

export {
  HttpTransport, DEFAULT_BASE_URL, DEFAULT_TIMEOUTS, isOutcomeUnknown,
  type Credentials, type HttpTransportOptions, type RequestOptions, type TimeoutConfig,
} from './transport/http.js';
export { ENDPOINTS, resolvePath, type EndpointName, type EndpointSpec } from './transport/endpoints.js';
export { shouldRetry, DEFAULT_RETRY_POLICY, type RetryPolicy } from './transport/retry.js';

/**
 * The duck-typed contract the TS integrations rely on.
 *
 * `synap-mastra`, `synap-eve` and `synap-claude-agent-ts` each redeclare this
 * shape locally today. Exporting it here means they can `import type` it
 * instead, so a change to the client's surface breaks their build rather than
 * their runtime.
 */
export interface SynapSdkLike {
  user: { context: { fetch: (options?: Record<string, unknown>) => Promise<unknown> } };
  customer: { context: { fetch: (options?: Record<string, unknown>) => Promise<unknown> } };
  client: { context: { fetch: (options?: Record<string, unknown>) => Promise<unknown> } };
  memories: { create: (options: { document: string } & Record<string, unknown>) => Promise<unknown> };
}
