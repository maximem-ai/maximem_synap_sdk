// ─── Provider configuration ──────────────────────────────────────────────────

export interface SynapProviderOptions {
  /** API key (sk-...). Falls back to SYNAP_API_KEY env var. */
  apiKey?: string;
  /** Override API base URL. Defaults to https://synap-cloud-prod.maximem.ai */
  baseUrl?: string;
  /** Override gRPC host. Defaults to synap-cloud-prod.maximem.ai */
  grpcHost?: string;
  /** Override gRPC port. Defaults to 443 */
  grpcPort?: number;
  /** Disable TLS for gRPC (local dev only). Defaults to true */
  grpcUseTls?: boolean;
}

// ─── Per-call model options ───────────────────────────────────────────────────

export interface SynapModelOptions {
  /** User ID for context scoping */
  userId?: string;
  /** Customer/org ID for context scoping */
  customerId?: string;
  /** Conversation ID for session-scoped context */
  conversationId?: string;
  /** Which context types to fetch. Defaults to all. */
  contextTypes?: Array<'facts' | 'preferences' | 'episodes' | 'emotions' | 'temporal_events'>;
  /** Max context items to fetch. Defaults to 10. */
  maxContextResults?: number;
  /** Write conversation turn back to Synap memory. Defaults to true. */
  writeMemory?: boolean;
  /** Inject context as system prompt prefix. Defaults to true. */
  injectContext?: boolean;
}

// ─── Context types ────────────────────────────────────────────────────────────

export interface ContextItem {
  id: string;
  content: string;
  contextType: string;
  confidence: number;
  source: string;
  eventDate?: string;
  validUntil?: string;
  temporalCategory?: string;
}

export interface ConversationContext {
  summary: string | null;
  currentState: Record<string, unknown>;
  keyExtractions: Record<string, unknown>;
  recentTurns: Array<{ role: string; content: string; timestamp: string }>;
  compactionId: string | null;
  conversationId: string | null;
}

export interface FetchedContext {
  facts: ContextItem[];
  preferences: ContextItem[];
  episodes: ContextItem[];
  emotions: ContextItem[];
  temporalEvents: ContextItem[];
  conversationContext: ConversationContext | null;
  /** Where the context came from */
  source: 'anticipation' | 'cache' | 'cloud';
  correlationId: string;
  // ─── Telemetry provenance (drives ContextUsedEvent / ContextAssembledEvent) ──
  /** Merged/source bundle id this context was served from ('' for HTTP cloud fetch). */
  bundleId: string;
  /** IDs of every context item served (no content — ids only, privacy rule). */
  servedItemIds: string[];
  /** Server-side source bundles that contributed items (anticipation merges). */
  sourceBundleIds: string[];
  /** Total tokens reported for the assembled bundle. */
  totalTokens: number;
  /** How the bundle was assembled — matches ContextAssembledEvent.assembly_source. */
  assemblySource: 'anticipation_cache' | 'http_cache' | 'cloud' | 'hybrid';
  /** Whether this fetch was served from a cache (anticipation or HTTP). */
  cacheHit: boolean;
}

// ─── Anticipation cache types (mirrors Python AnticipationCache) ──────────────

export type CachedBundleType = 'anticipation' | 'user_summary' | 'reactive' | 'compaction_update';

export interface CachedBundle {
  bundleId: string;
  itemsByType: Record<string, RawContextItem[]>;
  conversationContext: RawConversationContext | null;
  bundleType: CachedBundleType;
  userId: string;
  customerId: string;
  conversationId: string;
  searchKeywords: string[];
  searchQueries: string[];
  /** Source bundle ids that contributed items (for ContextUsedEvent attribution). */
  sourceBundleIds: string[];
  /** Total tokens reported by the server for this bundle. */
  totalTokens: number;
  /** Server-assigned 0..1 confidence. Carried for telemetry; not used for ranking
   *  (matches the Python SDK, which ranks on BM25 only). */
  bundleConfidence: number;
  /** MACA pattern that produced this bundle. Carried for per-pattern attribution. */
  originPatternId: string;
  storedAt: number;
  ttl: number;
}

export interface RawContextItem {
  item_id: string;
  content: string;
  context_type: string;
  source: string;
  confidence: number;
  similarity_score: number;
  relevance_score: number;
  scope: string;
  entity_id: string;
  event_date: string;
  valid_until: string;
  temporal_category: string;
  temporal_confidence: number;
}

export interface RawConversationContext {
  summary: string;
  current_state_json: string;
  key_extractions_json: string;
  recent_turns: Array<{ role: string; content: string; timestamp: string }>;
  compaction_id: string;
  compacted_at: string;
  conversation_id: string;
}

// ─── gRPC message types (mirrors synap_service.proto) ────────────────────────

export interface ConversationEventMsg {
  event_type: string;
  conversation_id: string;
  user_id: string;
  role: string;
  content: string;
  customer_id: string;
  session_id: string;
  metadata: Record<string, string>;
  timestamp_ms: number;
  search_queries: string[];
  context_types: string[];
}

export interface ContextBundleMsg {
  bundle_id: string;
  decision_id: string;
  items_by_type: Record<string, { items: RawContextItem[] }>;
  total_tokens: number;
  bundle_type: string;
  anticipation_user_id: string;
  anticipation_customer_id: string;
  anticipation_conversation_id: string;
  search_keywords: string[];
  search_queries: string[];
  conversation_context?: RawConversationContext;
  // Section 16 — bundle composition (proto fields 21-23).
  bundle_confidence: number;
  origin_pattern_id: string;
  ttl_hint_seconds: number;
}

// Client → server learning-loop telemetry (StreamEvent oneof fields 4 & 5).
// IDs and metadata only — never raw prompts or item content.
export interface ContextUsedEventMsg {
  bundle_id: string;
  conversation_id: string;
  user_id: string;
  customer_id: string;
  served_item_ids: string[];
  timestamp_ms: number;
  scope: string;
  source_bundle_ids: string[];
}

export interface ContextAssembledEventMsg {
  correlation_id: string;
  conversation_id: string;
  user_id: string;
  customer_id: string;
  final_item_ids: string[];
  final_total_tokens: number;
  compaction_id: string;
  recent_turn_count: number;
  compaction_end_timestamp: string;
  assembly_source: string;
  assembly_duration_ms: number;
  cache_hit: boolean;
  timestamp_ms: number;
  sdk_version: string;
}

// ─── Credentials ──────────────────────────────────────────────────────────────

export interface Credentials {
  api_key: string;
  client_id: string;
  instance_id: string;
}
