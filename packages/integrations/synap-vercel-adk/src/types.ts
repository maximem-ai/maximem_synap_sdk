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
}

// ─── Anticipation cache types (mirrors Python AnticipationCache) ──────────────

export interface CachedBundle {
  itemsByType: Record<string, RawContextItem[]>;
  conversationContext: RawConversationContext | null;
  bundleType: 'anticipation' | 'user_summary' | 'compaction_update';
  userId: string;
  customerId: string;
  conversationId: string;
  searchKeywords: string[];
  searchQueries: string[];
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
}

// ─── Credentials ──────────────────────────────────────────────────────────────

export interface Credentials {
  api_key: string;
  client_id: string;
  instance_id: string;
}
