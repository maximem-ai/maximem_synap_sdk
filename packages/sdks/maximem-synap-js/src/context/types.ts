/**
 * Context shapes.
 *
 * There are deliberately TWO of these, and they are both live in production:
 *
 *  - **Raw snake_case** (`RawContext`), returned by the namespaced surface
 *    (`client.user.context.fetch(...)`), which passes `result.context` through
 *    untouched.
 *  - **Normalised camelCase** (`NormalisedContext`), returned by the legacy
 *    top-level methods (`client.fetchUserContext(...)`).
 *
 * Converging them would be a clean improvement and would also break one of the
 * two sets of existing consumers, so it is a 2.0 change, not a port change.
 * See gotcha G-B in the migration plan. Do not "tidy" this.
 */

export type Json = Record<string, unknown>;

// ─── Raw wire shapes (snake_case) ─────────────────────────────────────────────

export interface RawContextItem {
  id?: string;
  item_id?: string;
  content?: string;
  confidence?: number;
  source?: string;
  extracted_at?: string | null;
  metadata?: Json;
  event_date?: string | null;
  valid_until?: string | null;
  temporal_category?: string | null;
  temporal_confidence?: number;
  [key: string]: unknown;
}

export interface RawConversationContext {
  summary?: string | null;
  current_state?: Json;
  current_state_json?: string;
  key_extractions?: Json;
  key_extractions_json?: string;
  recent_turns?: unknown[];
  compaction_id?: string | null;
  compacted_at?: string | null;
  conversation_id?: string | null;
  [key: string]: unknown;
}

export interface RawContext {
  facts?: RawContextItem[];
  preferences?: RawContextItem[];
  episodes?: RawContextItem[];
  emotions?: RawContextItem[];
  temporal_events?: RawContextItem[];
  conversation_context?: RawConversationContext | null;
  metadata?: Json;
  [key: string]: unknown;
}

// ─── Normalised shapes (camelCase) ────────────────────────────────────────────

export interface TemporalFields {
  eventDate: string | null;
  validUntil: string | null;
  temporalCategory: string | null;
  temporalConfidence: number;
}

export interface Fact extends TemporalFields {
  id: string;
  content: string;
  confidence: number;
  source: string;
  extractedAt: string | null;
  metadata: Json;
}

export interface Preference extends TemporalFields {
  id: string;
  category: string;
  content: string;
  strength: number;
  source: string;
  extractedAt: string | null;
  metadata: Json;
}

export interface Episode extends TemporalFields {
  id: string;
  summary: string;
  occurredAt: string | null;
  significance: number;
  participants: unknown[];
  metadata: Json;
}

export interface Emotion extends TemporalFields {
  id: string;
  emotionType: string;
  intensity: number;
  detectedAt: string | null;
  context: string;
  metadata: Json;
}

export interface TemporalEvent {
  id: string;
  content: string;
  eventDate: string | null;
  validUntil: string | null;
  /**
   * Nullable despite the `''` default in the normaliser: `pickDefined` skips
   * only `undefined`, so an explicit server `null` is preserved and the
   * default never applies. The declared type follows the runtime, not the
   * default.
   */
  temporalCategory: string | null;
  temporalConfidence: number;
  confidence: number;
  source: string;
  extractedAt: string | null;
  metadata: Json;
}

export interface ConversationContext {
  summary: string | null;
  currentState: Json;
  keyExtractions: Json;
  recentTurns: unknown[];
  compactionId: string | null;
  compactedAt: string | null;
  conversationId: string | null;
}

export interface ContextMetadata {
  correlationId: string;
  ttlSeconds: number;
  source: string;
  retrievedAt: string | null;
  compactionApplied: boolean | null;
}

export interface NormalisedContext {
  facts: Fact[];
  preferences: Preference[];
  episodes: Episode[];
  emotions: Emotion[];
  temporalEvents: TemporalEvent[];
  conversationContext: ConversationContext | null;
  metadata: ContextMetadata;
  rawResponse: Json;
}

// ─── Request options ──────────────────────────────────────────────────────────

/**
 * Fetch options.
 *
 * Both spellings of every id are accepted. The removed wrapper did this
 * (`pickDefined(args.user_id, args.userId)`) and callers depend on it, so
 * rejecting one spelling would be a silent breaking change.
 */
export interface FetchOptions {
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  conversation_id?: string;
  conversationId?: string;
  search_query?: string[];
  searchQuery?: string[];
  max_results?: number;
  maxResults?: number;
  types?: string[];
  mode?: string;
  precision_level?: string;
  precisionLevel?: string;
  /** Sent only when explicitly false, matching the Python controllers. */
  include_conversation_context?: boolean;

  /**
   * `"in-conversation"` (default) or `"conversation-summary"`.
   *
   * Summary mode returns a caller profile plus previous-conversation
   * summaries. USER SCOPE ONLY: the other three routes reject it with 422.
   */
  context_mode?: string;
  contextMode?: string;
  /** Summary mode only. Include the caller profile. */
  include_profile?: boolean;
  includeProfile?: boolean;
  /** Summary mode only. Previous conversations to summarise, 0 to 20. */
  last_n_conversations?: number;
  lastNConversations?: number;
  /** Explicit scope ladder. Sent as `scope` when supplied. */
  scope_path?: Record<string, string>;
  scopePath?: Record<string, string>;
}
