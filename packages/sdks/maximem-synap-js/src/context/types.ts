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
  /**
   * The day the words were SAID, which is not the day the event happened.
   *
   * `event_date` is empty unless somebody stated when the event took place, so
   * without this field beside it an item whose turn stated no event date
   * arrives with no machine-readable date at all. Empty when the client never
   * told the server when the conversation happened; it is never the server's
   * own ingest or extraction clock.
   */
  said_at?: string | null;
  valid_until?: string | null;
  temporal_category?: string | null;
  temporal_confidence?: number;
  /**
   * The search queries of the fetch that found this item, in the order they
   * were sent. Absent from a server that predates the field, and on an item
   * served from the prefetch cache.
   */
  matched_queries?: string[];
  /**
   * Anything the server sends that this SDK version has no named field for.
   *
   * This index signature already made unknown keys reachable in the TYPE; what
   * dropped them was the normaliser, which built each item from a hand-picked
   * list of keys. So `event_end` and `event_precision` were computed by the
   * server, dropped by its own payload, and dropped a third time here, and a
   * caller could not read "sometime in April" even once the server sent it.
   */
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
  /**
   * `response_shape: "ranked"` only: every item once, best first across
   * types, each with `rank`, `category` and `metadata.rank_score`. The five
   * lists above are then views of these same items, by type.
   */
  items?: RawContextItem[];
  /** `response_shape: "ranked"` only: the chunks, quotes from source documents. */
  excerpts?: RawContextItem[];
  conversation_context?: RawConversationContext | null;
  metadata?: Json;
  [key: string]: unknown;
}

// ─── Normalised shapes (camelCase) ────────────────────────────────────────────

export interface TemporalFields {
  eventDate: string | null;
  /**
   * The rest of the event window. `eventDate` is its START.
   *
   * "last month" is stored as 2023-04-01 to 2023-04-30 at precision "month",
   * and reading only the start makes it look like the 1st, so arithmetic from
   * it is wrong by up to the width of the window. `eventPrecision` carries the
   * one distinction nothing else does: "approximate" has no end, so "around
   * November 20" and "on November 20" differ only by this label.
   */
  eventEnd: string | null;
  eventPrecision: string | null;
  /**
   * The day the words were SAID, which is not the day the event happened.
   *
   * `event_date` is empty unless somebody stated when the event took place, so
   * without this field beside it an item whose turn stated no event date
   * arrives with no machine-readable date at all. Empty when the client never
   * told the server when the conversation happened; it is never the server's
   * own ingest or extraction clock.
   */
  saidAt: string | null;
  validUntil: string | null;
  temporalCategory: string | null;
  temporalConfidence: number;
}

/**
 * Which of the fetch's search queries found an item. A fetch can carry several
 * and returns one set for all of them. Always an array: empty when the fetch
 * named no query or the server predates the field.
 */
export interface QueryAttribution {
  matchedQueries: string[];
}

export interface Fact extends TemporalFields, QueryAttribution {
  id: string;
  content: string;
  confidence: number;
  source: string;
  extractedAt: string | null;
  metadata: Json;
}

export interface Preference extends TemporalFields, QueryAttribution {
  id: string;
  category: string;
  content: string;
  strength: number;
  source: string;
  extractedAt: string | null;
  metadata: Json;
}

export interface Episode extends TemporalFields, QueryAttribution {
  id: string;
  summary: string;
  occurredAt: string | null;
  significance: number;
  participants: unknown[];
  metadata: Json;
}

export interface Emotion extends TemporalFields, QueryAttribution {
  id: string;
  emotionType: string;
  intensity: number;
  detectedAt: string | null;
  context: string;
  metadata: Json;
}

export interface TemporalEvent extends QueryAttribution {
  id: string;
  content: string;
  eventDate: string | null;
  /** The rest of the window. See the note on `TemporalFields.eventEnd`. */
  eventEnd: string | null;
  eventPrecision: string | null;
  /** See the note on `TemporalFields.saidAt`. */
  saidAt: string | null;
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

/**
 * Why a fetch may be thinner than the stores can support. Null when healthy.
 *
 * `degraded` says something went wrong, `level` how thin the answer is
 * ("partial" | "minimal" | "failed"), and `code` whether a retry is worth it
 * ("timeout" | "overloaded" | "unavailable" | "internal"). The server never
 * names which of its services broke.
 *
 * It had no slot here, and that is how a real failure stayed invisible: a
 * retrieval that lost its embedding call returned 200 with zero items for a
 * user holding 122 memories, and the caller could not tell that from "no
 * relevant memories".
 */
export interface RetrievalHealth {
  degraded: boolean;
  level: string | null;
  code: string | null;
  warnings: string[];
}

export interface NormalisedContext {
  facts: Fact[];
  preferences: Preference[];
  episodes: Episode[];
  emotions: Emotion[];
  temporalEvents: TemporalEvent[];
  conversationContext: ConversationContext | null;
  metadata: ContextMetadata;
  retrieval: RetrievalHealth | null;
  rawResponse: Json;
  /**
   * `response_shape: "ranked"` only, and absent otherwise: every item once, best
   * first across types, as the wire sends it (snake_case, with `rank`,
   * `category` and `metadata.rank_score`). The five lists above hold the same
   * items by type, except the chunks, which are in `excerpts`.
   */
  items?: RawContextItem[];
  /** `response_shape: "ranked"` only, and absent otherwise: the chunks, quotes from source documents. */
  excerpts?: RawContextItem[];
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

  /**
   * The moment the question is being asked, so the server can resolve a period
   * the question names ("in March", "two months ago") instead of counting back
   * from now.
   *
   * Sent only when provided, like `precision_level`: the golden payloads in
   * CONTRACT/conformance/request_payloads.json are byte-for-byte, so an
   * always-present key would break every existing case.
   *
   * Lenient on the server: `"2023-03-11"`, `"2023/03/11 21:26"` and
   * `"2023/03/11 (Sat) 21:26"` all parse, and anything unreadable falls back to
   * now with a warning rather than failing the read.
   */
  query_date?: string;
  queryDate?: string;

  /**
   * How many items each search query may contribute when `search_query` holds
   * more than one, so a query with many strong matches cannot take every
   * place. `max_results` stays the ceiling on the whole response: three
   * queries at five each under `max_results: 10` return ten.
   *
   * Sent only when provided, like `query_date`. Left out, or with one query,
   * the response is the best `max_results` items overall, as before. `0` is
   * treated as left out.
   */
  max_results_per_query?: number;
  maxResultsPerQuery?: number;

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

  /**
   * `"by_type"` (default): the five typed lists, as every earlier SDK version
   * got them. `"ranked"`: also `items`, every item once in the order the
   * server ranked them across types, with chunks typed as `excerpts`; the
   * typed lists are then views of those items.
   *
   * Sent only when `"ranked"`, so a default fetch sends the body it always
   * did. A server that predates the parameter returns the five lists, and the
   * SDK then builds `items` from them by `metadata.rank_score`.
   */
  response_shape?: string;
  responseShape?: string;
}
