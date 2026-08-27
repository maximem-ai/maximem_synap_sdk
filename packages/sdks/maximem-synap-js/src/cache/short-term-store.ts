/**
 * Per-conversation short-term context, held locally.
 *
 * Mirrors Python's `cache/short_term_store.py`. This is what makes reads see
 * your own writes: a turn recorded a moment ago is buffered here and spliced
 * onto the next fetch, so a fast follow-up does not retrieve context that omits
 * what the user just said. Python's own comment on the overlay puts it plainly:
 * the pre-compaction window "is exactly when the playground's fast follow-ups
 * happen."
 *
 * LRU by conversation, with an absolute age cap. Everything here is per process
 * and holds only what this process wrote, so nothing is authoritative: the
 * server's view wins for the summary fields, and this only ever contributes the
 * verbatim tail.
 */

const DEFAULT_MAX_CONVERSATIONS = 100;
/** 12 hours, as in Python. */
const DEFAULT_MAX_AGE_MS = 12 * 60 * 60 * 1000;

export interface ShortTermTurn {
  role: string;
  content: string;
  /** ISO-8601. */
  timestamp: string;
}

export interface ShortTermEntry {
  conversationId: string;
  summary: string | null;
  factualParagraph: string | null;
  conversationalParagraph: string | null;
  currentState: Record<string, unknown>;
  keyExtractions: Record<string, unknown>;
  compactionId: string | null;
  compactedAt: string | null;
  endTimestamp: string | null;
  recentTurns: ShortTermTurn[];
  lastActivityAt: number;
}

export interface ShortTermStoreOptions {
  maxConversations?: number;
  maxAgeMs?: number;
  /** Injected in tests. Wall clock, because entries are compared to ISO stamps. */
  now?: () => number;
}

/** Parse an ISO-8601 stamp to epoch millis, or null. */
export function parseIso(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null;
  if (value instanceof Date) return value.getTime();
  if (typeof value !== 'string') return null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

/**
 * True when a turn is strictly after the cutoff.
 *
 * An unparseable timestamp KEEPS the turn. Dropping it would silently lose a
 * message, which is worse than briefly showing one the server has already
 * folded into its summary.
 */
function isAfter(turnTimestamp: unknown, cutoffMs: number): boolean {
  const parsed = parseIso(turnTimestamp);
  if (parsed === null) return true;
  return parsed > cutoffMs;
}

export class ShortTermStore {
  // Insertion order IS the LRU order: re-inserting on touch moves an entry to
  // the end, so the first key Map yields is the least recently used.
  readonly #entries = new Map<string, ShortTermEntry>();
  readonly #maxConversations: number;
  readonly #maxAgeMs: number;
  readonly #now: () => number;

  constructor(options: ShortTermStoreOptions = {}) {
    this.#maxConversations = options.maxConversations ?? DEFAULT_MAX_CONVERSATIONS;
    this.#maxAgeMs = options.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
    this.#now = options.now ?? (() => Date.now());
  }

  /** The entry, or null. Age-evicts on read, as Python does. */
  get(conversationId: string): ShortTermEntry | null {
    if (!conversationId) return null;
    const entry = this.#entries.get(conversationId);
    if (entry === undefined) return null;
    if (this.#now() - entry.lastActivityAt > this.#maxAgeMs) {
      this.#entries.delete(conversationId);
      return null;
    }
    this.#touch(conversationId, entry);
    return entry;
  }

  has(conversationId: string): boolean {
    return this.get(conversationId) !== null;
  }

  get size(): number {
    return this.#entries.size;
  }

  /**
   * Append a raw turn.
   *
   * Called from `record_message` and the stream's `send_message`. When no entry
   * exists yet (the first turn, before any compaction has arrived) a fresh one
   * is created with an empty summary; the next `compaction_update` fills the
   * summary fields in and prunes the turns it covers.
   */
  appendTurn(conversationId: string, role: string, content: string, timestamp?: string): void {
    if (!conversationId) return;
    const stamp = timestamp ?? new Date(this.#now()).toISOString();
    const entry = this.#entries.get(conversationId) ?? this.#blank(conversationId);
    entry.recentTurns.push({ role, content, timestamp: stamp });
    entry.lastActivityAt = parseIso(stamp) ?? this.#now();
    this.#touch(conversationId, entry);
    this.#evict();
  }

  /**
   * Apply a `compaction_update` bundle.
   *
   * Replaces the summary fields from the bundle's `conversation_context` and
   * prunes locally buffered turns at or before its `end_timestamp`. Turns that
   * arrived DURING the compaction window are kept by design: the server has not
   * seen them yet, which is the whole reason the local tail exists.
   */
  applyCompaction(bundle: Record<string, unknown>): void {
    if (typeof bundle !== 'object' || bundle === null) return;
    const cc = (bundle['conversation_context'] ?? {}) as Record<string, unknown>;
    const conversationId = String(
      cc['conversation_id'] ??
        bundle['anticipation_conversation_id'] ??
        bundle['conversation_id'] ??
        '',
    );
    if (!conversationId) return;

    const endTs = parseIso(cc['end_timestamp'] ?? cc['compacted_at']);
    const entry = this.#entries.get(conversationId) ?? this.#blank(conversationId);

    // `x || existing` throughout, matching Python: a blank field from the
    // server must not erase what is already held.
    entry.summary = (cc['summary'] as string) || entry.summary;
    entry.factualParagraph = (cc['factual_paragraph'] as string) || entry.factualParagraph;
    entry.conversationalParagraph =
      (cc['conversational_paragraph'] as string) || entry.conversationalParagraph;
    entry.currentState = { ...((cc['current_state'] as Record<string, unknown>) ?? {}) };
    entry.keyExtractions = { ...((cc['key_extractions'] as Record<string, unknown>) ?? {}) };
    entry.compactionId = (cc['compaction_id'] as string) || entry.compactionId;
    entry.compactedAt = (cc['compacted_at'] as string) || entry.compactedAt;
    entry.endTimestamp = (cc['end_timestamp'] as string) || entry.endTimestamp;

    if (endTs !== null && entry.recentTurns.length > 0) {
      entry.recentTurns = entry.recentTurns.filter((t) => isAfter(t.timestamp, endTs));
    }
    entry.lastActivityAt = this.#now();
    this.#touch(conversationId, entry);
    this.#evict();
  }

  invalidate(conversationId: string): void {
    this.#entries.delete(conversationId);
  }

  clear(): void {
    this.#entries.clear();
  }

  #blank(conversationId: string): ShortTermEntry {
    return {
      conversationId,
      summary: null,
      factualParagraph: null,
      conversationalParagraph: null,
      currentState: {},
      keyExtractions: {},
      compactionId: null,
      compactedAt: null,
      endTimestamp: null,
      recentTurns: [],
      lastActivityAt: this.#now(),
    };
  }

  /** Re-insert so this key becomes the most recently used. */
  #touch(conversationId: string, entry: ShortTermEntry): void {
    this.#entries.delete(conversationId);
    this.#entries.set(conversationId, entry);
  }

  #evict(): void {
    while (this.#entries.size > this.#maxConversations) {
      const oldest = this.#entries.keys().next();
      if (oldest.done === true) break;
      this.#entries.delete(oldest.value);
    }
  }
}
