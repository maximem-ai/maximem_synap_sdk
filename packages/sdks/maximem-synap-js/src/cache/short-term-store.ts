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
 *
 * Holding an entry is not the same as holding the conversation. A process that
 * restarted, evicted the entry, or shares the conversation with another process
 * has only the turns it recorded itself. So the server reports how many
 * messages it holds, on every write and on every compaction bundle, and
 * `isComplete()` says whether the entry can answer on its own. When it cannot,
 * the caller asks the server and hands the answer to `reseed`.
 */

import type { TurnStore } from './turn-store.js';

const DEFAULT_MAX_CONVERSATIONS = 100;
/** 12 hours, as in Python. */
const DEFAULT_MAX_AGE_MS = 12 * 60 * 60 * 1000;

export interface ShortTermTurn {
  role: string;
  content: string;
  /** ISO-8601. */
  timestamp: string;
  /** The server's id for the turn, when the write returned one. */
  message_id?: string;
}

/**
 * How every v3 context_block begins (server: compaction/v3/render.py). A
 * summary that begins this way IS the block, which is how a REST-fed entry is
 * recognised. Same constant as Python's `CONTEXT_BLOCK_PREFIX`.
 */
export const CONTEXT_BLOCK_PREFIX = '## Conversation so far (compacted';

/** The v3 block in a conversation_context payload, or null (v2 server). */
export function contextBlockOf(cc: Record<string, unknown>): string | null {
  const block = cc['context_block'];
  if (typeof block === 'string' && block) return block;
  const summary = cc['summary'];
  if (typeof summary === 'string' && summary.startsWith(CONTEXT_BLOCK_PREFIX)) return summary;
  return null;
}

export interface ShortTermEntry {
  conversationId: string;
  summary: string | null;
  /**
   * Compaction v3: the server's one framed block, printed as received and
   * once. When set, the paragraphs are not used (the server sends them empty).
   */
  contextBlock: string | null;
  factualParagraph: string | null;
  conversationalParagraph: string | null;
  currentState: Record<string, unknown>;
  keyExtractions: Record<string, unknown>;
  compactionId: string | null;
  compactedAt: string | null;
  endTimestamp: string | null;
  recentTurns: ShortTermTurn[];
  lastActivityAt: number;
  // --- Completeness bookkeeping. See `isComplete`. ---
  /** Messages the cached summary covers, from the start of the conversation. */
  compactedMessageCount: number;
  /** The highest message count the server has reported. null = never heard. */
  serverMessageCount: number | null;
  /**
   * The server has confirmed at least one write made by this process. A
   * process that only reads never hears a count again after it is seeded, so
   * it could not notice the conversation moving on without it.
   */
  participant: boolean;
  /** At some point this entry held everything the server held. */
  synced: boolean;
  /**
   * The server's count ran ahead of an entry that was in sync, so something
   * else is writing to this conversation. Latched: this process cannot see
   * those writes when they happen, only at its own next write.
   */
  shared: boolean;
}

/** What a write learned from the server about the turn and the conversation. */
export interface TurnConfirmation {
  /** The server's id for the turn. A turn already held under it is not appended twice. */
  messageId?: string | undefined;
  /** The conversation's message count as the server reported it on this write. */
  serverMessageCount?: number | null | undefined;
  /** Other writes this process still has outstanding for the conversation. */
  inFlight?: number | undefined;
}

/** What `/context-for-prompt` returned, as far as `reseed` needs it. */
export interface ServerPromptState {
  compactionId: string | null;
  recentMessages: ShortTermTurn[];
  compactedMessageCount: number;
  totalMessageCount: number;
}

export function localMessageCount(entry: ShortTermEntry): number {
  return entry.compactedMessageCount + entry.recentTurns.length;
}

/**
 * Whether an entry can stand in for the server's short-term context.
 *
 * True when the server holds nothing this entry lacks. Holding MORE than the
 * server has confirmed is fine: those are this process's own turns, sent and
 * not yet counted.
 */
export function isComplete(entry: ShortTermEntry): boolean {
  return (
    entry.participant &&
    entry.synced &&
    !entry.shared &&
    entry.serverMessageCount !== null &&
    entry.serverMessageCount <= localMessageCount(entry)
  );
}

/**
 * Whether `turns` already holds `turn`.
 *
 * By message id where both sides have one. Otherwise by role and text: a
 * streamed turn has no id (the server assigns it in a background write), and
 * the turns in a `fetch` response carry none either.
 */
function heldIn(turn: ShortTermTurn, turns: ShortTermTurn[]): boolean {
  return turns.some((held) =>
    turn.message_id && held.message_id
      ? held.message_id === turn.message_id
      : held.role === turn.role && held.content === turn.content,
  );
}

/**
 * The local turns that come after everything in `serverTurns`.
 *
 * Walks back from the newest local turn and stops at the first one the
 * server's list already holds. What is left is what this process wrote or sent
 * after the server's list was built.
 *
 * A turn the server rewrote before storing it (a privacy policy) does not
 * match its local copy by text, so it can come back from here as well as from
 * the server. That shows one turn twice; it never loses one.
 */
export function turnsAfter(
  localTurns: readonly ShortTermTurn[],
  serverTurns: ShortTermTurn[],
): ShortTermTurn[] {
  const newer: ShortTermTurn[] = [];
  for (let i = localTurns.length - 1; i >= 0; i -= 1) {
    const turn = localTurns[i] as ShortTermTurn;
    if (heldIn(turn, serverTurns)) break;
    newer.unshift(turn);
  }
  return newer;
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

  /**
   * Every turn appended here is also kept in the archive (a TurnStore), which
   * a compaction does not prune: the turns it covers stay searchable locally
   * (history-search.ts). Python: `ShortTermContextStore.attach_archive`.
   */
  #archive: TurnStore | null = null;

  constructor(options: ShortTermStoreOptions = {}) {
    this.#maxConversations = options.maxConversations ?? DEFAULT_MAX_CONVERSATIONS;
    this.#maxAgeMs = options.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
    this.#now = options.now ?? (() => Date.now());
  }

  attachArchive(archive: TurnStore | null): void {
    this.#archive = archive;
  }

  get archive(): TurnStore | null {
    return this.#archive;
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
   *
   * `confirmation` is what the write learned from the server. See
   * `TurnConfirmation` and `noteServerCount`.
   */
  appendTurn(
    conversationId: string,
    role: string,
    content: string,
    timestamp?: string,
    confirmation: TurnConfirmation = {},
  ): void {
    if (!conversationId) return;
    const stamp = timestamp ?? new Date(this.#now()).toISOString();
    if (this.#archive !== null) {
      try {
        this.#archive.add(conversationId, role, content, stamp, { messageId: confirmation.messageId ?? null });
      } catch {
        // The archive never blocks the prompt path.
      }
    }
    const entry = this.#entries.get(conversationId) ?? this.#blank(conversationId);
    const messageId = confirmation.messageId;
    const alreadyHeld =
      Boolean(messageId) && entry.recentTurns.some((t) => t.message_id === messageId);
    if (!alreadyHeld) {
      const turn: ShortTermTurn = { role, content, timestamp: stamp };
      if (messageId) turn.message_id = messageId;
      entry.recentTurns.push(turn);
    }
    const count = confirmation.serverMessageCount;
    if (count !== null && count !== undefined) {
      this.#noteCount(entry, Number(count), confirmation.inFlight ?? 0);
    }
    entry.lastActivityAt = parseIso(stamp) ?? this.#now();
    this.#touch(conversationId, entry);
    this.#evict();
  }

  /**
   * Record the server's message count for a conversation.
   *
   * The streamed write path's half of `appendTurn`'s confirmation: a streamed
   * turn is appended when it is sent, and the count arrives later, once the
   * server has written it. A count for a conversation this process holds
   * nothing for is ignored.
   */
  noteServerCount(conversationId: string, count: number): void {
    if (!conversationId) return;
    const entry = this.#entries.get(conversationId);
    if (entry !== undefined) this.#noteCount(entry, Number(count), 0);
  }

  /**
   * Compare the server's count with what the entry holds.
   *
   * The count is taken after one of this process's own writes, so it proves
   * the process takes part in the conversation.
   *
   * Server ahead of an entry that was in sync: something else wrote, and the
   * entry is marked shared for good. Server ahead of an entry that was never
   * in sync: this process joined late (a restart, an eviction), and `reseed`
   * will fix it. `inFlight` covers this process's own writes that the server
   * has counted and the entry has not appended yet, so two concurrent writes
   * do not look like a second writer.
   */
  #noteCount(entry: ShortTermEntry, count: number, inFlight: number): void {
    entry.participant = true;
    if (count > localMessageCount(entry) + inFlight) {
      if (entry.synced) entry.shared = true;
    } else if (entry.serverMessageCount === null) {
      // First word from the server, and nothing is missing.
      entry.synced = true;
    }
    entry.serverMessageCount = Math.max(entry.serverMessageCount ?? 0, count);
  }

  /**
   * Apply a `compaction_update` bundle.
   *
   * Replaces the summary fields from the bundle's `conversation_context` and
   * drops the buffered turns the new summary covers.
   *
   * Which turns those are is decided by count, not by clock. The bundle says
   * how many messages the summary covers and how many the conversation held
   * when the bundle was built, so the last `total - compacted` of its
   * `recent_turns` are exactly the uncovered messages. Turns this process
   * appended after the server took that count are kept behind them. Timestamps
   * cannot do this job: the REST write returns the time of the response and
   * not of the stored row, a caller may supply its own message time, and a
   * streamed turn is stamped when a background write runs.
   *
   * A bundle from a server that sends no counts falls back to the older
   * timestamp prune, and the entry is left out of sync so it is never used to
   * answer on its own.
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
    const compactedAt = parseIso(cc['compacted_at']);
    const compactedCount = Number(cc['compacted_message_count'] ?? 0) || 0;
    const totalCount = Number(cc['total_message_count'] ?? 0) || 0;
    const entry = this.#entries.get(conversationId) ?? this.#blank(conversationId);

    const heldAt = parseIso(entry.compactedAt);
    // An older compaction must not replace a newer one.
    if (compactedAt !== null && heldAt !== null && compactedAt < heldAt) return;

    // `x || existing` throughout, matching Python: a blank field from the
    // server must not erase what is already held.
    const block = contextBlockOf(cc);
    if (block) {
      // A v3 compaction replaces the whole record. Keeping a v2 paragraph
      // cached from an earlier push would print it next to the block, so the
      // paragraphs are cleared, not kept.
      entry.contextBlock = block;
      entry.summary = (cc['summary'] as string) || block;
      entry.factualParagraph = null;
      entry.conversationalParagraph = null;
    } else {
      entry.summary = (cc['summary'] as string) || entry.summary;
      entry.factualParagraph = (cc['factual_paragraph'] as string) || entry.factualParagraph;
      entry.conversationalParagraph =
        (cc['conversational_paragraph'] as string) || entry.conversationalParagraph;
    }
    entry.currentState = { ...((cc['current_state'] as Record<string, unknown>) ?? {}) };
    entry.keyExtractions = { ...((cc['key_extractions'] as Record<string, unknown>) ?? {}) };
    entry.compactionId = (cc['compaction_id'] as string) || entry.compactionId;
    entry.compactedAt = (cc['compacted_at'] as string) || entry.compactedAt;
    entry.endTimestamp = (cc['end_timestamp'] as string) || entry.endTimestamp;

    if (compactedCount > 0 && compactedCount <= totalCount) {
      const uncovered = totalCount - compactedCount;
      const serverTurns = Array.isArray(cc['recent_turns'])
        ? (cc['recent_turns'] as ShortTermTurn[])
        : [];
      const tail: ShortTermTurn[] =
        uncovered > 0 ? serverTurns.slice(-uncovered).map((t) => ({ ...t })) : [];
      // Turns appended here after the server counted. Only an entry that was
      // in sync knows how many of its own turns those are.
      const ahead = localMessageCount(entry) - totalCount;
      const newer =
        entry.synced && ahead > 0
          ? entry.recentTurns.slice(-ahead).filter((t) => !heldIn(t, tail))
          : [];
      entry.recentTurns = [...tail, ...newer];
      entry.compactedMessageCount = compactedCount;
      entry.serverMessageCount = Math.max(entry.serverMessageCount ?? 0, totalCount);
      // A short tail means the bundle did not carry every uncovered message,
      // so the entry cannot claim to hold them.
      entry.synced = tail.length === uncovered;
    } else {
      if (endTs !== null && entry.recentTurns.length > 0) {
        entry.recentTurns = entry.recentTurns.filter((t) => isAfter(t.timestamp, endTs));
      }
      entry.synced = false;
    }
    entry.lastActivityAt = this.#now();
    this.#touch(conversationId, entry);
    this.#evict();
  }

  /**
   * Bring an entry back in step from a `/context-for-prompt` response.
   *
   * Called after the server was asked because the entry was not complete. The
   * response lists every message the server's summary does not cover, which is
   * what `recentTurns` is meant to hold.
   *
   * Only an entry that already exists is reseeded. Creating one here would let
   * a process that never writes to the conversation answer from a snapshot it
   * has no way to refresh.
   *
   * If the server's summary is not the one cached here, the entry cannot
   * render it, so it stays out of sync until the next bundle brings the
   * summary itself.
   *
   * Turns held locally and absent from the response are kept when they come
   * after everything the response has: they were written or sent after the
   * server read its messages.
   */
  reseed(conversationId: string, state: ServerPromptState): void {
    if (!conversationId) return;
    const entry = this.#entries.get(conversationId);
    if (entry === undefined || entry.shared) return;
    entry.serverMessageCount = Math.max(
      entry.serverMessageCount ?? 0,
      Number(state.totalMessageCount) || 0,
    );
    if ((state.compactionId || null) !== entry.compactionId) {
      entry.synced = false;
      return;
    }
    const serverTurns = state.recentMessages.map((m) => ({ ...m }));
    entry.recentTurns = [...serverTurns, ...turnsAfter(entry.recentTurns, serverTurns)];
    entry.compactedMessageCount = Number(state.compactedMessageCount) || 0;
    entry.synced = true;
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
      contextBlock: null,
      factualParagraph: null,
      conversationalParagraph: null,
      currentState: {},
      keyExtractions: {},
      compactionId: null,
      compactedAt: null,
      endTimestamp: null,
      recentTurns: [],
      lastActivityAt: this.#now(),
      compactedMessageCount: 0,
      serverMessageCount: null,
      participant: false,
      synced: false,
      shared: false,
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
