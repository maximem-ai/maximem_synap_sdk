/**
 * Local store of a conversation's raw turns, for search after compaction.
 *
 * Port of `maximem_synap/cache/turn_store.py`. The short-term store keeps
 * only the turns a compaction has not covered yet; this store keeps every
 * turn this process saw or fetched, so the details a compaction leaves out
 * can be found again by a local search (history-search.ts).
 *
 * Python keeps it in a SQLite file under ~/.synap. This one holds it in
 * memory, with the same limits and the same answers, because the SDK also
 * runs in browsers and edge runtimes, and the main entry may not load a node
 * builtin eagerly (bundle-hygiene.test.ts). A turn this process did not see
 * arrives through `addMany` from the turns route, with its position.
 */

import { HISTORY_SEARCH } from '../behavior/contract.js';

export interface StoreLimits {
  max_conversations: number;
  max_turns_per_conversation: number;
  max_bytes_per_conversation: number;
  idle_ttl_seconds: number;
  max_file_bytes: number;
}

export interface StoredTurn {
  /** Insertion order in this store. */
  seq: number;
  role: string;
  content: string;
  timestamp: string;
  message_id: string | null;
  /** 1-based turn number on the server, when known. */
  position: number | null;
}

export interface TurnInput {
  role?: string;
  content?: string;
  timestamp?: string | Date | null;
  message_id?: string | null;
  position?: number | string | null;
}

const SWEEP_EVERY_WRITES = 50;

interface Conversation {
  lastActivity: number;
  turns: StoredTurn[];
}

function bytesOf(text: string): number {
  return new TextEncoder().encode(text).length;
}

export class TurnStore {
  readonly limits: StoreLimits;
  readonly onDisk = false;
  #now: () => number;
  #seq = 0;
  #writes = 0;
  #conversations = new Map<string, Conversation>();
  #vectors = new Map<string, Map<number, Float32Array>>();

  constructor(options: { limits?: StoreLimits; now?: () => number } = {}) {
    this.limits = options.limits ?? { ...HISTORY_SEARCH.store_limits };
    this.#now = options.now ?? (() => Date.now() / 1000);
  }

  add(
    conversationId: string,
    role: string,
    content: string,
    timestamp?: string | null,
    options: { messageId?: string | null; position?: number | null } = {},
  ): void {
    this.addMany(conversationId, [
      { role, content, timestamp: timestamp ?? '', message_id: options.messageId ?? null, position: options.position ?? null },
    ]);
  }

  /**
   * Insert turns not held yet; returns how many were new. A turn held without
   * an id or position (a streamed write) gets them when the same role and
   * text come back from the server.
   */
  addMany(conversationId: string, turns: Iterable<TurnInput>): number {
    if (!conversationId) return 0;
    const conv = this.#conversations.get(conversationId) ?? { lastActivity: 0, turns: [] };
    let added = 0;
    for (const turn of turns) {
      const role = String(turn.role || 'user');
      const content = String(turn.content ?? '');
      if (!content) continue;
      const messageId = turn.message_id ? String(turn.message_id) : null;
      const raw = turn.position;
      const position =
        typeof raw === 'number' && Number.isInteger(raw) ? raw : /^\d+$/.test(String(raw ?? '')) ? Number(raw) : null;
      const ts = turn.timestamp instanceof Date ? turn.timestamp.toISOString() : String(turn.timestamp ?? '');
      if (messageId !== null) {
        const held = conv.turns.find((t) => t.message_id === messageId);
        if (held !== undefined) {
          if (position !== null) held.position = position;
          continue;
        }
      }
      const same = conv.turns.find(
        (t) => t.role === role && t.content === content && (t.message_id === null || t.position === null),
      );
      if (same !== undefined) {
        same.message_id = same.message_id ?? messageId;
        same.position = position ?? same.position;
        continue;
      }
      conv.turns.push({ seq: ++this.#seq, role, content, timestamp: ts, message_id: messageId, position });
      added += 1;
    }
    conv.lastActivity = this.#now();
    this.#conversations.set(conversationId, conv);
    this.#trim(conv);
    this.#writes += 1;
    if (this.#writes % SWEEP_EVERY_WRITES === 0) this.sweep();
    return added;
  }

  #trim(conv: Conversation): void {
    const limits = this.limits;
    let size = conv.turns.reduce((n, t) => n + bytesOf(t.content), 0);
    while (conv.turns.length > limits.max_turns_per_conversation || size > limits.max_bytes_per_conversation) {
      const excess = Math.max(conv.turns.length - limits.max_turns_per_conversation, 1);
      const order = [...conv.turns].sort(
        (a, b) => (a.position ?? Number.MAX_SAFE_INTEGER) - (b.position ?? Number.MAX_SAFE_INTEGER) || a.seq - b.seq,
      );
      const gone = new Set(order.slice(0, excess).map((t) => t.seq));
      if (gone.size === 0) break;
      for (const t of conv.turns) if (gone.has(t.seq)) size -= bytesOf(t.content);
      conv.turns = conv.turns.filter((t) => !gone.has(t.seq));
      for (const vectors of this.#vectors.values()) for (const seq of gone) vectors.delete(seq);
    }
  }

  /** Idle conversations, then the least recently used beyond the count, then more until the bytes fit. */
  sweep(): void {
    const limits = this.limits;
    const now = this.#now();
    for (const [id, conv] of this.#conversations) {
      if (conv.lastActivity < now - limits.idle_ttl_seconds) this.drop(id);
    }
    const byRecent = [...this.#conversations.entries()].sort((a, b) => b[1].lastActivity - a[1].lastActivity);
    for (const [id] of byRecent.slice(limits.max_conversations)) this.drop(id);
    while (this.sizeBytes() > limits.max_file_bytes) {
      const oldest = this.conversations()[0];
      if (oldest === undefined) break;
      this.drop(oldest);
    }
  }

  drop(conversationId: string): void {
    const conv = this.#conversations.get(conversationId);
    if (conv === undefined) return;
    for (const vectors of this.#vectors.values()) for (const t of conv.turns) vectors.delete(t.seq);
    this.#conversations.delete(conversationId);
  }

  clear(): void {
    this.#conversations.clear();
    this.#vectors.clear();
  }

  close(): void {
    // Nothing to release in memory; kept for parity with the Python store.
  }

  /** In conversation order: by server position where known, else as written here. */
  turns(conversationId: string): StoredTurn[] {
    const conv = this.#conversations.get(conversationId);
    if (conv === undefined) return [];
    const bySeq = [...conv.turns].sort((a, b) => a.seq - b.seq);
    const key = new Map<number, number>();
    let top = 0;
    let step = 0;
    for (const t of bySeq) {
      if (t.position !== null) {
        key.set(t.seq, t.position);
        top = Math.max(top, t.position);
        step = 0;
      } else {
        step += 1;
        key.set(t.seq, top + step * 1e-6);
      }
    }
    return bySeq
      .sort((a, b) => (key.get(a.seq) ?? 0) - (key.get(b.seq) ?? 0) || a.seq - b.seq)
      .map((t) => ({ ...t }));
  }

  /** Turns the server also holds: archived tool results are local only. Python: `count`. */
  count(conversationId: string): number {
    return (this.#conversations.get(conversationId)?.turns ?? []).filter((t) => t.role !== 'tool').length;
  }

  sizeBytes(): number {
    let n = 0;
    for (const conv of this.#conversations.values()) for (const t of conv.turns) n += bytesOf(t.content);
    return n;
  }

  /** Conversation ids, least recently active first. */
  conversations(): string[] {
    return [...this.#conversations.entries()].sort((a, b) => a[1].lastActivity - b[1].lastActivity).map(([id]) => id);
  }

  vectors(seqs: readonly number[], model: string): Map<number, Float32Array> {
    const held = this.#vectors.get(model);
    const out = new Map<number, Float32Array>();
    if (held === undefined) return out;
    for (const seq of seqs) {
      const v = held.get(seq);
      if (v !== undefined) out.set(seq, v);
    }
    return out;
  }

  putVectors(model: string, items: Map<number, Float32Array>): void {
    const held = this.#vectors.get(model) ?? new Map<number, Float32Array>();
    for (const [seq, vec] of items) held.set(seq, vec);
    this.#vectors.set(model, held);
  }
}
