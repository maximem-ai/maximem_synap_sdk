/**
 * In-memory TTL cache for context bundles pushed over gRPC.
 *
 * Port of `maximem_synap/cache/anticipation_cache.py`.
 *
 * ## Why this is a faithful port and not a simplification
 *
 * `@maximem/synap-vercel-adk` already contains a native TS anticipation cache,
 * and it is a simplification: a hardcoded 30-minute TTL, a plain BM25 threshold
 * with no per-query derivation, and none of the gates below. Shipping that as
 * the SDK's cache would mean two retrieval behaviors under one brand, which is
 * gotcha G-C. Every gate here exists because a specific eval failed without it:
 *
 *  - **Recall bypass** runs BEFORE scoring. The measured false hit scored 2.86
 *    at coverage 1.00, so no score-side gate could ever have caught it.
 *  - **The novel-term gate** is suppressed below a 200-stem corpus, because at
 *    cold start ordinary English trips it on nearly every query.
 *  - **The per-query threshold** scales down for short queries, which otherwise
 *    cannot reach the nominal score at all.
 *  - **The coverage gate** catches a bundle that clears BM25 on one shared
 *    token ("account") while lacking the asked-for fact.
 *
 * All thresholds come from the shared behavior contract.
 */

import { BM25, tokenize } from '../cache/bm25.js';
import { CONTRACT, THRESHOLDS, isRecallQuery } from '../behavior/contract.js';
import { getEnvFlag, getEnvFloat } from '../util/env.js';

const EFFECTIVE_FLOOR = CONTRACT.thresholds.effective_floor.value;
const QUERY_TOKEN_SCALE = CONTRACT.thresholds.query_token_scale.value;
const REJECTED_CAP = CONTRACT.telemetry.rejected_items_cap.value;
const PREVIEW_CHARS = CONTRACT.telemetry.content_preview_chars;

/** Sentinel scope that matches any requester. */
export const ANY_SCOPE = '_any';

export interface BundleItem {
  item_id?: string;
  content?: string;
  [key: string]: unknown;
}

export interface ContextBundle {
  bundleId: string;
  entityId?: string | null;
  conversationId?: string | null;
  itemsByType: Record<string, BundleItem[]>;
  /** Server hint; may only SHORTEN the global TTL, never extend it. */
  ttlHintSeconds?: number | null;
  /** Which anticipation pattern produced this bundle. Diagnostics only. */
  bundleType?: string | null;
  /** The queries the server anticipated. Diagnostics only. */
  searchQueries?: readonly string[] | null;
}

interface ItemRecord {
  bundleId: string;
  itemType: string;
  content: string;
  tokens: string[];
  itemDict: BundleItem;
}

interface EntryRecord {
  bundleId: string;
  entityId: string;
  conversationId: string | null;
  bundleType: string | null;
  searchQueries: string[];
  /** Sliding lease, refreshed on hit unless TTL hints are honoured. */
  storedAt: number;
  /** Absolute creation time; the max-entry-age cap uses this, not the lease. */
  createdAt: number;
  ttlHintSeconds: number | null;
}

export interface LookupResult {
  itemsByType: Record<string, BundleItem[]>;
  bundleIds: string[];
  score: number;
  coverage: number;
}

export interface LookupParams {
  searchQuery?: readonly string[] | null;
  entityId?: string | null;
  customerId?: string | null;
  clientId?: string | null;
  conversationId?: string | null;
  maxItems?: number;
}

export type ExitReason =
  | 'empty'
  | 'recall_bypass'
  | 'no_query_tokens'
  | 'novel_term_gate'
  | 'empty_corpus'
  | 'no_items_above_threshold'
  | 'coverage_gate'
  | 'hit';

export interface LookupTelemetry {
  searchQuery: string[];
  entityId: string | null;
  customerId: string | null;
  clientId: string | null;
  conversationId: string | null;
  scopeFilterAccepted: string[];
  novelTermRatio: number | null;
  bm25Threshold: number | null;
  bm25QueryTokens: string[];
  itemsPicked: unknown[];
  itemsRejected: unknown[];
  coverage: number | null;
  hit: boolean;
  exitReason: ExitReason;
}

export interface SnapshotItemPreview {
  type: string;
  scope: string | null;
  content: string;
}

export interface SnapshotItemRecord {
  bundle_id: string;
  item_type: string;
  tokens: string[];
  content: string;
}

export interface SnapshotBundle {
  bundle_id: string;
  entity_id: string;
  conversation_id: string | null;
  bundle_type: string | null;
  search_queries: string[];
  scope_counts: Record<string, number>;
  total_items: number;
  items: SnapshotItemPreview[];
}

export interface AnticipationCacheSnapshot {
  total_entries: number;
  total_item_records: number;
  scope_breakdown_overall: Record<string, number>;
  corpus_vocab_size: number;
  corpus_vocab_sample: string[];
  item_records: SnapshotItemRecord[];
  bundles: SnapshotBundle[];
}

export interface AnticipationCacheOptions {
  ttlSeconds?: number;
  maxBundles?: number;
  /** Injected in tests. Defaults to a monotonic clock. */
  now?: () => number;
  onLookup?: (telemetry: LookupTelemetry) => void;
}

const DEFAULT_TTL_SECONDS = 1800;
const DEFAULT_MAX_BUNDLES = 50;

export class AnticipationCache {
  private readonly ttlSeconds: number;
  private readonly maxBundles: number;
  private readonly now: () => number;
  private readonly onLookup: ((t: LookupTelemetry) => void) | undefined;

  private entries = new Map<string, EntryRecord>();
  private items: ItemRecord[] = [];
  private corpusVocab = new Set<string>();
  private bm25: BM25 | null = null;
  private bm25Dirty = true;

  constructor(options: AnticipationCacheOptions = {}) {
    this.ttlSeconds = options.ttlSeconds ?? DEFAULT_TTL_SECONDS;
    this.maxBundles = options.maxBundles ?? DEFAULT_MAX_BUNDLES;
    this.now = options.now ?? monotonicNow;
    this.onLookup = options.onLookup;
  }

  get size(): number {
    return this.entries.size;
  }

  get itemCount(): number {
    return this.items.length;
  }

  store(bundle: ContextBundle): void {
    if (!bundle.bundleId) return;
    this.evictExpired();

    // Client-scope knowledge (FAQs, policies) arrives with no entity, and the
    // sentinel makes it visible to every requester.
    const entityId = bundle.entityId || ANY_SCOPE;

    if (this.entries.has(bundle.bundleId)) this.dropBundle(bundle.bundleId);

    const now = this.now();
    this.entries.set(bundle.bundleId, {
      bundleId: bundle.bundleId,
      entityId,
      conversationId: bundle.conversationId ?? null,
      bundleType: bundle.bundleType ?? null,
      searchQueries: [...(bundle.searchQueries ?? [])],
      storedAt: now,
      createdAt: now,
      ttlHintSeconds: bundle.ttlHintSeconds ?? null,
    });

    for (const [itemType, list] of Object.entries(bundle.itemsByType ?? {})) {
      for (const item of list ?? []) {
        const content = String(item.content ?? '');
        const tokens = tokenize(content);
        this.items.push({ bundleId: bundle.bundleId, itemType, content, tokens, itemDict: item });
        for (const t of tokens) this.corpusVocab.add(t);
      }
    }
    this.bm25Dirty = true;

    // LRU by lease age.
    while (this.entries.size > this.maxBundles) {
      let oldest: EntryRecord | null = null;
      for (const e of this.entries.values()) {
        if (oldest === null || e.storedAt < oldest.storedAt) oldest = e;
      }
      if (oldest === null) break;
      this.dropBundle(oldest.bundleId);
    }
  }

  lookup(params: LookupParams): LookupResult | null {
    const {
      searchQuery = null, entityId = null, customerId = null,
      clientId = null, conversationId = null, maxItems = 10,
    } = params;

    this.evictExpired();

    const accepted = this.buildAcceptedScope(entityId, customerId, clientId);
    const telemetry: LookupTelemetry = {
      searchQuery: (searchQuery ?? []).filter((q): q is string => Boolean(q)),
      entityId, customerId, clientId, conversationId,
      scopeFilterAccepted: [...accepted].sort(),
      novelTermRatio: null, bm25Threshold: null, bm25QueryTokens: [],
      itemsPicked: [], itemsRejected: [], coverage: null,
      hit: false, exitReason: 'empty',
    };

    if (this.entries.size === 0 || this.items.length === 0) {
      return this.miss(telemetry, 'empty');
    }

    const hasQuery = (searchQuery ?? []).some((q) => q && q.trim() !== '');
    if (!hasQuery) {
      return this.freshnessLookup(entityId, maxItems, telemetry);
    }

    // Recall bypass fires BEFORE scoring: a stale false-hit on a recall-shaped
    // question makes the agent deny a fact the store holds, and the measured
    // failure cleared every score-side gate.
    if (getEnvFlag(CONTRACT.recall_bypass.env_flag) && isRecallQuery(searchQuery)) {
      return this.miss(telemetry, 'recall_bypass');
    }

    const queryTokens = (searchQuery ?? []).flatMap((q) => tokenize(q));
    telemetry.bm25QueryTokens = queryTokens;
    if (queryTokens.length === 0) {
      return this.miss(telemetry, 'no_query_tokens');
    }

    const uniqueStems = new Set(queryTokens);
    let novel = 0;
    for (const s of uniqueStems) if (!this.corpusVocab.has(s)) novel++;
    const novelRatio = uniqueStems.size ? novel / uniqueStems.size : 0;
    telemetry.novelTermRatio = round4(novelRatio);

    // Only meaningful once the corpus has real vocabulary. Below the floor the
    // ratio is dominated by ordinary English and would miss on nearly anything.
    if (
      this.corpusVocab.size >= THRESHOLDS.minCorpusForNovelGate &&
      novelRatio >= THRESHOLDS.novelTerm
    ) {
      return this.miss(telemetry, 'novel_term_gate');
    }

    if (this.bm25Dirty || this.bm25 === null) {
      if (this.items.length === 0) return this.miss(telemetry, 'empty_corpus');
      this.bm25 = new BM25(this.items.map((i) => i.tokens));
      this.bm25Dirty = false;
    }

    const scores = this.bm25.scores(queryTokens);
    const validBundles = this.getValidBundleIds(accepted, conversationId);

    const effectiveThreshold = Math.max(
      EFFECTIVE_FLOOR,
      Math.min(THRESHOLDS.bm25, QUERY_TOKEN_SCALE * queryTokens.length),
    );
    telemetry.bm25Threshold = round4(effectiveThreshold);

    const picked: { score: number; item: ItemRecord }[] = [];
    const rejected: { score: number; payload: Record<string, unknown> }[] = [];

    for (let idx = 0; idx < scores.length; idx++) {
      const item = this.items[idx]!;
      const score = scores[idx]!;
      const reject = (reason: string) =>
        rejected.push({
          score,
          payload: {
            item_index: idx, bundle_id: item.bundleId, item_type: item.itemType,
            content_first_120: item.content.slice(0, PREVIEW_CHARS),
            bm25_score: round4(score), reason,
          },
        });
      if (score < effectiveThreshold) { reject('below_threshold'); continue; }
      if (!validBundles.has(item.bundleId)) { reject('scope_filter_excluded'); continue; }
      picked.push({ score, item });
    }

    rejected.sort((a, b) => b.score - a.score);
    telemetry.itemsRejected = rejected.slice(0, REJECTED_CAP).map((r) => r.payload);

    if (picked.length === 0) {
      return this.miss(telemetry, 'no_items_above_threshold');
    }

    picked.sort((a, b) => b.score - a.score);
    const top = picked.slice(0, maxItems);

    const pickedStems = new Set<string>();
    for (const { item } of top) for (const t of item.tokens) pickedStems.add(t);
    let overlap = 0;
    for (const s of uniqueStems) if (pickedStems.has(s)) overlap++;
    const coverage = uniqueStems.size ? overlap / uniqueStems.size : 1.0;
    telemetry.coverage = round4(coverage);

    // Observe-only unless explicitly configured. A malformed setting stays
    // observe-only rather than failing closed on every lookup.
    const coverageMin = getEnvFloat('SYNAP_SDK_CACHE_COVERAGE_MIN');
    if (coverageMin !== null && coverage < coverageMin) {
      // Gate BEFORE any side effect, so a rejected lookup leaves cache state
      // untouched and does not refresh a lease.
      return this.miss(telemetry, 'coverage_gate');
    }

    const itemsByType: Record<string, BundleItem[]> = {};
    const bundleIds = new Set<string>();
    for (const { item } of top) {
      (itemsByType[item.itemType] ??= []).push(item.itemDict);
      bundleIds.add(item.bundleId);
    }

    // Honouring a TTL hint also means a hit stops renewing the lease, otherwise
    // a hinted bundle under steady traffic would outlive its hint.
    const refreshOnHit = !getEnvFlag('SYNAP_SDK_CACHE_HONOR_TTL_HINT');
    if (refreshOnHit) {
      const now = this.now();
      for (const id of bundleIds) {
        const entry = this.entries.get(id);
        if (entry) entry.storedAt = now;
      }
    }

    telemetry.hit = true;
    telemetry.exitReason = 'hit';
    telemetry.itemsPicked = top.map(({ score, item }) => ({
      bundle_id: item.bundleId, item_type: item.itemType,
      content_first_120: item.content.slice(0, PREVIEW_CHARS), bm25_score: round4(score),
    }));
    this.fire(telemetry);

    return {
      itemsByType,
      bundleIds: [...bundleIds],
      score: top[0]!.score,
      coverage,
    };
  }

  /**
   * Drop this entity's bundles after a write, so the cache stops serving a
   * pre-write view. Gated on the flag: this is the write-path hook.
   */
  invalidateEntity(entityId: string): number {
    if (!getEnvFlag('SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE')) return 0;
    return this.dropEntity(entityId);
  }

  /**
   * Unconditionally drop this entity's bundles. Backs `cache.clear_user` and
   * `cache.clear_customer`, which are explicit user requests and so are not
   * gated on the invalidate-on-write flag.
   */
  dropEntity(entityId: string): number {
    let dropped = 0;
    for (const entry of [...this.entries.values()]) {
      // Never drop shared client-scope knowledge on one entity's behalf.
      if (entry.entityId === ANY_SCOPE) continue;
      if (entry.entityId === entityId) { this.dropBundle(entry.bundleId); dropped++; }
    }
    return dropped;
  }

  /**
   * A read-only diagnostic projection of the cache.
   *
   * Backs `client.anticipation_cache_snapshot()` and mirrors Python's shape
   * key for key, including snake_case keys: this output gets logged, diffed
   * and pasted into issues across both SDKs, so it is a wire format.
   *
   * Note the 140-character content previews. That is Python's literal, and it
   * is deliberately NOT the 120 of `telemetry.content_preview_chars` from the
   * contract, which caps lookup telemetry rather than this snapshot.
   */
  snapshot(): AnticipationCacheSnapshot {
    const SNAPSHOT_PREVIEW_CHARS = 140;
    const overallScope = new Map<string, number>();
    const itemsByBundle = new Map<string, ItemRecord[]>();
    for (const record of this.items) {
      const list = itemsByBundle.get(record.bundleId);
      if (list === undefined) itemsByBundle.set(record.bundleId, [record]);
      else list.push(record);
    }

    const bundles: SnapshotBundle[] = [];
    for (const [bundleId, entry] of this.entries) {
      const perBundle = new Map<string, number>();
      const itemPreviews: SnapshotItemPreview[] = [];
      for (const record of itemsByBundle.get(bundleId) ?? []) {
        // `scope` is absent on plenty of items; Python's Counter happily keys
        // on None, which JSON-serialises to null, so preserve the distinction
        // rather than coercing to a string.
        const scope = (record.itemDict['scope'] ?? null) as string | null;
        const key = scope === null ? '' : scope;
        overallScope.set(key, (overallScope.get(key) ?? 0) + 1);
        perBundle.set(key, (perBundle.get(key) ?? 0) + 1);
        itemPreviews.push({
          type: record.itemType,
          scope,
          content: record.content.slice(0, SNAPSHOT_PREVIEW_CHARS),
        });
      }
      let totalItems = 0;
      for (const n of perBundle.values()) totalItems += n;
      bundles.push({
        bundle_id: bundleId,
        entity_id: entry.entityId,
        conversation_id: entry.conversationId,
        bundle_type: entry.bundleType,
        search_queries: [...entry.searchQueries],
        scope_counts: Object.fromEntries(perBundle),
        total_items: totalItems,
        items: itemPreviews,
      });
    }

    return {
      total_entries: this.entries.size,
      total_item_records: this.items.length,
      scope_breakdown_overall: Object.fromEntries(overallScope),
      corpus_vocab_size: this.corpusVocab.size,
      corpus_vocab_sample: [...this.corpusVocab].sort().slice(0, 80),
      item_records: this.items.slice(0, 20).map((rec) => ({
        bundle_id: rec.bundleId,
        item_type: rec.itemType,
        tokens: rec.tokens.slice(0, 25),
        content: rec.content.slice(0, SNAPSHOT_PREVIEW_CHARS),
      })),
      bundles,
    };
  }

  /**
   * The freshest `user_summary` bundle for this entity, or null.
   *
   * `entityId` is REQUIRED, and a missing one refuses the lookup rather than
   * falling back to the freshest summary across all users. Python hardened this
   * for exactly that reason: the permissive version could splice user A's
   * summary into user B's response on the conversation-scope path.
   */
  lookupUserSummary(entityId: string | null | undefined): Record<string, unknown> | null {
    if (!entityId) return null;
    this.evictExpired();

    let freshest: EntryRecord | null = null;
    for (const entry of this.entries.values()) {
      if (entry.bundleType !== 'user_summary') continue;
      // ANY_SCOPE covers client-shared summaries, which are safe for everyone.
      if (entry.entityId !== entityId && entry.entityId !== ANY_SCOPE) continue;
      if (freshest === null || entry.storedAt > freshest.storedAt) freshest = entry;
    }
    if (freshest === null) return null;

    // Serving refreshes the sliding lease, unless the server's TTL hint is
    // being honoured, in which case the hint owns expiry.
    if (!getEnvFlag('SYNAP_SDK_CACHE_HONOR_TTL_HINT')) {
      freshest.storedAt = this.now();
    }
    return this.#bundleFor(freshest.bundleId);
  }

  /** Rebuild a bundle's items_by_type from the flat item records. */
  #bundleFor(bundleId: string): Record<string, unknown> {
    const itemsByType: Record<string, BundleItem[]> = {};
    for (const record of this.items) {
      if (record.bundleId !== bundleId) continue;
      (itemsByType[record.itemType] ??= []).push(record.itemDict);
    }
    return { bundle_id: bundleId, items_by_type: itemsByType };
  }

  /**
   * Drop every bundle belonging to one conversation.
   *
   * Backs the `compaction_update` signal on the Listen stream: a compaction
   * rewrites the conversation's summary, so any bundle built from the previous
   * state is stale and must not keep serving for the rest of its TTL.
   *
   * Unlike `dropEntity`, client-scope bundles are NOT spared: a bundle is only
   * dropped here if it is bound to this specific conversation, and a shared
   * client-scope bundle has no conversation id to match.
   */
  dropConversation(conversationId: string): number {
    if (!conversationId) return 0;
    let dropped = 0;
    for (const entry of [...this.entries.values()]) {
      if (entry.conversationId === conversationId) { this.dropBundle(entry.bundleId); dropped++; }
    }
    return dropped;
  }

  clear(): void {
    this.entries.clear();
    this.items = [];
    this.corpusVocab.clear();
    this.bm25 = null;
    this.bm25Dirty = true;
  }

  // ── internals ───────────────────────────────────────────────────────────────

  private miss(telemetry: LookupTelemetry, reason: ExitReason): null {
    telemetry.exitReason = reason;
    telemetry.hit = false;
    this.fire(telemetry);
    return null;
  }

  private fire(telemetry: LookupTelemetry): void {
    // Exception-safe by contract: a buggy hook must never break the SDK.
    try {
      this.onLookup?.(telemetry);
    } catch {
      /* ignored on purpose */
    }
  }

  private freshnessLookup(
    entityId: string | null,
    maxItems: number,
    telemetry: LookupTelemetry,
  ): LookupResult | null {
    const candidates = [...this.entries.values()]
      .filter((e) => entityId === null || e.entityId === entityId || e.entityId === ANY_SCOPE)
      .sort((a, b) => b.storedAt - a.storedAt);
    if (candidates.length === 0) return this.miss(telemetry, 'empty');

    const newest = candidates[0]!;
    const itemsByType: Record<string, BundleItem[]> = {};
    let n = 0;
    for (const item of this.items) {
      if (item.bundleId !== newest.bundleId) continue;
      if (n >= maxItems) break;
      (itemsByType[item.itemType] ??= []).push(item.itemDict);
      n++;
    }
    if (n === 0) return this.miss(telemetry, 'empty');

    telemetry.hit = true;
    telemetry.exitReason = 'hit';
    this.fire(telemetry);
    return { itemsByType, bundleIds: [newest.bundleId], score: 0, coverage: 1.0 };
  }

  /**
   * The widened scope-match set for a request.
   *
   * A narrower-scope request also matches broader-scope bundles for the same
   * customer or client. Falsy ids are dropped so an empty customer_id cannot
   * accidentally match bundles keyed at "".
   */
  private buildAcceptedScope(
    entityId: string | null, customerId: string | null, clientId: string | null,
  ): Set<string> {
    const accepted = new Set<string>([ANY_SCOPE]);
    if (entityId) accepted.add(entityId);
    if (customerId) accepted.add(customerId);
    if (clientId) accepted.add(clientId);
    return accepted;
  }

  private getValidBundleIds(accepted: Set<string>, conversationId: string | null): Set<string> {
    const valid = new Set<string>();
    for (const entry of this.entries.values()) {
      if (!accepted.has(entry.entityId)) continue;
      // A conversation-scoped bundle is only valid inside its conversation.
      if (entry.conversationId && conversationId && entry.conversationId !== conversationId) continue;
      valid.add(entry.bundleId);
    }
    return valid;
  }

  private effectiveTtl(entry: EntryRecord): number {
    if (!getEnvFlag('SYNAP_SDK_CACHE_HONOR_TTL_HINT')) return this.ttlSeconds;
    if (entry.ttlHintSeconds == null || entry.ttlHintSeconds <= 0) return this.ttlSeconds;
    // A hint may only shorten.
    return Math.min(this.ttlSeconds, entry.ttlHintSeconds);
  }

  private evictExpired(): void {
    const now = this.now();
    // Default cap is twice the sliding TTL; 0 disables. Without it a bundle
    // under steady traffic renews forever and can serve a retired value
    // indefinitely.
    const configuredMaxAge = getEnvFloat('SYNAP_SDK_CACHE_MAX_ENTRY_AGE');
    const maxAge = configuredMaxAge ?? this.ttlSeconds * 2;

    for (const entry of [...this.entries.values()]) {
      const idleExpired = now - entry.storedAt > this.effectiveTtl(entry);
      const ageExpired = maxAge > 0 && now - entry.createdAt > maxAge;
      if (idleExpired || ageExpired) this.dropBundle(entry.bundleId);
    }
  }

  private dropBundle(bundleId: string): void {
    if (!this.entries.delete(bundleId)) return;
    this.items = this.items.filter((i) => i.bundleId !== bundleId);
    // Vocabulary is rebuilt rather than decremented: a stem can be shared by
    // several bundles, so per-bundle subtraction would drop stems still in use.
    this.corpusVocab = new Set(this.items.flatMap((i) => i.tokens));
    this.bm25Dirty = true;
  }
}

function monotonicNow(): number {
  // performance.now() is monotonic and unaffected by wall-clock adjustments,
  // matching Python's time.monotonic(). Date.now() would let an NTP step
  // expire or resurrect every entry at once.
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now() / 1000
    : Date.now() / 1000;
}

function round4(n: number): number {
  return Number(n.toFixed(4));
}
