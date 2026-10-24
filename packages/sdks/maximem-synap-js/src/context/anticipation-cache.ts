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
 *  - **Matching by meaning** (CONTRACT semantic_match) decides a hit when a
 *    model is loaded, with a close match raced against the server. It replaced
 *    a hard-coded list of recall phrases that sent those questions to the
 *    server: a measured false hit scored 2.86 at coverage 1.00, which no
 *    word-score gate could catch.
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
import { CONTRACT, SEMANTIC_MATCH, THRESHOLDS } from '../behavior/contract.js';
import type { Encoder } from '../cache/history-search.js';
import { getEnv, getEnvFlag, getEnvFloat } from '../util/env.js';

const EFFECTIVE_FLOOR = CONTRACT.thresholds.effective_floor.value;
const QUERY_TOKEN_SCALE = CONTRACT.thresholds.query_token_scale.value;
const REJECTED_CAP = CONTRACT.telemetry.rejected_items_cap.value;
const PREVIEW_CHARS = CONTRACT.telemetry.content_preview_chars;

/** Sentinel scope that matches any requester. */
export const ANY_SCOPE = '_any';

// ── the rung a request or a bundle stands at ───────────────────────────────
//
// A pushed bundle is stamped with the scope rung the server retrieved it at,
// and a fetch names one through `scope_path`. Both are written the same way so
// they can be compared as strings, because nothing in this process can resolve
// anything: the server speaks in registry node ids and an SDK in a customer's
// app has never seen one, so the comparison happens in the vocabulary the
// CALLER uses, which is level keys and the client's own external ids.
//
// The recipe and its golden vectors live in the shared behavior contract
// (CONTRACT/behavior/anticipation.json, `rung_key`). It is written three times
// (here, the Python SDK, the server) because the producer is not an SDK, so
// the usual "port it once" rule cannot apply. Do NOT re-tune it here.
const RUNG_SEPARATOR = '|';
const RUNG_ASSIGN = '=';

/**
 * The rung a `scope_path` names, canonically, or null for no path.
 *
 * Sorted by level key because an object carries whatever order the caller
 * wrote it in, and the ladder's own order is not knowable here. Only the value
 * is escaped: level keys are `^[a-z][a-z0-9_]{0,63}$`, so they can hold
 * neither separator. Backslash is escaped first, or `a|b` and `a\pb` would
 * encode to the same string, which is a false match between two rungs.
 */
export function rungKey(
  scopePath: Record<string, string> | null | undefined,
): string | null {
  if (!scopePath) return null;
  const pairs: [string, string][] = [];
  for (const [k, v] of Object.entries(scopePath)) {
    if (k === '' || v === null || v === undefined || String(v) === '') continue;
    pairs.push([String(k), String(v)]);
  }
  if (pairs.length === 0) return null;
  pairs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return pairs
    .map(([k, v]) => {
      const escaped = v
        .split('\\').join('\\\\')
        .split(RUNG_SEPARATOR).join('\\p')
        .split(RUNG_ASSIGN).join('\\e');
      return `${k}${RUNG_ASSIGN}${escaped}`;
    })
    .join(RUNG_SEPARATOR);
}

/**
 * May a bundle stamped `bundleRung` answer a request at `requestRung`?
 *
 * Equality, and equality on purpose. A subset rule ("the bundle names more
 * rungs than you did, close enough") reads as the generous option and is the
 * dangerous one: the rung nobody named is exactly the rung that tells two
 * people sharing an external id apart, which is the case the whole ladder
 * exists for. Being wrong here costs a cold fetch; being wrong the other way
 * hands one person's context to another.
 *
 * null on either side means "named no rung". Both null is every request in
 * production today and it serves, so nothing changes for anyone who has not
 * turned nested scoping on. `''` normalises to null because a proto string
 * field has no null and an unstamped bundle arrives as the empty string.
 */
export function rungServes(
  bundleRung: string | null | undefined,
  requestRung: string | null | undefined,
): boolean {
  return (bundleRung || null) === (requestRung || null);
}

/**
 * A bundle the server pushed with no user or customer stamp, tied to a
 * conversation other than the one this request names (or to one when the
 * request names none).
 *
 * Nothing on such a bundle says whose it is: it is stamped ANY_SCOPE because
 * the request that triggered it named no user. Its conversation is the only
 * owner it has, so outside that conversation it is served to nobody. Without
 * this, a user-scope or client-scope fetch, which names no conversation, could
 * be answered from another user's conversation.
 */
export function unownedElsewhere(
  entryEntity: string, entryConversation: string | null, conversationId: string | null,
): boolean {
  return entryEntity === ANY_SCOPE && entryConversation !== null && entryConversation !== conversationId;
}

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
  /**
   * The scope rung the server retrieved this bundle at, as `rungKey` writes
   * one. Absent from anything an older server sent, which reads as "named no
   * rung" rather than as "matches anything".
   */
  scopeRung?: string | null;
}

interface ItemRecord {
  bundleId: string;
  itemType: string;
  content: string;
  tokens: string[];
  itemDict: BundleItem;
  /**
   * The item's vector under `vecModel`, filled off the lookup path once a model
   * is loaded. Null until then: the item is matched by BM25 alone.
   */
  vec: number[] | null;
  vecModel: string;
  /** The questions the memory answers (its HQEs), sent with the bundle item, and their vectors. */
  questions: string[];
  qvecs: number[][];
}

/** The asker's own pronouns in the third person memories are stored in. Python: `canonical_query`. */
export function canonicalQuery(query: string): string {
  let out = query;
  for (const [lang, rules] of Object.entries(SEMANTIC_MATCH.self_pronouns ?? {})) {
    if (lang.startsWith('$') || typeof rules === 'string') continue;
    for (const [pattern, repl] of rules) out = out.replace(new RegExp(pattern, 'gi'), repl);
  }
  return out;
}

/** Value patterns for the kinds of value the queries ask for. Python: `_asked_kind_patterns`. */
function askedDetailPatterns(queries: readonly string[]): RegExp[] {
  const out: RegExp[] = [];
  for (const [name, spec] of Object.entries(SEMANTIC_MATCH.asked_details ?? {})) {
    if (name.startsWith('$') || typeof spec !== 'object') continue;
    const q = new RegExp(spec.query, 'i');
    if (queries.some((x) => q.test(x))) out.push(new RegExp(spec.value, 'i'));
  }
  return out;
}

function itemQuestions(item: BundleItem): string[] {
  const qs = (item as { questions?: unknown }).questions;
  return Array.isArray(qs) ? qs.filter((q): q is string => typeof q === 'string' && q.trim() !== '').map((q) => q.trim()) : [];
}

/**
 * The copy of a stored item a hit hands out: the server's date label in front
 * of the content, and the item's `matched_queries` when a per-query share
 * decided them. Python: `as_served`.
 *
 * The label is the one the server prints when it answers the fetch itself
 * (`ContextItemProto.date_prefix`), so a memory reads "(said May 11, 2023) ..."
 * whichever side answered. Not written into the stored item: the cache matches
 * queries against that text, and the dates are not what a query is about. Left
 * alone when the content already opens with it, and when there is none: a
 * server that predates the field sends "". The `date_prefix` key itself is not
 * handed out.
 */
export function asServed(item: BundleItem, matchedQueries: readonly string[] | null = null): BundleItem {
  if (!('date_prefix' in item) && matchedQueries === null) return item;
  // Served inside the content, not beside it: a caller seeing both could add it twice.
  const { date_prefix: raw, ...rest } = item as BundleItem & { date_prefix?: unknown };
  const label = typeof raw === 'string' ? raw : '';
  const out: BundleItem = { ...rest };
  if (matchedQueries !== null) out['matched_queries'] = [...matchedQueries];
  const content = typeof out.content === 'string' ? out.content : '';
  if (label !== '' && !content.startsWith(label)) out.content = label + content;
  return out;
}

export const MATCHERS = ['bm25', 'semantic'] as const;
export type Matcher = (typeof MATCHERS)[number];

/** `SYNAP_SDK_CACHE_MATCHER` (bm25 or semantic) over the contract's mode. */
function matcherSetting(): Matcher {
  const raw = (getEnv('SYNAP_SDK_CACHE_MATCHER') ?? '').trim().toLowerCase();
  return (MATCHERS as readonly string[]).includes(raw) ? (raw as Matcher) : SEMANTIC_MATCH.mode;
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length && i < b.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
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
  scopeRung: string | null;
}

export interface LookupResult {
  itemsByType: Record<string, BundleItem[]>;
  bundleIds: string[];
  score: number;
  coverage: number;
  /**
   * A semantic match below the sure bar is "borderline": the caller may still
   * ask the server, with a short deadline. Python: `_match_band`.
   */
  band?: 'sure' | 'borderline';
}

export interface LookupParams {
  searchQuery?: readonly string[] | null;
  entityId?: string | null;
  customerId?: string | null;
  clientId?: string | null;
  conversationId?: string | null;
  /**
   * The rung this request names, from its `scope_path`. A SECOND, independent
   * dimension and NOT another tier of the entity funnel: the funnel widens and
   * the rung does not, because a rung is a boundary and a boundary that widens
   * is not one.
   */
  scopeRung?: string | null;
  maxItems?: number;
  /**
   * Each query's share of the `maxItems` places, by the server's rule
   * (`reservePerQuery`). Served items then carry `matched_queries`.
   */
  maxResultsPerQuery?: number | null;
  /**
   * For each of `queryVectors`, the query it is a form of (the query as asked,
   * or its third-person form), from `lookupAsync`.
   */
  queryOwners?: readonly string[] | null;
  /**
   * The search queries' vectors under the cache's encoder, from `lookupAsync`.
   * With them (and matcher "semantic") the lookup matches by meaning.
   */
  queryVectors?: readonly (readonly number[])[] | null;
}

export type ExitReason =
  | 'empty'
  | 'no_query_tokens'
  | 'novel_term_gate'
  | 'empty_corpus'
  | 'no_items_above_threshold'
  | 'coverage_gate'
  | 'asked_kind_missing'
  | 'hit';

export interface LookupTelemetry {
  searchQuery: string[];
  entityId: string | null;
  customerId: string | null;
  clientId: string | null;
  conversationId: string | null;
  scopeRung: string | null;
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
  /**
   * The local embedding model, usually handed over later with `setEncoder`
   * once it has loaded. Until then, and with matcher "bm25", lookups are BM25.
   */
  encoder?: Encoder | null;
  matcher?: Matcher;
  minCosine?: number;
  serveMinCosine?: number;
  askedKind?: boolean;
  itemQuestions?: boolean;
  canonical?: 'off' | 'canon' | 'both';
  bundleServeMinCosine?: number;
  sureMinCosine?: number;
}

const DEFAULT_TTL_SECONDS = 1800;
const DEFAULT_MAX_BUNDLES = 50;

/**
 * The queries a per-query share is counted over, as the server counts them:
 * distinct, in the order sent, empty strings dropped. Python: `distinct_queries`.
 */
export function distinctQueries(searchQuery: readonly (string | null | undefined)[] | null | undefined): string[] {
  return [...new Set((searchQuery ?? []).filter((q): q is string => typeof q === 'string' && q !== ''))];
}

/** The queries to share places between, or null for one ranked cut. Python: `_shares`. */
function sharesOf(searchQuery: readonly (string | null | undefined)[] | null | undefined, perQuery: number | null | undefined): string[] | null {
  if (!perQuery || perQuery <= 0) return null;
  const queries = distinctQueries(searchQuery);
  return queries.length > 1 ? queries : null;
}

/**
 * The server's per-query share (`_reserve_per_query`), on a cache hit.
 *
 * `ranked` is best first and `matched[i]` names the queries item `i` answers.
 * The queries take turns in the order the caller sent them, each taking its
 * best item not yet taken, until it has `perQuery` or nothing is left. `budget`
 * (`max_results`) is never exceeded. An item several queries found counts
 * against each and takes one place. Items no query accounts for fill what room
 * is left, best first. The result keeps the order the items came in. Python:
 * `reserve_per_query`.
 */
export function reservePerQuery<T>(
  ranked: readonly T[],
  budget: number,
  perQuery: number,
  queries: readonly string[],
  matched: readonly (readonly string[])[],
): { kept: T[]; matched: string[][] } {
  if (budget <= 0) return { kept: [], matched: [] };
  const byQuery = new Map(queries.map((q) => [q, ranked.map((_, i) => i).filter((i) => matched[i]!.includes(q))] as const));
  const cursor = new Map<string, number>(queries.map((q) => [q, 0]));
  const taken = new Map<string, number>(queries.map((q) => [q, 0]));
  const picked: number[] = [];
  const pickedSet = new Set<number>();

  let progressed = true;
  while (picked.length < budget && progressed) {
    progressed = false;
    for (const q of queries) {
      if (picked.length >= budget) break;
      if (taken.get(q)! >= perQuery) continue;
      const candidates = byQuery.get(q)!;
      let j = cursor.get(q)!;
      while (j < candidates.length && pickedSet.has(candidates[j]!)) j++;
      if (j >= candidates.length) { cursor.set(q, j); continue; }
      const i = candidates[j]!;
      cursor.set(q, j + 1);
      picked.push(i);
      pickedSet.add(i);
      for (const owner of matched[i]!) if (taken.has(owner)) taken.set(owner, taken.get(owner)! + 1);
      progressed = true;
    }
  }
  for (let i = 0; i < ranked.length; i++) {
    if (picked.length >= budget) break;
    if (!pickedSet.has(i) && !matched[i]!.some((q) => queries.includes(q))) {
      picked.push(i);
      pickedSet.add(i);
    }
  }
  const keep = [...picked].sort((a, b) => a - b);
  return { kept: keep.map((i) => ranked[i]!), matched: keep.map((i) => [...matched[i]!]) };
}

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

  // Semantic matching (CONTRACT.semantic_match). Python: the same fields.
  private encoder: Encoder | null;
  private readonly matcher: Matcher;
  private readonly minCosine: number;
  private readonly serveMinCosine: number;
  private readonly askedKind: boolean;
  private readonly itemQuestions: boolean;
  private readonly canonical: 'off' | 'canon' | 'both';
  private readonly bundleServeMinCosine: number;
  private readonly sureMinCosine: number;
  /** Embedding runs one batch after another; `whenEmbedded` awaits the queue. */
  private embedding: Promise<void> = Promise.resolve();

  constructor(options: AnticipationCacheOptions = {}) {
    this.ttlSeconds = options.ttlSeconds ?? DEFAULT_TTL_SECONDS;
    this.maxBundles = options.maxBundles ?? DEFAULT_MAX_BUNDLES;
    this.now = options.now ?? monotonicNow;
    this.onLookup = options.onLookup;
    this.encoder = options.encoder ?? null;
    this.matcher = options.matcher ?? matcherSetting();
    this.minCosine = options.minCosine ?? SEMANTIC_MATCH.min_cosine;
    this.serveMinCosine = options.serveMinCosine ?? SEMANTIC_MATCH.serve_min_cosine;
    this.askedKind = options.askedKind ?? SEMANTIC_MATCH.asked_kind;
    this.itemQuestions = options.itemQuestions ?? SEMANTIC_MATCH.item_questions;
    this.canonical = options.canonical ?? SEMANTIC_MATCH.canonical_query;
    this.bundleServeMinCosine = options.bundleServeMinCosine ?? SEMANTIC_MATCH.bundle_serve_min_cosine;
    this.sureMinCosine = options.sureMinCosine ?? SEMANTIC_MATCH.sure_min_cosine;
  }

  /**
   * Hand over the local embedding model once it has loaded. Items already
   * cached are embedded off the lookup path.
   */
  setEncoder(encoder: Encoder | null): void {
    this.encoder = encoder;
    this.embedItems(this.items);
  }

  /** True when lookups match by meaning: matcher "semantic" and a model loaded. */
  get semantic(): boolean {
    return this.matcher === 'semantic' && this.encoder !== null;
  }

  /** Resolves once every item stored so far has had its turn to be embedded. */
  whenEmbedded(): Promise<void> {
    return this.embedding;
  }

  /**
   * Give `items` their vectors, queued behind any earlier batch, so neither a
   * bundle arriving on the stream nor a lookup waits for the model. A lookup
   * before they are ready matches those items by BM25 alone.
   */
  private embedItems(items: readonly ItemRecord[]): void {
    const encoder = this.encoder;
    if (encoder === null || this.matcher !== 'semantic') return;
    const todo = items.filter((i) => i.vecModel !== encoder.model);
    if (todo.length === 0) return;
    this.embedding = this.embedding.then(async () => {
      try {
        const texts: string[] = [];
        for (const i of todo) {
          texts.push(i.content);
          if (this.itemQuestions) texts.push(...i.questions);
        }
        const vecs = await encoder.encode(texts);
        let k = 0;
        for (const item of todo) {
          const n = this.itemQuestions ? item.questions.length : 0;
          const v = vecs[k];
          if (v) {
            item.vec = Array.from(v);
            item.qvecs = vecs.slice(k + 1, k + 1 + n).map((x) => Array.from(x));
            item.vecModel = encoder.model;
          }
          k += 1 + n;
        }
      } catch {
        /* the items stay BM25-only */
      }
    });
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
      scopeRung: bundle.scopeRung || null,
    });

    for (const [itemType, list] of Object.entries(bundle.itemsByType ?? {})) {
      for (const item of list ?? []) {
        const content = String(item.content ?? '');
        const tokens = tokenize(content);
        this.items.push({
          bundleId: bundle.bundleId, itemType, content, tokens, itemDict: item,
          vec: null, vecModel: '', questions: itemQuestions(item), qvecs: [],
        });
        for (const t of tokens) this.corpusVocab.add(t);
      }
    }
    this.bm25Dirty = true;
    this.embedItems(this.items.filter((i) => i.bundleId === bundle.bundleId));

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

  /**
   * `lookup`, matching by meaning when a model is loaded: embeds the queries,
   * then looks up with their vectors. A query the model cannot embed is looked
   * up by BM25, as without a model. Python does this inside `lookup`.
   */
  async lookupAsync(params: LookupParams): Promise<LookupResult | null> {
    const encoder = this.encoder;
    let queries = (params.searchQuery ?? [])
      .filter((q): q is string => typeof q === 'string' && q.trim() !== '')
      .map((q) => q.trim())
      .slice(0, SEMANTIC_MATCH.max_queries);
    if (!this.semantic || encoder === null || queries.length === 0 || this.items.length === 0) {
      return this.lookup(params);
    }
    // The query as asked, and/or with the asker's pronouns in the third person
    // the memories are written in (canonical_query). Python: the same.
    const asked = [...queries];
    let owners = [...asked];
    if (this.canonical === 'canon' || this.canonical === 'both') {
      const canon = queries.map(canonicalQuery);
      const changed = canon.filter((c, i) => c !== queries[i]);
      queries = this.canonical === 'both' ? [...queries, ...changed] : canon;
      if (this.canonical === 'both') owners = [...asked, ...asked.filter((q, i) => canon[i] !== q)];
    }
    let queryVectors: number[][] | null = null;
    try {
      queryVectors = await (encoder.encodeQuery ?? encoder.encode).call(encoder, queries);
    } catch {
      queryVectors = null;
    }
    return this.lookup({ ...params, queryVectors, queryOwners: owners });
  }

  lookup(params: LookupParams): LookupResult | null {
    const {
      searchQuery = null, entityId = null, customerId = null,
      clientId = null, conversationId = null, scopeRung = null, maxItems = 10,
      queryVectors = null, queryOwners = null, maxResultsPerQuery = null,
    } = params;
    const shares = sharesOf(searchQuery, maxResultsPerQuery);

    this.evictExpired();

    const accepted = this.buildAcceptedScope(entityId, customerId, clientId);
    const telemetry: LookupTelemetry = {
      searchQuery: (searchQuery ?? []).filter((q): q is string => Boolean(q)),
      entityId, customerId, clientId, conversationId,
      scopeRung: scopeRung ?? null,
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
      return this.freshnessLookup(
        entityId ?? customerId, conversationId, scopeRung ?? null, maxItems, telemetry,
      );
    }

    const queryTokens = (searchQuery ?? []).flatMap((q) => tokenize(q));
    telemetry.bm25QueryTokens = queryTokens;
    if (this.semantic && queryVectors !== null && queryVectors.length > 0) {
      return this.semanticLookup(
        queryTokens, queryVectors, accepted, conversationId, scopeRung ?? null, maxItems, telemetry,
        shares, maxResultsPerQuery, queryOwners);
    }
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
    const validBundles = this.getValidBundleIds(accepted, conversationId, scopeRung ?? null);

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
    let top = picked.slice(0, maxItems);
    let matched: string[][] | null = null;
    if (shares !== null) {
      const perQ = this.bm25PerQuery(shares);
      const indexOf = new Map(this.items.map((it, i) => [it, i] as const));
      const credit = picked.map(({ item }) =>
        shares.filter((q) => perQ.get(q)!.scores[indexOf.get(item)!]! >= perQ.get(q)!.bar));
      ({ kept: top, matched } = reservePerQuery(picked, maxItems, maxResultsPerQuery!, shares, credit));
    }

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

    return this.serve(top, telemetry, coverage, 'bm25_score', matched);
  }

  /**
   * Match by meaning: a lookup is a hit when an item's cosine to a query
   * reaches `minCosine`, and the hit also serves the items at `serveMinCosine`
   * or above. An item with no vector yet is matched by BM25 with its novel-term
   * gate, as before. Python: `_semantic_lookup`, which explains the choices.
   */
  private semanticLookup(
    queryTokens: string[],
    queryVectors: readonly (readonly number[])[],
    accepted: Set<string>,
    conversationId: string | null,
    scopeRung: string | null,
    maxItems: number,
    telemetry: LookupTelemetry,
    shares: string[] | null = null,
    maxResultsPerQuery: number | null = null,
    queryOwners: readonly string[] | null = null,
  ): LookupResult | null {
    const model = this.encoder?.model ?? '';
    if (queryTokens.length > 0 && (this.bm25Dirty || this.bm25 === null) && this.items.length > 0) {
      this.bm25 = new BM25(this.items.map((i) => i.tokens));
      this.bm25Dirty = false;
    }
    const bm25Scores = queryTokens.length > 0 && this.bm25 ? this.bm25.scores(queryTokens) : [];
    const effectiveThreshold = Math.max(
      EFFECTIVE_FLOOR,
      Math.min(THRESHOLDS.bm25, QUERY_TOKEN_SCALE * Math.max(queryTokens.length, 1)),
    );
    telemetry.bm25Threshold = round4(effectiveThreshold);
    const uniqueStems = new Set(queryTokens);
    let novel = 0;
    for (const s of uniqueStems) if (!this.corpusVocab.has(s)) novel++;
    const novelRatio = uniqueStems.size ? novel / uniqueStems.size : 1;
    telemetry.novelTermRatio = round4(novelRatio);
    const wordRoute = queryTokens.length > 0 && !(
      this.corpusVocab.size >= THRESHOLDS.minCorpusForNovelGate && novelRatio >= THRESHOLDS.novelTerm
    );

    const validBundles = this.getValidBundleIds(accepted, conversationId, scopeRung);
    const picked: { score: number; item: ItemRecord }[] = [];
    const companions: { score: number; item: ItemRecord }[] = [];
    const rejected: { score: number; payload: Record<string, unknown> }[] = [];
    this.items.forEach((item, idx) => {
      const b = bm25Scores[idx] ?? 0;
      const reject = (score: number, cos: number | null, reason: string) =>
        rejected.push({
          score,
          payload: {
            item_index: idx, bundle_id: item.bundleId, item_type: item.itemType,
            content_first_120: item.content.slice(0, PREVIEW_CHARS),
            bm25_score: round4(b), cosine: cos === null ? null : round4(cos), reason,
          },
        });
      // Scope first: another user's items are never scored.
      if (!validBundles.has(item.bundleId)) return reject(b, null, 'scope_filter_excluded');
      let passes: boolean;
      let rank: number;
      let cos: number | null = null;
      if (item.vec !== null && item.vecModel === model) {
        // The memory's own text, and the questions it answers.
        const vecs = [item.vec, ...item.qvecs];
        cos = Math.max(...queryVectors.flatMap((q) => vecs.map((v) => cosine(q, v))));
        passes = cos >= this.minCosine;
        rank = cos;
      } else {
        passes = wordRoute && b >= effectiveThreshold;
        rank = this.minCosine; // not embedded yet: ranked at the threshold
      }
      if (!passes && cos !== null && cos >= Math.min(this.serveMinCosine, this.bundleServeMinCosine)) {
        companions.push({ score: cos, item });
        return;
      }
      if (!passes) return reject(rank, cos, 'below_threshold');
      picked.push({ score: rank, item });
    });

    rejected.sort((a, b) => b.score - a.score);
    telemetry.itemsRejected = rejected.slice(0, REJECTED_CAP).map((r) => r.payload);
    if (picked.length === 0) return this.miss(telemetry, 'no_items_above_threshold');
    picked.sort((a, b) => b.score - a.score);
    // Served with a hit: items above the serve bar, and items from a bundle
    // that holds a hit above the lower same-bundle bar.
    const hitBundles = new Set(picked.map((p) => p.item.bundleId));
    const served = companions.filter(
      (c) => c.score >= this.serveMinCosine || (hitBundles.has(c.item.bundleId) && c.score >= this.bundleServeMinCosine),
    );
    served.sort((a, b) => b.score - a.score);
    const candidates = [...picked, ...served];
    let top = candidates.slice(0, maxItems);
    let matched: string[][] | null = null;
    if (shares !== null) {
      // Each query's own best cosine to the item, the asked form and its
      // third-person form together; an item not embedded yet is credited by
      // each query's own BM25 score, as the BM25 path credits it. Python: the same.
      const owners = queryOwners ?? queryVectors.map((_, i) => telemetry.searchQuery[i] ?? '');
      const bar = Math.min(this.minCosine, this.serveMinCosine, this.bundleServeMinCosine);
      const perQ = this.bm25PerQuery(shares);
      const indexOf = new Map(this.items.map((it, i) => [it, i] as const));
      const credit = candidates.map(({ item }) => {
        if (item.vec !== null && item.vecModel === model) {
          const vecs = [item.vec, ...item.qvecs];
          const best = new Map<string, number>();
          queryVectors.forEach((qv, j) => {
            const c = Math.max(...vecs.map((v) => cosine(qv, v)));
            const owner = owners[j] ?? '';
            best.set(owner, Math.max(best.get(owner) ?? -1, c));
          });
          return shares.filter((q) => (best.get(q.trim()) ?? -1) >= bar);
        }
        const idx = indexOf.get(item)!;
        return shares.filter((q) => perQ.get(q)!.scores[idx]! >= perQ.get(q)!.bar);
      });
      ({ kept: top, matched } = reservePerQuery(candidates, maxItems, maxResultsPerQuery!, shares, credit));
    }

    // A question for a kind of value (a number to call, an amount, a date) is
    // not answered by items on its topic that hold no such value.
    if (this.askedKind) {
      const patterns = askedDetailPatterns(telemetry.searchQuery);
      if (patterns.length > 0 && !top.some(({ item }) => patterns.some((p) => p.test(item.content)))) {
        return this.miss(telemetry, 'asked_kind_missing');
      }
    }
    const hit = this.serve(top, telemetry, null, 'cosine', matched);
    // Below the sure bar a hit is borderline: the caller may still ask the
    // server, with a short deadline, and serve this only if it is late.
    hit.band = picked[0]!.score >= this.sureMinCosine ? 'sure' : 'borderline';
    return hit;
  }

  /** Each query's own BM25 scores over every cached item, and its bar. Python: `_bm25_per_query`. */
  private bm25PerQuery(queries: string[]): Map<string, { scores: number[]; bar: number }> {
    if ((this.bm25Dirty || this.bm25 === null) && this.items.length > 0) {
      this.bm25 = new BM25(this.items.map((i) => i.tokens));
      this.bm25Dirty = false;
    }
    const out = new Map<string, { scores: number[]; bar: number }>();
    for (const q of queries) {
      const toks = tokenize(q);
      const scores = toks.length > 0 && this.bm25 ? this.bm25.scores(toks) : this.items.map(() => 0);
      const bar = Math.max(EFFECTIVE_FLOOR, Math.min(THRESHOLDS.bm25, QUERY_TOKEN_SCALE * Math.max(toks.length, 1)));
      out.set(q, { scores, bar: toks.length > 0 ? bar : Infinity });
    }
    return out;
  }

  /** A hit: renew the leases of the bundles used, report, merge the items. */
  private serve(
    top: { score: number; item: ItemRecord }[],
    telemetry: LookupTelemetry,
    coverage: number | null,
    scoreKey: 'bm25_score' | 'cosine',
    matched: string[][] | null = null,
  ): LookupResult {
    // With a per-query share each item carries `matched_queries`, set on a
    // copy (`asServed`) with the date label: the stored item dicts are shared
    // by every later lookup.
    const itemsByType: Record<string, BundleItem[]> = {};
    const bundleIds = new Set<string>();
    top.forEach(({ item }, i) => {
      const dict = asServed(item.itemDict, matched === null ? null : matched[i]!);
      (itemsByType[item.itemType] ??= []).push(dict);
      bundleIds.add(item.bundleId);
    });

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
      content_first_120: item.content.slice(0, PREVIEW_CHARS), [scoreKey]: round4(score),
    }));
    this.fire(telemetry);

    return {
      itemsByType,
      bundleIds: [...bundleIds],
      score: top[0]!.score,
      coverage: coverage ?? 1.0,
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
  lookupUserSummary(
    entityId: string | null | undefined,
    scopeRung: string | null = null,
  ): Record<string, unknown> | null {
    if (!entityId) return null;
    this.evictExpired();

    let freshest: EntryRecord | null = null;
    for (const entry of this.entries.values()) {
      if (entry.bundleType !== 'user_summary') continue;
      if (!rungServes(entry.scopeRung, scopeRung)) continue;
      // ANY_SCOPE covers client-shared summaries, which are safe for everyone.
      if (entry.entityId !== entityId && entry.entityId !== ANY_SCOPE) continue;
      if (unownedElsewhere(entry.entityId, entry.conversationId, null)) continue;
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
    owner: string | null,
    conversationId: string | null,
    scopeRung: string | null,
    maxItems: number,
    telemetry: LookupTelemetry,
  ): LookupResult | null {
    // This path serves a WHOLE bundle on no query at all, so it is the least
    // filtered way out of this cache and needs the rung as much as the scored
    // path does. `owner` is the user, or the customer on a customer fetch. With
    // neither named it serves only ANY_SCOPE bundles: it used to serve the
    // newest bundle of ANY user, so a customer or client fetch with no query
    // got a stranger's items.
    const candidates = [...this.entries.values()]
      .filter((e) => rungServes(e.scopeRung, scopeRung))
      .filter((e) => e.entityId === ANY_SCOPE || (owner !== null && owner !== '' && e.entityId === owner))
      .filter((e) => !unownedElsewhere(e.entityId, e.conversationId, conversationId))
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

  /**
   * The rung is checked FIRST and it never widens. Everything else here widens
   * deliberately: a user-scope request reaches customer-shared bundles, and a
   * conversation-scope request reaches user-scope ones. A rung is the opposite
   * kind of thing, so it is checked before the tiers and no amount of widening
   * can put a cross-rung bundle back in.
   */
  private getValidBundleIds(
    accepted: Set<string>, conversationId: string | null, scopeRung: string | null,
  ): Set<string> {
    const valid = new Set<string>();
    for (const entry of this.entries.values()) {
      if (!rungServes(entry.scopeRung, scopeRung)) continue;
      if (!accepted.has(entry.entityId)) continue;
      // A conversation-scoped bundle is only valid inside its conversation.
      if (entry.conversationId && conversationId && entry.conversationId !== conversationId) continue;
      if (unownedElsewhere(entry.entityId, entry.conversationId, conversationId)) continue;
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
