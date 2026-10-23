/**
 * Loader for the shared SDK behavior contract.
 *
 * The values here are tuned against measured eval failures and are shared,
 * byte-identical, with the Python SDK. They live in data rather than code
 * specifically so the two SDKs cannot drift. See
 * `synap/sdk/CONTRACT/README.md`.
 *
 * `anticipation.json` is vendored into `src/` by
 * `synap/sdk/scripts/sync-behavior.sh` and imported (not read from disk) so
 * that the bundler inlines it. Reading it with `node:fs` at runtime would work
 * in Node and break in every bundled target: Edge, Workers, and anything under
 * webpack/turbopack, where `dist/` is rewritten and relative paths do not
 * survive. It is the same failure mode as `@grpc/proto-loader` reading a
 * `.proto` off disk (gotcha G-P).
 */

import contractJson from './anticipation.json' with { type: 'json' };

export interface RecallPattern {
  pattern: string;
  flags: string;
  note: string;
}

export interface BehaviorContract {
  version: number;
  recall_bypass: {
    note: string;
    env_flag: string;
    default: boolean;
    patterns: RecallPattern[];
  };
  thresholds: {
    bm25: { value: number; note: string };
    novel_term: { value: number; note: string };
    min_corpus_for_novel_gate: { value: number; note: string };
    effective_floor: { value: number; note: string };
    query_token_scale: { value: number; note: string };
  };
  coverage_gate: {
    note: string;
    env_flag: string;
    default: number | null;
    empty_query_coverage: number;
  };
  telemetry: {
    rejected_items_cap: { value: number; note: string };
    content_preview_chars: number;
  };
  bm25: {
    k1: number;
    b: number;
    idf_floor: number;
    min_token_len: number;
    min_stem_len: number;
    suffixes: string[];
    stop_words: string[];
    alias_token: { pattern: string; note: string };
  };
  semantic_match: {
    model: string;
    mode: 'bm25' | 'semantic';
    min_cosine: number;
    serve_min_cosine: number;
    asked_kind: boolean;
    max_queries: number;
    item_questions: boolean;
    canonical_query: 'off' | 'canon' | 'both';
    self_pronouns: Record<string, [string, string][] | string>;
    asked_details: Record<string, { query: string; value: string } | string>;
    bundle_serve_min_cosine: number;
    sure_min_cosine: number;
    race_deadline_ms: number;
  };
  history_search: {
    max_excerpts: number;
    max_words_per_excerpt: number;
    min_matched_terms: number;
    rrf_k: number;
    recency_weight: number;
    min_cosine?: number;
    previous_turns?: number;
    previous_turn_words?: number;
    rare_term_max_share?: number;
    tool_result_max_chars?: number;
    passages?: boolean;
    passages_per_turn?: number;
    early_query?: { pattern: string; flags?: string; note?: string };
    asked_kinds?: Record<string, { query: string; value: string } | string>;
    irregular_forms?: Record<string, string>;
    heading: string;
    store_limits: {
      max_conversations: number;
      max_turns_per_conversation: number;
      max_bytes_per_conversation: number;
      idle_ttl_seconds: number;
      max_file_bytes: number;
    };
  };
}

export const CONTRACT = contractJson as unknown as BehaviorContract;

/** ECMAScript flags we accept. Anything else is invalid in Python too. */
const ALLOWED_FLAGS = new Set(['i', 'm', 's']);

function compile(pattern: string, flags: string): RegExp {
  for (const ch of flags) {
    if (!ALLOWED_FLAGS.has(ch)) {
      throw new Error(
        `Unsupported regex flag '${ch}' in behavior contract. Flags must be ` +
          `valid in both Python and ECMAScript.`,
      );
    }
  }
  return new RegExp(pattern, flags);
}

/**
 * Compiled recall-bypass patterns, in contract order.
 *
 * Deliberately NOT using the `g` flag: a global RegExp carries `lastIndex`
 * between calls, so a shared instance would match on one query and then miss
 * on the next. That failure is intermittent and traffic-dependent, which is
 * the worst kind to debug.
 */
export const RECALL_PATTERNS: readonly RegExp[] = CONTRACT.recall_bypass.patterns.map((p) =>
  compile(p.pattern, p.flags ?? ''),
);

export const THRESHOLDS = {
  bm25: CONTRACT.thresholds.bm25.value,
  novelTerm: CONTRACT.thresholds.novel_term.value,
  minCorpusForNovelGate: CONTRACT.thresholds.min_corpus_for_novel_gate.value,
  /** Lower bound on the derived per-query threshold. */
  effectiveFloor: CONTRACT.thresholds.effective_floor.value,
  /** Points of score allowed per query token when deriving the threshold. */
  queryTokenScale: CONTRACT.thresholds.query_token_scale.value,
} as const;

export const BM25_PARAMS = CONTRACT.bm25;
export const STOP_WORDS: ReadonlySet<string> = new Set(CONTRACT.bm25.stop_words);
// Order is load-bearing: stem() returns on first match, so this must stay
// longest-first exactly as the contract lists it.
export const SUFFIXES: readonly string[] = CONTRACT.bm25.suffixes;
export const ALIAS_TOKEN_SOURCE = CONTRACT.bm25.alias_token.pattern;
/** Local search over a conversation's earlier turns (cache/history-search.ts). */
export const HISTORY_SEARCH = CONTRACT.history_search;
/** Matching fetch queries to prefetched items by meaning (context/anticipation-cache.ts). */
export const SEMANTIC_MATCH = CONTRACT.semantic_match;

/** True when any query string looks like an explicit memory-recall ask. */
export function isRecallQuery(searchQuery: readonly (string | null | undefined)[] | null | undefined): boolean {
  for (const q of searchQuery ?? []) {
    if (!q) continue;
    for (const pattern of RECALL_PATTERNS) {
      if (pattern.test(q)) return true;
    }
  }
  return false;
}
