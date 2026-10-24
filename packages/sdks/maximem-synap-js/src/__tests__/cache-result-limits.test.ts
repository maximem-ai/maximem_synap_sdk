/**
 * The result limits on a prefetch-cache hit decide like Python: a hit serves up
 * to the caller's `max_results`, and `max_results_per_query` gives each query
 * its share by the server's rule, with each served item naming the queries it
 * answers. Corpus: CONTRACT/conformance/cache_result_limits.json, captured from
 * the Python SDK by scripts/gen_cache_limits_corpus.py. A failure here is a real
 * divergence; fix the code, not the corpus.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AnticipationCache, reservePerQuery, type ContextBundle } from '../context/anticipation-cache.js';
import { tryServeFromCache } from '../context/anticipated.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(
  readFileSync(path.resolve(here, '../../../CONTRACT/conformance/cache_result_limits.json'), 'utf8'),
) as {
  reserve_cases: {
    name: string; budget: number; per_query: number; queries: string[]; matched: string[][];
    expected: { kept: string[]; matched: string[][] };
  }[];
  bundles: Record<string, unknown>[];
  vectors: Record<string, number[]>;
  lookup_cases: {
    name: string; matcher: 'semantic' | 'bm25'; search_query: string[]; max_items: number;
    per_query: number | null; served: string[] | null; matched_queries: Record<string, string[]> | null;
  }[];
};

describe('the per-query share, against the Python corpus', () => {
  it.each(corpus.reserve_cases)('$name', (c) => {
    const ranked = c.matched.map((_, i) => `i${i}`);
    const { kept, matched } = reservePerQuery(ranked, c.budget, c.per_query, c.queries, c.matched);
    expect({ kept, matched }).toEqual(c.expected);
  });
});

const encoder = {
  model: 'corpus',
  encode: (texts: string[]) => texts.map((t) => corpus.vectors[t] ?? new Array(8).fill(0)),
};

function toBundle(b: Record<string, unknown>): ContextBundle {
  return {
    bundleId: String(b['bundle_id']),
    entityId: (b['_anticipation_user_id'] as string | undefined) ?? null,
    conversationId: (b['_anticipation_conversation_id'] as string | undefined) ?? null,
    itemsByType: b['items_by_type'] as ContextBundle['itemsByType'],
  };
}

async function cacheFor(matcher: 'semantic' | 'bm25'): Promise<AnticipationCache> {
  const c = matcher === 'semantic'
    ? new AnticipationCache({ encoder, matcher: 'semantic' })
    : new AnticipationCache({ matcher: 'bm25' });
  for (const b of corpus.bundles) c.store(toBundle(b));
  await c.whenEmbedded();
  return c;
}

describe('lookups, against the Python corpus', () => {
  it('covers both matchers, a hit above ten, and a share', () => {
    expect(corpus.lookup_cases.some((c) => (c.served?.length ?? 0) > 10)).toBe(true);
    expect(corpus.lookup_cases.some((c) => c.per_query !== null && c.matched_queries !== null
      && Object.keys(c.matched_queries).length > 0)).toBe(true);
    expect(new Set(corpus.lookup_cases.map((c) => c.matcher))).toEqual(new Set(['semantic', 'bm25']));
  });

  it.each(corpus.lookup_cases)('$matcher $name', async (c) => {
    const cache = await cacheFor(c.matcher);
    const attempt = await tryServeFromCache(cache, 'user', {
      searchQuery: c.search_query, entityId: 'ana', clientId: 'cli', maxResultsPerQuery: c.per_query,
    }, c.max_items);
    const served = attempt.hit === null ? null : Object.values(attempt.hit.itemsByType).flat();
    // Python returns items in rank order; JS groups by type in rank order.
    const got = served === null ? null : served.map((i) => String((i as { id?: string }).id));
    expect(got === null ? null : [...got].sort()).toEqual(c.served === null ? null : [...c.served].sort());
    if (served !== null) {
      const matched = Object.fromEntries(served
        .filter((i) => 'matched_queries' in (i as object))
        .map((i) => [String((i as { id?: string }).id), (i as { matched_queries: string[] }).matched_queries]));
      expect(matched).toEqual(c.matched_queries ?? {});
    }
  });
});
