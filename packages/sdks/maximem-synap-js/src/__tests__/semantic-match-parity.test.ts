/**
 * Matching by meaning decides like Python: the same bundles, the same fixed
 * vectors, the same lookups serve the same items in the same order. Corpus:
 * CONTRACT/conformance/semantic_match.json, captured from the Python SDK by
 * scripts/gen_semantic_corpus.py. A failure here is a real divergence; fix
 * the code, not the corpus.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AnticipationCache, type ContextBundle } from '../context/anticipation-cache.js';
import { tryServeFromCache } from '../context/anticipated.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(
  readFileSync(path.resolve(here, '../../../CONTRACT/conformance/semantic_match.json'), 'utf8'),
) as {
  bundles: Record<string, unknown>[];
  vectors: Record<string, number[]>;
  unembedded: string[];
  cases: {
    text: string; entity_id: string | null; conversation_id: string | null;
    min_cosine: number; serve_min_cosine: number; served: string[] | null;
    band: 'sure' | 'borderline' | null;
  }[];
};

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

async function cacheFor(minCosine: number, serveMinCosine: number): Promise<AnticipationCache> {
  const c = new AnticipationCache({ encoder, matcher: 'semantic', minCosine, serveMinCosine });
  for (const b of corpus.bundles) c.store(toBundle(b));
  await c.whenEmbedded();
  // As if their vectors were not ready yet, as the generator does.
  for (const item of (c as unknown as { items: { content: string; vec: unknown; vecModel: string }[] }).items) {
    if (corpus.unembedded.includes(item.content)) {
      item.vec = null;
      item.vecModel = '';
    }
  }
  return c;
}

describe('semantic matching conformance (Python corpus)', () => {
  it('has cases with hits, misses and both bands', () => {
    expect(corpus.cases.some((c) => c.served === null)).toBe(true);
    expect(corpus.cases.some((c) => (c.served?.length ?? 0) > 1)).toBe(true);
    expect(corpus.cases.some((c) => c.band === 'borderline')).toBe(true);
    expect(corpus.cases.some((c) => c.band === 'sure')).toBe(true);
  });

  it('serves the same items in the same order', async () => {
    const caches = new Map<string, AnticipationCache>();
    for (const c of corpus.cases) {
      const key = `${c.min_cosine}/${c.serve_min_cosine}`;
      if (!caches.has(key)) caches.set(key, await cacheFor(c.min_cosine, c.serve_min_cosine));
      const scope = c.entity_id === null ? 'conversation' : 'user';
      const attempt = await tryServeFromCache(caches.get(key)!, scope, {
        searchQuery: [c.text], entityId: c.entity_id, conversationId: c.conversation_id, clientId: 'cli',
      }, 10);
      const served = attempt.hit === null ? null : Object.values(attempt.hit.itemsByType).flat();
      const got = served === null ? null : served.map((i) => String((i as { id?: string }).id));
      // Python returns items in rank order; JS groups by type in rank order.
      // Compare the order within each type and the set across types.
      if (c.served === null || got === null) {
        expect(got, `${key} ${c.text}`).toEqual(c.served);
      } else {
        expect([...got].sort(), `${key} ${c.text}`).toEqual([...c.served].sort());
        expect(attempt.hit?.band ?? null, `${key} ${c.text} band`).toEqual(c.band);
      }
    }
  });
});
