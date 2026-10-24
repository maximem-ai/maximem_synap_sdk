/**
 * A prefetch-cache hit serves each memory with the server's date label, as
 * Python does: the label (`ContextItemProto.date_prefix`) goes in front of the
 * content of the copy a hit hands out, never twice, and never into the stored
 * item, so it plays no part in matching. Corpus:
 * CONTRACT/conformance/cache_date_prefix.json, captured from the Python SDK by
 * scripts/gen_cache_date_prefix_corpus.py. A failure here is a real divergence;
 * fix the code, not the corpus.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AnticipationCache, asServed, type BundleItem, type ContextBundle } from '../context/anticipation-cache.js';
import { tryServeFromCache } from '../context/anticipated.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(
  readFileSync(path.resolve(here, '../../../CONTRACT/conformance/cache_date_prefix.json'), 'utf8'),
) as {
  serve_cases: { name: string; item: BundleItem; matched: string[] | null; expected: BundleItem }[];
  bundles: Record<string, unknown>[];
  vectors: Record<string, number[]>;
  lookup_cases: {
    name: string; matcher: 'semantic' | 'bm25'; search_query: string[]; max_items: number;
    served: { item_id: string; content: string; has_label_key: boolean }[] | null;
  }[];
};

describe('as served, against the Python corpus', () => {
  it.each(corpus.serve_cases)('$name', (c) => {
    const stored = structuredClone(c.item);
    expect(asServed(stored, c.matched)).toEqual(c.expected);
    expect(stored).toEqual(c.item);
  });

  it('serves an item without a label as stored', () => {
    const item = { item_id: 'm1', content: 'The user likes tea' };
    expect(asServed(item)).toBe(item);
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
    itemsByType: structuredClone(b['items_by_type']) as ContextBundle['itemsByType'],
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
  it.each(corpus.lookup_cases)('$matcher $name', async (c) => {
    const cache = await cacheFor(c.matcher);
    const attempt = await tryServeFromCache(cache, 'user', {
      searchQuery: c.search_query, entityId: 'ana', clientId: 'cli',
    }, c.max_items);
    const served = attempt.hit === null ? null : Object.values(attempt.hit.itemsByType).flat().map((i) => ({
      item_id: String(i.item_id), content: String(i.content), has_label_key: 'date_prefix' in i,
    }));
    // Python returns items in rank order; JS groups by type in rank order.
    const byId = (xs: { item_id: string }[]) => [...xs].sort((a, b) => a.item_id.localeCompare(b.item_id));
    expect(served === null ? null : byId(served)).toEqual(c.served === null ? null : byId(c.served));
  });

  it('leaves the stored items unlabelled', async () => {
    const cache = await cacheFor('bm25');
    const c = corpus.lookup_cases.find((x) => x.matcher === 'bm25' && x.name === 'both_topics')!;
    await tryServeFromCache(cache, 'user', { searchQuery: c.search_query, entityId: 'ana', clientId: 'cli' }, 10);
    const again = await tryServeFromCache(cache, 'user', { searchQuery: c.search_query, entityId: 'ana', clientId: 'cli' }, 10);
    const contents = Object.values(again.hit!.itemsByType).flat().map((i) => String(i.content));
    expect(contents.some((x) => x.startsWith('(said'))).toBe(true);
    expect(contents.every((x) => !x.startsWith('(said May 11, 2023, happened Mar 2023) (said'))).toBe(true);
  });
});

describe('the stream decoder, with the options stream-client uses', () => {
  it('reads date_prefix, and an older server\'s item decodes with an empty one', async () => {
    const protobuf = (await import('protobufjs')).default;
    const root = protobuf.Root.fromJSON(SYNAP_PROTO_DESCRIPTOR as any);
    const Item = root.lookupType('synap.v1.ContextItemProto');
    const opts = { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true };
    const set = Item.toObject(Item.decode(Item.encode({ item_id: 'a', content: 'x', date_prefix: '(said May 11, 2023) ' }).finish()), opts);
    const unset = Item.toObject(Item.decode(Item.encode({ item_id: 'b', content: 'x' }).finish()), opts);
    expect(set['date_prefix']).toBe('(said May 11, 2023) ');
    expect(unset['date_prefix']).toBe('');
    expect(asServed(set as BundleItem).content).toBe('(said May 11, 2023) x');
    const { date_prefix: _empty, ...withoutLabel } = unset;
    expect(asServed(unset as BundleItem)).toEqual(withoutLabel);
  });
});
