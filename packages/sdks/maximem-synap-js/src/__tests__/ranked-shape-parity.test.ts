import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFetchRequest, fetchContext } from '../context/fetch.js';
import { applyResponseShape, wireCategory, wireRankScore } from '../context/ranked.js';
import { formatForPrompt, mergeScopeResults } from '../context/unified.js';
import type { FetchOptions, RawContext, RawContextItem } from '../context/types.js';
import { InvalidInputError } from '../errors.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';

/**
 * `response_shape: "ranked"`, against the corpus captured from Python
 * (`synap/sdk/scripts/gen_ranked_corpus.py`). The Python half is
 * `synap/tests/sdk/test_response_shape.py`.
 *
 * Opt-in: a fetch that does not ask sends the body it always sent and gets the
 * raw context back untouched.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(
  readFileSync(
    path.resolve(here, '..', '..', '..', 'CONTRACT', 'conformance', 'ranked_shape.json'),
    'utf8',
  ),
) as {
  fixtures: Record<string, RawContext>;
  parse_cases: Array<{
    name: string; fixture: string; response_shape: string;
    expected: Record<string, unknown>;
  }>;
  merge_cases: Array<{
    name: string; scopes: Array<[string, string]>; include_scope: boolean;
    expected: { items: unknown[]; scope_map: Record<string, string>; total_items: number;
      formatted_context: string };
  }>;
};

const fixture = (key: string): RawContext =>
  JSON.parse(JSON.stringify(corpus.fixtures[key])) as RawContext;

const ids = (items: RawContextItem[] | undefined) =>
  (items ?? []).map((it) => (it.id || it.item_id || '') as string);

const summary = (items: RawContextItem[] | undefined) =>
  (items ?? []).map((it) => [it.id, it['rank'], wireCategory(it), wireRankScore(it)]);

const SCOPES = [
  { scope: 'user' as const, required: { user_id: 'u1' } },
  { scope: 'customer' as const, required: { customer_id: 'c1' } },
  { scope: 'client' as const, required: {} },
  { scope: 'conversation' as const, required: { conversation_id: 'conv-1' } },
];

describe('the request', () => {
  it.each(SCOPES)('$scope scope sends nothing by default', ({ scope, required }) => {
    const body = buildFetchRequest(scope, { ...required } as FetchOptions).body;
    expect('response_shape' in body).toBe(false);
  });

  it.each(SCOPES)('$scope scope sends nothing for by_type', ({ scope, required }) => {
    const body = buildFetchRequest(scope, { ...required, response_shape: 'by_type' } as FetchOptions).body;
    expect('response_shape' in body).toBe(false);
  });

  it.each(SCOPES)('$scope scope sends ranked', ({ scope, required }) => {
    const body = buildFetchRequest(scope, { ...required, responseShape: 'ranked' } as FetchOptions).body;
    expect(body['response_shape']).toBe('ranked');
  });

  it('rejects a bad value, as Python does', () => {
    expect(() => buildFetchRequest('user', { user_id: 'u1', response_shape: 'flat' } as FetchOptions))
      .toThrow(InvalidInputError);
  });
});

describe('parsing, against the Python corpus', () => {
  it.each(corpus.parse_cases)('$name', (c) => {
    const ctx = applyResponseShape(fixture(c.fixture), c.response_shape);
    expect({
      items: summary(ctx.items),
      fact_ids: ids(ctx.facts),
      preference_ids: ids(ctx.preferences),
      episode_ids: ids(ctx.episodes),
      emotion_ids: ids(ctx.emotions),
      temporal_event_ids: ids(ctx.temporal_events),
      excerpt_ids: ids(ctx.excerpts),
    }).toEqual(c.expected);
  });

  it('a by-type response is returned untouched', () => {
    const raw = fixture('legacy');
    expect(applyResponseShape(raw, 'by_type')).toBe(raw);
  });
});

describe('the cross-scope merge, against the Python corpus', () => {
  it.each(corpus.merge_cases)('$name', (c) => {
    const merged = mergeScopeResults(
      c.scopes.map(([scope, key]) => [scope, applyResponseShape(fixture(key), 'ranked')] as const),
    );
    expect(summary(merged.items)).toEqual(c.expected.items);
    expect(merged.scope_map).toEqual(c.expected.scope_map);
    expect(merged.total_items).toBe(c.expected.total_items);
    // Byte for byte: this is the string customers put in a system prompt.
    expect(formatForPrompt(merged, { includeScope: c.include_scope, includeConversationContext: true }))
      .toBe(c.expected.formatted_context);
  });
});

describe('fetchContext', () => {
  const transport = (context: RawContext) => ({
    checkCustomerId: () => undefined,
    request: async () => ({ context }),
  }) as unknown as Parameters<typeof fetchContext>[0];

  it('a ranked fetch from an old server comes back ranked', async () => {
    const ctx = await fetchContext(transport(fixture('legacy')), 'user',
      { user_id: 'u1', response_shape: 'ranked' } as FetchOptions);
    expect(summary(ctx.items).map((r) => r[0])).toEqual(['f1', 't1', 'e1', 'c1', 'm1', 'p1', 'f2', 'f9']);
  });

  it('a default fetch returns the raw context as before', async () => {
    const raw = fixture('legacy');
    const ctx = await fetchContext(transport(raw), 'user', { user_id: 'u1' } as FetchOptions);
    expect(ctx).toEqual(raw);
    expect('items' in ctx).toBe(false);
  });

  it('a prefetched hit is served ranked', async () => {
    const hit = fixture('cache_hit');
    const ctx = await fetchContext(transport({}), 'user',
      { user_id: 'u1', response_shape: 'ranked' } as FetchOptions,
      { beforeFetch: () => ({ response: hit, servedItemIds: [], bundleId: 'b' }) });
    expect(summary(ctx.items)).toEqual([
      ['f1', 1, 'facts', 0.91], ['t1', 2, 'temporal_events', 0.89], ['c1', 3, 'excerpts', 0.81]]);
  });
});

describe('the stream descriptor', () => {
  it('carries rank_score as an optional double on each item', () => {
    const item = (SYNAP_PROTO_DESCRIPTOR as any).nested.synap.nested.v1.nested.ContextItemProto;
    expect(item.fields.rank_score).toMatchObject({ type: 'double', id: 16 });
    expect(item.oneofs._rank_score.oneof).toEqual(['rank_score']);
  });
});

describe('the stream decoder, with the options stream-client uses', () => {
  it('leaves an unset rank_score absent, never 0, and reads a set one', async () => {
    const protobuf = (await import('protobufjs')).default;
    const root = protobuf.Root.fromJSON(SYNAP_PROTO_DESCRIPTOR as any);
    const Item = root.lookupType('synap.v1.ContextItemProto');
    const opts = { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true };
    const unset = Item.toObject(Item.decode(Item.encode({ item_id: 'a' }).finish()), opts);
    const zero = Item.toObject(Item.decode(Item.encode({ item_id: 'b', rank_score: 0 }).finish()), opts);
    const set = Item.toObject(Item.decode(Item.encode({ item_id: 'c', rank_score: 0.81 }).finish()), opts);
    expect(wireRankScore(unset as RawContextItem)).toBeNull();
    expect(wireRankScore(zero as RawContextItem)).toBe(0);
    expect(wireRankScore(set as RawContextItem)).toBe(0.81);
  });
});

describe('the deprecated camelCase methods keep a ranked result', () => {
  it('returns items and excerpts, and nothing extra for a by-type result', async () => {
    const { normalizeContextResponse } = await import('../context/normalize.js');
    const { applyResponseShape } = await import('../context/ranked.js');
    const wire = { items: [
      { id: 'm1', content: 'a fact', category: 'facts', rank: 1, confidence: 0.9, metadata: { rank_score: 0.9 } },
      { id: 'c1', content: 'a chunk quote', category: 'excerpts', rank: 2, confidence: 0.8, metadata: { rank_score: 0.8 } },
      { id: 'p1', content: 'a pref', category: 'preferences', rank: 3, confidence: 0.7, metadata: { rank_score: 0.7 } },
    ] };
    const ranked = normalizeContextResponse({ context: applyResponseShape(wire as never, 'ranked') } as never);
    expect(ranked.items?.map((i) => i['id'])).toEqual(['m1', 'c1', 'p1']);
    expect(ranked.excerpts?.map((i) => i['id'])).toEqual(['c1']);
    const byType = normalizeContextResponse({ context: { facts: [{ id: 'f', content: 'x' }] } } as never);
    expect(byType).not.toHaveProperty('items');
    expect(byType).not.toHaveProperty('excerpts');
  });
});
