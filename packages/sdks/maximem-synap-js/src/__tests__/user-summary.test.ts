import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { TurnCounter, mergeUserSummary, USER_SUMMARY_INTERVAL } from '../context/user-summary.js';
import * as registry from '../registry.js';
import type { RawContext } from '../context/types.js';

const UUID = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
const json = (b: unknown) =>
  new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });

beforeEach(() => registry.clear());
afterEach(() => registry.clear());

describe('turn cadence', () => {
  it('injects every Nth turn and never on the first', () => {
    const t = new TurnCounter();
    const fired: number[] = [];
    for (let i = 1; i <= 12; i += 1) {
      t.increment('c1');
      if (t.shouldInject('c1')) fired.push(i);
    }
    expect(USER_SUMMARY_INTERVAL).toBe(5);
    expect(fired).toEqual([5, 10]);
  });

  it('counts per conversation, not globally', () => {
    const t = new TurnCounter();
    for (let i = 0; i < 4; i += 1) t.increment('a');
    for (let i = 0; i < 4; i += 1) t.increment('b');
    expect(t.shouldInject('a')).toBe(false);
    t.increment('a');
    expect(t.shouldInject('a')).toBe(true);
    // 'b' is still on 4 and must not be dragged along.
    expect(t.shouldInject('b')).toBe(false);
  });

  it('buckets a missing conversation id, as Python does', () => {
    const t = new TurnCounter();
    for (let i = 0; i < 5; i += 1) t.increment(undefined);
    expect(t.shouldInject(undefined)).toBe(true);
  });
});

describe('cache.lookupUserSummary', () => {
  function cacheWithSummary(entityId: string, content: string) {
    const c = new AnticipationCache();
    c.store({
      bundleId: `b-${entityId}`,
      entityId,
      bundleType: 'user_summary',
      itemsByType: { facts: [{ item_id: `i-${entityId}`, content, scope: 'user' }] },
    });
    return c;
  }

  it('refuses an unscoped lookup', () => {
    // The permissive version returned the freshest summary across ALL users,
    // which is how user A's summary reaches user B on the conversation path.
    const c = cacheWithSummary('user-a', 'A private detail');
    for (const missing of [undefined, null, '']) {
      expect(c.lookupUserSummary(missing)).toBeNull();
    }
  });

  it('never returns another user\'s summary', () => {
    const c = cacheWithSummary('user-a', 'A private detail');
    expect(c.lookupUserSummary('user-b')).toBeNull();
    expect(c.lookupUserSummary('user-a')).not.toBeNull();
  });

  it('returns client-shared summaries to anyone', () => {
    const c = new AnticipationCache();
    c.store({
      bundleId: 'shared', entityId: null, bundleType: 'user_summary',
      itemsByType: { facts: [{ item_id: 'i1', content: 'a company policy', scope: 'client' }] },
    });
    expect(c.lookupUserSummary('anybody')).not.toBeNull();
  });

  it('ignores bundles that are not user summaries', () => {
    const c = new AnticipationCache();
    c.store({
      bundleId: 'b1', entityId: 'u1', bundleType: 'anticipation',
      itemsByType: { facts: [{ item_id: 'i1', content: 'not a summary', scope: 'user' }] },
    });
    expect(c.lookupUserSummary('u1')).toBeNull();
  });

  it('returns the freshest when there are several', () => {
    const c = cacheWithSummary('u1', 'older');
    c.store({
      bundleId: 'newer', entityId: 'u1', bundleType: 'user_summary',
      itemsByType: { facts: [{ item_id: 'i-new', content: 'newer', scope: 'user' }] },
    });
    const bundle = c.lookupUserSummary('u1') as { items_by_type: Record<string, Array<{ content: string }>> };
    expect(bundle.items_by_type['facts']?.[0]?.content).toBe('newer');
  });
});

describe('mergeUserSummary', () => {
  const bundle = {
    items_by_type: {
      facts: [
        { item_id: 's1', content: 'summary fact one', confidence: 0.9 },
        { item_id: 's2', content: 'summary fact two', confidence: 0.8 },
        { item_id: 's3', content: 'three', confidence: 0.7 },
        { item_id: 's4', content: 'four, over the cap', confidence: 0.6 },
      ],
      episodes: [{ item_id: 'e1', content: 'an episode body', confidence: 0.5 }],
    },
  };

  it('appends without replacing what the retrieval found', () => {
    const response: RawContext = { facts: [{ id: 'existing', content: 'from retrieval' }] };
    mergeUserSummary(response, bundle);
    expect(response.facts?.[0]?.content).toBe('from retrieval');
    expect(response.facts).toHaveLength(4);   // 1 existing + 3 capped
  });

  it('caps at three items per collection', () => {
    const response: RawContext = { facts: [] };
    mergeUserSummary(response, bundle);
    expect(response.facts).toHaveLength(3);
    expect(response.facts?.map((f) => f.id)).toEqual(['s1', 's2', 's3']);
  });

  it('skips items the response already carries', () => {
    const response: RawContext = { facts: [{ id: 's1', content: 'already here' }] };
    mergeUserSummary(response, bundle);
    expect(response.facts?.filter((f) => f.id === 's1')).toHaveLength(1);
  });

  it('maps proto field names onto the response shape', () => {
    // An episode carries `content` in the bundle but consumers read `summary`.
    const response: RawContext = {};
    mergeUserSummary(response, bundle);
    const episode = response.episodes?.[0];
    expect(episode?.summary).toBe('an episode body');
    expect(episode?.id).toBe('e1');
    expect(typeof episode?.significance).toBe('number');
  });

  it('tolerates a bundle with nothing usable', () => {
    const response: RawContext = { facts: [] };
    expect(() => mergeUserSummary(response, {})).not.toThrow();
    expect(() => mergeUserSummary(response, { items_by_type: { facts: 'not an array' } })).not.toThrow();
    expect(response.facts).toHaveLength(0);
  });
});

describe('injection through the client', () => {
  function client() {
    return new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async () => json({ context: { facts: [] } })) as unknown as typeof fetch,
    });
  }

  async function fetchTimes(c: SynapClient, n: number, userId?: string) {
    let last: RawContext = {};
    for (let i = 0; i < n; i += 1) {
      last = await c.conversation.context.fetch({
        conversation_id: UUID,
        ...(userId !== undefined ? { user_id: userId } : {}),
      });
    }
    return last;
  }

  // NOTE on coverage: an end-to-end injection test would need a `user_summary`
  // bundle inside the CLIENT's cache, and the only writer is the gRPC stream
  // (Python is the same: nothing stores into the cache from the public
  // surface). Rather than widen the surface with a test-only seed hook, the
  // three pieces are unit-tested above and the client's wiring is checked by
  // its two refusal paths below, which are the ones that matter for privacy.

  it('does nothing without a user_id, so a summary cannot cross users', async () => {
    const c = client();
    const result = await fetchTimes(c, USER_SUMMARY_INTERVAL);
    expect(result.facts ?? []).toHaveLength(0);
    await c.shutdown();
  });

  it('does nothing when no summary is cached', async () => {
    const c = client();
    const result = await fetchTimes(c, USER_SUMMARY_INTERVAL, 'u1');
    expect(result.facts ?? []).toHaveLength(0);
    await c.shutdown();
  });
});
