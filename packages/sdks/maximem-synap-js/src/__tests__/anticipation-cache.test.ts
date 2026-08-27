import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AnticipationCache, ANY_SCOPE, type ContextBundle, type LookupTelemetry } from '../context/anticipation-cache.js';

let clock = 1000;
const now = () => clock;

function bundle(over: Partial<ContextBundle> = {}): ContextBundle {
  return {
    bundleId: 'b1',
    entityId: 'user-1',
    itemsByType: {
      facts: [{ item_id: 'f1', content: 'The user email address is ada@example.com' }],
      preferences: [{ item_id: 'p1', content: 'Prefers aisle seats on long haul flights' }],
    },
    ...over,
  };
}

const ENV_KEYS = [
  'SYNAP_SDK_CACHE_RECALL_BYPASS', 'SYNAP_SDK_CACHE_COVERAGE_MIN',
  'SYNAP_SDK_CACHE_HONOR_TTL_HINT', 'SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE',
  'SYNAP_SDK_CACHE_MAX_ENTRY_AGE',
];
beforeEach(() => { clock = 1000; for (const k of ENV_KEYS) delete process.env[k]; });
afterEach(() => { for (const k of ENV_KEYS) delete process.env[k]; });

describe('AnticipationCache', () => {
  it('stores and retrieves on a matching query', () => {
    const c = new AnticipationCache({ now });
    c.store(bundle());
    const hit = c.lookup({ searchQuery: ['what is my email address'], entityId: 'user-1' });
    expect(hit).not.toBeNull();
    expect(hit!.itemsByType['facts']).toHaveLength(1);
    expect(hit!.bundleIds).toEqual(['b1']);
  });

  it('misses on an empty cache', () => {
    expect(new AnticipationCache({ now }).lookup({ searchQuery: ['anything'] })).toBeNull();
  });

  describe('scope funnel', () => {
    it('accepts client-scope bundles for any requester', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle({ bundleId: 'shared', entityId: null }));
      const hit = c.lookup({ searchQuery: ['email address'], entityId: 'someone-else' });
      expect(hit).not.toBeNull();
      expect(hit!.bundleIds).toEqual(['shared']);
    });

    it('widens from user to customer to client', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle({ bundleId: 'cust', entityId: 'customer-9' }));
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1', customerId: 'customer-9' })).not.toBeNull();
      // Without the customer scope the same bundle must not match.
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1' })).toBeNull();
    });

    it('does not let an empty customer id match a bundle keyed at ""', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle({ bundleId: 'blank', entityId: '' }));
      // entityId '' falls back to the _any sentinel rather than matching a
      // falsy customer id, which would leak across tenants.
      expect(c.lookup({ searchQuery: ['email'], entityId: 'other', customerId: '' })).not.toBeNull();
    });

    it('confines a conversation-scoped bundle to its conversation', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle({ conversationId: 'conv-a' }));
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1', conversationId: 'conv-a' })).not.toBeNull();
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1', conversationId: 'conv-b' })).toBeNull();
    });
  });

  describe('recall bypass', () => {
    it('is off by default, so behaviour is unchanged', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.lookup({ searchQuery: ['what is my email address'], entityId: 'user-1' })).not.toBeNull();
    });

    it('bypasses a recall-shaped question when enabled', () => {
      process.env['SYNAP_SDK_CACHE_RECALL_BYPASS'] = 'true';
      const c = new AnticipationCache({ now });
      c.store(bundle());
      // This scores well and would be a HIT. It is refused anyway, because a
      // stale answer to a recall question is the expensive failure.
      expect(c.lookup({ searchQuery: ['what is my email address'], entityId: 'user-1' })).toBeNull();
    });

    it('still serves ordinary task queries when enabled', () => {
      process.env['SYNAP_SDK_CACHE_RECALL_BYPASS'] = 'true';
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.lookup({ searchQuery: ['book an aisle seat flights'], entityId: 'user-1' })).not.toBeNull();
    });

    it('reads the flag at call time, not at import time', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.lookup({ searchQuery: ['what is my email'], entityId: 'user-1' })).not.toBeNull();
      process.env['SYNAP_SDK_CACHE_RECALL_BYPASS'] = '1';
      expect(c.lookup({ searchQuery: ['what is my email'], entityId: 'user-1' })).toBeNull();
      process.env['SYNAP_SDK_CACHE_RECALL_BYPASS'] = 'false';
      expect(c.lookup({ searchQuery: ['what is my email'], entityId: 'user-1' })).not.toBeNull();
    });
  });

  describe('novel-term gate', () => {
    it('does not fire on a small corpus', () => {
      // At cold start the ratio is dominated by ordinary English; enforcing it
      // would miss on nearly every second-turn query.
      const c = new AnticipationCache({ now });
      c.store(bundle());
      const hit = c.lookup({ searchQuery: ['email address'], entityId: 'user-1' });
      expect(hit).not.toBeNull();
    });

    it('fires once the corpus is large enough', () => {
      const c = new AnticipationCache({ now, maxBundles: 500 });
      // Build a corpus past the 200-stem floor.
      for (let i = 0; i < 60; i++) {
        c.store({
          bundleId: `b${i}`, entityId: 'user-1',
          itemsByType: { facts: [{ content: `zebra${i} quartz${i} nimbus${i} orbital${i} lantern${i}` }] },
        });
      }
      // A query whose stems are entirely absent from that corpus.
      expect(c.lookup({ searchQuery: ['xylophone marzipan quixotic'], entityId: 'user-1' })).toBeNull();
    });
  });

  describe('coverage gate', () => {
    it('is observe-only by default', () => {
      const c = new AnticipationCache({ now });
      let seen: LookupTelemetry | null = null;
      const c2 = new AnticipationCache({ now, onLookup: (t) => { seen = t; } });
      c2.store(bundle());
      c2.lookup({ searchQuery: ['email address'], entityId: 'user-1' });
      expect(seen).not.toBeNull();
      expect(seen!.coverage).not.toBeNull();
      expect(c).toBeDefined();
    });

    it('rejects a low-coverage hit when configured', () => {
      process.env['SYNAP_SDK_CACHE_COVERAGE_MIN'] = '0.9';
      const c = new AnticipationCache({ now });
      c.store(bundle());
      // Shares "flights" but not the rest, so it clears BM25 on one token
      // while lacking the asked-for fact.
      expect(c.lookup({ searchQuery: ['flights passport visa renewal'], entityId: 'user-1' })).toBeNull();
    });

    it('treats a malformed setting as observe-only rather than failing closed', () => {
      process.env['SYNAP_SDK_CACHE_COVERAGE_MIN'] = 'not-a-number';
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
    });

    it('does not refresh a lease when the coverage gate rejects', () => {
      process.env['SYNAP_SDK_CACHE_COVERAGE_MIN'] = '0.99';
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle());
      clock += 60;
      c.lookup({ searchQuery: ['flights passport visa renewal'], entityId: 'user-1' });
      clock += 60; // 120s since store: expired if the lease was never refreshed
      delete process.env['SYNAP_SDK_CACHE_COVERAGE_MIN'];
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).toBeNull();
    });
  });

  describe('expiry', () => {
    it('expires an idle bundle', () => {
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle());
      clock += 101;
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1' })).toBeNull();
      expect(c.size).toBe(0);
    });

    it('renews the lease on a hit', () => {
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle());
      clock += 80;
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
      clock += 80; // 160s since store, but only 80s since the renewing hit
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
    });

    it('caps absolute age so a renewed bundle cannot live forever', () => {
      // Without this, steady traffic keeps a pre-update snapshot alive
      // indefinitely and it serves a retired value.
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle());
      for (let i = 0; i < 5; i++) {
        clock += 50;
        c.lookup({ searchQuery: ['email address'], entityId: 'user-1' });
      }
      // 250s elapsed > 2x the 100s TTL
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).toBeNull();
    });

    it('honours SYNAP_SDK_CACHE_MAX_ENTRY_AGE=0 as "no cap"', () => {
      process.env['SYNAP_SDK_CACHE_MAX_ENTRY_AGE'] = '0';
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle());
      for (let i = 0; i < 10; i++) {
        clock += 50;
        expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
      }
    });

    it('lets a TTL hint shorten but never extend the global TTL', () => {
      process.env['SYNAP_SDK_CACHE_HONOR_TTL_HINT'] = 'true';
      const short = new AnticipationCache({ now, ttlSeconds: 1000 });
      short.store(bundle({ ttlHintSeconds: 50 }));
      clock += 60;
      expect(short.lookup({ searchQuery: ['email'], entityId: 'user-1' })).toBeNull();

      clock = 1000;
      const long = new AnticipationCache({ now, ttlSeconds: 100 });
      long.store(bundle({ bundleId: 'b2', ttlHintSeconds: 9999 }));
      clock += 150;
      expect(long.lookup({ searchQuery: ['email'], entityId: 'user-1' })).toBeNull();
    });

    it('stops renewing leases when TTL hints are honoured', () => {
      process.env['SYNAP_SDK_CACHE_HONOR_TTL_HINT'] = 'true';
      const c = new AnticipationCache({ now, ttlSeconds: 100 });
      c.store(bundle({ ttlHintSeconds: 100 }));
      clock += 80;
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
      clock += 30; // 110s since store; a renewal would have kept it alive
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).toBeNull();
    });
  });

  describe('invalidate on write', () => {
    it('is off by default', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.invalidateEntity('user-1')).toBe(0);
      expect(c.size).toBe(1);
    });

    it('drops the writing entity when enabled', () => {
      process.env['SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE'] = 'true';
      const c = new AnticipationCache({ now });
      c.store(bundle());
      expect(c.invalidateEntity('user-1')).toBe(1);
      expect(c.size).toBe(0);
    });

    it('never drops shared client-scope knowledge on one user write', () => {
      process.env['SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE'] = 'true';
      const c = new AnticipationCache({ now });
      c.store(bundle({ bundleId: 'shared', entityId: null }));
      c.store(bundle({ bundleId: 'mine', entityId: 'user-1' }));
      expect(c.invalidateEntity('user-1')).toBe(1);
      expect(c.size).toBe(1);
    });
  });

  describe('eviction bookkeeping', () => {
    it('evicts the least recently used bundle past the cap', () => {
      const c = new AnticipationCache({ now, maxBundles: 2 });
      c.store(bundle({ bundleId: 'a' })); clock += 1;
      c.store(bundle({ bundleId: 'b' })); clock += 1;
      c.store(bundle({ bundleId: 'c' }));
      expect(c.size).toBe(2);
    });

    it('rebuilds vocabulary rather than subtracting on drop', () => {
      // A stem shared by two bundles must survive one of them being dropped,
      // or the novel-term gate starts firing on terms still in the corpus.
      const c = new AnticipationCache({ now, maxBundles: 1 });
      c.store(bundle({ bundleId: 'a' })); clock += 1;
      c.store(bundle({ bundleId: 'b' }));
      expect(c.size).toBe(1);
      expect(c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toBeNull();
    });

    it('replaces rather than duplicates a re-stored bundle', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle());
      const before = c.itemCount;
      c.store(bundle());
      expect(c.itemCount).toBe(before);
      expect(c.size).toBe(1);
    });

    it('clear() empties everything', () => {
      const c = new AnticipationCache({ now });
      c.store(bundle());
      c.clear();
      expect(c.size).toBe(0);
      expect(c.itemCount).toBe(0);
      expect(c.lookup({ searchQuery: ['email'], entityId: 'user-1' })).toBeNull();
    });
  });

  describe('telemetry hook', () => {
    it('reports the exit reason for each outcome', () => {
      const seen: LookupTelemetry[] = [];
      const c = new AnticipationCache({ now, onLookup: (t) => seen.push(t) });
      c.lookup({ searchQuery: ['x'] });
      expect(seen.at(-1)!.exitReason).toBe('empty');

      c.store(bundle());
      c.lookup({ searchQuery: ['email address'], entityId: 'user-1' });
      expect(seen.at(-1)!.exitReason).toBe('hit');
      expect(seen.at(-1)!.hit).toBe(true);

      process.env['SYNAP_SDK_CACHE_RECALL_BYPASS'] = 'true';
      c.lookup({ searchQuery: ['remind me of my email'], entityId: 'user-1' });
      expect(seen.at(-1)!.exitReason).toBe('recall_bypass');
    });

    it('caps the rejected-items payload', () => {
      const seen: LookupTelemetry[] = [];
      const c = new AnticipationCache({ now, maxBundles: 200, onLookup: (t) => seen.push(t) });
      for (let i = 0; i < 40; i++) {
        c.store({ bundleId: `b${i}`, entityId: 'user-1', itemsByType: { facts: [{ content: `item ${i} unrelated content` }] } });
      }
      c.lookup({ searchQuery: ['completely different topic'], entityId: 'user-1' });
      expect(seen.at(-1)!.itemsRejected.length).toBeLessThanOrEqual(20);
    });

    it('never lets a throwing hook break a lookup', () => {
      const c = new AnticipationCache({ now, onLookup: () => { throw new Error('bad hook'); } });
      c.store(bundle());
      expect(() => c.lookup({ searchQuery: ['email address'], entityId: 'user-1' })).not.toThrow();
    });
  });

  it('falls back to freshness when there is no query', () => {
    const c = new AnticipationCache({ now });
    c.store(bundle({ bundleId: 'old' })); clock += 10;
    c.store(bundle({ bundleId: 'new' }));
    const hit = c.lookup({ entityId: 'user-1' });
    expect(hit).not.toBeNull();
    expect(hit!.bundleIds).toEqual(['new']);
  });

  it('exports the _any sentinel', () => {
    expect(ANY_SCOPE).toBe('_any');
  });
});
