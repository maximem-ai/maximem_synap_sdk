/**
 * Tests for src/context/anticipation-cache.ts
 *
 * Covers:
 *   - store / lookup: scope matching, scope mismatch → null
 *   - BM25 scoring: cache hit on relevant query, miss on unrelated query
 *   - Novel-term gate: >45% unknown terms → miss
 *   - No-query path: most-recent bundle returned
 *   - BM25 threshold: score below 1.5 → miss
 *   - TTL expiry: expired entries evicted before lookup
 *   - LRU cap: at 100 entries, oldest entry evicted on insert
 *   - lookupUserSummary: filters by bundleType === 'user_summary'
 *   - invalidateConversation: removes all entries for that conversationId
 *   - size(): reflects current entry count
 *   - bundleToContext: all item types mapped correctly
 *   - conversationContext mapping from bundle
 */

import { describe, it, expect, vi } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import type { CachedBundle, RawContextItem } from '../types.js';

// ─── Helper ─────────────────────────────────────────────────────────────────

function makeRawItem(content: string, overrides: Partial<RawContextItem> = {}): RawContextItem {
  return {
    item_id: 'item-1',
    content,
    context_type: 'fact',
    source: 'test',
    confidence: 0.9,
    similarity_score: 0.9,
    relevance_score: 0.9,
    scope: 'user',
    entity_id: '',
    event_date: '',
    valid_until: '',
    temporal_category: '',
    temporal_confidence: 0,
    ...overrides,
  };
}

function makeBundle(overrides: Partial<CachedBundle> = {}): CachedBundle {
  return {
    bundleId: 'bundle-1',
    itemsByType: {
      facts: [makeRawItem('User loves distributed systems and sharding')],
    },
    conversationContext: null,
    bundleType: 'anticipation',
    userId: 'u1',
    customerId: 'c1',
    conversationId: 'conv-1',
    searchKeywords: ['distributed', 'systems', 'sharding'],
    searchQueries: ['distributed systems sharding'],
    sourceBundleIds: ['bundle-1'],
    totalTokens: 100,
    bundleConfidence: 0.9,
    originPatternId: '',
    storedAt: Date.now(),
    ttl: 300_000,
    ...overrides,
  };
}

// ─── store / lookup ──────────────────────────────────────────────────────────

describe('AnticipationCache.store + lookup', () => {
  it('returns null on empty cache', () => {
    const cache = new AnticipationCache();
    expect(cache.lookup({ userId: 'u1' })).toBeNull();
  });

  it('returns a hit when scope and query match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const hit = cache.lookup({
      userId: 'u1',
      customerId: 'c1',
      conversationId: 'conv-1',
      searchQuery: ['distributed systems sharding'],
    });
    expect(hit).not.toBeNull();
    expect(hit!.facts).toHaveLength(1);
    expect(hit!.facts[0].content).toContain('distributed systems');
    expect(hit!.source).toBe('anticipation');
  });

  it('returns null when userId does not match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    expect(cache.lookup({ userId: 'other', searchQuery: ['distributed'] })).toBeNull();
  });

  it('returns null when customerId does not match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    expect(cache.lookup({ userId: 'u1', customerId: 'other-cust', searchQuery: ['distributed'] })).toBeNull();
  });

  it('returns null when conversationId does not match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    expect(cache.lookup({ userId: 'u1', conversationId: 'conv-other', searchQuery: ['distributed'] })).toBeNull();
  });

  it('matches when only userId is provided (no customerId/conversationId filter)', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    // no customerId / conversationId specified → scopeMatches allows it
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['distributed systems sharding'] });
    expect(hit).not.toBeNull();
  });

  it('returns most-recent bundle when no search query provided', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ userId: 'u2' }));
    const hit = cache.lookup({ userId: 'u2' });
    expect(hit).not.toBeNull();
  });

  it('sets source to "anticipation" on cache hit', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['distributed sharding'] });
    expect(hit?.source).toBe('anticipation');
  });
});

// ─── BM25 scoring and novel-term gate ────────────────────────────────────────

describe('AnticipationCache BM25 + novel-term gate', () => {
  it('returns null when >45% of query terms are unknown (novel-term gate)', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    // All tokens below are completely outside the cache vocabulary
    const hit = cache.lookup({
      userId: 'u1',
      searchQuery: ['completely foreign unrelated xyzzy tokens nobody knows'],
    });
    expect(hit).toBeNull();
  });

  it('returns null when BM25 score is too low for the query', () => {
    const cache = new AnticipationCache();
    // Bundle has only "database" content; query is about something entirely different
    cache.store(makeBundle({
      itemsByType: { facts: [makeRawItem('database indexing basics')] },
      searchKeywords: ['database', 'indexing'],
      searchQueries: ['database indexing'],
    }));
    // "painting art museum" shares no tokens with "database indexing"
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['painting art museum'] });
    expect(hit).toBeNull();
  });

  it('returns hit when query shares vocabulary with bundle', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    // "sharding" is in the bundle keywords — should score above BM25_THRESHOLD
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['sharding distributed'] });
    expect(hit).not.toBeNull();
  });
});

// ─── TTL expiry ──────────────────────────────────────────────────────────────

describe('AnticipationCache TTL expiry', () => {
  it('evicts expired entries before lookup', () => {
    vi.useFakeTimers();
    const cache = new AnticipationCache();
    cache.store(makeBundle({ ttl: 1_000 })); // 1-second TTL
    expect(cache.size()).toBe(1);

    // Advance time past TTL
    vi.advanceTimersByTime(2_000);
    const hit = cache.lookup({ userId: 'u1' });
    expect(hit).toBeNull();
    expect(cache.size()).toBe(0);

    vi.useRealTimers();
  });

  it('keeps entries that have not yet expired', () => {
    vi.useFakeTimers();
    const cache = new AnticipationCache();
    cache.store(makeBundle({ ttl: 60_000 }));

    vi.advanceTimersByTime(30_000); // not yet expired
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['distributed sharding'] });
    expect(hit).not.toBeNull();

    vi.useRealTimers();
  });
});

// ─── LRU capacity cap ────────────────────────────────────────────────────────

describe('AnticipationCache capacity cap (MAX_ENTRIES=100)', () => {
  it('evicts the oldest entry when 101st entry is stored', () => {
    const cache = new AnticipationCache();
    // Store 100 unique entries (different userId to get different keys)
    for (let i = 0; i < 100; i++) {
      cache.store(makeBundle({ userId: `user-${i}`, conversationId: `conv-${i}` }));
    }
    expect(cache.size()).toBe(100);

    // Store one more — should evict the oldest
    cache.store(makeBundle({ userId: 'user-extra', conversationId: 'conv-extra' }));
    expect(cache.size()).toBe(100); // still at cap
  });
});

// ─── lookupUserSummary ────────────────────────────────────────────────────────

describe('AnticipationCache.lookupUserSummary', () => {
  it('returns null on empty cache', () => {
    const cache = new AnticipationCache();
    expect(cache.lookupUserSummary('u1')).toBeNull();
  });

  it('finds user_summary bundles by userId', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ bundleType: 'user_summary', userId: 'u1' }));
    const result = cache.lookupUserSummary('u1');
    expect(result).not.toBeNull();
    expect(result!.bundleType).toBe('user_summary');
  });

  it('does not return anticipation bundles', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ bundleType: 'anticipation', userId: 'u1' }));
    expect(cache.lookupUserSummary('u1')).toBeNull();
  });

  it('does not return summary for a different userId', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ bundleType: 'user_summary', userId: 'u1' }));
    expect(cache.lookupUserSummary('other')).toBeNull();
  });

  it('evicts expired entries before lookup', () => {
    vi.useFakeTimers();
    const cache = new AnticipationCache();
    cache.store(makeBundle({ bundleType: 'user_summary', userId: 'u1', ttl: 100 }));
    vi.advanceTimersByTime(200);
    expect(cache.lookupUserSummary('u1')).toBeNull();
    vi.useRealTimers();
  });
});

// ─── invalidateConversation ──────────────────────────────────────────────────

describe('AnticipationCache.invalidateConversation', () => {
  it('removes all entries matching the conversationId', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ conversationId: 'conv-target', userId: 'u1' }));
    cache.store(makeBundle({ conversationId: 'conv-other', userId: 'u2' }));
    expect(cache.size()).toBe(2);
    cache.invalidateConversation('conv-target');
    expect(cache.size()).toBe(1);
    // The remaining entry should be for conv-other
    const hit = cache.lookup({ userId: 'u2', searchQuery: ['distributed sharding'] });
    expect(hit).not.toBeNull();
  });

  it('no-op when no entries match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ conversationId: 'conv-1' }));
    cache.invalidateConversation('conv-999');
    expect(cache.size()).toBe(1);
  });

  it('clears cache fully when all entries match', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ conversationId: 'conv-1', userId: 'u1' }));
    // Same conversationId but different userId → different bundle key
    // Note: bundle key is `${bundleType}:${userId}:${customerId}:${conversationId}`
    // So u1/conv-1 and u2/conv-1 are different keys but same conversationId
    cache.store(makeBundle({ conversationId: 'conv-1', userId: 'u2' }));
    cache.invalidateConversation('conv-1');
    expect(cache.size()).toBe(0);
  });
});

// ─── size() ──────────────────────────────────────────────────────────────────

describe('AnticipationCache.size', () => {
  it('returns 0 initially', () => {
    expect(new AnticipationCache().size()).toBe(0);
  });

  it('increments after store', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ userId: 'u1' }));
    expect(cache.size()).toBe(1);
    cache.store(makeBundle({ userId: 'u2' }));
    expect(cache.size()).toBe(2);
  });

  it('does not increment for duplicate key (upsert)', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ userId: 'u1' }));
    cache.store(makeBundle({ userId: 'u1' })); // same key
    expect(cache.size()).toBe(1);
  });
});

// ─── bundleToContext: all item types ─────────────────────────────────────────

describe('AnticipationCache bundleToContext item type mapping', () => {
  it('maps preferences, episodes, emotions, temporal_events from bundle', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({
      userId: 'u-all',
      itemsByType: {
        facts: [makeRawItem('A fact', { item_id: 'f1' })],
        preferences: [makeRawItem('Dark mode', { item_id: 'p1', context_type: 'preference' })],
        episodes: [makeRawItem('Past project', { item_id: 'e1', context_type: 'episode' })],
        emotions: [makeRawItem('Feeling good', { item_id: 'em1', context_type: 'emotion' })],
        temporal_events: [makeRawItem('Meeting Friday', { item_id: 't1', context_type: 'temporal_events' })],
      },
      searchKeywords: ['fact', 'dark', 'mode', 'past', 'project', 'feeling', 'meeting'],
      searchQueries: ['fact dark mode past project'],
    }));

    const hit = cache.lookup({ userId: 'u-all', searchQuery: ['fact dark mode past project feeling meeting'] });
    expect(hit).not.toBeNull();
    expect(hit!.facts).toHaveLength(1);
    expect(hit!.preferences).toHaveLength(1);
    expect(hit!.episodes).toHaveLength(1);
    expect(hit!.emotions).toHaveLength(1);
    expect(hit!.temporalEvents).toHaveLength(1);
    expect(hit!.preferences[0].content).toBe('Dark mode');
    expect(hit!.temporalEvents[0].content).toBe('Meeting Friday');
  });

  it('maps conversationContext from bundle', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({
      userId: 'u-cc',
      conversationContext: {
        summary: 'Bundle context summary',
        current_state_json: '{"step":2}',
        key_extractions_json: '{"topic":"deploy"}',
        recent_turns: [{ role: 'user', content: 'turn', timestamp: '2026-01-01T00:00:00Z' }],
        compaction_id: 'comp-99',
        compacted_at: '2026-01-01T00:00:00Z',
        conversation_id: 'conv-cc',
      },
      searchKeywords: ['bundle', 'context', 'summary'],
      searchQueries: ['bundle context summary'],
    }));
    const hit = cache.lookup({ userId: 'u-cc', searchQuery: ['bundle context summary'] });
    expect(hit).not.toBeNull();
    expect(hit!.conversationContext).not.toBeNull();
    expect(hit!.conversationContext!.summary).toBe('Bundle context summary');
    expect(hit!.conversationContext!.compactionId).toBe('comp-99');
    expect(hit!.conversationContext!.conversationId).toBe('conv-cc');
    expect(hit!.conversationContext!.currentState).toEqual({ step: 2 });
    expect(hit!.conversationContext!.recentTurns).toHaveLength(1);
  });

  it('returns null conversationContext when bundle has none', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({ userId: 'u-nocc', conversationContext: null }));
    const hit = cache.lookup({ userId: 'u-nocc', searchQuery: ['distributed sharding'] });
    expect(hit).not.toBeNull();
    expect(hit!.conversationContext).toBeNull();
  });

  it('handles malformed current_state_json gracefully', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({
      userId: 'u-bad-json',
      conversationContext: {
        summary: 'test',
        current_state_json: '{ invalid json >>>',
        key_extractions_json: '{}',
        recent_turns: [],
        compaction_id: '',
        compacted_at: '',
        conversation_id: 'conv-bad',
      },
      searchKeywords: ['test', 'summary'],
      searchQueries: ['test summary'],
    }));
    const hit = cache.lookup({ userId: 'u-bad-json', searchQuery: ['test summary'] });
    expect(hit).not.toBeNull();
    expect(hit!.conversationContext!.currentState).toEqual({});
  });
});

// ─── store overwrites existing key ───────────────────────────────────────────

describe('AnticipationCache store overwrites same key', () => {
  it('updates the bundle content when stored with same key', () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle({
      userId: 'u1',
      itemsByType: { facts: [makeRawItem('Old fact', { item_id: 'old' })] },
      searchKeywords: ['old', 'fact'],
      searchQueries: ['old fact'],
    }));
    cache.store(makeBundle({
      userId: 'u1',
      itemsByType: { facts: [makeRawItem('New fact updated', { item_id: 'new' })] },
      searchKeywords: ['new', 'fact', 'updated'],
      searchQueries: ['new fact updated'],
    }));
    expect(cache.size()).toBe(1);
    const hit = cache.lookup({ userId: 'u1', searchQuery: ['new fact updated'] });
    expect(hit).not.toBeNull();
    expect(hit!.facts[0].id).toBe('new');
  });
});
