/**
 * One user's prefetched context must never answer another user's fetch.
 *
 * One SDK process serves many users: the stream pushes every user's bundles
 * into one in-memory cache, and the user filter in the lookup is the only thing
 * keeping them apart. Each test here is a path that served another user's
 * items before 21 Oct 2026 (Python: synap/tests/sdk/test_anticipation_user_scope.py):
 *
 * - a customer or client fetch with no search query served the newest bundle
 *   of any user;
 * - a conversation fetch with no user and no query did the same;
 * - a bundle stamped with a conversation but no user (ANY_SCOPE) answered
 *   user- and client-scope fetches from anyone.
 */
import { describe, it, expect } from 'vitest';
import { AnticipationCache, type ContextBundle } from '../context/anticipation-cache.js';
import { tryServeFromCache } from '../context/anticipated.js';

const QUERY = ['pharmacy refill Lisinopril'];

function bundle(over: Partial<ContextBundle> = {}): ContextBundle {
  return {
    bundleId: 'a1',
    entityId: 'alice',
    itemsByType: { facts: [{ item_id: 'f1', content: "Alice's pharmacy refill is Lisinopril 10mg at Walgreens" }] },
    ...over,
  };
}

const served = async (c: AnticipationCache, scope: 'conversation' | 'user' | 'customer' | 'client', p: object) =>
  (await tryServeFromCache(c, scope, { clientId: 'cli', ...p }, 10)).hit !== null;

describe('anticipation cache: user scope', () => {
  it('a customer or client fetch with no query gets no user bundle', async () => {
    const c = new AnticipationCache();
    c.store(bundle());
    expect(await served(c, 'customer', { customerId: 'acme' })).toBe(false);
    expect(await served(c, 'client', {})).toBe(false);
  });

  it('a conversation fetch with no user gets no user bundle, with or without a query', async () => {
    const c = new AnticipationCache();
    c.store(bundle());
    expect(await served(c, 'conversation', { conversationId: 'conv-bob' })).toBe(false);
    expect(await served(c, 'conversation', { conversationId: 'conv-bob', searchQuery: QUERY })).toBe(false);
  });

  it('the owner still gets it with no query', async () => {
    const c = new AnticipationCache();
    c.store(bundle());
    expect(await served(c, 'user', { entityId: 'alice' })).toBe(true);
  });

  it('a customer bundle serves a customer fetch with no query', async () => {
    const c = new AnticipationCache();
    c.store(bundle({ entityId: 'acme' }));
    expect(await served(c, 'customer', { customerId: 'acme' })).toBe(true);
    expect(await served(c, 'customer', { customerId: 'globex' })).toBe(false);
  });

  it('a bundle with a conversation and no user stays in its conversation', async () => {
    const c = new AnticipationCache();
    c.store(bundle({ entityId: null, conversationId: 'conv-alice' }));
    expect(await served(c, 'user', { entityId: 'bob', searchQuery: QUERY })).toBe(false);
    expect(await served(c, 'client', { searchQuery: QUERY })).toBe(false);
    expect(await served(c, 'client', {})).toBe(false);
    expect(await served(c, 'conversation', { entityId: 'bob', conversationId: 'conv-bob', searchQuery: QUERY })).toBe(false);
    expect(await served(c, 'conversation', { conversationId: 'conv-alice', searchQuery: QUERY })).toBe(true);
  });

  it('an unowned conversation summary is not a user summary', async () => {
    const c = new AnticipationCache();
    c.store(bundle({ entityId: null, conversationId: 'conv-alice', bundleType: 'user_summary' }));
    expect(c.lookupUserSummary('bob')).toBeNull();
  });
});
