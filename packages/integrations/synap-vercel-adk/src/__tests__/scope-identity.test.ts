/**
 * A path and the ids fixed beside it must name the same identity. The server
 * lets the path's `user` and `customer` win over the flat ids in the same body,
 * so a disagreement would read and write another identity's memory. Refused at
 * construction (middleware), and again where the public helpers are called
 * directly. Every assertion is on what was (not) sent.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AnticipationCache } from '../context/anticipation-cache.js';
import { fetchContext } from '../context/http-fetcher.js';
import { createSynapMiddleware } from '../middleware.js';
import { recordTurn } from '../memory/turns.js';
import { SynapScopeError } from '../index.js';
import { fingerprint } from '../scope-guard.js';
import type { CachedBundle, Credentials } from '../types.js';
import { rungKey } from '../util/scope.js';

const PATH = { customer: 'acme', user: 'alice' };
const creds: Credentials = { api_key: 'sk', client_id: 'cli', instance_id: 'inst' };

function stubFetch() {
  const f = vi.fn().mockResolvedValue({
    ok: true, status: 200, statusText: 'OK', text: async () => '{}', json: async () => ({}),
  });
  vi.stubGlobal('fetch', f);
  return f;
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('createSynapMiddleware', () => {
  const make = (o: object) =>
    createSynapMiddleware({ credentials: creds, anticipationCache: new AnticipationCache(), grpcClient: null, ...o });

  it('refuses another userId and another customerId, before any request', () => {
    const f = stubFetch();
    expect(() => make({ userId: 'bob', customerId: 'acme', scopePath: PATH })).toThrow(SynapScopeError);
    expect(() => make({ userId: 'alice', customerId: 'globex', scopePath: PATH })).toThrow(SynapScopeError);
    expect(f).not.toHaveBeenCalled();
  });

  it('the refusal names no raw id', () => {
    let err: Error | undefined;
    try { make({ userId: 'bob-secret', customerId: 'acme', scopePath: PATH }); } catch (e) { err = e as Error; }
    expect(err).toBeInstanceOf(SynapScopeError);
    expect(err!.message).toContain(fingerprint('bob-secret'));
    for (const raw of ['bob-secret', 'alice', 'acme']) expect(err!.message).not.toContain(raw);
  });

  it('matching ids pass; an empty customerId passes', () => {
    expect(() => make({ userId: 'alice', customerId: 'acme', scopePath: PATH })).not.toThrow();
    expect(() => make({ userId: 'alice', scopePath: PATH })).not.toThrow();
  });

  it('B2C: customer == user passes with no customerId or an equal one', () => {
    const b2c = { customer: 'alice', user: 'alice' };
    expect(() => make({ userId: 'alice', scopePath: b2c })).not.toThrow();
    expect(() => make({ userId: 'alice', customerId: 'alice', scopePath: b2c })).not.toThrow();
    expect(() => make({ userId: 'alice', customerId: 'rival', scopePath: b2c })).toThrow(SynapScopeError);
  });

  it('a path with no user / customer key accepts any ids; no path checks nothing', () => {
    expect(() => make({ userId: 'bob', customerId: 'globex', scopePath: { team: 'eng' } })).not.toThrow();
    expect(() => make({ userId: 'bob', customerId: 'globex' })).not.toThrow();
    expect(() => make({ userId: 'bob', customerId: 'globex', scopePath: {} })).not.toThrow();
  });
});

describe('the public helpers called directly', () => {
  it('fetchContext throws and sends nothing', async () => {
    const f = stubFetch();
    await expect(fetchContext({
      credentials: creds, searchQuery: ['q'],
      modelOptions: { userId: 'bob', customerId: 'acme', scopePath: PATH },
    })).rejects.toBeInstanceOf(SynapScopeError);
    expect(f).not.toHaveBeenCalled();
  });

  it('recordTurn never throws: it writes nothing on the stream or over REST and logs fingerprints only', async () => {
    const f = stubFetch();
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const sendConversationEvent = vi.fn(async () => true);
    const getStream = vi.fn(async () => ({ sendConversationEvent }) as never);
    const ok = await recordTurn({
      credentials: creds, role: 'user', content: 'hi', getStream,
      modelOptions: { userId: 'bob-secret', customerId: 'acme', conversationId: 'c1', scopePath: PATH },
    });
    expect(ok).toBe(false);
    expect(sendConversationEvent).not.toHaveBeenCalled();
    expect(f).not.toHaveBeenCalled();
    expect(err).toHaveBeenCalledOnce();
    const logged = String(err.mock.calls[0]![0]);
    expect(logged).toContain(fingerprint('bob-secret'));
    for (const raw of ['bob-secret', 'alice', 'acme']) expect(logged).not.toContain(raw);
  });

  it('recordTurn with matching ids still writes, with the path', async () => {
    const sendConversationEvent = vi.fn(async () => true);
    const ok = await recordTurn({
      credentials: creds, role: 'user', content: 'hi',
      getStream: async () => ({ sendConversationEvent }) as never,
      modelOptions: { userId: 'alice', customerId: 'acme', conversationId: 'c1', scopePath: PATH },
    });
    expect(ok).toBe(true);
    expect(sendConversationEvent).toHaveBeenCalledWith(expect.objectContaining({ scope: PATH }));
  });
});

describe('AnticipationCache.lookupUserSummary is rung-aware', () => {
  const summary = (over: Partial<CachedBundle> = {}): CachedBundle => ({
    bundleId: 's', itemsByType: {}, conversationContext: null, bundleType: 'user_summary',
    userId: 'alice', customerId: 'acme', conversationId: '', searchKeywords: [], searchQueries: [],
    sourceBundleIds: [], totalTokens: 1, bundleConfidence: 1, originPatternId: '',
    storedAt: Date.now(), ttl: 300_000, ...over,
  });
  const RUNG = rungKey(PATH)!;

  it('serves a stamped summary only to a request naming that level', () => {
    const cache = new AnticipationCache();
    cache.store(summary({ scopeRung: RUNG }));
    expect(cache.lookupUserSummary('alice', PATH)).not.toBeNull();
    expect(cache.lookupUserSummary('alice', { customer: 'acme', team: 'x', user: 'alice' })).toBeNull();
    expect(cache.lookupUserSummary('alice')).toBeNull();
  });

  it('does not serve an unstamped summary to a request that names a level', () => {
    const cache = new AnticipationCache();
    cache.store(summary());
    expect(cache.lookupUserSummary('alice', PATH)).toBeNull();
  });

  it('without a path on either side it serves exactly as before', () => {
    const cache = new AnticipationCache();
    cache.store(summary());
    expect(cache.lookupUserSummary('alice')).not.toBeNull();
    expect(cache.lookupUserSummary('alice', {})).not.toBeNull();
  });
});
