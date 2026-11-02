/**
 * scopePath: the level of a deeper scope ladder, on every request this package
 * makes. The wire key is `scope` everywhere (REST bodies and the stream
 * conversation event's map field 14), omitted when absent or empty. Every
 * assertion is on what was actually sent.
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AnticipationCache } from '../context/anticipation-cache.js';
import { fetchContext } from '../context/http-fetcher.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { createSynapMiddleware } from '../middleware.js';
import { recordTurn } from '../memory/turns.js';
import type { CachedBundle, Credentials, SynapModelOptions } from '../types.js';
import { rungKey, rungServes, scopeField } from '../util/scope.js';
import { startGrpcServer, type TestServer } from './helpers/grpc-server.js';

const PATH = { customer: 'acme', team: 'eng', user: 'u1' };
const creds: Credentials = { api_key: 'sk', client_id: 'cli', instance_id: 'inst' };
const hasKey = (o: unknown, k: string) => Object.keys(o as object).includes(k);

function okFetch(body: unknown = {}) {
  return vi.fn().mockResolvedValue({
    ok: true, status: 200, statusText: 'OK',
    text: async () => JSON.stringify(body), json: async () => body,
  });
}
const sentBody = (f: ReturnType<typeof vi.fn>, i = 0) =>
  JSON.parse((f.mock.calls[i]![1] as { body: string }).body) as Record<string, unknown>;

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('scopeField', () => {
  it('is the wire key `scope`, a copy, and nothing when absent or empty', () => {
    const f = scopeField(PATH);
    expect(f).toEqual({ scope: PATH });
    expect(f.scope).not.toBe(PATH);
    expect(scopeField(undefined)).toEqual({});
    expect(scopeField(null)).toEqual({});
    expect(scopeField({})).toEqual({});
  });
});

describe('context fetch bodies (all four scope endpoints)', () => {
  const cases: Array<[string, Partial<SynapModelOptions>, string]> = [
    ['conversation', { conversationId: 'c1', userId: 'u1', customerId: 'acme' }, '/v1/context/conversation/fetch'],
    ['user', { userId: 'u1', customerId: 'acme' }, '/v1/context/user/fetch'],
    ['customer', { customerId: 'acme' }, '/v1/context/customer/fetch'],
    ['client', {}, '/v1/context/client/fetch'],
  ];
  for (const [name, opts, endpoint] of cases) {
    it(`${name}: carries scope when set, and not otherwise`, async () => {
      const f = okFetch();
      vi.stubGlobal('fetch', f);
      await fetchContext({ credentials: creds, modelOptions: { ...opts, scopePath: PATH }, searchQuery: ['q'] });
      expect(String(f.mock.calls[0]![0])).toContain(endpoint);
      expect(sentBody(f)['scope']).toEqual(PATH);

      for (const scopePath of [undefined, {}]) {
        const g = okFetch();
        vi.stubGlobal('fetch', g);
        await fetchContext({ credentials: creds, modelOptions: { ...opts, scopePath }, searchQuery: ['q'] });
        expect(hasKey(sentBody(g), 'scope')).toBe(false);
      }
    });
  }

  it('without a path the user body is exactly what it always was', async () => {
    const f = okFetch();
    vi.stubGlobal('fetch', f);
    await fetchContext({ credentials: creds, modelOptions: { userId: 'u1', customerId: 'acme' }, searchQuery: ['q'] });
    expect(sentBody(f)).toEqual({
      search_query: ['q'], max_results: 10, types: ['all'], mode: 'fast', user_id: 'u1', customer_id: 'acme',
    });
  });
});

describe('recordTurn', () => {
  const base = { credentials: creds, role: 'user' as const, content: 'hello' };

  it('on the stream: the event carries scope', async () => {
    const sendConversationEvent = vi.fn(async () => true);
    await recordTurn({
      ...base, modelOptions: { userId: 'u1', conversationId: 'c1', scopePath: PATH },
      getStream: async () => ({ sendConversationEvent }) as never,
    });
    expect((sendConversationEvent.mock.calls[0] as unknown[])[0]).toMatchObject({ scope: PATH, event_type: 'user_message' });
  });

  it('over REST: the body carries scope', async () => {
    const f = okFetch();
    await recordTurn({
      ...base, modelOptions: { userId: 'u1', conversationId: 'c1', scopePath: PATH },
      getStream: async () => null, fetchImpl: f as never,
    });
    expect(sentBody(f)['scope']).toEqual(PATH);
  });

  it('without a path: neither the event nor the REST body has the key, and the REST body is unchanged', async () => {
    for (const scopePath of [undefined, {}]) {
      const sendConversationEvent = vi.fn(async () => true);
      await recordTurn({
        ...base, modelOptions: { userId: 'u1', conversationId: 'c1', scopePath },
        getStream: async () => ({ sendConversationEvent }) as never,
      });
      expect(hasKey((sendConversationEvent.mock.calls[0] as unknown[])[0], 'scope')).toBe(false);

      const f = okFetch();
      await recordTurn({
        ...base, modelOptions: { userId: 'u1', conversationId: 'c1', scopePath },
        getStream: async () => null, fetchImpl: f as never,
      });
      expect(sentBody(f)).toEqual({
        conversation_id: 'c1', role: 'user', content: 'hello', user_id: 'u1', metadata: { source: 'vercel_ai_sdk' },
      });
    }
  });
});

describe('the real stream client and a real Listen server', () => {
  let server: TestServer | null = null;
  let client: GrpcStreamClient | null = null;
  afterEach(async () => { await client?.disconnect(); await server?.close(); server = null; client = null; });

  const event = (extra: object) => ({
    event_type: 'user_message', conversation_id: 'c1', user_id: 'u1', role: 'user', content: 'hi',
    customer_id: '', session_id: '', metadata: {}, timestamp_ms: 1, search_queries: [], context_types: [], ...extra,
  });
  async function connect() {
    server = await startGrpcServer();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined,
      { host: '127.0.0.1', port: server.port, useTls: false });
    await client.connect();
  }
  const waitFor = async (n: number) => {
    for (let i = 0; i < 100 && server!.events.length < n; i++) await new Promise((r) => setTimeout(r, 10));
  };

  it('the map field `scope` reaches the server; an event without one arrives with an empty map', async () => {
    await connect();
    expect(await client!.sendConversationEvent(event({ scope: PATH }))).toBe(true);
    expect(await client!.sendConversationEvent(event({ content: 'no path' }))).toBe(true);
    await waitFor(2);
    expect(server!.events[0]!['scope']).toEqual(PATH);
    expect(server!.events[1]!['scope']).toEqual({});
  });
});

describe('through the middleware', () => {
  it('both turns carry scope; the tool call event does not', async () => {
    const sent: Array<Record<string, unknown>> = [];
    const grpcClient = {
      isConnected: true,
      async sendConversationEvent(e: Record<string, unknown>) { sent.push(e); return true; },
      async sendContextUsedEvent() {}, async sendContextAssembledEvent() {},
    };
    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: new AnticipationCache(), grpcClient: grpcClient as never,
      userId: 'u1', conversationId: 'c1', injectContext: false, scopePath: PATH,
    });
    await mw.wrapGenerate!({
      doGenerate: async () => ({
        text: 'reply', toolCalls: [{ toolName: 't', toolCallId: 'k', args: {} }],
      }),
      params: { prompt: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }] },
    } as never);
    await new Promise((r) => setTimeout(r, 20));
    const byType = (t: string) => sent.filter((e) => e['event_type'] === t);
    expect(byType('user_message')[0]!['scope']).toEqual(PATH);
    expect(byType('assistant_message')[0]!['scope']).toEqual(PATH);
    expect(byType('tool_call')).toHaveLength(1);
    expect(hasKey(byType('tool_call')[0], 'scope')).toBe(false);
  });

  it('the HTTP fallback fetch carries scope, and the conversation-only history read does not', async () => {
    const f = vi.fn(async (url: string) => ({
      ok: true, status: 200, statusText: 'OK',
      text: async () => '{}', json: async () => ({ items_by_type: {}, formatted_context: '' }),
      url,
    }));
    vi.stubGlobal('fetch', f);
    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: new AnticipationCache(), grpcClient: null,
      userId: 'u1', conversationId: 'c1', writeMemory: false, scopePath: PATH,
    });
    await mw.transformParams!({
      params: { prompt: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }] }, type: 'generate',
    } as never);
    const calls = f.mock.calls as unknown as Array<[string, { method: string; body?: string }]>;
    const fetchCall = calls.find((c) => c[0].includes('/v1/context/'))!;
    expect(JSON.parse(fetchCall[1].body!)['scope']).toEqual(PATH);
    const history = calls.find((c) => c[0].includes('/context-for-prompt'))!;
    expect(history[1].method).toBe('GET');
    expect(history[0]).not.toMatch(/scope/);
    expect(history[1].body).toBeUndefined();
  });

  it('without a path nothing the middleware sends carries one', async () => {
    const sent: Array<Record<string, unknown>> = [];
    const grpcClient = {
      isConnected: true,
      async sendConversationEvent(e: Record<string, unknown>) { sent.push(e); return true; },
      async sendContextUsedEvent() {}, async sendContextAssembledEvent() {},
    };
    const f = okFetch({ items_by_type: {} });
    vi.stubGlobal('fetch', f);
    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: new AnticipationCache(), grpcClient: grpcClient as never,
      userId: 'u1', conversationId: 'c1',
    });
    const params = { prompt: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }] };
    await mw.transformParams!({ params, type: 'generate' } as never);
    await mw.wrapGenerate!({ doGenerate: async () => ({ text: 'reply' }), params } as never);
    await new Promise((r) => setTimeout(r, 20));
    expect(sent.length).toBeGreaterThan(0);
    for (const e of sent) expect(hasKey(e, 'scope')).toBe(false);
    for (const c of f.mock.calls) {
      const body = (c[1] as { body?: string }).body;
      if (body) expect(hasKey(JSON.parse(body), 'scope')).toBe(false);
    }
  });
});

// ── the anticipation cache keeps a bundle for one level away from another ────

function bundle(over: Partial<CachedBundle> = {}): CachedBundle {
  return {
    bundleId: 'b', itemsByType: { facts: [{
      item_id: 'i', content: 'loves distributed systems sharding', context_type: 'facts', source: 's',
      confidence: 1, similarity_score: 1, relevance_score: 1, scope: 'user', entity_id: '', event_date: '',
      valid_until: '', temporal_category: '', temporal_confidence: 0,
    }] },
    conversationContext: null, bundleType: 'anticipation', userId: 'u1', customerId: 'c1', conversationId: '',
    searchKeywords: ['distributed', 'systems', 'sharding'], searchQueries: ['distributed systems sharding'],
    sourceBundleIds: ['b'], totalTokens: 1, bundleConfidence: 1, originPatternId: '',
    storedAt: Date.now(), ttl: 300_000, ...over,
  };
}
const Q = { userId: 'u1', customerId: 'c1', searchQuery: ['distributed systems sharding'] };

describe('AnticipationCache and the level', () => {
  const RUNG = rungKey(PATH)!;
  const OTHER = { customer: 'acme', team: 'finance', user: 'u1' };

  it('serves a bundle stamped with the same level', () => {
    const cache = new AnticipationCache();
    cache.store(bundle({ scopeRung: RUNG }));
    expect(cache.lookup({ ...Q, scopePath: PATH })).not.toBeNull();
  });
  it('does not serve it to another level, nor to a request that names none', () => {
    const cache = new AnticipationCache();
    cache.store(bundle({ scopeRung: RUNG }));
    expect(cache.lookup({ ...Q, scopePath: OTHER })).toBeNull();
    expect(cache.lookup({ ...Q })).toBeNull();
  });
  it('does not serve an unstamped bundle to a request that names a level', () => {
    const cache = new AnticipationCache();
    cache.store(bundle());
    expect(cache.lookup({ ...Q, scopePath: PATH })).toBeNull();
  });
  it('without a path on either side it serves exactly as before', () => {
    const cache = new AnticipationCache();
    cache.store(bundle());
    expect(cache.lookup({ ...Q })).not.toBeNull();
    expect(cache.lookup({ ...Q, scopePath: {} })).not.toBeNull();
  });
  it('two levels do not overwrite each other', () => {
    const cache = new AnticipationCache();
    cache.store(bundle({ scopeRung: RUNG }));
    cache.store(bundle({ scopeRung: rungKey(OTHER)! }));
    expect(cache.size()).toBe(2);
  });
  it('the stream client stamps a pushed bundle with anticipation_scope_rung', () => {
    const cache = new AnticipationCache();
    const client = new GrpcStreamClient(creds, cache, undefined, { host: '127.0.0.1', port: 1, useTls: false });
    (client as unknown as { handleBundle(b: unknown): void }).handleBundle({
      bundle_id: 'b9', bundle_type: 'anticipation', items_by_type: {
        facts: { items: bundle().itemsByType['facts'] },
      },
      anticipation_user_id: 'u1', anticipation_customer_id: 'c1', anticipation_conversation_id: '',
      search_keywords: ['distributed', 'systems', 'sharding'], search_queries: ['distributed systems sharding'],
      anticipation_scope_rung: RUNG,
    });
    expect(cache.lookup({ ...Q, scopePath: PATH })).not.toBeNull();
    expect(cache.lookup({ ...Q })).toBeNull();
  });
});

describe('the middleware asks the cache for its own level', () => {
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'distributed systems sharding' }] }];
  const OTHER = { customer: 'acme', team: 'finance', user: 'u1' };

  it('a bundle stamped for another level is not served: the HTTP fetch runs instead', async () => {
    const cache = new AnticipationCache();
    cache.store(bundle({ scopeRung: rungKey(OTHER)!, customerId: 'acme' }));
    const f = okFetch({ items_by_type: {} });
    vi.stubGlobal('fetch', f);
    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: cache, grpcClient: null,
      userId: 'u1', customerId: 'acme', writeMemory: false, scopePath: PATH,
    });
    await mw.transformParams!({ params: { prompt }, type: 'generate' } as never);
    expect(f.mock.calls.some((c) => String(c[0]).includes('/v1/context/'))).toBe(true);
  });

  it('a bundle stamped for this level is served: no HTTP fetch', async () => {
    const cache = new AnticipationCache();
    cache.store(bundle({ scopeRung: rungKey(PATH)!, customerId: 'acme' }));
    const f = okFetch({ items_by_type: {} });
    vi.stubGlobal('fetch', f);
    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: cache, grpcClient: null,
      userId: 'u1', customerId: 'acme', writeMemory: false, scopePath: PATH,
    });
    await mw.transformParams!({ params: { prompt }, type: 'generate' } as never);
    expect(f.mock.calls.some((c) => String(c[0]).includes('/v1/context/'))).toBe(false);
  });
});

describe('rungKey and rungServes match the shared contract', () => {
  const file = path.resolve(path.dirname(fileURLToPath(import.meta.url)),
    '../../../../synap/sdk/CONTRACT/behavior/anticipation.json');
  it('every golden vector', () => {
    if (!existsSync(file)) return; // the monorepo only
    const vectors = (JSON.parse(readFileSync(file, 'utf8')) as {
      rung_key: { vectors: Array<{ path: Record<string, string>; key: string | null }> };
    }).rung_key.vectors;
    expect(vectors.length).toBeGreaterThan(5);
    for (const v of vectors) expect(rungKey(v.path)).toBe(v.key);
  });
  it('serves on equality only; absent matches absent', () => {
    expect(rungServes('', null)).toBe(true);
    expect(rungServes(undefined, null)).toBe(true);
    expect(rungServes('customer=a', 'customer=a')).toBe(true);
    expect(rungServes('customer=a', null)).toBe(false);
    expect(rungServes(null, 'customer=a')).toBe(false);
    expect(rungServes('customer=a', 'customer=a|user=b')).toBe(false);
  });
});
