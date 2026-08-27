import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SynapClient } from '../client.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import {
  scopeLookupParams, buildAnticipationResponse, servedItemIds, tryServeFromCache,
} from '../context/anticipated.js';
import * as registry from '../registry.js';

beforeEach(() => registry.clear());
afterEach(() => registry.clear());

/** A cache holding one bundle for `entityId`, matching "refund" queries. */
function seeded(entityId: string | null, extra: Record<string, unknown> = {}) {
  const cache = new AnticipationCache();
  cache.store({
    bundleId: 'b1',
    entityId,
    bundleType: 'anticipation',
    itemsByType: {
      facts: [
        { item_id: 'i1', content: 'Refunds take five business days to process', scope: 'user' },
        { item_id: 'i2', content: 'Refund requests go to the billing team', scope: 'user' },
      ],
    },
    ...extra,
  });
  return cache;
}

const QUERY = ['how long do refunds take'];

describe('scope widening', () => {
  const all = {
    searchQuery: QUERY, entityId: 'u1', customerId: 'c1', clientId: 'cli', conversationId: 'conv',
  };

  it('conversation scope keeps every identifier', () => {
    expect(scopeLookupParams('conversation', all)).toEqual(all);
  });

  it('user scope drops the conversation binding', () => {
    // A user-scope fetch is not bound to one thread.
    expect(scopeLookupParams('user', all).conversationId).toBeUndefined();
    expect(scopeLookupParams('user', all).entityId).toBe('u1');
  });

  it('customer scope refuses to widen to user bundles', () => {
    // Widening here would leak one visitor's data into another visitor's
    // customer fetch.
    const scoped = scopeLookupParams('customer', all);
    expect(scoped.entityId).toBeNull();
    expect(scoped.customerId).toBe('c1');
    expect(scoped.clientId).toBe('cli');
  });

  it('client scope narrows furthest', () => {
    const scoped = scopeLookupParams('client', all);
    expect(scoped.entityId).toBeNull();
    expect(scoped.customerId).toBeNull();
    expect(scoped.clientId).toBe('cli');
  });
});

describe('serving from the cache', () => {
  it('answers a matching query without the network', () => {
    const attempt = tryServeFromCache(seeded('u1'), 'user', {
      searchQuery: QUERY, entityId: 'u1', clientId: 'cli',
    }, 10);
    expect(attempt.hit).not.toBeNull();
    expect(attempt.response?.facts).toBeDefined();
  });

  it('marks the response as locally served', () => {
    const attempt = tryServeFromCache(seeded('u1'), 'user', {
      searchQuery: QUERY, entityId: 'u1',
    }, 10);
    // How a caller, and the metering that reads it, tells a local hit from a
    // billed fetch.
    expect(attempt.response?.metadata?.['source']).toBe('anticipation');
    expect(attempt.response?.metadata?.['cache_hit']).toBe(true);
  });

  it('falls through when nothing matches', () => {
    const attempt = tryServeFromCache(seeded('u1'), 'user', {
      searchQuery: ['something completely unrelated to any of it'], entityId: 'u1',
    }, 10);
    expect(attempt.response).toBeNull();
  });

  it('never serves one user a bundle stored for another', () => {
    const attempt = tryServeFromCache(seeded('user-a'), 'user', {
      searchQuery: QUERY, entityId: 'user-b',
    }, 10);
    expect(attempt.response).toBeNull();
  });

  it('degrades to the network when the cache throws', () => {
    // A fault in the cache must never fail a retrieval the server could answer.
    const broken = { lookup() { throw new Error('cache exploded'); } } as unknown as AnticipationCache;
    const attempt = tryServeFromCache(broken, 'user', { searchQuery: QUERY, entityId: 'u1' }, 10);
    expect(attempt.response).toBeNull();
  });

  it('collects served item ids for the learning-loop event', () => {
    const hit = seeded('u1').lookup({ searchQuery: QUERY, entityId: 'u1' });
    expect(hit).not.toBeNull();
    if (hit !== null) expect(servedItemIds(hit).length).toBeGreaterThan(0);
  });

  it('carries the conversation context through', () => {
    const hit = {
      itemsByType: { facts: [{ item_id: 'i1', content: 'x' }] },
      bundleIds: ['b1'], score: 1, coverage: 1,
    };
    const response = buildAnticipationResponse(hit);
    expect(response.facts).toHaveLength(1);
  });
});

/**
 * End to end: a real gRPC server pushes a bundle into a real SynapClient's
 * cache, and the next fetch is answered without touching the network.
 *
 * This is the regression that matters. `store()` was wired and `lookup()` was
 * never called, so the stream filled a cache nothing read and every fetch was
 * a billed round trip. Only a test that drives BOTH halves can catch that.
 */
describe('the client serves a fetch from a stream-delivered bundle', () => {
  async function startServer() {
    const grpc = await import('@grpc/grpc-js');
    const protoLoader = await import('@grpc/proto-loader');
    const { SYNAP_PROTO_DESCRIPTOR } = await import('../grpc/descriptor.js');
    const pkgDef = protoLoader.fromJSON(
      SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoLoader.fromJSON>[0],
      { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true },
    );
    const pkg = grpc.loadPackageDefinition(pkgDef) as unknown as {
      synap: { v1: { SynapService: { service: never } } };
    };
    let push: ((m: Record<string, unknown>) => void) | null = null;
    const queued: Array<Record<string, unknown>> = [];
    let ready: () => void = () => {};
    const connected = new Promise<void>((r) => { ready = r; });

    const server = new grpc.Server();
    server.addService(pkg.synap.v1.SynapService.service, {
      Listen: (call: { on: (e: string, h: (a?: unknown) => void) => void; write: (m: unknown) => void; end: () => void }) => {
        push = (m) => call.write(m);
        for (const m of queued.splice(0)) call.write(m);
        ready();
        call.on('data', () => {});
        call.on('end', () => call.end());
        call.on('error', () => {});
      },
      IngestTelemetry: () => {},
    });
    const port = await new Promise<number>((resolve, reject) => {
      server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (e, p) =>
        e ? reject(e) : resolve(p));
    });
    return {
      port, connected,
      push: (m: Record<string, unknown>) => { if (push) push(m); else queued.push(m); },
      stop: () => new Promise<void>((r) => server.tryShutdown(() => r())),
    };
  }

  async function until(predicate: () => boolean, ms = 3000) {
    const deadline = Date.now() + ms;
    while (!predicate()) {
      if (Date.now() > deadline) throw new Error('timed out');
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it('answers from the cache and never calls fetch', async () => {
    const server = await startServer();
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ context: { facts: [] } }), {
        status: 200, headers: { 'content-type': 'application/json' },
      }));
    const client = new SynapClient({
      apiKey: 'k', _force_new: true, clientId: 'cli', fetchImpl: fetchImpl as never,
    });

    await client.instance.listen({ host: '127.0.0.1', port: server.port, use_tls: false });
    await server.connected;

    server.push({
      context_bundle: {
        bundle_id: 'b-stream',
        bundle_type: 'anticipation',
        anticipation_user_id: 'u1',
        search_queries: ['refund policy'],
        items_by_type: {
          facts: { items: [{ item_id: 'i1', content: 'Refunds take five business days to process', scope: 'user' }] },
        },
      },
    });
    await until(() => client.anticipation_cache_snapshot().total_entries > 0);

    const before = fetchImpl.mock.calls.length;
    const ctx = await client.user.context.fetch({
      user_id: 'u1', search_query: ['how long do refunds take'],
    });

    // Served locally: no extra network call, and the response says so.
    expect(fetchImpl.mock.calls.length).toBe(before);
    expect(ctx.metadata?.['source']).toBe('anticipation');
    expect(ctx.facts?.[0]?.content).toContain('Refunds take five business days');

    await client.instance.stop_listening();
    await client.shutdown();
    await server.stop();
  });

  it('still goes to the network when the query does not match', async () => {
    const server = await startServer();
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ context: { facts: [{ id: 'srv', content: 'from server' }] } }), {
        status: 200, headers: { 'content-type': 'application/json' },
      }));
    const client = new SynapClient({
      apiKey: 'k', _force_new: true, clientId: 'cli', fetchImpl: fetchImpl as never,
    });
    await client.instance.listen({ host: '127.0.0.1', port: server.port, use_tls: false });
    await server.connected;
    server.push({
      context_bundle: {
        bundle_id: 'b-stream', bundle_type: 'anticipation', anticipation_user_id: 'u1',
        items_by_type: { facts: { items: [{ item_id: 'i1', content: 'Refunds take five business days', scope: 'user' }] } },
      },
    });
    await until(() => client.anticipation_cache_snapshot().total_entries > 0);

    const ctx = await client.user.context.fetch({
      user_id: 'u1', search_query: ['what is the weather in Lisbon today'],
    });
    expect(ctx.facts?.[0]?.content).toBe('from server');
    await client.instance.stop_listening();
    await client.shutdown();
    await server.stop();
  });
});

describe('the network path, when nothing is cached', () => {
  it('goes to the server with a cold cache', async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ context: { facts: [{ id: 'f1', content: 'from server' }] } }), {
        status: 200, headers: { 'content-type': 'application/json' },
      }));
    const c = new SynapClient({ apiKey: 'k', _force_new: true, fetchImpl: fetchImpl as never });
    const ctx = await c.user.context.fetch({ user_id: 'u1', search_query: QUERY });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(ctx.facts?.[0]?.content).toBe('from server');
    // Not marked as anticipation, because it was not served locally.
    expect(ctx.metadata?.['source']).not.toBe('anticipation');
    await c.shutdown();
  });
});
