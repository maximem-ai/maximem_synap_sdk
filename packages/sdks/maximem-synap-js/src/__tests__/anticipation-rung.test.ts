/**
 * A prefetched bundle may only be served back at the rung it was fetched for.
 *
 * The Python counterpart is `synap/tests/scoping/test_prefetch_carries_the_rung.py`,
 * and the two must stay in step: this is a boundary, so a fix that lands on one
 * SDK and not the other is a boundary that exists in one language.
 *
 * The cache matched a stored bundle on entity id and conversation id, and a
 * rung is neither, so a bundle retrieved for one rung was servable to a caller
 * standing at another, entirely inside the customer's own process. No request
 * leaves it, so nothing in any server log shows it happening.
 *
 * The rule is EQUALITY. A subset rule ("the bundle names more rungs than you
 * did, close enough") reads as the generous option and is the dangerous one:
 * the rung nobody named is precisely the rung that distinguishes two people who
 * share an external id, which is the whole reason the ladder exists. Being
 * wrong this way costs a cold fetch; being wrong the other way hands one
 * person's context to another.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AnticipationCache, rungKey, rungServes } from '../context/anticipation-cache.js';
import { scopeLookupParams, tryServeFromCache } from '../context/anticipated.js';
import { SynapClient } from '../client.js';
import * as registry from '../registry.js';

beforeEach(() => registry.clear());
afterEach(() => registry.clear());

/**
 * A real gRPC server speaking the committed descriptor.
 *
 * Copied from `anticipated-serving.test.ts` rather than shared, deliberately:
 * a helper shared between two suites is a helper one of them can quietly
 * change, and this one is the thing that makes the end-to-end assertion below
 * mean "over the real wire" rather than "through a stub I wrote".
 */
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

const PAYMENTS = 'customer=acme|team=payments|user=dana';
const SECURITY = 'customer=acme|team=security|user=dana';
const QUERY = ['how long do refunds take'];
const CONVERSATION = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';

/** A cache holding one bundle, stamped at `scopeRung`. */
function seeded(scopeRung: string | null, bundleId = 'b1') {
  const cache = new AnticipationCache();
  cache.store({
    bundleId,
    entityId: 'dana',
    scopeRung,
    bundleType: 'anticipation',
    itemsByType: {
      facts: [{ item_id: 'i1', content: 'Refunds take five business days to process' }],
    },
  });
  return cache;
}

const look = (cache: AnticipationCache, scopeRung: string | null) =>
  cache.lookup({ searchQuery: QUERY, entityId: 'dana', scopeRung });

describe('the rung key', () => {
  it('is written the way a caller writes a scope path', () => {
    expect(rungKey({ customer: 'acme', team: 'payments', user: 'dana' })).toBe(PAYMENTS);
  });

  it('does not depend on the order the caller wrote the path in', () => {
    // An object carries insertion order and the ladder's own order is not
    // knowable here, so the key is sorted. Without that, one rung written two
    // ways is two rungs and the cache never hits.
    expect(rungKey({ user: 'dana', customer: 'acme', team: 'payments' })).toBe(PAYMENTS);
  });

  it('names no rung for an absent or empty path', () => {
    // Every caller in production today.
    expect(rungKey(null)).toBeNull();
    expect(rungKey(undefined)).toBeNull();
    expect(rungKey({})).toBeNull();
  });

  it('cannot be forged by a separator inside an external id', () => {
    // External ids are the client's own strings and can hold anything.
    expect(rungKey({ customer: 'a|b' })).not.toBe(rungKey({ customer: 'a', b: 'x' }));
    // The escape-order case: `a|b` and the literal `a\pb` must stay distinct.
    expect(rungKey({ customer: 'a|b' })).not.toBe(rungKey({ customer: 'a\\pb' }));
  });

  it('matches the shared contract vectors, which the Python SDK also asserts', () => {
    // Three implementations of one recipe (this SDK, the Python SDK, the
    // server) because the producer is not an SDK, so the usual "port it once"
    // rule cannot apply. The vectors are the only thing left holding them
    // together.
    const here = path.dirname(fileURLToPath(import.meta.url));
    const contract = JSON.parse(
      readFileSync(path.resolve(here, '../behavior/anticipation.json'), 'utf8'),
    ) as { rung_key: { vectors: { path: Record<string, string>; key: string | null }[] } };
    expect(contract.rung_key.vectors.length).toBeGreaterThanOrEqual(8);
    for (const c of contract.rung_key.vectors) {
      expect(rungKey(c.path)).toBe(c.key);
    }
  });
});

describe('rungServes', () => {
  it('is equality, with the empty string read as no rung', () => {
    // A proto string field has no null, so an unstamped bundle crosses the
    // wire as ''. Reading that as a rung named '' would take the whole fleet's
    // prefetch dark.
    expect(rungServes(null, null)).toBe(true);
    expect(rungServes('', null)).toBe(true);
    expect(rungServes(PAYMENTS, PAYMENTS)).toBe(true);
    expect(rungServes(PAYMENTS, SECURITY)).toBe(false);
    expect(rungServes(PAYMENTS, null)).toBe(false);
    expect(rungServes(null, PAYMENTS)).toBe(false);
  });
});

describe('the cache refuses to cross a rung', () => {
  it('does not serve one team the bundle another team prefetched', () => {
    const cache = seeded(PAYMENTS);
    expect(look(cache, SECURITY)).toBeNull();
    // POSITIVE CONTROL, in the same test. A refusal on its own proves nothing:
    // the same assertion passes against a cache that is simply broken.
    expect(look(cache, PAYMENTS)).not.toBeNull();
  });

  it('does not serve a rung-specific bundle to a caller who named no rung', () => {
    // The direction a subset rule gets wrong. A caller who named nothing is
    // standing at whatever their ids resolve to, which is not necessarily the
    // rung this bundle was fetched for.
    const cache = seeded(PAYMENTS);
    expect(look(cache, null)).toBeNull();
  });

  it('does not serve an unstamped bundle to a caller who named a rung', () => {
    const cache = seeded(null);
    expect(look(cache, PAYMENTS)).toBeNull();
    // POSITIVE CONTROL: it still answers a caller who named nothing.
    expect(look(cache, null)).not.toBeNull();
  });

  it('changes nothing for a client who names no rung', () => {
    // 170 of the 171 production clients, asserted rather than assumed.
    const cache = seeded(null);
    const hit = look(cache, null);
    expect(hit?.bundleIds).toEqual(['b1']);
  });

  it('refuses a no-query freshness serve across a rung', () => {
    // The least filtered way out of this cache: a whole bundle, no query.
    const cache = seeded(PAYMENTS);
    expect(cache.lookup({ searchQuery: null, entityId: 'dana', scopeRung: SECURITY })).toBeNull();
    expect(cache.lookup({ searchQuery: null, entityId: 'dana', scopeRung: PAYMENTS })).not.toBeNull();
  });

  it('refuses a user summary across a rung', () => {
    const cache = new AnticipationCache();
    cache.store({
      bundleId: 'b-sum', entityId: 'dana', scopeRung: PAYMENTS, bundleType: 'user_summary',
      itemsByType: { facts: [{ item_id: 'i1', content: 'Dana prefers email' }] },
    });
    expect(cache.lookupUserSummary('dana', SECURITY)).toBeNull();
    expect(cache.lookupUserSummary('dana', PAYMENTS)).not.toBeNull();
  });
});

describe('the rung reaches the cache from a fetch', () => {
  it('travels through every scope unchanged', () => {
    // Unlike the ids beside it, a rung never widens, so there is no scope for
    // which dropping it would be the safe thing to do.
    const all = {
      searchQuery: QUERY, entityId: 'dana', customerId: 'acme', clientId: 'cli',
      conversationId: 'conv', scopeRung: PAYMENTS,
    };
    for (const scope of ['conversation', 'user', 'customer', 'client'] as const) {
      expect(scopeLookupParams(scope, all).scopeRung).toBe(PAYMENTS);
    }
  });

  it('is honoured by tryServeFromCache', () => {
    const cache = seeded(PAYMENTS);
    expect(
      tryServeFromCache(cache, 'user', { searchQuery: QUERY, entityId: 'dana', scopeRung: SECURITY }, 10).response,
    ).toBeNull();
    expect(
      tryServeFromCache(cache, 'user', { searchQuery: QUERY, entityId: 'dana', scopeRung: PAYMENTS }, 10).response,
    ).not.toBeNull();
  });

  it('carries the rung into the conversation summary splice', async () => {
    // `#injectUserSummary` runs from a CALLBACK, not from the fetch arguments,
    // so the scope path has to travel with it through
    // `createConversationNamespace`'s `onFetched`. Nothing else in this suite
    // reaches that argument, and without it a summary prefetched at one rung
    // is spliced into a response for a caller standing at another: the exact
    // failure `lookupUserSummary` was hardened for once already, one level
    // down. Python needs no such plumbing because its call site has the path.
    const server = await startServer();
    const client = new SynapClient({
      apiKey: 'k', _force_new: true, clientId: 'cli',
      fetchImpl: (async () =>
        new Response(JSON.stringify({ context: { facts: [] } }), {
          status: 200, headers: { 'content-type': 'application/json' },
        })) as never,
    });

    await client.instance.listen({ host: '127.0.0.1', port: server.port, use_tls: false });
    await server.connected;
    server.push({
      context_bundle: {
        bundle_id: 'b-sum',
        bundle_type: 'user_summary',
        anticipation_user_id: 'dana',
        anticipation_scope_rung: PAYMENTS,
        items_by_type: {
          facts: { items: [{ item_id: 's1', content: 'Dana prefers email over calls', scope: 'user' }] },
        },
      },
    });
    await until(() => client.anticipation_cache_snapshot().total_entries > 0);

    const fetchAt = async (team: string) => {
      let last: Record<string, unknown> = {};
      // The splice fires every fifth turn on a conversation.
      for (let i = 0; i < 5; i += 1) {
        last = (await client.conversation.context.fetch({
          conversation_id: CONVERSATION, user_id: 'dana',
          scope_path: { customer: 'acme', team, user: 'dana' },
        } as never)) as unknown as Record<string, unknown>;
      }
      return last;
    };

    const wrong = await fetchAt('security');
    expect((wrong['facts'] as unknown[] | undefined) ?? []).toHaveLength(0);

    // POSITIVE CONTROL: the rung that prefetched it still gets it spliced in,
    // or this is testing that the splice is broken rather than that it is
    // bounded.
    const right = await fetchAt('payments');
    expect((right['facts'] as { content?: string }[] | undefined) ?? []).toHaveLength(1);

    await client.instance.stop_listening();
    await client.shutdown();
    await server.stop();
  });

  it('reaches the cache from a real stream, over the real proto', async () => {
    // The whole chain in one test: a server writes the rung into
    // `anticipation_scope_rung`, the descriptor carries it, the stream client
    // stores it, and a fetch naming a DIFFERENT rung goes to the network while
    // one naming the same rung is served locally.
    //
    // The network call is the positive control. Without it "refused" and
    // "the cache was empty" look identical, and this feature has confused
    // those two before.
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
        bundle_id: 'b-payments',
        bundle_type: 'anticipation',
        anticipation_user_id: 'dana',
        anticipation_scope_rung: PAYMENTS,
        search_queries: ['refund policy'],
        items_by_type: {
          facts: { items: [{ item_id: 'i1', content: 'Refunds take five business days to process', scope: 'user' }] },
        },
      },
    });
    await until(() => client.anticipation_cache_snapshot().total_entries > 0);

    const before = fetchImpl.mock.calls.length;
    const wrongRung = await client.user.context.fetch({
      user_id: 'dana', search_query: QUERY,
      scope_path: { customer: 'acme', team: 'security', user: 'dana' },
    });
    expect(fetchImpl.mock.calls.length).toBe(before + 1);
    expect(wrongRung.metadata?.['source']).not.toBe('anticipation');

    const rightRung = await client.user.context.fetch({
      user_id: 'dana', search_query: QUERY,
      scope_path: { customer: 'acme', team: 'payments', user: 'dana' },
    });
    expect(fetchImpl.mock.calls.length).toBe(before + 1);
    expect(rightRung.metadata?.['source']).toBe('anticipation');

    await client.instance.stop_listening();
    await client.shutdown();
    await server.stop();
  });
});
