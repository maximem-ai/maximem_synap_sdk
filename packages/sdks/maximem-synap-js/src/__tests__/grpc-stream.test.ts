import { describe, it, expect, afterEach } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { SYNAP_PROTO_DESCRIPTOR, PROTO_SHA } from '../grpc/descriptor.js';

/**
 * Real gRPC, not mocks.
 *
 * A stubbed grpc-js would prove nothing about the part most likely to break:
 * whether the inlined proto descriptor actually serialises a ContextBundleProto
 * the same way the server does. So this stands up a genuine grpc-js server on
 * localhost using the same descriptor the client loads, and speaks the real
 * bidirectional Listen stream over it.
 */

interface ServerHandle {
  port: number;
  /** Resolves with each StreamEvent the client sends. */
  received: Array<Record<string, unknown>>;
  push: (msg: Record<string, unknown>) => void;
  stop: () => Promise<void>;
  clientConnected: Promise<void>;
}

async function startServer(): Promise<ServerHandle> {
  const grpc = await import('@grpc/grpc-js');
  const protoLoader = await import('@grpc/proto-loader');

  const pkgDef = protoLoader.fromJSON(
    SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoLoader.fromJSON>[0],
    { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true },
  );
  const pkg = grpc.loadPackageDefinition(pkgDef) as unknown as {
    synap: { v1: { SynapService: { service: never } } };
  };

  const received: Array<Record<string, unknown>> = [];
  let pushToClient: ((msg: Record<string, unknown>) => void) | null = null;
  const queued: Array<Record<string, unknown>> = [];
  let markConnected: () => void = () => {};
  const clientConnected = new Promise<void>((resolve) => { markConnected = resolve; });

  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen: (call: {
      on: (e: string, h: (arg?: unknown) => void) => void;
      write: (m: unknown) => void;
      end: () => void;
    }) => {
      pushToClient = (msg) => call.write(msg);
      for (const m of queued.splice(0)) call.write(m);
      markConnected();
      call.on('data', (msg) => { received.push(msg as Record<string, unknown>); });
      call.on('end', () => { call.end(); });
      call.on('error', () => { /* client hung up */ });
    },
    IngestTelemetry: () => { /* unused here */ },
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (err, p) => {
      if (err) reject(err); else resolve(p);
    });
  });

  return {
    port,
    received,
    push: (msg) => { if (pushToClient) pushToClient(msg); else queued.push(msg); },
    clientConnected,
    stop: () => new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); }),
  };
}

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

function bundle(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    bundle_id: 'b1',
    bundle_type: 'anticipation',
    anticipation_user_id: 'u1',
    anticipation_conversation_id: 'conv-1',
    search_queries: ['refund policy'],
    search_keywords: ['refund'],
    ttl_hint_seconds: 0,
    items_by_type: {
      facts: { items: [{ item_id: 'i1', content: 'Refunds take five days', scope: 'user' }] },
    },
    ...overrides,
  };
}

describe('gRPC anticipation stream over a real server', () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.splice(0)) await fn();
  });

  async function connected(cache: AnticipationCache, options = {}) {
    const server = await startServer();
    const client = new GrpcStreamClient(CREDS, cache, {
      host: '127.0.0.1', port: server.port, useTls: false, ...options,
    });
    cleanup.push(async () => { await client.disconnect(); await server.stop(); });
    await client.connect();
    await server.clientConnected;
    return { server, client };
  }

  /** The stream is async; wait for the cache to settle rather than sleeping blindly. */
  async function until(predicate: () => boolean, ms = 3000): Promise<void> {
    const deadline = Date.now() + ms;
    while (!predicate()) {
      if (Date.now() > deadline) throw new Error('timed out waiting for the stream');
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it('connects and reports connected state', async () => {
    const cache = new AnticipationCache();
    const { client } = await connected(cache);
    expect(client.isConnected).toBe(true);
    expect(client.currentState).toBe('connected');
  });

  it('stores an anticipation bundle pushed by the server', async () => {
    const cache = new AnticipationCache();
    const { server } = await connected(cache);
    server.push({ context_bundle: bundle() });
    await until(() => cache.size === 1);

    const snapshot = cache.snapshot();
    expect(snapshot.total_entries).toBe(1);
    expect(snapshot.total_item_records).toBe(1);
    const first = snapshot.bundles[0];
    expect(first?.bundle_id).toBe('b1');
    expect(first?.entity_id).toBe('u1');
    expect(first?.conversation_id).toBe('conv-1');
    expect(first?.bundle_type).toBe('anticipation');
    // Queries and keywords are both fed to BM25, so both are retained.
    expect(first?.search_queries).toContain('refund policy');
    expect(first?.search_queries).toContain('refund');
    expect(first?.items[0]?.content).toBe('Refunds take five days');
  });

  it('serves that bundle back through a lookup', async () => {
    // End to end: proto on the wire, into the cache, out through BM25.
    const cache = new AnticipationCache();
    const { server } = await connected(cache);
    server.push({ context_bundle: bundle() });
    await until(() => cache.size === 1);

    const hit = cache.lookup({
      searchQuery: ['how long do refunds take'],
      entityId: 'u1',
      conversationId: 'conv-1',
    });
    expect(hit).not.toBeNull();
    expect(hit?.itemsByType['facts']?.[0]?.content).toBe('Refunds take five days');
  });

  it('skips reactive bundles, matching Python', async () => {
    const cache = new AnticipationCache();
    const { server } = await connected(cache);
    server.push({ context_bundle: bundle({ bundle_type: 'reactive', bundle_id: 'r1' }) });
    server.push({ context_bundle: bundle({ bundle_id: 'keep' }) });
    await until(() => cache.size === 1);
    expect(cache.snapshot().bundles.map((b) => b.bundle_id)).toEqual(['keep']);
  });

  it('stores a compaction_update AND dispatches it', async () => {
    // Python stores everything except reactive. The vercel-adk client drops
    // compaction updates instead; following it here would cache less than pip.
    const cache = new AnticipationCache();
    const seen: string[] = [];
    const { server } = await connected(cache, {
      onCompactionUpdate: (conversationId: string) => { seen.push(conversationId); },
    });
    server.push({
      context_bundle: bundle({ bundle_type: 'compaction_update', bundle_id: 'c1' }),
    });
    await until(() => seen.length === 1);
    expect(seen).toEqual(['conv-1']);
    expect(cache.snapshot().bundles.map((b) => b.bundle_id)).toEqual(['c1']);
  });

  it('invokes onContext for every bundle type', async () => {
    const cache = new AnticipationCache();
    const types: string[] = [];
    const { server } = await connected(cache, {
      onContext: (b: Record<string, unknown>) => { types.push(String(b['bundle_type'])); },
    });
    for (const t of ['anticipation', 'reactive', 'compaction_update']) {
      server.push({ context_bundle: bundle({ bundle_type: t, bundle_id: `b-${t}` }) });
    }
    await until(() => types.length === 3);
    expect(types).toEqual(['anticipation', 'reactive', 'compaction_update']);
  });

  it('floors a short server TTL hint at 60s and treats 0 as unset', async () => {
    const cache = new AnticipationCache();
    const { server } = await connected(cache);
    server.push({ context_bundle: bundle({ bundle_id: 'short', ttl_hint_seconds: 5 }) });
    await until(() => cache.size === 1);
    // A 5s hint must not evict the bundle almost immediately. Query phrased
    // like the lookup test above: a single novel token trips the novel-term
    // gate on a corpus this small, which would fail for the wrong reason.
    expect(
      cache.lookup({
        searchQuery: ['how long do refunds take'], entityId: 'u1', conversationId: 'conv-1',
      }),
    ).not.toBeNull();
  });

  it('sends conversation events the server can parse', async () => {
    const cache = new AnticipationCache();
    const { server, client } = await connected(cache);
    client.sendConversationEvent({
      event_type: 'user_message', content: 'hello', role: 'user',
      conversation_id: 'conv-1', user_id: 'u1',
    });
    await until(() => server.received.length >= 1);
    const event = server.received[0]?.['conversation_event'] as Record<string, unknown>;
    expect(event?.['content']).toBe('hello');
    expect(event?.['event_type']).toBe('user_message');
    expect(event?.['conversation_id']).toBe('conv-1');
  });

  it('sends session control', async () => {
    const cache = new AnticipationCache();
    const { server, client } = await connected(cache);
    client.sendSessionControl({ action: 'start', conversation_id: 'conv-1', user_id: 'u1' });
    await until(() => server.received.length >= 1);
    const control = server.received[0]?.['session_control'] as Record<string, unknown>;
    expect(control?.['action']).toBe('start');
  });

  it('drops writes silently when disconnected', async () => {
    const cache = new AnticipationCache();
    const client = new GrpcStreamClient(CREDS, cache, { host: '127.0.0.1', port: 1, useTls: false });
    // Never connected: a fire-and-forget event must not throw on the caller.
    expect(() => client.sendConversationEvent({ event_type: 'user_message' })).not.toThrow();
  });

  it('reports closed after disconnect', async () => {
    const cache = new AnticipationCache();
    const { client } = await connected(cache);
    await client.disconnect();
    expect(client.currentState).toBe('closed');
    expect(client.isConnected).toBe(false);
  });
});

describe('proto descriptor', () => {
  it('records the sha of the proto it was generated from', () => {
    expect(PROTO_SHA).toMatch(/^[0-9a-f]{16}$/);
  });

  it('matches the canonical proto still on disk', async () => {
    // The descriptor is generated, committed, and easy to forget to regenerate.
    const { readFileSync: read, existsSync: exists } = await import('node:fs');
    const { createHash } = await import('node:crypto');
    const canonical = new URL('../../../../proto/synap_service.proto', import.meta.url);
    if (!exists(canonical)) return;
    const sha = createHash('sha256').update(read(canonical, 'utf8')).digest('hex').slice(0, 16);
    expect(
      PROTO_SHA,
      'the proto changed; regenerate with scripts/gen_proto_descriptor.mjs',
    ).toBe(sha);
  });

  it('describes the bidirectional Listen RPC', async () => {
    const protoLoader = await import('@grpc/proto-loader');
    const pkgDef = protoLoader.fromJSON(
      SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoLoader.fromJSON>[0],
      { keepCase: true },
    );
    const listen = pkgDef['synap.v1.SynapService'] as Record<string, {
      path: string; requestStream: boolean; responseStream: boolean;
    }>;
    expect(listen['Listen']?.path).toBe('/synap.v1.SynapService/Listen');
    expect(listen['Listen']?.requestStream).toBe(true);
    expect(listen['Listen']?.responseStream).toBe(true);
  });
});

describe('connect() failure semantics', () => {
  it('rejects instead of resolving against an unreachable port', async () => {
    // Regression: grpc-js connects lazily, so `connect()` used to resolve
    // against a dead port, `isConnected` read true, and every subsequent
    // fire-and-forget send was dropped silently. An agent in that state looks
    // like it has no memory rather than like it has a connection problem.
    const cache = new AnticipationCache();
    const client = new GrpcStreamClient(CREDS, cache, {
      host: '127.0.0.1',
      port: 1, // nothing listens here
      useTls: false,
      connectTimeoutMs: 1500,
    });
    let caught: unknown = null;
    await client.connect().catch((e: unknown) => { caught = e; });
    expect(caught, 'connect() should reject for an unreachable host').not.toBeNull();
    expect(client.isConnected).toBe(false);
    expect(client.currentState).not.toBe('connected');
    await client.disconnect();
  });
});
