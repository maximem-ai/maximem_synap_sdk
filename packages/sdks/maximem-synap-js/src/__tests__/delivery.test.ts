/**
 * Delivery: an event survives a dropped stream, and arrives once.
 *
 * Mirrors `python/tests/test_delivery.py`. Before this, `#write` returned early
 * and silently whenever the stream was down, so every turn that happened during
 * a reconnect was simply gone — JS did not even have the send queue Python had.
 *
 * Two buffers, because there are two ways to lose an event:
 *
 * - queued: there was no stream when we tried.
 * - unacknowledged: the write went out and the stream broke before the server
 *   said it had it. That is the one that loses a turn mid-conversation rather
 *   than during a visible outage.
 *
 * Real grpc-js against a real server on localhost, like `grpc-stream.test.ts`
 * and for the same reason: the interesting part is what actually goes over the
 * wire, and a stub would assert the shape this file already believes in.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

interface ServerHandle {
  port: number;
  received: Array<Record<string, unknown>>;
  push: (msg: Record<string, unknown>) => void;
  dropStream: () => void;
  stop: () => Promise<void>;
  clientConnected: () => Promise<void>;
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
  let current: { write: (m: unknown) => void; end: () => void } | null = null;
  let markConnected: () => void = () => {};
  let connected = new Promise<void>((resolve) => { markConnected = resolve; });

  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen: (call: {
      on: (e: string, h: (arg?: unknown) => void) => void;
      write: (m: unknown) => void;
      end: () => void;
    }) => {
      current = call;
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
    push: (msg) => current?.write(msg),
    dropStream: () => {
      // End the call from the server side: the client sees the stream go and
      // reconnects, which is the event these tests are about.
      connected = new Promise<void>((resolve) => { markConnected = resolve; });
      current?.end();
      current = null;
    },
    clientConnected: () => connected,
    stop: () => new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); }),
  };
}

function turn(overrides: Record<string, unknown> = {}) {
  return {
    event_type: 'user_message',
    conversation_id: 'c1',
    user_id: 'u1',
    content: 'where is my order',
    role: 'user',
    ...overrides,
  };
}

function eventIds(received: Array<Record<string, unknown>>): string[] {
  return received
    .filter((m) => m['conversation_event'])
    .map((m) => String(
      (m['conversation_event'] as Record<string, unknown>)['event_id'] ?? '',
    ));
}

describe('delivery over a real stream', () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.splice(0)) await fn();
  });

  async function connected(options = {}) {
    const server = await startServer();
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: server.port, useTls: false, ...options,
    });
    cleanup.push(async () => { await client.disconnect(); await server.stop(); });
    await client.connect();
    await server.clientConnected();
    return { server, client };
  }

  async function until(predicate: () => boolean, ms = 4000): Promise<void> {
    const deadline = Date.now() + ms;
    while (!predicate()) {
      if (Date.now() > deadline) throw new Error('timed out waiting for the stream');
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it('mints an id when the caller sends none', async () => {
    const { server, client } = await connected();
    client.sendConversationEvent(turn());
    await until(() => eventIds(server.received).length === 1);
    expect(eventIds(server.received)[0]).toBeTruthy();
  });

  it('keeps the id the caller supplied', async () => {
    const { server, client } = await connected();
    client.sendConversationEvent(turn({ event_id: 'mine' }));
    await until(() => eventIds(server.received).length === 1);
    expect(eventIds(server.received)).toEqual(['mine']);
  });

  it('gives two events two ids', async () => {
    const { server, client } = await connected();
    client.sendConversationEvent(turn());
    client.sendConversationEvent(turn());
    await until(() => eventIds(server.received).length === 2);
    expect(new Set(eventIds(server.received)).size).toBe(2);
  });

  it('replays an unacknowledged event after the stream drops', async () => {
    // The case that loses a turn mid-conversation: the write went out, the
    // stream broke before the server acknowledged it, and nothing knew.
    const { server, client } = await connected({ reconnectDelayMs: 10 });
    client.sendConversationEvent(turn({ event_id: 'ev-1' }));
    await until(() => eventIds(server.received).includes('ev-1'));

    server.dropStream();
    await server.clientConnected();

    await until(() => eventIds(server.received).filter((i) => i === 'ev-1').length >= 2);
    const seen = eventIds(server.received).filter((i) => i === 'ev-1');
    expect(seen.length).toBeGreaterThanOrEqual(2);
    expect(new Set(seen).size).toBe(1);   // the SAME id, which is what dedupes
  });

  it('does not replay an event the server acknowledged', async () => {
    const { server, client } = await connected({ reconnectDelayMs: 10 });
    client.sendConversationEvent(turn({ event_id: 'ev-1' }));
    await until(() => eventIds(server.received).includes('ev-1'));

    server.push({ event_ack: { event_ids: ['ev-1'], timestamp_ms: Date.now() } });
    await new Promise((r) => setTimeout(r, 100));

    server.dropStream();
    await server.clientConnected();
    await new Promise((r) => setTimeout(r, 200));

    expect(eventIds(server.received).filter((i) => i === 'ev-1')).toHaveLength(1);
  });

  it('queues an event sent while the stream is down, and sends it on reconnect', async () => {
    const { server, client } = await connected({ reconnectDelayMs: 10 });
    server.dropStream();

    client.sendConversationEvent(turn({ event_id: 'ev-while-down' }));

    await server.clientConnected();
    await until(() => eventIds(server.received).includes('ev-while-down'));
    expect(eventIds(server.received)).toContain('ev-while-down');
  });

  it('never throws on the caller, connected or not', async () => {
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: 1, useTls: false,
    });
    expect(() => client.sendConversationEvent(turn())).not.toThrow();
    await expect(client.disconnect()).resolves.toBeUndefined();
  });
});
