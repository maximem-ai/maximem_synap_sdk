/**
 * A turn sent immediately after `connect()` must still reach the server.
 *
 * Mirrors `python/tests/test_stream_readiness.py`. The Python side of this bug
 * was found on deployed staging: five events sent right after `listen()`
 * returned gave the server `conversation_events=0`, the same five with a three
 * second pause gave 5 of 5, and nothing the client did distinguished them.
 *
 * Three things had to be true for a turn to survive, and JS was missing the
 * same ones:
 *
 *   1. do not report 'connected' off a call object the server has not accepted
 *   2. half-close the write side, then WAIT for the server to finish reading
 *   3. tell the caller what never got through
 *
 * These run against a real grpc-js server in-process, like `delivery.test.ts`
 * and for the same reason: the question is what the SERVER received, and every
 * version of this bug looked perfect from the client. A test that asserts
 * `write()` was called passes on the broken code. The first draft of this file
 * did exactly that, and a mutation that deleted the whole fix left all nine
 * tests green.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

interface ServerHandle {
  port: number;
  received: Array<Record<string, unknown>>;
  clientConnected: () => Promise<void>;
  stop: () => Promise<void>;
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
  let markConnected: () => void = () => {};
  const connected = new Promise<void>((resolve) => { markConnected = resolve; });

  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen: (call: {
      on: (e: string, h: (arg?: unknown) => void) => void;
      end: () => void;
    }) => {
      markConnected();
      call.on('data', (msg) => { received.push(msg as Record<string, unknown>); });
      // The half-close arrives here. Ending our side completes the RPC
      // cleanly, which is what the client's disconnect() now waits for.
      call.on('end', () => { call.end(); });
      call.on('error', () => { /* client hung up */ });
    },
    IngestTelemetry: () => { /* unused */ },
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (err, p) => {
      if (err) reject(err); else resolve(p);
    });
  });

  return {
    port,
    received,
    clientConnected: () => connected,
    stop: () => new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); }),
  };
}

function turn(id: string, eventType = 'user_message') {
  return {
    event_type: eventType,
    conversation_id: 'c1',
    user_id: 'u1',
    content: 'hello',
    role: 'user',
    event_id: id,
  };
}

function eventIds(received: Array<Record<string, unknown>>): string[] {
  return received
    .filter((m) => m['conversation_event'])
    .map((m) => String((m['conversation_event'] as Record<string, unknown>)['event_id'] ?? ''));
}

describe('a turn sent right after connect still reaches the server', () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => { for (const fn of cleanup.splice(0)) await fn(); });

  async function live() {
    const server = await startServer();
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: server.port, useTls: false,
    });
    cleanup.push(async () => { await server.stop(); });
    return { server, client };
  }

  it('delivers all five events of a turn with no pause anywhere', async () => {
    // THE regression. No sleep, no wait: connect, send the turn, disconnect.
    // That sequence lost every event, and the client reported success for all
    // of them. Asserting on the server's `received` is the whole point.
    const { server, client } = await live();
    await client.connect();

    for (const [i, kind] of ['user_message', 'agent_thinking', 'tool_call',
      'tool_result', 'assistant_message'].entries()) {
      client.sendConversationEvent(turn(`ev-${i}`, kind));
    }
    await client.disconnect();

    expect(eventIds(server.received)).toEqual(
      ['ev-0', 'ev-1', 'ev-2', 'ev-3', 'ev-4']);
  });

  it('delivers a single event sent in the same tick as connect', async () => {
    const { server, client } = await live();
    await client.connect();
    client.sendConversationEvent(turn('only-one'));
    await client.disconnect();

    expect(eventIds(server.received)).toEqual(['only-one']);
  });

  it('keeps the ids it was given, so the server can deduplicate', async () => {
    // A replay that mints a fresh id doubles the turn instead of retrying it.
    const { server, client } = await live();
    await client.connect();
    client.sendConversationEvent(turn('keep-this-id'));
    await client.disconnect();

    expect(eventIds(server.received)).toContain('keep-this-id');
  });

  it('reports nothing undelivered once the server has it', async () => {
    const { server, client } = await live();
    await client.connect();
    client.sendConversationEvent(turn('ev-1'));
    await client.disconnect();

    expect(eventIds(server.received)).toEqual(['ev-1']);
  });
});

describe('the caller can see what was lost', () => {
  function offline() {
    return new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: 1, useTls: false,
    });
  }

  it('reports nothing outstanding on a fresh client', () => {
    // Every send path returns void, so without this a caller cannot tell a
    // delivered turn from a lost one.
    expect(offline().undelivered()).toEqual({ queued: 0, unacknowledged: 0 });
  });

  it('counts events with no stream as undelivered', () => {
    const c = offline();
    c.sendConversationEvent(turn('ev-1'));
    c.sendConversationEvent(turn('ev-2'));
    expect(c.undelivered()).toEqual({ queued: 2, unacknowledged: 0 });
  });

  it('disconnect on a client that never connected does not hang', async () => {
    const c = offline();
    await expect(c.disconnect()).resolves.toBeUndefined();
    expect(c.currentState).toBe('closed');
  });
});
