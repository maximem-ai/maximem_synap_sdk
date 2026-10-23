/**
 * The server refuses work while an instance's memory architecture is built.
 *
 * Mirrors `python/tests/test_instance_not_ready.py`. Two surfaces, one
 * behaviour in both SDKs:
 *
 *  - HTTP 409 `{"error": "instance_not_ready", ...}` becomes
 *    `InstanceNotReadyError`, a `ConflictError`, thrown at once and never retried.
 *  - A `throttle` stream signal with metadata code `instance_not_ready` holds the
 *    events (never dropped, never acknowledged), logs ONE warning, replays after
 *    the server's retry_after_seconds, and resumes. No reconnect.
 *
 * The stream half runs against a real in-process grpc-js server, like
 * `stream-readiness.test.ts`: the question is what the SERVER sees.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { GrpcStreamClient, NOT_READY_MAX_HELD } from '../grpc/stream-client.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';
import {
  ConflictError, InstanceNotReadyError, TranscriptConflictError, SynapPermanentError,
  isSynapError,
} from '../errors.js';
import { HttpTransport } from '../transport/http.js';

function body(over: Record<string, unknown> = {}) {
  const flat = {
    error: 'instance_not_ready',
    message: "This instance's memory architecture is still being built.",
    state: 'building',
    retry_after_seconds: 420,
    instance_id: 'inst_1',
    ...over,
  };
  return { ...flat, detail: { code: 'instance_not_ready', ...flat } };
}

function jsonResponse(status: number, payload: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

async function call409(payload: unknown, headers: Record<string, string> = {}) {
  let calls = 0;
  const fetchImpl = (async () => { calls += 1; return jsonResponse(409, payload, headers); }) as typeof fetch;
  const transport = new HttpTransport({
    credentials: { apiKey: 'k', clientId: 'c', instanceId: 'i' }, fetchImpl,
    retryPolicy: { maxAttempts: 3, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
  let error: unknown;
  try { await transport.request('memories_create', { body: {} }); }
  catch (e) { error = e; }
  return { error, calls };
}

describe('HTTP 409 instance_not_ready', () => {
  it('is parsed into the typed error, raised once, never retried', async () => {
    const { error, calls } = await call409(body(), { 'Retry-After': '420' });
    expect(error).toBeInstanceOf(InstanceNotReadyError);
    const e = error as InstanceNotReadyError;
    expect(e.retryAfterSeconds).toBe(420);
    expect(e.instanceId).toBe('inst_1');
    expect(e.state).toBe('building');
    expect(e.message).toContain('still being built');
    expect(e.code).toBe('instance_not_ready');
    expect(calls).toBe(1);
  });

  it('is a ConflictError, so existing handlers keep working', async () => {
    const { error } = await call409(body());
    expect(error).toBeInstanceOf(ConflictError);
    expect(error).toBeInstanceOf(SynapPermanentError);
    expect(error).not.toBeInstanceOf(TranscriptConflictError);
    expect(isSynapError(error)).toBe(true);
    expect(new InstanceNotReadyError('x')).toBeInstanceOf(ConflictError);
  });

  it('reads the detail copy alone, and the Retry-After header as a fallback', async () => {
    const b = body();
    const detailOnly = { detail: b.detail };
    const { error } = await call409(detailOnly, { 'Retry-After': '90' });
    expect((error as InstanceNotReadyError).retryAfterSeconds).toBe(420);

    const noNumber = body();
    delete (noNumber as Record<string, unknown>)['retry_after_seconds'];
    delete (noNumber.detail as Record<string, unknown>)['retry_after_seconds'];
    const viaHeader = await call409(noNumber, { 'Retry-After': '30' });
    expect((viaHeader.error as InstanceNotReadyError).retryAfterSeconds).toBe(30);
  });

  it('leaves a real conflict a plain ConflictError', async () => {
    const { error } = await call409({ detail: { code: 'other', message: 'busy' } });
    expect(error).toBeInstanceOf(ConflictError);
    expect(error).not.toBeInstanceOf(InstanceNotReadyError);
  });
});

// ---------------------------------------------------------------- the stream

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

interface Server {
  port: number;
  received: Array<Record<string, unknown>>;
  accepted: string[];
  listens: () => number;
  setReady: (v: boolean) => void;
  stop: () => Promise<void>;
}

async function startServer(): Promise<Server> {
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
  const accepted: string[] = [];
  let ready = false;
  let listens = 0;
  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen: (call: {
      on: (e: string, h: (arg?: unknown) => void) => void;
      write: (m: unknown) => void;
      end: () => void;
    }) => {
      listens += 1;
      call.on('data', (raw) => {
        const msg = raw as Record<string, unknown>;
        received.push(msg);
        const ev = msg['conversation_event'] as Record<string, unknown> | undefined;
        if (!ev) return;
        if (!ready) {
          call.write({ signal: {
            signal_type: 'throttle',
            reason: 'instance_not_ready: building',
            metadata: {
              code: 'instance_not_ready', state: 'building',
              retry_after_seconds: '60', instance_id: 'inst_0123456789abcdef',
            },
          } });
        } else {
          accepted.push(String(ev['event_id']));
          call.write({ event_ack: { event_ids: [String(ev['event_id'])] } });
        }
      });
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
    port, received, accepted, listens: () => listens, setReady: (v) => { ready = v; },
    stop: () => new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); }),
  };
}

const turn = (id: string) => ({
  event_type: 'user_message', conversation_id: 'c1', user_id: 'u1',
  content: 'hello', role: 'user', event_id: id,
});
const ids = (received: Array<Record<string, unknown>>) =>
  received.filter((m) => m['conversation_event'])
    .map((m) => String((m['conversation_event'] as Record<string, unknown>)['event_id']));
const until = async (cond: () => boolean, ms = 4000) => {
  const end = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > end) throw new Error('condition not met in time');
    await new Promise((r) => setTimeout(r, 10));
  }
};

describe('throttle instance_not_ready on the live stream', () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.splice(0)) await fn();
    vi.restoreAllMocks();
  });

  it('holds the events, warns once, replays, and resumes without reconnecting', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'info').mockImplementation(() => {});
    const server = await startServer();
    cleanup.push(() => server.stop());
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: server.port, useTls: false, outboxPath: '',
      // Probe quickly; the grace must outlast a localhost round trip.
      sleep: (ms) => new Promise((r) => setTimeout(r, ms >= 1000 ? 60 : 60)),
    });
    await client.connect();

    client.sendConversationEvent(turn('ev-1'));
    client.sendConversationEvent(turn('ev-2'));
    await until(() => client.instanceNotReady !== null);

    // Kept, never acknowledged, and still on the same connection.
    expect(client.undelivered().unacknowledged).toBe(2);
    expect(client.currentState).toBe('connected');
    expect(client.instanceNotReady?.retryAfterSeconds).toBe(60);
    expect(client.instanceNotReady?.instanceId).toBe('inst_0123456789abcdef');

    // Let it replay into a few more refusals: still one warning.
    await until(() => ids(server.received).length >= 6);
    const notReadyWarnings = warn.mock.calls.filter((c) => String(c[0]).includes('still building'));
    expect(notReadyWarnings).toHaveLength(1);
    expect(client.undelivered().unacknowledged).toBe(2);

    // A send during the hold is held, not written.
    const before = ids(server.received).length;
    client.sendConversationEvent(turn('ev-3'));
    expect(client.undelivered().unacknowledged).toBe(3);
    expect(ids(server.received).length).toBe(before);

    // The build finishes: the next replay is accepted and acknowledged.
    server.setReady(true);
    await until(() => client.instanceNotReady === null && client.undelivered().unacknowledged === 0);
    expect(new Set(ids(server.received))).toEqual(new Set(['ev-1', 'ev-2', 'ev-3']));
    expect(server.listens()).toBe(1); // never reconnected

    // Normal again: written straight out and acknowledged.
    client.sendConversationEvent(turn('ev-4'));
    await until(() => ids(server.received).includes('ev-4') && client.undelivered().unacknowledged === 0);
    await client.disconnect();
  });

  async function live() {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'info').mockImplementation(() => {});
    const server = await startServer();
    cleanup.push(() => server.stop());
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: server.port, useTls: false, outboxPath: '',
      sleep: (ms) => new Promise((r) => setTimeout(r, ms >= 1000 ? 60 : 60)),
    });
    await client.connect();
    return { warn, server, client };
  }

  it('delivers 250 events sent during the hold, in order, once each', async () => {
    const { warn, server, client } = await live();
    client.sendConversationEvent(turn('ev-0'));
    await until(() => client.instanceNotReady !== null);
    for (let i = 1; i < 250; i += 1) client.sendConversationEvent(turn(`ev-${i}`));
    expect(client.undelivered().unacknowledged).toBe(250); // the 100 cap evicted nothing
    server.setReady(true);
    await until(() => client.undelivered().unacknowledged === 0 && client.instanceNotReady === null, 8000);
    const want = Array.from({ length: 250 }, (_, i) => `ev-${i}`);
    expect(new Set(server.accepted).size).toBe(250);
    expect(server.accepted).toHaveLength(250); // none duplicated
    expect(server.accepted).toEqual(want); // in order
    const text = warn.mock.calls.map((c) => String(c[0])).join('\n');
    expect(text).not.toContain('DROPPED');
    expect(text).not.toContain('Nothing is lost');
    await client.disconnect();
  });

  it('at the cap drops the oldest and says exactly how many', async () => {
    const { warn, server, client } = await live();
    client.sendConversationEvent(turn('ev-0'));
    await until(() => client.instanceNotReady !== null);
    for (let i = 1; i < NOT_READY_MAX_HELD + 100; i += 1) client.sendConversationEvent(turn(`ev-${i}`));
    expect(client.undelivered().unacknowledged).toBe(NOT_READY_MAX_HELD);
    server.setReady(true);
    await until(() => client.undelivered().unacknowledged === 0 && client.instanceNotReady === null, 20000);
    const drops = warn.mock.calls.map((c) => String(c[0])).filter((m) => m.includes('DROPPED'));
    expect(drops).toHaveLength(1);
    expect(drops[0]).toContain('100 held event(s)');
    expect(server.accepted).toHaveLength(NOT_READY_MAX_HELD);
    expect(server.accepted[0]).toBe('ev-100');
    await client.disconnect();
  }, 30000);
});
