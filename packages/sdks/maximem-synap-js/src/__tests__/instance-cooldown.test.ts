/**
 * Client-side cooldown for `instance_not_ready`, and `instance_build_failed`.
 *
 * Mirrors `python/tests/test_instance_cooldown.py`.
 *
 * After the server refuses a memory call with 409 `instance_not_ready` the SDK
 * throws the same error locally, with no network call, until the server's wait
 * (capped at 300 s) has passed. `instance_build_failed` is not time based: it
 * parses to its own error, sets no cooldown and is never retried.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { AnticipationCache } from '../context/anticipation-cache.js';
import {
  GrpcStreamClient, NOT_READY_MAX_HELD,
  BUILD_FAILED_PROBE_INTERVAL_MS, BUILD_FAILED_PROBE_JITTER_MS,
} from '../grpc/stream-client.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';
import {
  ConflictError, InstanceBuildFailedError, InstanceNotReadyError, SynapPermanentError,
} from '../errors.js';
import { HttpTransport, isGatedPath } from '../transport/http.js';

function notReadyBody(wait: number | null = 120) {
  const flat: Record<string, unknown> = {
    error: 'instance_not_ready',
    message: "This instance's memory architecture is still being built.",
    state: 'building', instance_id: 'inst_1',
  };
  if (wait !== null) flat['retry_after_seconds'] = wait;
  return { ...flat, detail: { code: 'instance_not_ready', ...flat } };
}

function failedBody() {
  const flat = {
    error: 'instance_build_failed', message: 'The build failed.', state: 'failed',
    instance_id: 'inst_1', attempts: 3, hint: 'Run: maximem-synap instances rebuild inst_1',
  };
  return { ...flat, detail: { code: 'instance_build_failed', ...flat } };
}

type Reply = [number, unknown];

/** Fake network: counts requests, plays back a script, then answers 200. */
function fake(...script: Reply[]) {
  const seen: Array<{ method: string; url: string }> = [];
  const queue = [...script];
  let fallback: Reply = [200, { ok: true }];
  const fetchImpl = (async (url: string, init: RequestInit) => {
    seen.push({ method: String(init.method), url: new URL(url).pathname });
    const [status, body] = queue.shift() ?? fallback;
    return new Response(JSON.stringify(body), {
      status, headers: { 'content-type': 'application/json' },
    });
  }) as unknown as typeof fetch;
  return { fetchImpl, seen, setFallback: (r: Reply) => { fallback = r; } };
}

function make(f: ReturnType<typeof fake>, opts: { retry?: boolean; instanceId?: string } = {}) {
  const clock = { now: 1000 };
  const transport = new HttpTransport({
    credentials: { apiKey: 'k', clientId: 'c', instanceId: opts.instanceId ?? 'inst_1' },
    fetchImpl: f.fetchImpl,
    clock: () => clock.now,
    retryPolicy: opts.retry
      ? { maxAttempts: 3, backoffBase: 0, backoffMax: 0, backoffJitter: false }
      : null,
  });
  return { transport, clock };
}

const write = (t: HttpTransport) => t.request('memories_create', { body: {} });

async function reject(p: Promise<unknown>): Promise<unknown> {
  try { await p; } catch (e) { return e; }
  throw new Error('expected a rejection');
}

describe('cooldown after instance_not_ready', () => {
  it('refuses the second call locally with no network call', async () => {
    const f = fake([409, notReadyBody(120)]);
    const { transport } = make(f);
    const first = await reject(write(transport)) as InstanceNotReadyError;
    expect(first.fromCooldown).toBe(false);
    expect(first.retryAfterSeconds).toBe(120);
    const e = await reject(write(transport)) as InstanceNotReadyError;
    expect(f.seen).toHaveLength(1);
    expect(e).toBeInstanceOf(InstanceNotReadyError);
    expect(e).toBeInstanceOf(ConflictError);
    expect(e.fromCooldown).toBe(true);
    expect(e.retryAfterSeconds).toBe(120);
    expect(e.message).toBe(first.message);
    expect(e.instanceId).toBe('inst_1');
    expect(e.state).toBe('building');
    expect(e.code).toBe('instance_not_ready');
  });

  it('reports the remaining wait, rounded up, never below 1', async () => {
    const f = fake([409, notReadyBody(120)]);
    const { transport, clock } = make(f);
    await reject(write(transport));
    clock.now += 30_400;
    expect(((await reject(write(transport))) as InstanceNotReadyError).retryAfterSeconds).toBe(90);
    clock.now += 89_000;
    expect(((await reject(write(transport))) as InstanceNotReadyError).retryAfterSeconds).toBe(1);
  });

  it('sends a real request once the wait has passed', async () => {
    const f = fake([409, notReadyBody(60)]);
    const { transport, clock } = make(f);
    await reject(write(transport));
    clock.now += 60_000;
    await expect(write(transport)).resolves.toEqual({ ok: true });
    expect(f.seen).toHaveLength(2);
  });

  it('is cleared by a successful call that was already in flight', async () => {
    let release: () => void = () => {};
    const hold = new Promise<void>((r) => { release = r; });
    let n = 0;
    const fetchImpl = (async () => {
      n += 1;
      if (n === 1) { await hold; return new Response('{"ok":true}', { status: 200 }); }
      if (n === 2) return new Response(JSON.stringify(notReadyBody(300)), { status: 409 });
      return new Response('{"ok":true}', { status: 200 });
    }) as unknown as typeof fetch;
    const transport = new HttpTransport({
      credentials: { apiKey: 'k', clientId: 'c', instanceId: 'inst_1' }, fetchImpl,
      retryPolicy: null, clock: () => 1000,
    });
    const a = write(transport);
    await Promise.resolve();
    await reject(write(transport)); // B: refused, arms the cooldown
    expect(((await reject(write(transport))) as InstanceNotReadyError).fromCooldown).toBe(true);
    expect(n).toBe(2);
    release();
    await a; // A succeeds: the instance is ready
    await expect(write(transport)).resolves.toEqual({ ok: true });
    expect(n).toBe(3);
  });

  it('caps the stored hint at 300 s', async () => {
    const f = fake([409, notReadyBody(5000)]);
    const { transport } = make(f);
    expect(((await reject(write(transport))) as InstanceNotReadyError).retryAfterSeconds).toBe(5000);
    expect(((await reject(write(transport))) as InstanceNotReadyError).retryAfterSeconds).toBe(300);
  });

  it('sets no cooldown when the server gave no hint', async () => {
    const f = fake([409, notReadyBody(null)]);
    const { transport } = make(f);
    await reject(write(transport));
    await write(transport);
    expect(f.seen).toHaveLength(2);
  });

  it('is per instance id', async () => {
    const f = fake([409, notReadyBody(120)]);
    const { transport } = make(f);
    await reject(write(transport));
    transport.updateCredentials({ instanceId: 'inst_2' });
    await expect(write(transport)).resolves.toEqual({ ok: true });
    expect(f.seen).toHaveLength(2);
  });

  it('is not shared between two transports', async () => {
    const a = fake([409, notReadyBody(120)]);
    const b = fake();
    await reject(write(make(a).transport));
    await write(make(b).transport);
    expect(b.seen).toHaveLength(1);
  });

  it('serves 50 concurrent callers the error with one request in total', async () => {
    const f = fake([409, notReadyBody(120)]);
    const { transport } = make(f);
    await reject(write(transport));
    const results = await Promise.allSettled(Array.from({ length: 50 }, () => write(transport)));
    expect(results.every((r) => r.status === 'rejected'
      && (r.reason as InstanceNotReadyError).fromCooldown === true)).toBe(true);
    expect(f.seen).toHaveLength(1);
  });

  it('survives concurrent first calls without error', async () => {
    const f = fake();
    f.setFallback([409, notReadyBody(120)]);
    const { transport } = make(f);
    const results = await Promise.allSettled(Array.from({ length: 20 }, () => write(transport)));
    expect(results.every((r) => r.status === 'rejected'
      && r.reason instanceof InstanceNotReadyError)).toBe(true);
    const after = await reject(write(transport)) as InstanceNotReadyError;
    expect(after.fromCooldown).toBe(true);
  });

  it('is never retried by the retry policy', async () => {
    const f = fake([409, notReadyBody(120)]);
    const { transport } = make(f, { retry: true });
    await reject(write(transport));
    expect(f.seen).toHaveLength(1);
  });
});

describe('which paths are gated', () => {
  const gated: Array<[string, string]> = [
    ['POST', '/v1/context/conversation/fetch'], ['POST', '/v1/context/user/fetch'],
    ['POST', '/v1/context/customer/fetch'], ['POST', '/v1/context/client/fetch'],
    ['POST', '/v1/conversations/messages'], ['POST', '/v1/conversations/messages/batch'],
    ['POST', '/v1/conversations/ingest'], ['POST', '/v1/events/batch'],
    ['POST', '/api/v1/memories/create'], ['POST', '/api/v1/memories/batch'],
    ['POST', '/api/v1/memories/upload'], ['POST', '/v1/migrations'],
    ['POST', '/v1/migrations/abc/run'], ['POST', '/v1/migrations/abc/approve'],
  ];
  const open: Array<[string, string]> = [
    ['GET', '/api/v1/auth/whoami'], ['GET', '/health'], ['POST', '/api/v1/auth/init'],
    ['GET', '/api/v1/memories/status/ing_1'], ['GET', '/api/v1/memories/m1'],
    ['POST', '/api/v1/memories/forget'], ['GET', '/v1/credits/balance'],
    ['POST', '/v1/telemetry/batch'], ['POST', '/v1/conversations/compact'],
    ['GET', '/v1/conversations/c1/compaction/status'], ['GET', '/v1/conversations/messages'],
    ['GET', '/v1/users/u1/profile'], ['POST', '/v1/conversations/c1/messages'],
    ['GET', '/v1/migrations'],
  ];

  it('matches exactly the routes the server gates', () => {
    for (const [m, p] of gated) expect(isGatedPath(m, p), `${m} ${p}`).toBe(true);
    for (const [m, p] of open) expect(isGatedPath(m, p), `${m} ${p}`).toBe(false);
  });

  it('lets non-gated endpoints through to the network during a cooldown', async () => {
    const f = fake([409, notReadyBody(300)]);
    const { transport } = make(f);
    await reject(write(transport));
    await transport.request('whoami');
    await transport.request('credits_balance');
    await transport.request('memories_status', { pathParams: { ingestion_id: 'i1' } });
    expect(f.seen).toHaveLength(4);
    expect(((await reject(write(transport))) as InstanceNotReadyError).fromCooldown).toBe(true);
  });

  it('a 409 on a non-gated endpoint sets no cooldown', async () => {
    const f = fake([409, notReadyBody(300)]);
    const { transport } = make(f);
    await reject(transport.request('whoami'));
    await write(transport);
    expect(f.seen).toHaveLength(2);
  });
});

describe('HTTP 409 instance_build_failed', () => {
  it('is parsed with its fields', async () => {
    const f = fake([409, failedBody()]);
    const { transport } = make(f);
    const e = await reject(write(transport)) as InstanceBuildFailedError;
    expect(e).toBeInstanceOf(InstanceBuildFailedError);
    expect(e).toBeInstanceOf(ConflictError);
    expect(e).toBeInstanceOf(SynapPermanentError);
    expect(e).not.toBeInstanceOf(InstanceNotReadyError);
    expect(e.instanceId).toBe('inst_1');
    expect(e.attempts).toBe(3);
    expect(e.hint).toBe('Run: maximem-synap instances rebuild inst_1');
    expect(e.message).toBe('The build failed.');
    expect(e.code).toBe('instance_build_failed');
  });

  it('reads the detail copy alone', async () => {
    const f = fake([409, { detail: failedBody().detail }]);
    const { transport } = make(f);
    expect(((await reject(write(transport))) as InstanceBuildFailedError).attempts).toBe(3);
  });

  it('sets no cooldown and is never retried', async () => {
    const f = fake([409, failedBody()], [409, failedBody()]);
    const { transport } = make(f, { retry: true });
    await reject(write(transport));
    expect(f.seen).toHaveLength(1); // not retried
    await reject(write(transport));
    expect(f.seen).toHaveLength(2); // every call is sent
    await expect(write(transport)).resolves.toEqual({ ok: true }); // rebuilt: works
  });
});

// ---------------------------------------------------------------- the stream

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

type Mode = 'failed' | 'building' | 'ready';

async function startServer() {
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
  let mode: Mode = 'failed';
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
        if (mode === 'failed') {
          call.write({ signal: {
            signal_type: 'throttle', reason: 'instance_build_failed: attempts exhausted',
            metadata: {
              code: 'instance_build_failed', state: 'failed', series: '1',
              instance_id: 'inst_0123456789abcdef', attempts: '3',
              hint: 'Run: maximem-synap instances rebuild inst_0123456789abcdef',
            },
          } });
        } else if (mode === 'building') {
          call.write({ signal: {
            signal_type: 'throttle', reason: 'instance_not_ready: building',
            metadata: {
              code: 'instance_not_ready', state: 'building',
              retry_after_seconds: '60', instance_id: 'inst_0123456789abcdef',
            },
          } });
        } else {
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
    port, received, listens: () => listens, setMode: (m: Mode) => { mode = m; },
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

describe('throttle instance_build_failed on the live stream', () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.splice(0)) await fn();
    vi.restoreAllMocks();
  });

  async function live() {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'info').mockImplementation(() => {});
    const server = await startServer();
    cleanup.push(() => server.stop());
    const sleeps: number[] = [];
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: server.port, useTls: false, outboxPath: '',
      sleep: (ms) => { sleeps.push(ms); return new Promise((r) => setTimeout(r, 60)); },
    });
    await client.connect();
    return { warn, server, client, sleeps };
  }

  it('holds events, warns ONCE with the hint, probes slowly, never reconnects, then recovers', async () => {
    const { warn, server, client, sleeps } = await live();
    client.sendConversationEvent(turn('ev-1'));
    client.sendConversationEvent(turn('ev-2'));
    await until(() => client.instanceNotReady !== null);

    expect(client.undelivered().unacknowledged).toBe(2);
    expect(client.currentState).toBe('connected');
    expect(client.instanceNotReady?.code).toBe('instance_build_failed');

    // Several probes, each refused again: still one warning, still one connection.
    await until(() => ids(server.received).length >= 6);
    const failedWarnings = warn.mock.calls.map((c) => String(c[0]))
      .filter((m) => m.includes('build FAILED'));
    expect(failedWarnings).toHaveLength(1);
    expect(failedWarnings[0]).toContain('maximem-synap instances rebuild inst_0123456789abcdef');
    expect(server.listens()).toBe(1);

    // The waits asked for are the slow fixed interval plus jitter, not the 5-120 s hint range.
    const probeWaits = sleeps.filter((ms) => ms >= 1000 && ms !== 3000); // 3000 = replay grace
    expect(probeWaits.length).toBeGreaterThan(0);
    for (const ms of probeWaits) {
      expect(ms).toBeGreaterThanOrEqual(BUILD_FAILED_PROBE_INTERVAL_MS);
      expect(ms).toBeLessThanOrEqual(BUILD_FAILED_PROBE_INTERVAL_MS + BUILD_FAILED_PROBE_JITTER_MS);
    }

    // A rebuild succeeded: the next probe is accepted and the stream resumes by itself.
    server.setMode('ready');
    await until(() => client.instanceNotReady === null && client.undelivered().unacknowledged === 0);
    expect(server.listens()).toBe(1);
    client.sendConversationEvent(turn('ev-3'));
    await until(() => ids(server.received).includes('ev-3') && client.undelivered().unacknowledged === 0);
    await client.disconnect();
  });

  it('keeps the 5000 cap and the real value is 5000', async () => {
    expect(NOT_READY_MAX_HELD).toBe(5000);
    const { server, client } = await live();
    client.sendConversationEvent(turn('ev-0'));
    await until(() => client.instanceNotReady !== null);
    for (let i = 1; i < NOT_READY_MAX_HELD + 50; i += 1) client.sendConversationEvent(turn(`ev-${i}`));
    expect(client.undelivered().unacknowledged).toBe(NOT_READY_MAX_HELD);
    server.setMode('ready');
    await until(() => client.undelivered().unacknowledged === 0 && client.instanceNotReady === null, 20000);
    await client.disconnect();
  }, 30000);

  it('instance_not_ready still uses its own message and hint', async () => {
    const { warn, server, client, sleeps } = await live();
    server.setMode('building');
    client.sendConversationEvent(turn('ev-1'));
    await until(() => client.instanceNotReady !== null);
    await until(() => ids(server.received).length >= 3);
    const msgs = warn.mock.calls.map((c) => String(c[0]));
    expect(msgs.filter((m) => m.includes('still building'))).toHaveLength(1);
    expect(msgs.some((m) => m.includes('build FAILED'))).toBe(false);
    expect(client.instanceNotReady?.code).toBe('instance_not_ready');
    expect(sleeps.length).toBeGreaterThan(0);
    server.setMode('ready');
    await until(() => client.instanceNotReady === null && client.undelivered().unacknowledged === 0);
    await client.disconnect();
  });
});
