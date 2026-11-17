/**
 * A6-01 on the SDK side: an error signal that names an event drops that event.
 * Python twin: python/tests/test_refused_event_signal.py.
 *
 * Real grpc-js server and the descriptor the client loads.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'i' };

async function server() {
  const pkgDef = protoLoader.fromJSON(SYNAP_PROTO_DESCRIPTOR as never, {
    keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true,
  });
  const pkg = grpc.loadPackageDefinition(pkgDef) as any;
  const received: any[] = [];
  let write: ((m: unknown) => void) | null = null;
  const srv = new grpc.Server();
  srv.addService(pkg.synap.v1.SynapService.service, {
    Listen(call: any) {
      write = (m) => call.write(m);
      call.on('data', (m: unknown) => received.push(m));
      call.on('end', () => call.end());
      call.on('error', () => {});
    },
    IngestTelemetry() {},
  });
  const port: number = await new Promise((res, rej) =>
    srv.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (e, p) => (e ? rej(e) : res(p))));
  return { port, received, push: (m: unknown) => write?.(m), stop: () => srv.forceShutdown() };
}

const until = async (p: () => boolean, ms = 3000) => {
  const end = Date.now() + ms;
  while (!p()) {
    if (Date.now() > end) throw new Error('timed out');
    await new Promise((r) => setTimeout(r, 10));
  }
};

const cleanup: Array<() => Promise<void> | void> = [];
afterEach(async () => { for (const c of cleanup.splice(0).reverse()) await c(); vi.restoreAllMocks(); });

describe('an error signal that names an event', () => {
  it('drops it from the unacknowledged set, tells the caller, and keeps the stream open', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = await server();
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(),
      { host: '127.0.0.1', port: s.port, useTls: false, outboxPath: '' } as never);
    cleanup.push(async () => { await client.disconnect(); s.stop(); });
    await client.connect();
    client.sendConversationEvent({ event_type: 'user_message', conversation_id: 'c1', user_id: 'u',
      content: 'hi', event_id: 'ev-1' } as never);
    await until(() => s.received.length === 1);
    expect(client.undelivered().unacknowledged).toBe(1);

    s.push({ signal: { signal_type: 'error', reason: 'customer_id is required', metadata: { event_id: 'ev-1' } } });
    await until(() => client.undelivered().unacknowledged === 0);
    expect(warn.mock.calls.flat().join(' ')).toContain('will not be retried');
    expect(client.isConnected).toBe(true);
  });

  it('with no event named keeps the old behaviour', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = await server();
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(),
      { host: '127.0.0.1', port: s.port, useTls: false, outboxPath: '' } as never);
    cleanup.push(async () => { await client.disconnect(); s.stop(); });
    await client.connect();
    client.sendConversationEvent({ event_type: 'user_message', conversation_id: 'c1', user_id: 'u',
      content: 'hi', event_id: 'ev-1' } as never);
    await until(() => s.received.length === 1);
    s.push({ signal: { signal_type: 'error', reason: 'nope', metadata: {} } });
    await until(() => warn.mock.calls.flat().join(' ').includes('stream stays open'));
    expect(client.undelivered().unacknowledged).toBe(1);
  });
});

describe('restoring from the journal', () => {
  it('still sends events journaled ten minutes ago after a restart', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dir = mkdtempSync(path.join(tmpdir(), 'synap-journal-'));
    cleanup.push(() => rmSync(dir, { recursive: true, force: true }));
    const old = Date.now() - 10 * 60 * 1000;

    // Process 1: sends an event the server never acknowledges, then goes away.
    const s1 = await server();
    const first = new GrpcStreamClient(CREDS, new AnticipationCache(),
      { host: '127.0.0.1', port: s1.port, useTls: false, outboxPath: dir } as never);
    await first.connect();
    first.sendConversationEvent({ event_type: 'user_message', conversation_id: 'c1', user_id: 'u',
      content: 'old', event_id: 'ev-old', sent_at_ms: old } as never);
    await until(() => s1.received.length === 1);
    await first.disconnect();
    s1.stop();

    // Process 2, ten minutes later: the journal's event goes out on its stream.
    const s2 = await server();
    const second = new GrpcStreamClient(CREDS, new AnticipationCache(),
      { host: '127.0.0.1', port: s2.port, useTls: false, outboxPath: dir } as never);
    cleanup.push(async () => { await second.disconnect(); s2.stop(); });
    await second.connect();
    await until(() => s2.received.length >= 1);
    expect(s2.received[0].conversation_event.event_id).toBe('ev-old');
  }, 15000);
});
