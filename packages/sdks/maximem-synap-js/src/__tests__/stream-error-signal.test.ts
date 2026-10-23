/**
 * A server error signal is logged and does not end the stream.
 *
 * The server sends `signal_type: "error"` for one refused event or
 * session_start and keeps the stream open. Python tore the stream down for
 * it and reconnected; JS dropped it without a word. Both now log it, and the
 * stream keeps delivering: the bundle sent after the error still arrives.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import * as registry from '../registry.js';

afterEach(() => { registry.clear(); vi.restoreAllMocks(); });

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

describe('server error signal', () => {
  it('is logged and the stream keeps delivering', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const server = await startServer();
    const client = new SynapClient({
      _force_new: true, apiKey: 'k', clientId: 'cli', instanceId: 'inst_0123456789abcdef',
      fetchImpl: (async () => new Response('{}')) as never,
    });
    await client.instance.listen({ host: '127.0.0.1', port: server.port, use_tls: false });
    await server.connected;

    server.push({ signal: { signal_type: 'error', reason: 'customer_id is required on a B2B instance.' } });
    server.push({
      context_bundle: {
        bundle_id: 'b-after', bundle_type: 'anticipation', anticipation_user_id: 'u1',
        items_by_type: { facts: { items: [{ item_id: 'i1', content: 'still delivering', scope: 'user' }] } },
      },
    });
    await until(() => client.anticipation_cache_snapshot().total_entries > 0);

    expect(warn.mock.calls.some((c) => String(c[0]).includes('customer_id is required on a B2B instance'))).toBe(true);
    expect(client.instance.is_listening).toBe(true);

    await client.instance.stop_listening();
    await client.shutdown();
    await server.stop();
  });
});
