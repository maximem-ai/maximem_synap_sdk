/**
 * A6-04: one stream failure causes exactly one reconnect.
 *
 * grpc-js emits 'error' and then 'end' for the same dead call, and both handlers
 * started a reconnect, so the client opened two streams and kept the second,
 * leaving the first live and unowned (later teardown closed the good one).
 *
 * Real grpc-js server, same descriptor the client loads.
 */
import { describe, it, expect } from 'vitest';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { AnticipationCache } from '../context/anticipation-cache.js';

async function serverThatFailsTheFirstStream() {
  const pkgDef = protoLoader.fromJSON(SYNAP_PROTO_DESCRIPTOR as never, {
    keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true,
  });
  const pkg = grpc.loadPackageDefinition(pkgDef) as any;
  let listens = 0;
  const open = new Set<number>();
  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen(call: any) {
      const id = ++listens;
      open.add(id);
      call.on('data', () => {});
      call.on('cancelled', () => open.delete(id));
      call.on('end', () => {});
      if (id === 1) {
        setTimeout(() => {
          open.delete(id);
          call.emit('error', { code: grpc.status.UNAVAILABLE, details: 'boom' });
        }, 100);
      }
    },
    IngestTelemetry() {},
  });
  const port: number = await new Promise((res, rej) =>
    server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (e, p) => (e ? rej(e) : res(p))));
  return { port, server, listens: () => listens, open };
}

describe('stream reconnect', () => {
  it('opens exactly one new stream after one failure', async () => {
    const s = await serverThatFailsTheFirstStream();
    const c = new GrpcStreamClient(
      { apiKey: 'k', clientId: 'c', instanceId: 'i' }, new AnticipationCache(),
      { host: '127.0.0.1', port: s.port, useTls: false, outboxPath: '',
        sleep: () => new Promise((r) => setTimeout(r, 20)) } as never,
    );
    await c.connect();
    await new Promise((r) => setTimeout(r, 1500));
    expect(s.listens()).toBe(2);
    expect(s.open.size).toBe(1);
    await c.disconnect();
    s.server.forceShutdown();
  }, 20000);
});
