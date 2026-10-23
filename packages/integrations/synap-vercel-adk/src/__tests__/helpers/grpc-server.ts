/**
 * A local Synap `Listen` server for tests: answers heartbeats and records every
 * StreamEvent the client writes. Insecure, on an ephemeral port.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

const here = path.dirname(fileURLToPath(import.meta.url));
const PROTO = path.resolve(here, '../../../proto/synap_service.proto');

export interface TestServer {
  port: number;
  /** Every `conversation_event` received, in order. */
  events: Array<Record<string, unknown>>;
  /** Number of Listen streams opened. */
  streams: () => number;
  close: () => Promise<void>;
}

export async function startGrpcServer(): Promise<TestServer> {
  const def = protoLoader.loadSync(PROTO, {
    keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true,
  });
  const pkg = grpc.loadPackageDefinition(def) as unknown as {
    synap: { v1: { SynapService: { service: grpc.ServiceDefinition } } };
  };
  const events: Array<Record<string, unknown>> = [];
  let streams = 0;
  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen(call: grpc.ServerDuplexStream<Record<string, unknown>, unknown>) {
      streams++;
      call.on('data', (msg: Record<string, unknown>) => {
        if (msg['conversation_event']) events.push(msg['conversation_event'] as Record<string, unknown>);
        if (msg['heartbeat_ping']) call.write({ heartbeat_pong: { timestamp_ms: Date.now() } });
      });
      call.on('end', () => call.end());
      call.on('error', () => undefined);
    },
    IngestTelemetry(call: grpc.ServerReadableStream<unknown, unknown>, cb: grpc.sendUnaryData<unknown>) {
      call.on('data', () => undefined);
      call.on('end', () => cb(null, { status: 'ok', events_received: 0 }));
    },
  } as never);

  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p));
  });
  return {
    port, events, streams: () => streams,
    close: () => new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); setTimeout(() => { server.forceShutdown(); resolve(); }, 500).unref(); }),
  };
}
