/**
 * GrpcStreamClient against a real local `Listen` server.
 *
 * What a mock cannot show: that connect() reports an unreachable host instead
 * of claiming to be connected, and that an open stream does not keep a
 * finished script alive (grpc-js refs the HTTP/2 session for as long as a call
 * is open).
 */
import net from 'node:net';
import { describe, it, expect, afterEach } from 'vitest';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { startGrpcServer, type TestServer } from './helpers/grpc-server.js';

const creds = { api_key: 'k', client_id: 'c', instance_id: 'i' };
const event = (content: string) => ({
  event_type: 'user_message', conversation_id: 'c1', user_id: 'u1', role: 'user', content,
  customer_id: '', session_id: '', metadata: {}, timestamp_ms: 1, search_queries: [], context_types: [],
});

/** Sockets to `port` that would keep the process alive. */
function refedSocketsTo(port: number): number {
  const handles = (process as unknown as { _getActiveHandles(): unknown[] })._getActiveHandles();
  return handles.filter((h) => h instanceof net.Socket && h.remotePort === port && (h as unknown as { hasRef(): boolean }).hasRef()).length;
}

describe('GrpcStreamClient', () => {
  let server: TestServer | null = null;
  let client: GrpcStreamClient | null = null;
  afterEach(async () => {
    await client?.disconnect();
    await server?.close();
    server = null; client = null;
  });

  it('connects, and a turn written to it reaches the server', async () => {
    server = await startGrpcServer();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined, { host: '127.0.0.1', port: server.port, useTls: false });
    await client.connect();
    expect(client.isConnected).toBe(true);
    expect(await client.sendConversationEvent(event('hello'))).toBe(true);
    for (let i = 0; i < 100 && server.events.length === 0; i++) await new Promise((r) => setTimeout(r, 10));
    expect(server.events[0]).toMatchObject({ event_type: 'user_message', content: 'hello' });
  });

  it('refuses to claim it is connected to a host that is not there', async () => {
    const dead = await startGrpcServer();
    const port = dead.port;
    await dead.close();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined, { host: '127.0.0.1', port, useTls: false });
    const started = Date.now();
    await expect(client.connect()).rejects.toThrow(/unreachable/);
    expect(Date.now() - started).toBeLessThan(3000);
    expect(client.isConnected).toBe(false);
    // And a write says so, so the caller can send it over REST.
    expect(await client.sendConversationEvent(event('lost?'))).toBe(false);
  });

  it('an open stream does not keep the process alive', async () => {
    server = await startGrpcServer();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined, { host: '127.0.0.1', port: server.port, useTls: false });
    await client.connect();
    await client.sendConversationEvent(event('hello'));
    await new Promise((r) => setTimeout(r, 50));
    expect(client.isConnected).toBe(true);
    expect(refedSocketsTo(server.port)).toBe(0);
  });

  it('an HTTP/2 session to any other host keeps its ref', async () => {
    server = await startGrpcServer();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined, { host: '127.0.0.1', port: server.port, useTls: false });
    await client.connect();
    const other = await startGrpcServer();
    // The module object grpc-js require()s; the ESM namespace is a snapshot.
    const http2 = (await import('node:http2')).default;
    const session = http2.connect(`http://127.0.0.1:${other.port}`);
    const ours = http2.connect(`http://127.0.0.1:${server.port}`);
    try {
      // The wrapper gives a Synap session its own no-op ref(); anything else
      // keeps the prototype's, which pins the process while a request is open.
      expect(Object.prototype.hasOwnProperty.call(session, 'ref')).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(ours, 'ref')).toBe(true);
    } finally {
      ours.destroy();
      session.destroy();
      await other.close();
    }
  });

  it('disconnect() closes the channel', async () => {
    server = await startGrpcServer();
    client = new GrpcStreamClient(creds, new AnticipationCache(), undefined, { host: '127.0.0.1', port: server.port, useTls: false });
    await client.connect();
    await client.disconnect();
    expect(client.isConnected).toBe(false);
    await new Promise((r) => setTimeout(r, 100));
    const handles = (process as unknown as { _getActiveHandles(): unknown[] })._getActiveHandles();
    expect(handles.filter((h) => h instanceof net.Socket && h.remotePort === server!.port && !h.destroyed)).toHaveLength(0);
  });
});
