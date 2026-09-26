/**
 * D12: a process killed mid-turn loses nothing after restart.
 *
 * Mirrors `python/tests/test_durability.py`. `delivery.test.ts` covers the two
 * in-memory buffers, which survive a dropped connection. Neither survives the
 * process: a hard kill between recording a turn and the server acknowledging
 * it took that turn with it, and nothing anywhere showed it had happened.
 *
 * A killed process is simulated the only honest way available: put events
 * through one client, throw it away WITHOUT calling disconnect(), and build a
 * second client on the same storage root. No flush, no teardown, which is what
 * a SIGKILL gives you. A test that disconnects first is testing the flush, not
 * the crash.
 *
 * These drive the journal and the buffers directly rather than standing up a
 * real server, because the question here is what is on disk across a restart,
 * and `delivery.test.ts` already covers the wire.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, appendFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';
import { OutboxJournal } from '../grpc/outbox-journal.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

let root: string;
beforeEach(async () => {
  // `node:fs` is resolved once per process and the stream client does it
  // inside connect(). These tests drive the journal directly, so they have
  // to do it themselves; without it every journal disables itself.
  await OutboxJournal.init();
  root = mkdtempSync(join(tmpdir(), 'synap-outbox-'));
});
afterEach(() => { rmSync(root, { recursive: true, force: true }); });

function client(): GrpcStreamClient {
  return new GrpcStreamClient(CREDS, new AnticipationCache(), { outboxPath: root });
}

function turn(id: string) {
  return {
    event_type: 'user_message',
    conversation_id: 'c1',
    user_id: 'u1',
    content: 'my card is not working',
    role: 'user',
    event_id: id,
  };
}

/** What the next start would put back into its send queue, in order. */
function restored(): string[] {
  const { queued, unacked } = new OutboxJournal(root, CREDS.instanceId).load();
  return [...unacked, ...queued].map((e) => String(e['event_id']));
}

describe('a killed process loses nothing', () => {
  it('a queued event comes back on the next start', () => {
    // No stream, so the event sits in the send queue and the process dies
    // there. Nothing is flushed and nothing is closed.
    const dead = client();
    dead.sendConversationEvent(turn('ev-1'));
    expect(restored()).toEqual(['ev-1']);
  });

  it('order is kept across the restart', () => {
    const dead = client();
    for (let i = 0; i < 5; i += 1) dead.sendConversationEvent(turn(`ev-${i}`));
    expect(restored()).toEqual(['ev-0', 'ev-1', 'ev-2', 'ev-3', 'ev-4']);
  });

  it('the id survives, so the server can deduplicate the replay', () => {
    // The whole retry story rests on the id being the SAME one. A restart that
    // mints a fresh id turns every recovered event into a duplicate turn,
    // which is worse than the loss it was fixing.
    const dead = client();
    dead.sendConversationEvent(turn('ev-keep-this-id'));
    expect(restored()).toEqual(['ev-keep-this-id']);
  });

  it('an event the buffer deliberately dropped is not resurrected', () => {
    // The depth and age bounds exist to stop an outage growing the buffer
    // without limit. Replaying what they evicted would quietly undo them.
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1' });
    j.recordDropped('ev-1');
    expect(restored()).toEqual([]);
  });

  it('an acknowledged event does not come back', () => {
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordUnacked({ event_id: 'ev-1' });
    j.recordAcked('ev-1');
    expect(restored()).toEqual([]);
  });

  it('a queued event later written is not replayed twice', () => {
    // One event, two lines. Restoring it from both lists would double the
    // turn, and the server's dedupe is the last line of defence, not the first.
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1' });
    j.recordUnacked({ event_id: 'ev-1' });
    expect(restored()).toEqual(['ev-1']);
  });

  it('a clean disconnect with nothing outstanding leaves nothing to replay', async () => {
    const c = client();
    await c.disconnect();
    expect(restored()).toEqual([]);
  });
});

describe('the journal itself', () => {
  it('a torn last line keeps everything before it', () => {
    // A kill mid-append leaves a partial line. Losing the whole file because
    // its last byte is missing turns a one-event loss into a hundred-event one.
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1' });
    j.recordQueued({ event_id: 'ev-2' });
    appendFileSync(j.path, '{"op":"q","e":{"event_id":"ev-3","cont', 'utf8');
    expect(restored()).toEqual(['ev-1', 'ev-2']);
  });

  it('compaction keeps what is outstanding and forgets the rest', () => {
    const j = new OutboxJournal(root, CREDS.instanceId);
    for (let i = 0; i < 20; i += 1) {
      j.recordUnacked({ event_id: `ev-${i}` });
      j.recordAcked(`ev-${i}`);
    }
    j.recordQueued({ event_id: 'still-here' });
    j.compact([{ event_id: 'still-here' }], []);
    expect(restored()).toEqual(['still-here']);
    expect(readFileSync(j.path, 'utf8').trim().split('\n')).toHaveLength(1);
  });

  it('an empty compaction removes the file', () => {
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1' });
    expect(existsSync(j.path)).toBe(true);
    j.compact([], []);
    expect(existsSync(j.path)).toBe(false);
  });

  it('two instances do not share a file', () => {
    // One file per instance id. Two instances in one process sharing a journal
    // would replay each other's events onto the wrong stream.
    new OutboxJournal(root, 'inst_aaa').recordQueued({ event_id: 'from-a' });
    new OutboxJournal(root, 'inst_bbb').recordQueued({ event_id: 'from-b' });
    expect(new OutboxJournal(root, 'inst_aaa').load().queued.map((e) => e['event_id']))
      .toEqual(['from-a']);
    expect(new OutboxJournal(root, 'inst_bbb').load().queued.map((e) => e['event_id']))
      .toEqual(['from-b']);
  });

  it('a record is one line, whatever the content holds', () => {
    // An append has to be one line, or a newline inside somebody's message
    // corrupts the record before it as well as the one being written.
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1', content: 'a\nb' });
    const lines = readFileSync(j.path, 'utf8').trim().split('\n');
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0] as string).e.content).toBe('a\nb');
  });

  it('no filesystem means no journal and no crash', () => {
    // Workers, edge runtimes and the browser. The SDK has to behave exactly as
    // it did before: buffered events survive a reconnect and not a restart.
    const j = new OutboxJournal(root, CREDS.instanceId, null);
    expect(j.enabled).toBe(false);
    j.recordQueued({ event_id: 'ev-1' });      // must not throw
    j.recordAcked('ev-1');
    j.compact([{ event_id: 'x' }], []);
    expect(j.load()).toEqual({ queued: [], unacked: [] });
  });

  it('a client with no filesystem still buffers and still sends', () => {
    const c = new GrpcStreamClient(CREDS, new AnticipationCache(), { outboxPath: '' });
    expect(() => c.sendConversationEvent(turn('ev-1'))).not.toThrow();
  });

  it('the file format matches Python byte for byte', () => {
    // Same root, same filename, same op letters, same field names. A Node
    // process and a Python process on one machine read each other's journals,
    // and the parity suites cannot see a file format.
    const j = new OutboxJournal(root, CREDS.instanceId);
    j.recordQueued({ event_id: 'ev-1' });
    j.recordUnacked({ event_id: 'ev-2' });
    j.recordAcked('ev-2');
    j.recordDropped('ev-3');
    const ops = readFileSync(j.path, 'utf8').trim().split('\n')
      .map((l) => JSON.parse(l) as { op: string });
    expect(ops.map((o) => o.op)).toEqual(['q', 'u', 'a', 'd']);
    expect(j.path.endsWith(`/outbox/${CREDS.instanceId}.jsonl`)).toBe(true);
  });
});

describe('the restore actually runs, against a real server', () => {
  // ⚠ The tests above drive the journal directly, so they prove the FILE is
  // right and nothing about the client reading it back. A mutation that
  // deleted `#restoreFromJournal`'s body entirely left all fifteen of them
  // green. These close that hole: they go through `connect()` and assert on
  // what the SERVER received, which is the only thing that settles it.
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => { for (const fn of cleanup.splice(0)) await fn(); });

  async function server() {
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
    const received: Array<Record<string, unknown>> = [];
    const srv = new grpc.Server();
    srv.addService(pkg.synap.v1.SynapService.service, {
      Listen: (call: { on: (e: string, h: (a?: unknown) => void) => void; end: () => void }) => {
        call.on('data', (m) => { received.push(m as Record<string, unknown>); });
        call.on('end', () => { call.end(); });
        call.on('error', () => { /* client hung up */ });
      },
      IngestTelemetry: () => { /* unused */ },
    });
    const port = await new Promise<number>((resolve, reject) => {
      srv.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(),
        (e, p) => { if (e) reject(e); else resolve(p); });
    });
    cleanup.push(() => new Promise<void>((r) => { srv.tryShutdown(() => r()); }));
    return { port, received };
  }

  function ids(received: Array<Record<string, unknown>>): string[] {
    return received
      .filter((m) => m['conversation_event'])
      .map((m) => String(
        (m['conversation_event'] as Record<string, unknown>)['event_id'] ?? ''));
  }

  it('a killed process\'s events go out on the next start', async () => {
    // Process one: no server to reach, so the events queue and the journal
    // takes them. It is then abandoned without disconnect(), which is what a
    // SIGKILL leaves behind.
    const dead = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: 1, useTls: false, outboxPath: root,
    });
    for (let i = 0; i < 3; i += 1) dead.sendConversationEvent(turn(`ev-${i}`));
    expect(restored()).toEqual(['ev-0', 'ev-1', 'ev-2']);

    // Process two: same storage root, a server that is actually up.
    const s = await server();
    const reborn = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: s.port, useTls: false, outboxPath: root,
    });
    await reborn.connect();
    await reborn.disconnect();

    expect(ids(s.received)).toEqual(['ev-0', 'ev-1', 'ev-2']);
  });

  it('a fresh journal replays nothing', async () => {
    const s = await server();
    const c = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1', port: s.port, useTls: false, outboxPath: root,
    });
    await c.connect();
    await c.disconnect();
    expect(ids(s.received)).toEqual([]);
  });
});
