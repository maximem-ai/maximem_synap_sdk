/**
 * A conversation id that is not a UUID and the UUID the server derives from it
 * are ONE conversation, for every local store.
 *
 * Mirrors `python/tests/test_conversation_id_normalisation.py`. The stream and
 * the server accept any string and derive `uuid5(NAMESPACE_URL, id)` from one
 * that is not a UUID; the REST reads need that UUID. The SDK buffered a
 * streamed turn in the short-term store under the raw id and read under the
 * derived one, so the next fetch did not carry the turn just said.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SynapClient } from '../client.js';
import { ShortTermStore } from '../cache/short-term-store.js';
import { unownedElsewhere } from '../context/anticipation-cache.js';
import { CompactionSubscribers } from '../conversation/interface.js';
import { TurnCounter } from '../context/user-summary.js';
import { normalizeConversationId } from '../util/conversation-id.js';
import * as registry from '../registry.js';

const RAW = 'pgv2_testconv72';
const DERIVED = normalizeConversationId(RAW);

/** [input, what the server's `coerce_conversation_id` returns], produced by that Python function. */
const SERVER: Array<[string, string]> = [
  ["pgv2_testconv72", "37cd0242-c7ca-556c-9455-bc1433387c27"],
  ["pgv2_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "d775d9d7-d8ce-5080-9bc7-ee5503f9e260"],
  ["conv_123", "cd0e4cf4-dbd7-51e2-9ddd-507811242407"],
  ["conv-abc-123", "2b6d8f45-2c28-501f-8ee2-f14f6d0c759a"],
  ["a", "e1407479-3136-56c0-9908-bb02fb0339e2"],
  [" ", "e0ccaace-c027-52a5-a807-2065497f8860"],
  [" leading", "41aadfe2-3164-58b9-b91f-f566d3970bcb"],
  ["trailing ", "58d512c0-3b4e-5722-b39b-29eb6e168319"],
  ["UPPER", "0e3f0a4d-5d1c-53ae-b7f9-d470182d58d8"],
  ["MiXeD_Case", "9706be50-b0a8-574f-98db-06c5d865d5bb"],
  ["with space", "58d42ded-a991-562a-b92b-ea7bd5175b18"],
  ["with/slash", "1e7053d3-f196-53b0-b412-b6b57dbc988a"],
  ["with?query=1&x=2", "a75d9d06-7323-574d-8fb0-97df573cbf75"],
  ["emoji_\ud83d\ude00", "0244d107-e84e-55c1-88d4-4d7248afa72c"],
  ["caf\u00e9", "79d4db4c-73a9-5cec-bd37-4fc862e2120d"],
  ["\u4f1a\u8bdd-1", "f76176e4-532c-5c2c-bf1d-c3c9c6aab69e"],
  ["0", "035c4ea0-d73b-5bde-bd6f-c806b04f2ec3"],
  ["12345", "a82b9066-2032-5bba-8bf9-8e865b09025c"],
  ["-----", "af24274c-5b62-58a2-bcd2-6abdbea174aa"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "cc9ebb9b-0982-50ad-96f7-ffb791ee8fc4"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "d7c769b2-da3b-59fa-bdd6-2f3e8c5ed303"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "3626e090-976f-515f-95a7-d3a479e5f76b"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "9d07bf9d-4264-5f1e-9682-bc55c8180b46"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "b84cc5b4-1bb5-52d8-b0b5-e6e5342d7368"],
  ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "2497c2dd-1eb2-577e-942e-f84f31eeae77"],
  ["3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5", "657de0d1-5805-589f-a49c-c7604df9a159"],
  ["3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5cd", "1e7a0ef9-dbd5-5af9-a2fc-1bc68378cb97"],
  ["gggggggggggggggggggggggggggggggg", "15edab69-7c9e-5004-bab7-a062eb5a653c"],
  ["urn:uuid:nope", "c2ff9bad-8ba1-5f18-9183-ee76b06693b4"],
  ["{not-a-uuid}", "a292a1a0-c6ed-5e3d-ab7d-762dde731359"],
  ["new\nline", "d7555da4-0424-5dd2-9929-a8bb335ebe9b"],
  ["tab\there", "b870aacd-4929-5a91-852a-579a6dbb2f9e"],
  ["http://example.com/c/1", "a92d34de-e114-5db8-acf0-eda3b51ad918"],
  ["session:abc:42", "9ca099c4-e2f7-55d0-8919-1a3439b2a760"],
  ["3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c", "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"],
  ["3F6B1A2C-4D5E-6F7A-8B9C-0D1E2F3A4B5C", "3F6B1A2C-4D5E-6F7A-8B9C-0D1E2F3A4B5C"],
  ["3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5c", "3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5c"],
  ["{3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c}", "{3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c}"],
  ["urn:uuid:3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c", "urn:uuid:3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"],
  ["uuid:3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c", "uuid:3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"],
  ["{{3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c}}", "{{3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c}}"],
  ["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
  ["+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "+aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
  [" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", " aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
  ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "],
  ["aaaaaaaa_aaaaaaaa_aaaaaaaa_aaaaaaa", "c23bd684-41a2-544b-9807-88a57015da6a"],
  ["----3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5c", "----3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5c"],
  ["3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c ", "85dd8049-89ff-5544-ae90-45ea977eb863"],
  ["-3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5", "80ee63b0-4fce-5805-bfd2-487809debd7a"],
];

beforeEach(() => { registry.clear(); });
afterEach(() => { registry.clear(); vi.doUnmock('../grpc/stream-client.js'); vi.resetModules(); });

describe('normalizeConversationId is the server function', () => {
  it('derives the UUID the server derives', () => {
    expect(DERIVED).toBe('37cd0242-c7ca-556c-9455-bc1433387c27');
  });

  it.each(SERVER)('%j', (input, expected) => {
    expect(normalizeConversationId(input)).toBe(expected);
    expect(normalizeConversationId(normalizeConversationId(input))).toBe(expected);
  });

  it('leaves empty values alone', () => {
    expect(normalizeConversationId('')).toBe('');
    expect(normalizeConversationId(undefined)).toBeUndefined();
    expect(normalizeConversationId(null)).toBeNull();
  });
});

describe('the local stores treat both spellings as one conversation', () => {
  it('short-term store: append under one, read under the other', () => {
    const s = new ShortTermStore();
    s.appendTurn(RAW, 'user', 'a');
    expect(s.get(DERIVED)?.recentTurns.map((t) => t.content)).toEqual(['a']);
    expect(s.size).toBe(1);
    s.noteServerCount(DERIVED, 4);
    expect(s.get(RAW)?.serverMessageCount).toBe(4);
    s.invalidate(DERIVED);
    expect(s.get(RAW)).toBeNull();
  });

  it('a server count named with the raw id reaches the derived entry', () => {
    const s = new ShortTermStore();
    s.appendTurn(DERIVED, 'user', 'a');
    s.noteServerCount(RAW, 4);
    expect(s.get(DERIVED)?.serverMessageCount).toBe(4);
  });

  it('a compaction bundle named with the derived id lands on the raw entry', () => {
    const s = new ShortTermStore();
    s.appendTurn(RAW, 'user', 'a');
    s.applyCompaction({ conversation_context: { conversation_id: DERIVED, summary: 'S', compacted_message_count: 0, total_message_count: 1 } });
    expect(s.size).toBe(1);
  });

  it('compaction subscribers hear either spelling', () => {
    const subs = new CompactionSubscribers();
    const heard: unknown[] = [];
    subs.add(RAW, (b) => { heard.push(b); });
    subs.dispatch(DERIVED, { x: 1 }, () => {});
    expect(heard).toEqual([{ x: 1 }]);
    expect(subs.removeAll(DERIVED)).toBe(1);
    const other: unknown[] = [];
    subs.add(DERIVED, (b) => { other.push(b); });
    subs.dispatch(RAW, { y: 2 }, () => {});
    expect(other).toEqual([{ y: 2 }]);
  });

  it('the turn cadence counts both spellings together', () => {
    const t = new TurnCounter();
    for (const id of [RAW, DERIVED, RAW, DERIVED, RAW]) t.increment(id);
    expect(t.shouldInject(DERIVED)).toBe(true);
  });

  it('an unstamped bundle tied to a conversation is the same conversation under either spelling', () => {
    expect(unownedElsewhere('_any', RAW, DERIVED)).toBe(false);
    expect(unownedElsewhere('_any', DERIVED, RAW)).toBe(false);
    expect(unownedElsewhere('_any', RAW, 'other')).toBe(true);
  });
});

const json = (b: unknown) =>
  new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });

describe('a turn sent on the stream is in the next read', () => {
  async function listening() {
    const controls: Array<Record<string, unknown>> = [];
    const fake = {
      isConnected: true, lastError: null,
      sendConversationEvent() {},
      sendSessionControl(i: Record<string, unknown>) { controls.push(i); },
      sendContextUsed() {}, sendContextAssembled() {},
      async connect() {}, async disconnect() {},
    };
    vi.doMock('../grpc/stream-client.js', () => ({
      GrpcStreamClient: class { isConnected = true; lastError = null; constructor() { return fake as never; } },
    }));
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async () =>
        json({ context: { facts: [], conversation_context: { summary: 's', recent_turns: [], compaction_id: 'c1' } } })
      ) as unknown as typeof fetch,
    });
    await c.instance.listen({ host: 'localhost', port: 1, useTls: false });
    return { c, controls };
  }

  it('the fetch under the derived UUID carries the turn sent under the raw id', async () => {
    const { c } = await listening();
    await c.instance.send_message({ content: 'the marker', conversation_id: RAW, user_id: 'u1' });
    const ctx = await c.conversation.context.fetch({ conversation_id: DERIVED, search_query: ['travel plans'] });
    const turns = (ctx.conversation_context?.['recent_turns'] ?? []) as Array<{ content: string }>;
    expect(turns.map((t) => t.content)).toEqual(['the marker']);
    await c.shutdown();
  });

  it("a session opened under the raw id ends under the derived one; the wire keeps the caller's spelling", async () => {
    const { c, controls } = await listening();
    await c.instance.send_message({ content: 'hi', conversation_id: RAW, user_id: 'u1' });
    await c.instance.end_session(DERIVED);
    expect(controls.map((x) => [x['action'], x['conversation_id']])).toEqual([['start', RAW], ['end', DERIVED]]);
    await c.shutdown();
  });
});
