/**
 * The local short-term store only answers when it holds the whole conversation.
 *
 * Mirrors Python's `synap/tests/sdk/test_short_term_completeness.py`, case for
 * case. `get_context_for_prompt` under `sdk_st_authoritative` used to render
 * from the store whenever it held anything at all. A process that had
 * restarted, or evicted the entry, or shared the conversation with another
 * process, held one turn and returned it as the conversation.
 *
 * The server now reports how many messages it holds, on every write and on
 * every compaction bundle. These tests pin what the store does with that number.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import { ShortTermStore, isComplete, type ShortTermTurn } from '../cache/short-term-store.js';
import { storedBatchRows } from '../conversation/interface.js';
import { overlayLocalRecentTurns } from '../context/overlay.js';
import type { RawContext } from '../context/types.js';
import * as registry from '../registry.js';

const CONV = '0b0f6a52-6f0e-4c58-9f0b-2f6f4d1f3a11';
const STAMP = '2026-10-08T12:00:00+00:00';

const roleOf = (n: number) => (n % 2 ? 'user' : 'assistant');

/** Append turn `n` the way the REST write does, with the server's count. */
function write(store: ShortTermStore, n: number, count?: number, inFlight = 0): void {
  store.appendTurn(CONV, roleOf(n), `turn ${n}`, undefined, {
    messageId: `m${n}`,
    serverMessageCount: count,
    inFlight,
  });
}

const serverTurn = (n: number): ShortTermTurn => ({
  role: roleOf(n), content: `turn ${n}`, timestamp: STAMP, message_id: `m${n}`,
});

const range = (from: number, to: number) =>
  Array.from({ length: to - from + 1 }, (_, i) => from + i);

function bundle(o: {
  compacted: number; total: number; tail: ShortTermTurn[]; compactionId?: string; at?: string;
}): Record<string, unknown> {
  return {
    bundle_type: 'compaction_update',
    anticipation_conversation_id: CONV,
    conversation_context: {
      conversation_id: CONV,
      summary: `summary of ${o.compacted}`,
      compaction_id: o.compactionId ?? 'cmp-1',
      compacted_at: o.at ?? '2026-10-08T12:10:00+00:00',
      recent_turns: o.tail.map((t) => ({ role: t.role, content: t.content, timestamp: t.timestamp })),
      compacted_message_count: o.compacted,
      total_message_count: o.total,
    },
  };
}

const contents = (store: ShortTermStore) => store.get(CONV)!.recentTurns.map((t) => t.content);
const complete = (store: ShortTermStore) => isComplete(store.get(CONV)!);

describe('when the store may answer', () => {
  it('a conversation recorded from its first turn is complete', () => {
    const store = new ShortTermStore();
    write(store, 1, 1);
    write(store, 2, 2);
    expect(complete(store)).toBe(true);
  });

  it('turns with no count from the server are never complete', () => {
    const store = new ShortTermStore();
    write(store, 1);
    write(store, 2);
    expect(complete(store)).toBe(false);
  });

  it('a process that joined late is not complete', () => {
    const store = new ShortTermStore();
    write(store, 7, 7);
    expect(complete(store)).toBe(false);
    expect(store.get(CONV)!.shared).toBe(false);
  });

  it('reseeding from the server makes it complete', () => {
    const store = new ShortTermStore();
    write(store, 7, 7);
    store.reseed(CONV, {
      compactionId: null, recentMessages: range(1, 7).map(serverTurn),
      compactedMessageCount: 0, totalMessageCount: 7,
    });
    expect(complete(store)).toBe(true);
    expect(contents(store)).toEqual(range(1, 7).map((n) => `turn ${n}`));
  });

  it('own streamed turns ahead of the count are still complete', () => {
    const store = new ShortTermStore();
    store.appendTurn(CONV, 'user', 'turn 1');
    store.noteServerCount(CONV, 1);
    store.appendTurn(CONV, 'assistant', 'turn 2');
    expect(complete(store)).toBe(true);
  });

  it('a count for an unknown conversation is ignored', () => {
    const store = new ShortTermStore();
    store.noteServerCount(CONV, 4);
    expect(store.get(CONV)).toBeNull();
  });
});

describe('another writer', () => {
  it('the server running ahead of a synced entry marks it shared', () => {
    const store = new ShortTermStore();
    write(store, 1, 1);
    write(store, 3, 3); // someone else wrote turn 2
    expect(store.get(CONV)!.shared).toBe(true);
    expect(complete(store)).toBe(false);
  });

  it('shared is not undone by a reseed', () => {
    const store = new ShortTermStore();
    write(store, 1, 1);
    write(store, 3, 3);
    store.reseed(CONV, {
      compactionId: null, recentMessages: [1, 2, 3].map(serverTurn),
      compactedMessageCount: 0, totalMessageCount: 3,
    });
    expect(complete(store)).toBe(false);
  });

  it("two of this process's own concurrent writes are not a second writer", () => {
    const store = new ShortTermStore();
    write(store, 1, 2, 1);
    expect(store.get(CONV)!.shared).toBe(false);
    write(store, 2, 2);
    expect(complete(store)).toBe(true);
    expect(store.get(CONV)!.shared).toBe(false);
  });
});

describe('compaction by count', () => {
  function synced(turns: number): ShortTermStore {
    const store = new ShortTermStore();
    for (const n of range(1, turns)) write(store, n, n);
    return store;
  }

  it('turns that arrived while compaction ran are kept', () => {
    const store = synced(10);
    store.applyCompaction(bundle({ compacted: 8, total: 10, tail: [serverTurn(9), serverTurn(10)] }));
    expect(contents(store)).toEqual(['turn 9', 'turn 10']);
    expect(store.get(CONV)!.compactedMessageCount).toBe(8);
    expect(complete(store)).toBe(true);
  });

  it('turns recorded after the bundle was built are kept behind the tail', () => {
    const store = synced(10);
    store.applyCompaction(bundle({ compacted: 8, total: 9, tail: [serverTurn(9)] }));
    expect(contents(store)).toEqual(['turn 9', 'turn 10']);
    expect(complete(store)).toBe(true);
  });

  it('a tail the summary already covers is not used', () => {
    const store = synced(8);
    store.applyCompaction(bundle({ compacted: 8, total: 8, tail: range(3, 8).map(serverTurn) }));
    expect(contents(store)).toEqual([]);
    expect(complete(store)).toBe(true);
  });

  it('a bundle brings a late joiner back in step', () => {
    const store = new ShortTermStore();
    write(store, 10, 10);
    store.applyCompaction(bundle({ compacted: 8, total: 10, tail: [serverTurn(9), serverTurn(10)] }));
    expect(contents(store)).toEqual(['turn 9', 'turn 10']);
    expect(complete(store)).toBe(true);
  });

  it('a bundle without counts leaves the entry out of sync', () => {
    const store = synced(4);
    store.applyCompaction(bundle({ compacted: 0, total: 0, tail: [] }));
    expect(complete(store)).toBe(false);
  });

  it('a process that only receives bundles is not complete', () => {
    const store = new ShortTermStore();
    store.applyCompaction(bundle({ compacted: 8, total: 9, tail: [serverTurn(9)] }));
    expect(complete(store)).toBe(false);
  });

  it('an older bundle does not replace a newer one', () => {
    const store = synced(8);
    store.applyCompaction(bundle({
      compacted: 8, total: 8, tail: [], compactionId: 'cmp-2', at: '2026-10-08T12:20:00+00:00',
    }));
    store.applyCompaction(bundle({
      compacted: 4, total: 4, tail: [], compactionId: 'cmp-1', at: '2026-10-08T12:10:00+00:00',
    }));
    expect(store.get(CONV)!.compactionId).toBe('cmp-2');
    expect(store.get(CONV)!.compactedMessageCount).toBe(8);
  });
});

describe('reseed', () => {
  it('does not create an entry', () => {
    const store = new ShortTermStore();
    store.reseed(CONV, {
      compactionId: null, recentMessages: [serverTurn(1)],
      compactedMessageCount: 0, totalMessageCount: 1,
    });
    expect(store.get(CONV)).toBeNull();
  });

  it('a summary the store does not hold keeps it out of sync', () => {
    const store = new ShortTermStore();
    write(store, 10, 10);
    store.reseed(CONV, {
      compactionId: 'cmp-9', recentMessages: [serverTurn(10)],
      compactedMessageCount: 9, totalMessageCount: 10,
    });
    expect(complete(store)).toBe(false);
  });

  it('a turn written after the server read is kept', () => {
    const store = new ShortTermStore();
    write(store, 7, 7);
    write(store, 8, 8);
    store.reseed(CONV, {
      compactionId: null, recentMessages: range(1, 7).map(serverTurn),
      compactedMessageCount: 0, totalMessageCount: 7,
    });
    expect(contents(store)).toEqual(range(1, 8).map((n) => `turn ${n}`));
    expect(complete(store)).toBe(true);
  });

  it('a turn the reseed brought in is not appended again', () => {
    const store = new ShortTermStore();
    write(store, 1, 3);
    store.reseed(CONV, {
      compactionId: null, recentMessages: [1, 2, 3].map(serverTurn),
      compactedMessageCount: 0, totalMessageCount: 3,
    });
    write(store, 3, 3);
    expect(contents(store)).toEqual(['turn 1', 'turn 2', 'turn 3']);
  });
});

describe('get_context_for_prompt', () => {
  beforeEach(() => { registry.clear(); });
  afterEach(() => { registry.clear(); delete process.env['SYNAP_SDK_ST_AUTHORITATIVE']; });

  /** The two REST calls the SDK makes, over one in-memory conversation. */
  function cloud() {
    const messages: ShortTermTurn[] = [];
    const state = { promptCalls: 0 };
    const json = (b: unknown) =>
      new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });
    const fetchImpl = (async (url: string, init?: { body?: string }) => {
      if (String(url).includes('/context-for-prompt')) {
        state.promptCalls += 1;
        return json({
          formatted_context: messages.map((m) => m.content).join('\n'),
          available: messages.length > 0,
          recent_messages: [...messages],
          recent_message_count: messages.length,
          compacted_message_count: 0,
          total_message_count: messages.length,
          compaction_id: null,
        });
      }
      const body = JSON.parse(init?.body ?? '{}') as { role?: string; content?: string };
      const n = messages.length + 1;
      messages.push({
        role: body.role ?? 'user', content: body.content ?? '', timestamp: STAMP, message_id: `m${n}`,
      });
      return json({
        message_id: `m${n}`, conversation_id: CONV, session_id: 's1',
        recorded_at: '2026-10-08T12:00:01+00:00', conversation_message_count: n,
      });
    }) as unknown as typeof fetch;
    return { state, fetchImpl };
  }

  const sdk = (fetchImpl: typeof fetch, options: Record<string, unknown> = {}) =>
    new SynapClient({
      apiKey: 'k', _force_new: true, clientId: 'cli_test', instanceId: 'inst_0123456789abcdef', fetchImpl, sdk_st_authoritative: true, ...options,
    });

  const say = (c: SynapClient, content: string) =>
    c.conversation.record_message({
      conversation_id: CONV, user_id: 'u1', customer_id: 'c1', role: 'user', content,
    });

  const prompt = async (c: SynapClient) =>
    String((await c.conversation.context.get_context_for_prompt({ conversation_id: CONV }))['formatted_context']);

  it('a complete store answers without the server', async () => {
    const { state, fetchImpl } = cloud();
    const c = sdk(fetchImpl);
    await say(c, 'alpha');
    await say(c, 'beta');
    const text = await prompt(c);
    expect(text).toContain('alpha');
    expect(text).toContain('beta');
    expect(state.promptCalls).toBe(0);
    await c.shutdown();
  });

  it('is off by default: with nothing set it asks the server every time', async () => {
    // Off because a process that reads before it writes cannot know that
    // another process recorded a turn in between.
    const { state, fetchImpl } = cloud();
    const c = new SynapClient({ apiKey: 'k', _force_new: true, clientId: 'cli_test', instanceId: 'inst_0123456789abcdef', fetchImpl });
    await say(c, 'alpha');
    expect(await prompt(c)).toContain('alpha');
    expect(await prompt(c)).toContain('alpha');
    expect(state.promptCalls).toBe(2);
    await c.shutdown();
  });

  it('the env var turns it on, and off over the option', async () => {
    process.env['SYNAP_SDK_ST_AUTHORITATIVE'] = '1';
    let { state, fetchImpl } = cloud();
    let c = new SynapClient({ apiKey: 'k', _force_new: true, clientId: 'cli_test', instanceId: 'inst_0123456789abcdef', fetchImpl });
    await say(c, 'alpha');
    await prompt(c);
    expect(state.promptCalls).toBe(0);
    await c.shutdown();

    process.env['SYNAP_SDK_ST_AUTHORITATIVE'] = '0';
    ({ state, fetchImpl } = cloud());
    c = sdk(fetchImpl);
    await say(c, 'alpha');
    await prompt(c);
    expect(state.promptCalls).toBe(1);
    await c.shutdown();
  });

  it('after a restart it asks the server once then answers locally', async () => {
    const { state, fetchImpl } = cloud();
    const before = sdk(fetchImpl);
    for (const word of ['alpha', 'beta', 'gamma']) await say(before, word);
    await before.shutdown();

    const after = sdk(fetchImpl);
    await say(after, 'delta');
    let text = await prompt(after);
    for (const word of ['alpha', 'beta', 'gamma', 'delta']) expect(text).toContain(word);
    expect(state.promptCalls).toBe(1);

    await say(after, 'epsilon');
    text = await prompt(after);
    for (const word of ['alpha', 'beta', 'gamma', 'delta', 'epsilon']) expect(text).toContain(word);
    expect(state.promptCalls).toBe(1);
    await after.shutdown();
  });
});

describe('a partly failed batch', () => {
  // `results` lists only what was stored. The failures are in `errors`, by index.
  const batch = (...contents: string[]) =>
    contents.map((content) => ({ conversation_id: CONV, role: 'user', content }));
  const row = (n: number) => ({ message_id: `m${n}`, conversation_message_count: n });
  const pairs = (bodies: Array<Record<string, unknown>>, result: unknown) =>
    storedBatchRows(bodies, result).map(([b, r]) => [b['content'], r['message_id']]);

  it('a clean batch pairs up in order', () => {
    const result = { results: [row(1), row(2), row(3)], errors: [] };
    expect(pairs(batch('a', 'b', 'c'), result)).toEqual([['a', 'm1'], ['b', 'm2'], ['c', 'm3']]);
  });

  it('a failure at the front does not shift the rest', () => {
    const result = { results: [row(1), row(2)], errors: [{ index: 0, error: 'nope' }] };
    expect(pairs(batch('fail', 'second', 'third'), result)).toEqual([
      ['second', 'm1'], ['third', 'm2'],
    ]);
  });

  it('a failure in the middle is skipped', () => {
    const result = { results: [row(1), row(2)], errors: [{ index: 1, error: 'nope' }] };
    expect(pairs(batch('a', 'fail', 'c'), result)).toEqual([['a', 'm1'], ['c', 'm2']]);
  });

  it('a result that cannot be lined up mirrors nothing', () => {
    expect(pairs(batch('a', 'b', 'c'), { results: [row(1)], errors: [] })).toEqual([]);
  });
});

describe('fetch overlay', () => {
  const fetched = (turns: ShortTermTurn[], compactionId: string | null = null): RawContext =>
    ({
      conversation_context: {
        summary: 'server summary', current_state: {}, key_extractions: {},
        recent_turns: turns.map((t) => ({ role: t.role, content: t.content, timestamp: t.timestamp })),
        compaction_id: compactionId,
      },
    }) as unknown as RawContext;

  const overlaid = (store: ShortTermStore, response: RawContext): string[] => {
    overlayLocalRecentTurns(response, store, CONV, true);
    const turns = (response.conversation_context?.['recent_turns'] ?? []) as ShortTermTurn[];
    return turns.map((t) => t.content);
  };

  it("a restarted process keeps the server's turns", () => {
    // It holds one turn. That used to replace the four the server sent.
    const store = new ShortTermStore();
    write(store, 4, 4);
    expect(overlaid(store, fetched(range(1, 4).map(serverTurn)))).toEqual(
      range(1, 4).map((n) => `turn ${n}`),
    );
  });

  it('a turn the server has not stored yet is still added', () => {
    const store = new ShortTermStore();
    store.appendTurn(CONV, 'user', 'just sent');
    expect(overlaid(store, fetched([]))).toEqual(['just sent']);
  });

  it("a turn sent after a restart is added behind the server's", () => {
    const store = new ShortTermStore();
    store.appendTurn(CONV, 'user', 'just sent');
    expect(overlaid(store, fetched(range(1, 3).map(serverTurn)))).toEqual([
      'turn 1', 'turn 2', 'turn 3', 'just sent',
    ]);
  });

  it('a complete store supplies the whole tail', () => {
    const store = new ShortTermStore();
    for (const n of range(1, 3)) write(store, n, n);
    expect(overlaid(store, fetched([serverTurn(1)]))).toEqual(['turn 1', 'turn 2', 'turn 3']);
  });

  it('a complete store under an older summary does not replace the tail', () => {
    const store = new ShortTermStore();
    for (const n of range(1, 4)) write(store, n, n);
    expect(overlaid(store, fetched([serverTurn(4)], 'cmp-new'))).toEqual(['turn 4']);
  });

  it("a shared conversation keeps the server's turns", () => {
    const store = new ShortTermStore();
    write(store, 1, 1);
    write(store, 3, 3); // another process wrote turn 2
    expect(overlaid(store, fetched([1, 2, 3].map(serverTurn)))).toEqual([
      'turn 1', 'turn 2', 'turn 3',
    ]);
  });

  it("does not put the server's id for a turn into the response", () => {
    const store = new ShortTermStore();
    write(store, 1, 1);
    const response = fetched([]);
    overlayLocalRecentTurns(response, store, CONV, true);
    const turns = (response.conversation_context?.['recent_turns'] ?? []) as object[];
    expect(Object.keys(turns[0] as object).sort()).toEqual(['content', 'role', 'timestamp']);
  });
});
