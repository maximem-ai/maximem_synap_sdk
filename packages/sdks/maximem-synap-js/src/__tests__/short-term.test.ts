import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import { ShortTermStore, parseIso } from '../cache/short-term-store.js';
import { budgetRecentTurns, verbatimOverlayEnabled, overlayLocalRecentTurns } from '../context/overlay.js';
import * as registry from '../registry.js';
import type { RawContext } from '../context/types.js';

const UUID = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
const json = (b: unknown) =>
  new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });

beforeEach(() => { registry.clear(); });
afterEach(() => { registry.clear(); delete process.env['SYNAP_ST_VERBATIM_OVERLAY']; });

describe('short-term store', () => {
  it('buffers turns per conversation', () => {
    const s = new ShortTermStore();
    s.appendTurn('c1', 'user', 'first');
    s.appendTurn('c1', 'assistant', 'second');
    s.appendTurn('c2', 'user', 'other');
    expect(s.get('c1')?.recentTurns.map((t) => t.content)).toEqual(['first', 'second']);
    expect(s.get('c2')?.recentTurns).toHaveLength(1);
    expect(s.size).toBe(2);
  });

  it('ignores an empty conversation id rather than bucketing under ""', () => {
    const s = new ShortTermStore();
    s.appendTurn('', 'user', 'orphan');
    expect(s.size).toBe(0);
    expect(s.get('')).toBeNull();
  });

  it('age-evicts on read', () => {
    let now = 1_000_000;
    const s = new ShortTermStore({ maxAgeMs: 1000, now: () => now });
    s.appendTurn('c1', 'user', 'x', new Date(now).toISOString());
    expect(s.get('c1')).not.toBeNull();
    now += 1001;
    expect(s.get('c1')).toBeNull();
    expect(s.size).toBe(0);
  });

  it('evicts least-recently-used past the cap', () => {
    const s = new ShortTermStore({ maxConversations: 2 });
    s.appendTurn('a', 'user', '1');
    s.appendTurn('b', 'user', '2');
    s.get('a');                       // touch 'a', so 'b' becomes oldest
    s.appendTurn('c', 'user', '3');
    expect(s.get('b')).toBeNull();
    expect(s.get('a')).not.toBeNull();
    expect(s.get('c')).not.toBeNull();
  });

  describe('applyCompaction', () => {
    it('prunes turns the compaction covers, keeping later ones', () => {
      const s = new ShortTermStore();
      s.appendTurn('c1', 'user', 'before', '2026-01-01T10:00:00Z');
      s.appendTurn('c1', 'user', 'during', '2026-01-01T12:00:00Z');
      s.applyCompaction({
        conversation_context: {
          conversation_id: 'c1',
          summary: 'a summary',
          compaction_id: 'comp-1',
          end_timestamp: '2026-01-01T11:00:00Z',
        },
      });
      const entry = s.get('c1');
      // Turns after the cutoff survive by design: the server has not seen them.
      expect(entry?.recentTurns.map((t) => t.content)).toEqual(['during']);
      expect(entry?.summary).toBe('a summary');
      expect(entry?.compactionId).toBe('comp-1');
    });

    it('keeps a turn whose timestamp will not parse', () => {
      // Dropping it would silently lose a message, which is worse than briefly
      // showing one the server already folded in.
      const s = new ShortTermStore();
      s.appendTurn('c1', 'user', 'unparseable', 'not-a-timestamp');
      s.applyCompaction({
        conversation_context: { conversation_id: 'c1', end_timestamp: '2026-01-01T11:00:00Z' },
      });
      expect(s.get('c1')?.recentTurns).toHaveLength(1);
    });

    it('does not let a blank server field erase what is held', () => {
      const s = new ShortTermStore();
      s.applyCompaction({
        conversation_context: { conversation_id: 'c1', summary: 'real summary', compaction_id: 'c-1' },
      });
      s.applyCompaction({ conversation_context: { conversation_id: 'c1', summary: '' } });
      expect(s.get('c1')?.summary).toBe('real summary');
      expect(s.get('c1')?.compactionId).toBe('c-1');
    });

    it('falls back to the anticipation conversation id', () => {
      const s = new ShortTermStore();
      s.applyCompaction({
        anticipation_conversation_id: 'c9',
        conversation_context: { summary: 'from a bundle' },
      });
      expect(s.get('c9')?.summary).toBe('from a bundle');
    });

    it('ignores a bundle with no conversation id at all', () => {
      const s = new ShortTermStore();
      s.applyCompaction({ conversation_context: { summary: 'nowhere to put this' } });
      expect(s.size).toBe(0);
    });
  });

  it('parses ISO stamps including a Z suffix', () => {
    expect(parseIso('2026-01-01T00:00:00Z')).toBe(Date.parse('2026-01-01T00:00:00Z'));
    expect(parseIso('2026-01-01T00:00:00+00:00')).toBe(Date.parse('2026-01-01T00:00:00Z'));
    for (const bad of [null, undefined, '', 'nonsense', 42]) expect(parseIso(bad)).toBeNull();
  });
});

describe('overlay budgeting', () => {
  const turn = (content: string) => ({ role: 'user', content, timestamp: '2026-01-01T00:00:00Z' });

  it('caps at the server window of 20 turns', () => {
    const turns = Array.from({ length: 30 }, (_, i) => turn(`turn ${i}`));
    const budgeted = budgetRecentTurns(turns);
    expect(budgeted).toHaveLength(20);
    // The NEWEST 20, since the caller is following up on the latest turn.
    expect(budgeted[19]?.content).toBe('turn 29');
  });

  it('drops oldest first when over the token budget', () => {
    // ~4 chars per token, 2000-token budget: 3 turns of 4000 chars is 3000
    // tokens, so the oldest has to go.
    const turns = [turn('a'.repeat(4000)), turn('b'.repeat(4000)), turn('c'.repeat(100))];
    const budgeted = budgetRecentTurns(turns);
    expect(budgeted.length).toBeLessThan(3);
    expect(budgeted[budgeted.length - 1]?.content.startsWith('c')).toBe(true);
  });

  it('is on by default and switchable off', () => {
    expect(verbatimOverlayEnabled()).toBe(true);
    process.env['SYNAP_ST_VERBATIM_OVERLAY'] = 'off';
    expect(verbatimOverlayEnabled()).toBe(false);
    process.env['SYNAP_ST_VERBATIM_OVERLAY'] = '1';
    expect(verbatimOverlayEnabled()).toBe(true);
  });

  it('leaves a response alone when the store is cold', () => {
    const response: RawContext = { facts: [] };
    overlayLocalRecentTurns(response, new ShortTermStore(), 'c1');
    expect(response.conversation_context).toBeUndefined();
  });

  it('never throws into the fetch path', () => {
    const broken = { get() { throw new Error('store exploded'); } } as unknown as ShortTermStore;
    const response: RawContext = { facts: [] };
    expect(() => overlayLocalRecentTurns(response, broken, 'c1')).not.toThrow();
  });
});

describe('read-your-writes, end to end', () => {
  /** Server answers every fetch with a conversation_context missing the new turn. */
  function client(serverTurns: unknown[] = []) {
    return new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async (url: string) =>
        String(url).includes('/messages')
          ? json({ message_id: 'm1' })
          // fetchContext returns `result.context`, so everything the caller
          // sees is nested here, conversation_context included.
          : json({
              context: {
                facts: [],
                conversation_context: {
                  summary: 'server summary',
                  recent_turns: serverTurns,
                  compaction_id: 'server-comp',
                },
              },
            })) as unknown as typeof fetch,
    });
  }

  it('surfaces a turn the server has not persisted yet', async () => {
    const c = client([]);           // server knows about nothing
    await c.conversation.record_message({
      conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'I fly from Berlin',
    });
    const ctx = await c.conversation.context.fetch({ conversation_id: UUID });
    const turns = (ctx.conversation_context?.['recent_turns'] ?? []) as Array<{ content: string }>;
    // Without the overlay this is empty and the agent looks like it forgot.
    expect(turns.map((t) => t.content)).toEqual(['I fly from Berlin']);
    await c.shutdown();
  });

  it('leaves the server in charge of the summary fields', async () => {
    const c = client([]);
    await c.conversation.record_message({
      conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'x',
    });
    const ctx = await c.conversation.context.fetch({ conversation_id: UUID });
    expect(ctx.conversation_context?.['summary']).toBe('server summary');
    expect(ctx.conversation_context?.['compaction_id']).toBe('server-comp');
    await c.shutdown();
  });

  it('does not overlay when the kill-switch is set', async () => {
    process.env['SYNAP_ST_VERBATIM_OVERLAY'] = 'off';
    const c = client([{ role: 'user', content: 'only what the server has' }]);
    await c.conversation.record_message({
      conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'local only',
    });
    const ctx = await c.conversation.context.fetch({ conversation_id: UUID });
    const turns = (ctx.conversation_context?.['recent_turns'] ?? []) as Array<{ content: string }>;
    expect(turns.map((t) => t.content)).toEqual(['only what the server has']);
    await c.shutdown();
  });

  it('does not buffer a turn whose write failed', async () => {
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      retryPolicy: { maxAttempts: 1 },
      fetchImpl: (async (url: string) =>
        String(url).includes('/messages')
          ? new Response(JSON.stringify({ detail: 'nope' }), { status: 500 })
          : json({ context: { facts: [] } })) as unknown as typeof fetch,
    });
    await c.conversation.record_message({
      conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'never landed',
    }).catch(() => {});
    const ctx = await c.conversation.context.fetch({ conversation_id: UUID });
    // Surfacing a message that does not exist server-side would be worse than
    // briefly missing one.
    expect(ctx.conversation_context).toBeUndefined();
    await c.shutdown();
  });

  it('buffers every message of a batch', async () => {
    const c = client([]);
    await c.conversation.record_messages_batch([
      { conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'one' },
      { conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'assistant', content: 'two' },
    ]);
    const ctx = await c.conversation.context.fetch({ conversation_id: UUID });
    const turns = (ctx.conversation_context?.['recent_turns'] ?? []) as Array<{ content: string }>;
    expect(turns.map((t) => t.content)).toEqual(['one', 'two']);
    await c.shutdown();
  });
});
