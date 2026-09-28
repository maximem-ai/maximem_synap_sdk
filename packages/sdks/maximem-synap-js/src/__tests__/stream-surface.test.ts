/**
 * The stream surface: sessions, typed tool events, and the role trap.
 *
 * Mirrors `python/tests/test_stream_surface.py` case for case. Three things,
 * all of which were wrong or missing on both sides:
 *
 * 1. Neither SDK ever sent `session_control`. The server runs its warm-up
 *    prefetch on `session_start`, so the first turn of every conversation was
 *    a cold fetch by construction, and `session_end` is what writes the turn's
 *    telemetry row. The SDK opens and closes sessions itself now.
 * 2. `role` and `event_type` have to agree and nothing checked. Reasoning went
 *    out as role 'assistant', which the server read before the event type, so
 *    every reasoning step was filed as the assistant's reply to the user.
 * 3. `event_type` was a free string, so a typo reached the server and fell
 *    through every classifier into UNKNOWN, where nothing read it.
 */
import { describe, expect, it, vi } from 'vitest';

import { createInstanceNamespace } from '../instance/interface.js';
import { AnticipationCache } from '../context/anticipation-cache.js';

interface SentEvent { [k: string]: unknown }

function harness(opts: { onTurn?: boolean } = {}) {
  const events: SentEvent[] = [];
  const controls: SentEvent[] = [];
  const turns: Array<{ conversationId: string; role: string; content: string }> = [];

  const fakeClient = {
    isConnected: true,
    lastError: null,
    sendConversationEvent(event: SentEvent) { events.push(event); },
    sendSessionControl(input: SentEvent) { controls.push(input); },
    sendContextUsed() {},
    sendContextAssembled() {},
    async connect() {},
    async disconnect() {},
  };

  const ns = createInstanceNamespace(
    () => ({ apiKey: 'k', clientId: 'c', instanceId: 'i' }),
    new AnticipationCache(),
    opts.onTurn === false ? undefined : (turn) => { turns.push(turn); },
  );

  // The stream is created inside listen() behind a lazy import. Reach past it:
  // these tests are about what the namespace SENDS, not about grpc-js.
  (ns as unknown as { _test_setClient?: unknown });
  return { ns, events, controls, turns, fakeClient };
}

/**
 * `listen()` builds its own client, so swap it in through the module's own
 * lazy import. Simpler and more honest than faking grpc-js: point the
 * namespace at a stub by calling listen with a stubbed constructor.
 */
async function listening(opts: { onTurn?: boolean } = {}) {
  const h = harness(opts);
  vi.doMock('../grpc/stream-client.js', () => ({
    GrpcStreamClient: class {
      isConnected = true;
      lastError = null;
      constructor() { return h.fakeClient as never; }
    },
  }));
  await h.ns.listen({ host: 'localhost', port: 1, useTls: false });
  return h;
}

describe('the session opens itself', () => {
  it('the first event opens a session', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', conversation_id: 'c1', user_id: 'u1' });
    expect(h.controls.map((c) => [c['action'], c['conversation_id']]))
      .toEqual([['start', 'c1']]);
  });

  it('the second event does not open another', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'a', conversation_id: 'c1', user_id: 'u1' });
    await h.ns.send_message({ content: 'b', conversation_id: 'c1', user_id: 'u1' });
    expect(h.controls).toHaveLength(1);
  });

  it('a second conversation gets its own', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'a', conversation_id: 'c1', user_id: 'u1' });
    await h.ns.send_message({ content: 'b', conversation_id: 'c2', user_id: 'u1' });
    expect(h.controls.map((c) => c['conversation_id'])).toEqual(['c1', 'c2']);
  });

  it('the session id rides on the events', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', conversation_id: 'c1', user_id: 'u1' });
    expect(h.events[0]?.['session_id']).toBe(h.controls[0]?.['session_id']);
    expect(h.events[0]?.['session_id']).toBeTruthy();
  });

  it('a caller supplied session id wins', async () => {
    const h = await listening();
    await h.ns.send_message({
      content: 'hello', conversation_id: 'c1', user_id: 'u1', session_id: 'mine',
    });
    expect(h.events[0]?.['session_id']).toBe('mine');
  });

  it('an event with no conversation opens nothing', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', user_id: 'u1' });
    expect(h.controls).toEqual([]);
  });

  it('stop_listening closes what it opened', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', conversation_id: 'c1', user_id: 'u1' });
    await h.ns.stop_listening();
    expect(h.controls.map((c) => c['action'])).toEqual(['start', 'end']);
  });

  it('end_session closes one conversation', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'a', conversation_id: 'c1', user_id: 'u1' });
    await h.ns.send_message({ content: 'b', conversation_id: 'c2', user_id: 'u1' });
    await h.ns.end_session('c1');
    const last = h.controls[h.controls.length - 1];
    expect([last?.['action'], last?.['conversation_id']]).toEqual(['end', 'c1']);
  });

  it('ending an unopened session is a no-op', async () => {
    const h = await listening();
    await h.ns.end_session('never-started');
    expect(h.controls).toEqual([]);
  });
});

describe('a session is only opened when it can be accepted', () => {
  // The server refuses a session_start with no user_id, and writing one only
  // says it went out, not that it was accepted. Opening on an event with no
  // user_id marked the conversation as open, never retried, and left
  // end_session naming a session the server never had.
  it('no user id opens nothing, and the event still goes out', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', conversation_id: 'c1' });
    expect(h.controls).toEqual([]);
    expect(h.events).toHaveLength(1);
  });

  it('a later event with the ids opens it', async () => {
    const h = await listening();
    await h.ns.send_message({ content: 'hello', conversation_id: 'c1' });
    await h.ns.send_message({ content: 'again', conversation_id: 'c1', user_id: 'u1' });
    expect(h.controls.map((c) => c['action'])).toEqual(['start']);
  });

  it('a tool event without ids does not open a bogus session', async () => {
    const h = await listening();
    await h.ns.record_tool_call({ tool_name: 'lookup', conversation_id: 'c1' });
    expect(h.controls).toEqual([]);
  });
});

describe('the typed tool methods', () => {
  it('a tool call names the role for you', async () => {
    const h = await listening();
    await h.ns.record_tool_call({
      tool_name: 'lookup_order', tool_args: { id: 42 }, tool_call_id: 'call_1',
      conversation_id: 'c1', user_id: 'u1',
    });
    const e = h.events[0];
    expect(e?.['event_type']).toBe('tool_call');
    expect(e?.['role']).toBe('assistant');
    expect(e?.['tool_name']).toBe('lookup_order');
    expect(e?.['tool_args_json']).toBe('{"id":42}');
    expect(e?.['tool_call_id']).toBe('call_1');
  });

  it('a tool result travels in its own field', async () => {
    const h = await listening();
    await h.ns.record_tool_result({
      result: { status: 'shipped' }, tool_name: 'lookup_order',
      tool_call_id: 'call_1', conversation_id: 'c1', user_id: 'u1',
    });
    const e = h.events[0];
    expect(e?.['event_type']).toBe('tool_result');
    expect(e?.['role']).toBe('tool');
    expect(e?.['tool_result_json']).toBe('{"status":"shipped"}');
    expect(e?.['content']).toBe('');
  });

  it('a plain text result keeps its quotes off', async () => {
    const h = await listening();
    await h.ns.record_tool_result({
      result: 'shipped', conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.events[0]?.['tool_result_json']).toBe('shipped');
  });

  it('a call and a result can be tied together', async () => {
    const h = await listening();
    await h.ns.record_tool_call({
      tool_name: 'lookup', tool_call_id: 'call_1', conversation_id: 'c1', user_id: 'u1',
    });
    await h.ns.record_tool_result({
      result: 'ok', tool_call_id: 'call_1', conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.events[0]?.['tool_call_id']).toBe(h.events[1]?.['tool_call_id']);
  });
});

describe('reasoning names no role', () => {
  it('it does not claim to be the assistant reply', async () => {
    const h = await listening();
    await h.ns.record_thinking({
      content: 'plan the lookup', conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.events[0]?.['event_type']).toBe('agent_thinking');
    expect(h.events[0]?.['role']).toBe('');
  });

  it('it opens a session too', async () => {
    const h = await listening();
    await h.ns.record_thinking({ content: 'plan', conversation_id: 'c1', user_id: 'u1' });
    expect(h.controls.map((c) => c['action'])).toEqual(['start']);
  });
});

describe('an unknown event type is refused', () => {
  it('a typo throws rather than travelling', async () => {
    const h = await listening();
    await expect(h.ns.send_message({
      content: 'x', event_type: 'tool_reslt', conversation_id: 'c1',
    })).rejects.toThrow(/tool_reslt/);
    expect(h.events).toEqual([]);
  });

  it('the message lists the real ones', async () => {
    const h = await listening();
    await expect(h.ns.send_message({
      content: 'x', event_type: 'nope', conversation_id: 'c1',
    })).rejects.toThrow(/tool_result/);
  });
});

describe('only turns reach the local short-term buffer', () => {
  it('a user message is buffered', async () => {
    const h = await listening();
    await h.ns.send_message({
      content: 'hello', conversation_id: 'c1', user_id: 'u1',
      event_type: 'user_message',
    });
    expect(h.turns).toHaveLength(1);
  });

  it('a tool result is not', async () => {
    // It was. Every event went into the buffer whatever its type, so a tool
    // result was spliced back into the caller's own prompt as a turn somebody
    // had said. Python only ever mirrored user and assistant messages.
    const h = await listening();
    await h.ns.record_tool_result({
      result: 'shipped', conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.turns).toEqual([]);
  });

  it('neither is a tool call', async () => {
    const h = await listening();
    await h.ns.record_tool_call({
      tool_name: 'lookup', conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.turns).toEqual([]);
  });
});
