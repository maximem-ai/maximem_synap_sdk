/**
 * A tool result the caller's framework returned must still reach the server.
 *
 * Mirrors `python/tests/test_tool_payload_serialisation.py` case for case.
 *
 * The bug was silent twice over. The namespace serialised `tool_result` with a
 * bare `JSON.stringify`, which throws on a circular object and on a `BigInt`,
 * and which returns `undefined` rather than a string for a function. Every
 * integration reports through a helper that swallows exceptions on purpose, so
 * a telemetry call can never break somebody's agent loop. The throw therefore
 * went nowhere and the whole tool event was never sent.
 *
 * JS fails on a narrower set of inputs than Python does, because
 * `JSON.stringify` walks an ordinary object's own enumerable properties and is
 * happy with a class instance or a `Date`. That makes the JS version of this
 * bug rarer and no less silent, and the fix has to be here anyway or the two
 * SDKs disagree on what a tool result is.
 *
 * Assertions are on what the transport RECEIVED. The broken version did not
 * throw either.
 */
import { describe, expect, it, vi } from 'vitest';

import { createInstanceNamespace } from '../instance/interface.js';
import { AnticipationCache } from '../context/anticipation-cache.js';

interface SentEvent { [k: string]: unknown }

async function listening() {
  const events: SentEvent[] = [];
  const fakeClient = {
    isConnected: true,
    lastError: null,
    sendConversationEvent(event: SentEvent) { events.push(event); },
    sendSessionControl() {},
    sendContextUsed() {},
    sendContextAssembled() {},
    async connect() {},
    async disconnect() {},
  };
  const ns = createInstanceNamespace(
    () => ({ apiKey: 'k', clientId: 'c', instanceId: 'i' }),
    new AnticipationCache(),
  );
  vi.doMock('../grpc/stream-client.js', () => ({
    GrpcStreamClient: class {
      isConnected = true;
      lastError = null;
      constructor() { return fakeClient as never; }
    },
  }));
  await ns.listen({ host: 'localhost', port: 1, useTls: false });
  return { ns, events };
}

/** The shapes that used to take their whole event down with them. */
function cyclic(): Record<string, unknown> {
  const d: Record<string, unknown> = { name: 'root' };
  d['self'] = d;
  return d;
}

function throwingToJson(): unknown {
  return { toJSON() { throw new Error('the caller decided not to'); } };
}

const UNSERIALISABLE: Array<[string, unknown]> = [
  ['reference cycle', cyclic()],
  ['bigint', { count: BigInt(9007199254740993n) }],
  ['bare bigint', BigInt(42n)],
  ['a toJSON that throws', throwingToJson()],
  ['a function', () => 'nope'],
  ['a symbol', Symbol('nope')],
];

describe('the event survives whatever the tool returned', () => {
  for (const [label, result] of UNSERIALISABLE) {
    it(`an unserialisable result is still sent: ${label}`, async () => {
      const h = await listening();
      await h.ns.record_tool_result({
        result, tool_name: 'search',
        conversation_id: 'c1', user_id: 'u1',
      });

      // The event exists at all. This is the whole bug: it did not.
      expect(h.events).toHaveLength(1);
      expect(h.events[0]?.['event_type']).toBe('tool_result');
    });

    it(`and it carries something a reader can use: ${label}`, async () => {
      const h = await listening();
      await h.ns.record_tool_result({
        result, conversation_id: 'c1', user_id: 'u1',
      });

      const body = h.events[0]?.['tool_result_json'];
      expect(typeof body).toBe('string');
      expect(body).not.toBe('');
    });
  }

  it('a tool call with unserialisable args is still sent', async () => {
    // `tool_args_json` had exactly the same bare stringify. A tool called with
    // a cyclic argument lost its tool_call event, so the tool_result that
    // followed had nothing to pair with.
    const h = await listening();
    await h.ns.record_tool_call({
      tool_name: 'search', tool_args: cyclic() as Record<string, unknown>,
      conversation_id: 'c1', user_id: 'u1',
    });

    expect(h.events).toHaveLength(1);
    expect(h.events[0]?.['event_type']).toBe('tool_call');
    expect(h.events[0]?.['tool_args_json']).not.toBe('');
  });
});

describe('nothing that already worked changed', () => {
  // A tool result is read by a model. A dict turning into its `String()` form
  // would quietly swap JSON for something else on every well-behaved tool.

  it('a plain object result is still json', async () => {
    const h = await listening();
    await h.ns.record_tool_result({
      result: { rows: 3, ok: true }, conversation_id: 'c1', user_id: 'u1',
    });

    expect(JSON.parse(String(h.events[0]?.['tool_result_json'])))
      .toEqual({ rows: 3, ok: true });
  });

  it('a string result still travels unquoted', async () => {
    // JSON.stringify would wrap a tool's plain-text answer in quotes and the
    // agent would read the quotes as part of the result.
    const h = await listening();
    await h.ns.record_tool_result({
      result: 'plain text', conversation_id: 'c1', user_id: 'u1',
    });

    expect(h.events[0]?.['tool_result_json']).toBe('plain text');
  });

  it('an array result is still json', async () => {
    const h = await listening();
    await h.ns.record_tool_result({
      result: [1, 2, 3], conversation_id: 'c1', user_id: 'u1',
    });
    expect(JSON.parse(String(h.events[0]?.['tool_result_json']))).toEqual([1, 2, 3]);
  });

  it('ordinary tool args are still json', async () => {
    const h = await listening();
    await h.ns.record_tool_call({
      tool_name: 'search', tool_args: { q: 'hello' },
      conversation_id: 'c1', user_id: 'u1',
    });
    expect(JSON.parse(String(h.events[0]?.['tool_args_json']))).toEqual({ q: 'hello' });
  });

  it('a null result is still the empty string, not "null"', async () => {
    // A tool that returns nothing must not report the four characters `null`
    // to the anticipation agent as if that were its answer.
    const h = await listening();
    await h.ns.record_tool_result({
      result: null, conversation_id: 'c1', user_id: 'u1',
    });
    expect(h.events[0]?.['tool_result_json']).toBe('');
  });
});

describe('the two SDKs agree on the degrade', () => {
  it('both keep the structure around an unserialisable leaf', async () => {
    // Python's `default=str` stringifies the leaf and keeps the JSON around
    // it. JS's replacer does the same for the one leaf type it has. If either
    // side ever falls back to stringifying the WHOLE value, this catches it.
    const h = await listening();
    await h.ns.record_tool_result({
      result: { rows: 3, count: BigInt(5n) },
      conversation_id: 'c1', user_id: 'u1',
    });

    expect(JSON.parse(String(h.events[0]?.['tool_result_json'])))
      .toEqual({ rows: 3, count: '5' });
  });
});
