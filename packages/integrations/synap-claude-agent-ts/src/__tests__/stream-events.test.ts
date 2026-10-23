/**
 * The shared stream helper: the two rules, and the traps it exists to stop.
 *
 * Mirrors `synap_integrations_common`'s Python tests. Everything here is a
 * behaviour that was wrong at least once in a real integration:
 *
 *  - `isListening` is a METHOD in JS and a PROPERTY in Python. Reading it
 *    without calling it gives a truthy function object, so every report gets
 *    attempted against a closed stream.
 *  - an absent `customer_id` sent as `''` is REFUSED by a B2C instance, so an
 *    ordinary turn becomes an error.
 *  - a `tool_args` that is a JSON string is dropped by the SDK without a word,
 *    so the tool call arrives claiming the tool takes no arguments.
 *  - a hook that throws takes down the caller's agent run.
 *
 * Every assertion is on what reached the fake SDK. A test that asserts a call
 * "did not throw" passes on code that reports nothing at all.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  endSession,
  ensureStream,
  fetchCompactedHistory,
  isSynapHistory,
  recordTurn,
  reportReasoning,
  reportToolCall,
  reportToolResult,
  reportTurn,
  setAutoListen,
  spliceHistory,
  streamIsActive,
} from '../stream-events.js';

interface Call { method: string; options: unknown }

function fakeSdk(opts: { listening?: boolean; throws?: boolean } = {}) {
  const calls: Call[] = [];
  const record = (method: string) => async (options: unknown) => {
    if (opts.throws === true) throw new Error('the stream is having a bad day');
    calls.push({ method, options });
  };
  const sdk = {
    instance: {
      isListening: () => opts.listening !== false,
      send_message: record('send_message'),
      record_tool_call: record('record_tool_call'),
      record_tool_result: record('record_tool_result'),
      record_thinking: record('record_thinking'),
      end_session: record('end_session'),
    },
  };
  return { sdk, calls };
}

function opts(calls: Call[], i = 0): Record<string, unknown> {
  return calls[i]?.options as Record<string, unknown>;
}

describe('silent when there is no stream', () => {
  it('reports nothing when the stream is not listening', async () => {
    const { sdk, calls } = fakeSdk({ listening: false });
    await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' });
    await reportToolCall(sdk, { toolName: 't', userId: 'u1' });
    await reportToolResult(sdk, { result: 'r', userId: 'u1' });
    await reportReasoning(sdk, { content: 'thinking', userId: 'u1' });
    await endSession(sdk, 'c1');
    expect(calls).toEqual([]);
  });

  it('says so in its return value, so a caller can fall back to REST', async () => {
    // This is the whole of the stream-first rule: an integration writes over
    // REST only when the stream said no. A helper that always returned void
    // would make every integration write twice or not at all.
    const { sdk } = fakeSdk({ listening: false });
    expect(await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' }))
      .toBe(false);
  });

  it('confirms when it did send', async () => {
    const { sdk } = fakeSdk();
    expect(await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' }))
      .toBe(true);
  });

  it('an sdk with no instance namespace answers no rather than throwing', async () => {
    // An older SDK, or a user's own stub. `streamIsActive` reading through a
    // missing namespace used to be the only thing between a hook and a crash.
    const old = { conversation: { record_message: async () => {} } };
    expect(streamIsActive(old)).toBe(false);
    expect(await reportTurn(old, { role: 'user', content: 'hi', userId: 'u1' }))
      .toBe(false);
  });

  it('an sdk missing just the one method answers no', async () => {
    const partial = { instance: { isListening: () => true } };
    expect(await reportToolCall(partial, { toolName: 't', userId: 'u1' }))
      .toBe(false);
  });

  it('null and undefined answer no', () => {
    expect(streamIsActive(null)).toBe(false);
    expect(streamIsActive(undefined)).toBe(false);
  });

  it('isListening is CALLED, not just read', () => {
    // A function object is truthy. Reading the property instead of calling it
    // makes every closed stream look open, which is the JS-only half of this
    // bug: Python's is a plain property and reads correctly.
    const spy = vi.fn(() => false);
    expect(streamIsActive({ instance: { isListening: spy } })).toBe(false);
    expect(spy).toHaveBeenCalled();
  });

  it('an isListening that throws answers no', () => {
    expect(streamIsActive({
      instance: { isListening: () => { throw new Error('half built'); } },
    })).toBe(false);
  });
});

describe('never throws', () => {
  it('swallows a failing report and says it did not send', async () => {
    const { sdk } = fakeSdk({ throws: true });
    expect(await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' }))
      .toBe(false);
    expect(await reportToolCall(sdk, { toolName: 't', userId: 'u1' })).toBe(false);
    expect(await reportToolResult(sdk, { result: 'r', userId: 'u1' })).toBe(false);
    expect(await reportReasoning(sdk, { content: 'x', userId: 'u1' })).toBe(false);
    expect(await endSession(sdk, 'c1')).toBe(false);
  });
});

describe('the role and the event type are set together', () => {
  it('a user turn is user_message with role user', async () => {
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' });
    expect(opts(calls)).toMatchObject({ role: 'user', event_type: 'user_message' });
  });

  it('an assistant turn is assistant_message with role assistant', async () => {
    // The event the anticipation agent acts on. An integration that reports
    // everything else and not this one gets no anticipation at all.
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, { role: 'assistant', content: 'yes', userId: 'u1' });
    expect(opts(calls)).toMatchObject({
      role: 'assistant', event_type: 'assistant_message',
    });
  });

  it('any role that is not user is treated as the assistant', async () => {
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, { role: 'model', content: 'yes', userId: 'u1' });
    expect(opts(calls)).toMatchObject({ role: 'assistant' });
  });
});

describe('an absent id is dropped, never sent empty', () => {
  it('no customer_id key at all when there is none', async () => {
    // A B2C instance REFUSES a call carrying a customer_id. Sending '' turns
    // every ordinary turn on a B2C instance into a refusal.
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' });
    expect(opts(calls)).not.toHaveProperty('customer_id');
  });

  it('an explicitly empty customer_id is dropped too', async () => {
    // ⚠ The shape that actually happens. An integration holds `customerId` as
    // a string and defaults it to '', so it reaches here as '' and not as
    // undefined. Only guarding undefined leaves every B2C turn refused.
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, {
      role: 'user', content: 'hi', userId: 'u1', customerId: '',
    });
    expect(opts(calls)).not.toHaveProperty('customer_id');
  });

  it('an explicitly empty conversation_id is dropped too', async () => {
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, {
      role: 'user', content: 'hi', userId: 'u1', conversationId: '',
    });
    expect(opts(calls)).not.toHaveProperty('conversation_id');
  });

  it('carries a customer_id when there is one', async () => {
    const { sdk, calls } = fakeSdk();
    await reportTurn(sdk, {
      role: 'user', content: 'hi', userId: 'u1', customerId: 'c1',
    });
    expect(opts(calls)).toMatchObject({ customer_id: 'c1' });
  });

  it('drops an empty tool_call_id rather than pairing on ""', async () => {
    // Two tool calls with id '' pair with each other's results.
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolCallId: '' });
    expect(opts(calls)).not.toHaveProperty('tool_call_id');
  });

  it('keeps a tool_call_id when there is one', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolCallId: 'call_1' });
    expect(opts(calls)).toMatchObject({ tool_call_id: 'call_1' });
  });

  it('a call and its result carry the same id', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolCallId: 'x' });
    await reportToolResult(sdk, { result: 'r', userId: 'u1', toolCallId: 'x' });
    expect(opts(calls, 0)['tool_call_id']).toBe(opts(calls, 1)['tool_call_id']);
  });
});

describe('the payload is not an id and is never dropped', () => {
  // ⚠ The divergence this guards. Python passes `result` and `content`
  // positionally, so its cleaner cannot reach them. A blanket "drop every
  // empty value" here would make the two SDKs disagree about what an empty
  // tool result means, and the disagreement would be silent.

  it('an empty-string tool result is still reported', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolResult(sdk, { result: '', userId: 'u1' });
    expect(calls).toHaveLength(1);
    expect(opts(calls)).toHaveProperty('result', '');
  });

  it('a null tool result is still reported', async () => {
    // A tool that found nothing said something. It is not the same as a tool
    // that was never called.
    const { sdk, calls } = fakeSdk();
    await reportToolResult(sdk, { result: null, userId: 'u1' });
    expect(calls).toHaveLength(1);
    expect(opts(calls)).toHaveProperty('result', null);
  });

  it('a false tool result is still reported', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolResult(sdk, { result: false, userId: 'u1' });
    expect(opts(calls)).toHaveProperty('result', false);
  });

  it('a zero step index is still reported', async () => {
    // Step 0 is the first step, not a missing one.
    const { sdk, calls } = fakeSdk();
    await reportReasoning(sdk, { content: 'first', userId: 'u1', stepIndex: 0 });
    expect(opts(calls)).toHaveProperty('step_index', 0);
  });
});

describe('tool arguments reach the server in a shape the SDK forwards', () => {
  // The SDK forwards `tool_args` only when it is a plain object and drops
  // anything else without a word, so the call arrives claiming the tool takes
  // no arguments.

  it('a plain object goes through unchanged', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolArgs: { q: 'hi' } });
    expect(opts(calls)['tool_args']).toEqual({ q: 'hi' });
  });

  it('a JSON string is parsed, not dropped', async () => {
    // LiveKit hands `FunctionCall.arguments` over as a JSON string.
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, {
      toolName: 't', userId: 'u1', toolArgs: '{"q": "hi"}',
    });
    expect(opts(calls)['tool_args']).toEqual({ q: 'hi' });
  });

  it('a string that is not JSON is wrapped, not dropped', async () => {
    // LangChain's `input_str`. Better a named field than no arguments at all.
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolArgs: 'raw text' });
    expect(opts(calls)['tool_args']).toEqual({ input: 'raw text' });
  });

  it('a JSON string holding a bare value is wrapped, not spread', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolArgs: '42' });
    expect(opts(calls)['tool_args']).toEqual({ input: '42' });
  });

  it('a Map becomes an object', async () => {
    // Semantic Kernel hands over a Mapping that is not a plain object.
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, {
      toolName: 't', userId: 'u1', toolArgs: new Map([['q', 'hi']]),
    });
    expect(opts(calls)['tool_args']).toEqual({ q: 'hi' });
  });

  it('an array is wrapped rather than sent as an object', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1', toolArgs: [1, 2] });
    expect(opts(calls)['tool_args']).toEqual({ input: [1, 2] });
  });

  it('no arguments means no tool_args key', async () => {
    const { sdk, calls } = fakeSdk();
    await reportToolCall(sdk, { toolName: 't', userId: 'u1' });
    expect(opts(calls)).not.toHaveProperty('tool_args');
  });
});

describe('reasoning', () => {
  it('reports the thought', async () => {
    const { sdk, calls } = fakeSdk();
    await reportReasoning(sdk, {
      content: 'I should search', userId: 'u1', thoughtType: 'plan',
    });
    expect(calls[0]?.method).toBe('record_thinking');
    expect(opts(calls)).toMatchObject({
      content: 'I should search', thought_type: 'plan',
    });
  });

  it('an empty thought is not an event', async () => {
    // A turn with no reasoning is an ordinary turn, not a gap to fill.
    const { sdk, calls } = fakeSdk();
    expect(await reportReasoning(sdk, { content: '', userId: 'u1' })).toBe(false);
    expect(calls).toEqual([]);
  });
});

describe('end of session', () => {
  it('passes the conversation id positionally, not as an object', async () => {
    // `end_session(conversationId)` is the one method on the namespace that
    // does NOT take an options object. Passing one closes nothing.
    const { sdk, calls } = fakeSdk();
    await endSession(sdk, 'c1');
    expect(calls[0]).toEqual({ method: 'end_session', options: 'c1' });
  });

  it('no conversation id is not a session to end', async () => {
    const { sdk, calls } = fakeSdk();
    expect(await endSession(sdk, '')).toBe(false);
    expect(calls).toEqual([]);
  });
});

// ------------------------------------------------- the stream opens itself

/**
 * An SDK whose stream is closed until `listen()` is called, the way the real
 * one behaves: `instance.is_listening` is a getter, `listen()` throws
 * `ListeningAlreadyActiveError` when a stream exists.
 */
function lazySdk(opts: {
  listenFails?: boolean;
  neverOpens?: boolean;
  listenHangs?: boolean;
  restThrows?: boolean;
} = {}) {
  const calls: Call[] = [];
  const rest: unknown[] = [];
  const state = { listening: false, listens: 0 };
  const record = (method: string) => async (options: unknown) => {
    calls.push({ method, options });
  };
  const sdk = {
    instance: {
      get is_listening() { return state.listening; },
      async listen() {
        if (state.listening) throw new Error('Listening is already active on this instance');
        state.listens += 1;
        if (opts.listenHangs === true) return new Promise<void>(() => {});
        await Promise.resolve();
        if (opts.listenFails === true) throw new Error('connection refused');
        if (opts.neverOpens !== true) state.listening = true;
      },
      send_message: record('send_message'),
      record_tool_call: record('record_tool_call'),
      record_tool_result: record('record_tool_result'),
      record_thinking: record('record_thinking'),
      end_session: record('end_session'),
    },
    conversation: {
      async record_message(options: unknown) {
        if (opts.restThrows === true) throw new Error('503');
        rest.push(options);
      },
    },
  };
  return { sdk, calls, rest, state };
}

describe('the first report opens the stream', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('opens the stream on the first turn and sends it there', async () => {
    const { sdk, calls, state } = lazySdk();
    expect(streamIsActive(sdk)).toBe(false);
    expect(await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1', conversationId: 'c1' }))
      .toBe(true);
    expect(state.listens).toBe(1);
    expect(streamIsActive(sdk)).toBe(true);
    expect(calls[0]?.method).toBe('send_message');
  });

  it('reads the real SDK\'s `is_listening` getter, not only a method', () => {
    // The published SDK has no `instance.isListening()`; that method lives on
    // its private `_internal`. Reading only the method made every real SDK
    // look permanently closed.
    expect(streamIsActive({ instance: { is_listening: true } })).toBe(true);
    expect(streamIsActive({ instance: { is_listening: false } })).toBe(false);
  });

  it('a stream the caller opened is used and never reopened', async () => {
    const { sdk, state } = lazySdk();
    await sdk.instance.listen();
    expect(await ensureStream(sdk)).toBe(true);
    expect(state.listens).toBe(1);
  });

  it('concurrent first reports share one open', async () => {
    const { sdk, calls, state } = lazySdk();
    const results = await Promise.all([
      reportTurn(sdk, { role: 'user', content: 'a', userId: 'u1', conversationId: 'c1' }),
      reportToolCall(sdk, { toolName: 't', userId: 'u1', conversationId: 'c1' }),
      reportReasoning(sdk, { content: 'x', userId: 'u1', conversationId: 'c1' }),
    ]);
    expect(results).toEqual([true, true, true]);
    expect(state.listens).toBe(1);
    expect(calls).toHaveLength(3);
  });

  it('tool calls, tool results and reasoning also open it', async () => {
    for (const report of [
      (sdk: unknown) => reportToolCall(sdk, { toolName: 't', userId: 'u1' }),
      (sdk: unknown) => reportToolResult(sdk, { result: 'r', userId: 'u1' }),
      (sdk: unknown) => reportReasoning(sdk, { content: 'x', userId: 'u1' }),
    ]) {
      const { sdk, state } = lazySdk();
      expect(await report(sdk)).toBe(true);
      expect(state.listens).toBe(1);
    }
  });

  it('a failed open is not retried for a minute, and warns once', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state } = lazySdk({ listenFails: true });
    expect(await ensureStream(sdk)).toBe(false);
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(1);
    vi.advanceTimersByTime(61_000);
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(2);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('a listen() that never answers is given up on after ten seconds', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk } = lazySdk({ listenHangs: true });
    const pending = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(10_001);
    expect(await pending).toBe(false);
  });

  it('a listen() that resolves without a stream counts as a failure', async () => {
    const { sdk, state } = lazySdk({ neverOpens: true });
    expect(await ensureStream(sdk)).toBe(false);
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(1);
  });

  it('SYNAP_AUTO_LISTEN=false leaves the stream closed', async () => {
    vi.stubEnv('SYNAP_AUTO_LISTEN', 'false');
    const { sdk, state } = lazySdk();
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(0);
  });

  it('setAutoListen(sdk, false) leaves it closed for that SDK only', async () => {
    const off = lazySdk();
    const on = lazySdk();
    setAutoListen(off.sdk, false);
    expect(await ensureStream(off.sdk)).toBe(false);
    expect(await ensureStream(on.sdk)).toBe(true);
    expect(off.state.listens).toBe(0);
  });

  it('setAutoListen(sdk, true) beats the environment', async () => {
    vi.stubEnv('SYNAP_AUTO_LISTEN', 'false');
    const { sdk } = lazySdk();
    setAutoListen(sdk, true);
    expect(await ensureStream(sdk)).toBe(true);
  });

  it('an opted-out SDK that already has a stream still uses it', async () => {
    const { sdk, calls } = lazySdk();
    await sdk.instance.listen();
    setAutoListen(sdk, false);
    expect(await reportTurn(sdk, { role: 'user', content: 'hi', userId: 'u1' })).toBe(true);
    expect(calls).toHaveLength(1);
  });

  it('never throws on things that are not an SDK', async () => {
    expect(await ensureStream(null)).toBe(false);
    expect(await ensureStream(undefined)).toBe(false);
    expect(await ensureStream('sdk')).toBe(false);
    expect(await ensureStream({ instance: null })).toBe(false);
  });

  it('endSession never opens a stream', async () => {
    const { sdk, calls, state } = lazySdk();
    expect(await endSession(sdk, 'c1')).toBe(false);
    expect(state.listens).toBe(0);
    expect(calls).toEqual([]);
  });

  it('a tool result with no stream is archived locally and never sent', async () => {
    const archive = vi.fn();
    const { sdk, calls } = lazySdk({ neverOpens: true });
    (sdk as unknown as { archive_tool_result: typeof archive }).archive_tool_result = archive;
    await reportToolResult(sdk, { result: 'r', toolName: 't', userId: 'u1', conversationId: 'c1' });
    expect(archive).toHaveBeenCalledWith('c1', 'r', { tool_name: 't' });
    expect(calls).toEqual([]);
  });
});

// ------------------------------------- one open, however many copies of this file

/**
 * An SDK that models the real one's stream client, which the older fakes do
 * not: `listen()` creates the client at once and `connect` takes `connectMs`;
 * `listen()` throws `ListeningAlreadyActiveError` for as long as a client
 * exists (connecting, connected or reconnecting), while `is_listening` is true
 * only when connected. `dropFor(ms)` puts the client into a reconnect.
 */
class ListeningAlreadyActiveError extends Error {
  override name = 'ListeningAlreadyActiveError';
}

function clientSdk(opts: { connectMs?: number; failFirst?: number } = {}) {
  const state = { listens: 0, hasClient: false, connected: false, failures: 0 };
  const sdk = {
    instance: {
      get is_listening() { return state.connected; },
      async listen() {
        if (state.hasClient) throw new ListeningAlreadyActiveError('Listening is already active on this instance');
        state.listens += 1;
        if (state.failures < (opts.failFirst ?? 0)) {
          state.failures += 1;
          await new Promise((r) => setTimeout(r, 5));
          throw new Error('connect timed out');
        }
        state.hasClient = true;
        await new Promise((r) => setTimeout(r, opts.connectMs ?? 20));
        state.connected = true;
      },
      async send_message(_options?: unknown) {},
    },
  };
  const dropFor = (ms: number) => {
    state.connected = false;
    setTimeout(() => { state.connected = true; }, ms);
  };
  return { sdk, state, dropFor };
}

describe('one open per SDK, whatever imports this file', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.resetModules();
  });

  it('two copies of the module share one open', async () => {
    // eve bundles the helper separately for hooks and for instructions; each
    // bundle is its own module scope, so a module-level map is one per copy.
    const first = await import('../stream-events.js');
    vi.resetModules();
    const second = await import('../stream-events.js');
    expect(second.ensureStream).not.toBe(first.ensureStream);
    const { sdk, state } = clientSdk({ connectMs: 50 });
    const results = await Promise.all([first.ensureStream(sdk), second.ensureStream(sdk)]);
    expect(results).toEqual([true, true]);
    expect(state.listens).toBe(1);
  });

  it('opt-out is shared across copies too', async () => {
    const first = await import('../stream-events.js');
    vi.resetModules();
    const second = await import('../stream-events.js');
    const { sdk, state } = clientSdk();
    first.setAutoListen(sdk, false);
    expect(await second.ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(0);
  });

  it('keeps the state on the SDK under a registry symbol, out of enumeration', async () => {
    const { sdk } = clientSdk();
    await ensureStream(sdk);
    expect(Object.keys(sdk)).toEqual(['instance']);
    expect(Object.getOwnPropertySymbols(sdk)).toContain(Symbol.for('@maximem/synap-stream-events.state'));
  });

  it('still opens one stream on an SDK that cannot take a property', async () => {
    const { sdk, state } = clientSdk({ connectMs: 20 });
    Object.freeze(sdk);
    const results = await Promise.all([ensureStream(sdk), ensureStream(sdk)]);
    expect(results).toEqual([true, true]);
    expect(state.listens).toBe(1);
  });
});

describe('a stream that is already on its way is used, not failed', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  async function opened(sdk: object): Promise<void> {
    const p = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(100);
    expect(await p).toBe(true);
  }

  it('a reconnecting stream (listen() says already active) is waited for', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state, dropFor } = clientSdk();
    await opened(sdk);
    dropFor(500);
    const pending = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(600);
    expect(await pending).toBe(true);
    // The one real open; the second call asked, was refused, and waited.
    expect(state.listens).toBe(1);
    expect(warn).not.toHaveBeenCalled();
    // No cooldown: the next report goes straight over the stream.
    expect(await ensureStream(sdk)).toBe(true);
  });

  it('a turn is sent on the stream while it reconnects, not over REST', async () => {
    vi.useFakeTimers();
    const { sdk, dropFor } = clientSdk();
    const rest: unknown[] = [];
    (sdk as unknown as { conversation: unknown }).conversation = {
      record_message: async (o: unknown) => { rest.push(o); },
    };
    const sent: unknown[] = [];
    sdk.instance.send_message = async (o?: unknown) => { sent.push(o); };
    await opened(sdk);
    dropFor(400);
    const turn = recordTurn(sdk, { role: 'user', content: 'hi', userId: 'u1', conversationId: 'c1' });
    await vi.advanceTimersByTimeAsync(500);
    expect(await turn).toBe(true);
    expect(sent).toHaveLength(1);
    expect(rest).toEqual([]);
  });

  it('gives up on a stream that never comes back, after three seconds', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, dropFor } = clientSdk();
    await opened(sdk);
    dropFor(60_000);
    const pending = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(3_100);
    expect(await pending).toBe(false);
  });

  it('an error that is not already-active is a failure at once', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state } = clientSdk({ failFirst: 1 });
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(1);
  });
});

describe('the first failed open is retried quickly, later ones cool down', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('retries once after three seconds and the stream comes up', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state } = clientSdk({ failFirst: 1 });
    const first = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(50);
    expect(await first).toBe(false);
    // Inside three seconds: REST, no new listen().
    await vi.advanceTimersByTimeAsync(2_000);
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(1);
    await vi.advanceTimersByTimeAsync(1_100);
    const retry = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(100);
    expect(await retry).toBe(true);
    expect(state.listens).toBe(2);
  });

  it('a second failure waits out the full minute', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state } = clientSdk({ failFirst: 3 });
    for (const advance of [50, 3_100]) {
      const p = ensureStream(sdk);
      await vi.advanceTimersByTimeAsync(advance);
      expect(await p).toBe(false);
      if (advance === 50) await vi.advanceTimersByTimeAsync(3_100);
    }
    expect(state.listens).toBe(2);
    // Three and then twenty seconds after the second failure: still cooling down.
    await vi.advanceTimersByTimeAsync(3_100);
    expect(await ensureStream(sdk)).toBe(false);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(await ensureStream(sdk)).toBe(false);
    expect(state.listens).toBe(2);
    await vi.advanceTimersByTimeAsync(40_000);
    const third = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(50);
    expect(await third).toBe(false);
    expect(state.listens).toBe(3);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('a listen() that timed out is not repeated once it has finished', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    // connect takes twelve seconds: past the ten-second give-up.
    const { sdk, state } = clientSdk({ connectMs: 12_000 });
    const first = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(10_100);
    expect(await first).toBe(false);
    await vi.advanceTimersByTimeAsync(3_000);
    expect(await ensureStream(sdk)).toBe(true);
    expect(state.listens).toBe(1);
  });

  it('a retry while that listen() is still running does not start a second stream', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, state } = clientSdk({ connectMs: 30_000 });
    const first = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(10_100);
    expect(await first).toBe(false);
    await vi.advanceTimersByTimeAsync(3_000);
    const retry = ensureStream(sdk);
    await vi.advanceTimersByTimeAsync(10_100);
    expect(await retry).toBe(false);
    expect(state.listens).toBe(1);
  });
});

describe('recordTurn: stream first, REST second, never both', () => {
  it('opens the stream and does not write REST', async () => {
    const { sdk, calls, rest } = lazySdk();
    expect(await recordTurn(sdk, {
      role: 'assistant', content: 'yes', conversationId: 'c1', userId: 'u1',
    })).toBe(true);
    expect(calls).toHaveLength(1);
    expect(opts(calls)).toMatchObject({ event_type: 'assistant_message' });
    expect(rest).toEqual([]);
  });

  it('falls back to REST when the stream cannot open', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { sdk, calls, rest } = lazySdk({ listenFails: true });
    expect(await recordTurn(sdk, {
      role: 'user', content: 'hi', conversationId: 'c1', userId: 'u1', customerId: '',
    })).toBe(true);
    expect(calls).toEqual([]);
    expect(rest).toEqual([{ conversation_id: 'c1', role: 'user', content: 'hi', user_id: 'u1' }]);
  });

  it('sends a customer_id on REST only when there is one', async () => {
    const { sdk, rest } = lazySdk({ neverOpens: true });
    await recordTurn(sdk, { role: 'user', content: 'hi', conversationId: 'c1', userId: 'u1', customerId: 'cu' });
    expect(rest[0]).toMatchObject({ customer_id: 'cu' });
  });

  it('is false, and writes nothing, for empty text', async () => {
    const { sdk, calls, rest, state } = lazySdk();
    expect(await recordTurn(sdk, { role: 'user', content: '  ', conversationId: 'c1' })).toBe(false);
    expect(state.listens).toBe(0);
    expect(calls).toEqual([]);
    expect(rest).toEqual([]);
  });

  it('is false with no stream and no conversation id (REST needs one)', async () => {
    const { sdk, rest } = lazySdk({ neverOpens: true });
    expect(await recordTurn(sdk, { role: 'user', content: 'hi' })).toBe(false);
    expect(rest).toEqual([]);
  });

  it('swallows a failing REST write', async () => {
    const { sdk } = lazySdk({ neverOpens: true, restThrows: true });
    expect(await recordTurn(sdk, { role: 'user', content: 'hi', conversationId: 'c1' })).toBe(false);
  });

  it('is false when the SDK has neither a stream nor REST', async () => {
    expect(await recordTurn({}, { role: 'user', content: 'hi', conversationId: 'c1' })).toBe(false);
  });
});

// ------------------------------------------------------ Synap's history

function historySdk(response: unknown) {
  return {
    conversation: {
      context: { get_context_for_prompt: vi.fn(async () => response) },
    },
  };
}

describe('fetchCompactedHistory', () => {
  it('wraps the compaction block and keeps the turns after it', async () => {
    const sdk = historySdk({
      context_block: '## Conversation so far (compacted at 5)\nthey like tea',
      history_block: 'earlier: asked about kettles',
      recent_messages: [
        { role: 'user', content: 'and coffee?' },
        { role: 'assistant', content: 'also fine' },
        { role: 'system', content: 'dropped' },
        { role: 'user', content: '   ' },
      ],
      compaction_id: 'cmp-1',
    });
    const h = await fetchCompactedHistory(sdk, 'c1');
    expect(h?.preamble).toBe(
      '<synap_conversation_history>\n## Conversation so far (compacted at 5)\nthey like tea\n\n'
      + 'earlier: asked about kettles\n</synap_conversation_history>',
    );
    expect(h?.turns).toEqual([['user', 'and coffee?'], ['assistant', 'also fine']]);
    expect(h?.compactionId).toBe('cmp-1');
    expect(sdk.conversation.context.get_context_for_prompt)
      .toHaveBeenCalledWith({ conversation_id: 'c1' });
  });

  it('is null when no compaction exists yet', async () => {
    expect(await fetchCompactedHistory(historySdk({ context_block: null, recent_messages: [] }), 'c1'))
      .toBeNull();
    expect(await fetchCompactedHistory(historySdk({ context_block: '  ' }), 'c1')).toBeNull();
  });

  it('is null on an SDK without the method, with no conversation, and on error', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(await fetchCompactedHistory({}, 'c1')).toBeNull();
    expect(await fetchCompactedHistory(historySdk({ context_block: 'x' }), '')).toBeNull();
    const failing = { conversation: { context: { get_context_for_prompt: async () => { throw new Error('500'); } } } };
    expect(await fetchCompactedHistory(failing, 'c1')).toBeNull();
  });
});

describe('isSynapHistory', () => {
  it('recognises the preamble and the SDK compaction block, nothing else', () => {
    expect(isSynapHistory('<synap_conversation_history>\nx')).toBe(true);
    expect(isSynapHistory('  \n## Conversation so far (compacted at 3)')).toBe(true);
    expect(isSynapHistory('hello')).toBe(false);
    expect(isSynapHistory(42)).toBe(false);
  });
});

describe('spliceHistory', () => {
  type Msg = { role: string; text: string };
  const fns = {
    roleOf: (m: Msg) => m.role,
    textOf: (m: Msg) => m.text,
    makeMessage: (role: string, text: string): Msg => ({ role, text }),
  };
  const history = {
    preamble: '<synap_conversation_history>\nsummary\n</synap_conversation_history>',
    turns: [['user', 'q5'], ['assistant', 'a5']] as const,
  };

  it('replaces every earlier turn with the preamble and Synap\'s turns', () => {
    const out = spliceHistory<Msg>([
      { role: 'system', text: 'be nice' },
      { role: 'user', text: 'q1' }, { role: 'assistant', text: 'a1' },
      { role: 'user', text: 'q5' }, { role: 'assistant', text: 'a5' },
      { role: 'user', text: 'q6' },
    ], history, fns);
    expect(out).toEqual([
      { role: 'system', text: 'be nice' },
      { role: 'system', text: history.preamble },
      { role: 'user', text: 'q5' }, { role: 'assistant', text: 'a5' },
      { role: 'user', text: 'q6' },
    ]);
  });

  it('keeps the current turn whole, tool messages included', () => {
    const out = spliceHistory<Msg>([
      { role: 'user', text: 'old' }, { role: 'tool', text: 'old result' },
      { role: 'user', text: 'now' }, { role: 'assistant', text: 'calling' }, { role: 'tool', text: 'result' },
    ], history, fns);
    expect(out.slice(-3)).toEqual([
      { role: 'user', text: 'now' }, { role: 'assistant', text: 'calling' }, { role: 'tool', text: 'result' },
    ]);
    expect(out.some((m) => m.text === 'old result')).toBe(false);
  });

  it('drops a Synap copy of the current user message, and what followed it', () => {
    const out = spliceHistory<Msg>(
      [{ role: 'user', text: 'q5' }, { role: 'assistant', text: 'a5' }, { role: 'user', text: 'q6' }],
      { preamble: 'P', turns: [['user', 'q5'], ['assistant', 'a5'], ['user', 'q6']] },
      fns,
    );
    expect(out.filter((m) => m.text === 'q6')).toHaveLength(1);
  });

  it('keeps framework turns Synap has not recorded yet', () => {
    const out = spliceHistory<Msg>([
      { role: 'user', text: 'q5' }, { role: 'assistant', text: 'a5' },
      { role: 'user', text: 'q5b' }, { role: 'assistant', text: 'a5b' },
      { role: 'user', text: 'q6' },
    ], history, fns);
    expect(out.map((m) => m.text)).toEqual([history.preamble, 'q5', 'a5', 'q5b', 'a5b', 'q6']);
  });

  it('does not keep a previous splice\'s preamble as a system message', () => {
    const out = spliceHistory<Msg>([
      { role: 'system', text: history.preamble },
      { role: 'user', text: 'q6' },
    ], history, fns);
    expect(out.filter((m) => m.text === history.preamble)).toHaveLength(1);
  });
});
