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
import { describe, expect, it, vi } from 'vitest';

import {
  endSession,
  reportReasoning,
  reportToolCall,
  reportToolResult,
  reportTurn,
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
