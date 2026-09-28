// What Synap receives when a Mastra Agent runs with a live stream open.
//
// Every assertion is on what reached the fake SDK. A test that asserts a call
// "did not throw" passes on code that reports nothing at all, and reporting
// nothing is the exact bug this work exists to fix.
//
// Two contracts are under test:
//
//   1. The reporter turns one saved Mastra message into the right stream
//      events, with the right ids, in the right order.
//   2. `SynapMemory.saveMessages` is stream-first, REST-fallback, never both —
//      and, for someone with no stream, byte-identical to what it always did.

import { describe, expect, it, vi } from 'vitest';

import { SynapMemory } from '../memory.js';
import { SynapMastraStreamReporter } from '../stream.js';
import { makeSdk } from './helpers.js';
import type { SynapSdkLike } from '../types.js';

interface StreamCall {
  method: string;
  options: unknown;
}

const THREAD = 'thread-1';

/** A Synap SDK that also carries the `instance` stream namespace. */
function streamingSdk(opts: { listening?: boolean; throws?: boolean } = {}) {
  const calls: StreamCall[] = [];
  const record = (method: string) =>
    vi.fn(async (options: unknown) => {
      if (opts.throws === true) throw new Error('the stream is having a bad day');
      calls.push({ method, options });
    });
  const sdk = {
    ...makeSdk(),
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

/** The options of the first call to `method`, or undefined. */
function sentTo(calls: StreamCall[], method: string): Record<string, unknown> | undefined {
  return calls.find((c) => c.method === method)?.options as Record<string, unknown> | undefined;
}

function methodsIn(calls: StreamCall[]): string[] {
  return calls.map((c) => c.method);
}

/** One Mastra DB message, with whatever parts the test cares about. */
function message(
  role: string,
  parts: unknown[],
  opts: { id?: string; threadId?: string } = {},
) {
  return {
    id: opts.id === undefined ? `m-${role}-1` : opts.id,
    role,
    createdAt: new Date('2026-06-08T00:00:00.000Z'),
    threadId: opts.threadId === undefined ? THREAD : opts.threadId,
    resourceId: 'alice',
    type: 'text',
    content: { format: 2 as const, parts },
  };
}

function textPart(text: string) {
  return { type: 'text', text };
}

function reasoningPart(reasoning: string) {
  return { type: 'reasoning', reasoning, details: [{ type: 'text', text: reasoning }] };
}

function toolPart(invocation: Record<string, unknown>) {
  return { type: 'tool-invocation', toolInvocation: invocation };
}

/** The flattened text of a message, the way `saveMessages` computes it. */
function textOf(parts: unknown[]): string {
  return parts
    .filter((p): p is { type: string; text: string } =>
      !!p && typeof p === 'object' && (p as { type?: string }).type === 'text')
    .map((p) => p.text)
    .join('');
}

function reporter(sdk: unknown, options = {}) {
  return new SynapMastraStreamReporter(sdk, 'alice', '', options);
}

// ── the reporter ─────────────────────────────────────────────────────────────

describe('silent for a caller who never opened a stream', () => {
  it('reports nothing at all when isListening says no', async () => {
    const { sdk, calls } = streamingSdk({ listening: false });
    const parts = [textPart('where is my order')];
    const sent = await reporter(sdk).report(message('user', parts), {
      threadId: THREAD, role: 'user', text: textOf(parts),
    });
    expect(calls).toEqual([]);
    expect(sent).toBe(false);
  });

  it('reports nothing when the sdk has no instance namespace', async () => {
    // The shape of an older SDK, or of a user's own stub.
    const plain = makeSdk();
    const parts = [textPart('hi')];
    expect(
      await reporter(plain).report(message('user', parts), {
        threadId: THREAD, role: 'user', text: 'hi',
      }),
    ).toBe(false);
  });
});

describe('a turn becomes the event the server persists', () => {
  it('a user message is user_message with the thread as the conversation', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [textPart('where is my order')];
    const sent = await reporter(sdk).report(message('user', parts), {
      threadId: THREAD, role: 'user', text: textOf(parts),
    });
    expect(sent).toBe(true);
    expect(sentTo(calls, 'send_message')).toMatchObject({
      role: 'user',
      event_type: 'user_message',
      content: 'where is my order',
      conversation_id: THREAD,
      user_id: 'alice',
    });
  });

  it('an assistant message is assistant_message, the event anticipation wakes on', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [textPart('it ships Friday')];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: textOf(parts),
    });
    expect(sentTo(calls, 'send_message')).toMatchObject({
      role: 'assistant',
      event_type: 'assistant_message',
      content: 'it ships Friday',
    });
  });

  it('a system message is not a turn and is never reported', async () => {
    // The stream has no event for the instructions, and reporting one as an
    // assistant turn would wake anticipation with the system prompt.
    const { sdk, calls } = streamingSdk();
    const parts = [textPart('You are a helpful assistant.')];
    const sent = await reporter(sdk).report(message('system', parts), {
      threadId: THREAD, role: 'system', text: textOf(parts),
    });
    expect(calls).toEqual([]);
    expect(sent).toBe(false);
  });

  it('an empty customer_id is dropped rather than sent', async () => {
    // A B2C instance refuses a call carrying a customer_id at all.
    const { sdk, calls } = streamingSdk();
    await reporter(sdk).report(message('user', [textPart('hi')]), {
      threadId: THREAD, role: 'user', text: 'hi',
    });
    expect(sentTo(calls, 'send_message')).not.toHaveProperty('customer_id');
  });

  it('carries the customer_id on a B2B instance', async () => {
    const { sdk, calls } = streamingSdk();
    const b2b = new SynapMastraStreamReporter(sdk, 'alice', 'cust-9', {});
    await b2b.report(message('user', [textPart('hi')]), {
      threadId: THREAD, role: 'user', text: 'hi',
    });
    expect(sentTo(calls, 'send_message')).toMatchObject({ customer_id: 'cust-9' });
  });

  it('a message with no thread has no conversation to belong to', async () => {
    const { sdk, calls } = streamingSdk();
    const sent = await reporter(sdk).report(
      message('user', [textPart('hi')], { threadId: '' }),
      { threadId: '', role: 'user', text: 'hi' },
    );
    expect(calls).toEqual([]);
    expect(sent).toBe(false);
  });
});

describe('the middle of the turn, which has no other home', () => {
  it('reports a reasoning part as a thinking event', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [reasoningPart('I should check the order table'), textPart('one moment')];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: textOf(parts),
    });
    expect(sentTo(calls, 'record_thinking')).toMatchObject({
      content: 'I should check the order table',
      thought_type: 'model_thought',
      step_index: 1,
      conversation_id: THREAD,
    });
  });

  it('reports a tool call with Mastra’s own call id', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({
        state: 'call', toolCallId: 'call_abc', toolName: 'lookup_order',
        args: { orderId: 'A-1' },
      }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(sentTo(calls, 'record_tool_call')).toMatchObject({
      tool_name: 'lookup_order',
      tool_args: { orderId: 'A-1' },
      tool_call_id: 'call_abc',
    });
  });

  it('a tool call and its result carry the same id, so they pair', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({
        state: 'result', toolCallId: 'call_abc', toolName: 'lookup_order',
        args: { orderId: 'A-1' }, result: { status: 'shipped' },
      }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(sentTo(calls, 'record_tool_call')?.['tool_call_id']).toBe('call_abc');
    expect(sentTo(calls, 'record_tool_result')).toMatchObject({
      tool_call_id: 'call_abc',
      tool_name: 'lookup_order',
      result: { status: 'shipped' },
    });
  });

  it('a failed tool still said something, and says it', async () => {
    // Mastra puts the failure in errorText, not in result. Anticipation reads
    // a search that failed very differently from one still running.
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({
        state: 'output-error', toolCallId: 'call_err', toolName: 'lookup_order',
        args: {}, errorText: 'order service timed out',
      }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(sentTo(calls, 'record_tool_result')).toMatchObject({
      result: 'order service timed out',
      tool_call_id: 'call_err',
    });
  });

  it('a partial call has not finished being asked for, so it is not reported', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({ state: 'partial-call', toolCallId: 'call_p', toolName: 'lookup_order', args: {} }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(methodsIn(calls)).toEqual([]);
  });

  it('tool arguments go through the helper, so a JSON string is not dropped', async () => {
    // The SDK forwards tool_args only when it is a plain object and drops
    // anything else without a word: the call would arrive claiming the tool
    // takes no arguments.
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({ state: 'call', toolCallId: 'c1', toolName: 't', args: '{"q":"hi"}' }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(sentTo(calls, 'record_tool_call')?.['tool_args']).toEqual({ q: 'hi' });
  });

  it('falls back to rawInput when Mastra could not parse the arguments', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({ state: 'call', toolCallId: 'c1', toolName: 't', rawInput: { q: 'raw' } }),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(sentTo(calls, 'record_tool_call')?.['tool_args']).toEqual({ q: 'raw' });
  });

  it('reports a message that is only tool calls, which carries no text', async () => {
    // This is why the report sits above saveMessages' text guard: an assistant
    // message whose only parts are tool calls would otherwise be skipped, and
    // the tool events have no other durable home.
    const { sdk, calls } = streamingSdk();
    const parts = [
      { type: 'step-start' },
      toolPart({ state: 'call', toolCallId: 'c1', toolName: 'lookup_order', args: {} }),
    ];
    const sent = await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(methodsIn(calls)).toEqual(['record_tool_call']);
    // No turn was sent, so the caller's REST path is still the only record of
    // a turn — there just is not one on this message.
    expect(sent).toBe(false);
  });

  it('reports why, then what it did, then what it said', async () => {
    // assistant_message is the event anticipation wakes on. Waking it before
    // the tool calls it is meant to reason about defeats the point.
    const { sdk, calls } = streamingSdk();
    const parts = [
      reasoningPart('check the order'),
      toolPart({
        state: 'result', toolCallId: 'c1', toolName: 'lookup_order',
        args: {}, result: 'shipped',
      }),
      textPart('it shipped'),
    ];
    await reporter(sdk).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: textOf(parts),
    });
    expect(methodsIn(calls)).toEqual([
      'record_thinking', 'record_tool_call', 'record_tool_result', 'send_message',
    ]);
  });

  it('honours reportThoughts: false without losing the rest', async () => {
    const { sdk, calls } = streamingSdk();
    const parts = [reasoningPart('private'), textPart('hello')];
    await reporter(sdk, { reportThoughts: false }).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: textOf(parts),
    });
    expect(methodsIn(calls)).toEqual(['send_message']);
  });

  it('honours reportToolResults: false but still reports the call', async () => {
    // A tool result is the caller's own customer data. The call itself is not.
    const { sdk, calls } = streamingSdk();
    const parts = [
      toolPart({
        state: 'result', toolCallId: 'c1', toolName: 't', args: {}, result: 'secret',
      }),
    ];
    await reporter(sdk, { reportToolResults: false }).report(message('assistant', parts), {
      threadId: THREAD, role: 'assistant', text: '',
    });
    expect(methodsIn(calls)).toEqual(['record_tool_call']);
  });
});

describe('Mastra re-saves a message it has already saved', () => {
  it('reports the turn once and tells the caller the server has it', async () => {
    // Filling a tool result into an existing assistant message moves that
    // message back to the unsaved set, so the same text arrives twice.
    const { sdk, calls } = streamingSdk();
    const r = reporter(sdk);
    const m = message('assistant', [textPart('it shipped')]);
    const info = { threadId: THREAD, role: 'assistant', text: 'it shipped' };

    expect(await r.report(m, info)).toBe(true);
    expect(await r.report(m, info)).toBe(true);

    expect(calls.filter((c) => c.method === 'send_message')).toHaveLength(1);
  });

  it('does not re-report a tool call when its result arrives on the second save', async () => {
    const { sdk, calls } = streamingSdk();
    const r = reporter(sdk);
    const called = message('assistant', [
      toolPart({ state: 'call', toolCallId: 'c1', toolName: 't', args: { q: 1 } }),
    ]);
    const resolved = message('assistant', [
      toolPart({
        state: 'result', toolCallId: 'c1', toolName: 't', args: { q: 1 }, result: 'ok',
      }),
    ]);
    const info = { threadId: THREAD, role: 'assistant', text: '' };

    await r.report(called, info);
    await r.report(resolved, info);
    // And a third save of the settled message repeats neither.
    await r.report(resolved, info);

    expect(methodsIn(calls)).toEqual(['record_tool_call', 'record_tool_result']);
  });

  it('does not re-report the reasoning on a message saved twice', async () => {
    const { sdk, calls } = streamingSdk();
    const r = reporter(sdk);
    const m = message('assistant', [reasoningPart('check the order')]);
    const info = { threadId: THREAD, role: 'assistant', text: '' };

    await r.report(m, info);
    await r.report(m, info);

    expect(methodsIn(calls)).toEqual(['record_thinking']);
  });

  it('reports a message again when its text actually changed', async () => {
    const { sdk, calls } = streamingSdk();
    const r = reporter(sdk);
    const info = { threadId: THREAD, role: 'assistant', text: 'it' };
    await r.report(message('assistant', [textPart('it')]), info);
    await r.report(message('assistant', [textPart('it shipped')]), {
      ...info, text: 'it shipped',
    });
    expect(calls.filter((c) => c.method === 'send_message').map((c) =>
      (c.options as Record<string, unknown>)['content'])).toEqual(['it', 'it shipped']);
  });

  it('does not suppress a turn whose send failed, so the retry still goes out', async () => {
    // Remembering a key before the send succeeded would lose the turn twice
    // over: once to the failure, once to the de-duplication.
    const calls: StreamCall[] = [];
    let failNext = true;
    const sdk = {
      ...makeSdk(),
      instance: {
        isListening: () => true,
        send_message: vi.fn(async (options: unknown) => {
          if (failNext) {
            failNext = false;
            throw new Error('stream hiccup');
          }
          calls.push({ method: 'send_message', options });
        }),
      },
    };
    const r = reporter(sdk);
    const m = message('assistant', [textPart('it shipped')]);
    const info = { threadId: THREAD, role: 'assistant', text: 'it shipped' };

    expect(await r.report(m, info)).toBe(false);
    expect(await r.report(m, info)).toBe(true);
    expect(calls).toHaveLength(1);
  });

  it('a message with no id is reported every time, because nothing identifies it', async () => {
    const { sdk, calls } = streamingSdk();
    const r = reporter(sdk);
    const info = { threadId: THREAD, role: 'user', text: 'yes' };
    await r.report(message('user', [textPart('yes')], { id: '' }), info);
    await r.report(message('user', [textPart('yes')], { id: '' }), info);
    expect(calls.filter((c) => c.method === 'send_message')).toHaveLength(2);
  });
});

describe('never breaks the agent run', () => {
  it('a message that throws while being read is skipped, not raised', async () => {
    // A hook runs inside the Agent's own save path. An exception from
    // telemetry there ends the run, and it must instead fall back to REST.
    const { sdk, calls } = streamingSdk();
    const hostile = {
      id: 'm1',
      role: 'assistant',
      get content(): unknown {
        throw new Error('this message is not what it looks like');
      },
    };
    const sent = await reporter(sdk).report(hostile, {
      threadId: THREAD, role: 'assistant', text: 'hello',
    });
    expect(sent).toBe(false);
    expect(calls).toEqual([]);
  });

  it('a malformed message is skipped rather than thrown', async () => {
    // Both shapes a hand-built or half-migrated message arrives in.
    for (const broken of [
      { id: 'm1', role: 'assistant', content: null },
      { id: 'm2', role: 'assistant', content: { format: 2, parts: 'not an array' } },
    ]) {
      const { sdk, calls } = streamingSdk();
      const sent = await reporter(sdk).report(broken, {
        threadId: THREAD, role: 'assistant', text: 'hello',
      });
      // The parts were unreadable, but the turn itself still went out.
      expect(sent).toBe(true);
      expect(methodsIn(calls)).toEqual(['send_message']);
    }
  });
});

// ── SynapMemory.saveMessages: stream first, REST second, never both ──────────

function memoryOver(sdk: unknown) {
  return new SynapMemory({ sdk: sdk as SynapSdkLike, userId: 'alice' });
}

function recordMessage(sdk: unknown) {
  return (sdk as SynapSdkLike).conversation.record_message as ReturnType<typeof vi.fn>;
}

describe('SynapMemory.saveMessages with a stream open', () => {
  it('sends the turn on the stream and does NOT also write it over REST', async () => {
    // The server persists a turn it receives on the stream. Writing it over
    // REST as well stores the same turn twice.
    const { sdk, calls } = streamingSdk();
    const m = message('user', [textPart('where is my order')]);
    await memoryOver(sdk).saveMessages({ messages: [m] });

    expect(sentTo(calls, 'send_message')).toMatchObject({
      event_type: 'user_message', content: 'where is my order',
    });
    expect(recordMessage(sdk)).not.toHaveBeenCalled();
  });

  it('reports the tool calls on a message that carries no text', async () => {
    const { sdk, calls } = streamingSdk();
    const m = message('assistant', [
      toolPart({ state: 'call', toolCallId: 'c1', toolName: 'lookup_order', args: { id: 1 } }),
    ]);
    await memoryOver(sdk).saveMessages({ messages: [m] });

    expect(sentTo(calls, 'record_tool_call')).toMatchObject({
      tool_name: 'lookup_order', tool_call_id: 'c1',
    });
  });

  it('falls back to REST when the stream refuses the turn', async () => {
    const { sdk } = streamingSdk({ throws: true });
    const m = message('user', [textPart('hi')]);
    await memoryOver(sdk).saveMessages({ messages: [m] });

    expect(recordMessage(sdk)).toHaveBeenCalledWith({
      conversation_id: THREAD,
      role: 'user',
      content: 'hi',
      user_id: 'alice',
      customer_id: '',
    });
  });

  it('keeps a system message on the REST path, where it always was', async () => {
    const { sdk, calls } = streamingSdk();
    const m = message('system', [textPart('You are helpful.')]);
    await memoryOver(sdk).saveMessages({ messages: [m] });

    expect(calls).toEqual([]);
    expect(recordMessage(sdk)).toHaveBeenCalledTimes(1);
  });

  it('still returns the messages it handled', async () => {
    const { sdk } = streamingSdk();
    const m = message('user', [textPart('hi')]);
    const result = await memoryOver(sdk).saveMessages({ messages: [m] });
    expect(result).toEqual({ messages: [m] });
  });
});

describe('SynapMemory.saveMessages for someone who never opted in', () => {
  it('is exactly the REST write it has always been', async () => {
    const sdk = makeSdk();
    const m = message('user', [textPart('where is my order')]);
    const result = await memoryOver(sdk).saveMessages({ messages: [m] });

    expect(recordMessage(sdk)).toHaveBeenCalledTimes(1);
    expect(recordMessage(sdk)).toHaveBeenCalledWith({
      conversation_id: THREAD,
      role: 'user',
      content: 'where is my order',
      user_id: 'alice',
      customer_id: '',
    });
    expect(result).toEqual({ messages: [m] });
  });

  it('writes both turns of an exchange, once each', async () => {
    const sdk = makeSdk();
    await memoryOver(sdk).saveMessages({
      messages: [
        message('user', [textPart('hi')], { id: 'm1' }),
        message('assistant', [textPart('hello')], { id: 'm2' }),
      ],
    });
    expect(recordMessage(sdk)).toHaveBeenCalledTimes(2);
  });

  it('a stream that exists but is not listening changes nothing', async () => {
    const { sdk, calls } = streamingSdk({ listening: false });
    await memoryOver(sdk).saveMessages({
      messages: [message('user', [textPart('hi')])],
    });
    expect(calls).toEqual([]);
    expect(recordMessage(sdk)).toHaveBeenCalledTimes(1);
  });
});
