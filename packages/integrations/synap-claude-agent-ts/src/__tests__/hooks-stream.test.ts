/**
 * The four Synap stream events this integration can actually produce.
 *
 * Before this, `createSynapHooks` installed one hook and reported one thing:
 * the user's prompt, over REST. The stream carries five per-turn events and
 * the agent SDK gives us a hook for four of them. The fifth, `agent_thinking`,
 * has no hook at all: `HookEvent` has 27 members and not one carries the
 * model's reasoning text.
 *
 * Two properties matter more than any individual mapping, and both were wrong
 * somewhere in the fleet before:
 *
 *   1. **Stream first, REST only as a fallback, never both.** The server
 *      persists a turn it receives on the stream itself. Calling
 *      `record_message` after a successful report writes the turn twice.
 *   2. **Byte-identical behaviour with no stream.** Most users never call
 *      `listen()`. For them every one of these hooks must be a no-op, and the
 *      REST path must behave exactly as it did.
 *
 * Assertions are on what the fake SDK RECEIVED. A test that only checks a hook
 * returned `{}` passes on a hook that reports nothing whatsoever, which is
 * precisely the bug.
 */
import { describe, expect, it, vi } from 'vitest';

import { createSynapHooks } from '../hooks.js';
import type { SynapSdkLike } from '../types.js';

interface StreamCall { method: string; options: unknown }

/**
 * An SDK with a live `instance` namespace, the way a user who called
 * `listen()` has one. `listening: false` is the far more common shape.
 */
function makeSdk(opts: { listening?: boolean } = {}) {
  const stream: StreamCall[] = [];
  const recordMessage = vi.fn(async () => ({}));
  const push = (method: string) => async (options: unknown) => {
    stream.push({ method, options });
  };
  const sdk = {
    fetch: vi.fn(async () => ({ formatted_context: '', facts: [] })),
    conversation: { record_message: recordMessage },
    memories: { create: vi.fn(async () => ({ ingestion_id: 'i' })) },
    instance: {
      isListening: () => opts.listening === true,
      send_message: push('send_message'),
      record_tool_call: push('record_tool_call'),
      record_tool_result: push('record_tool_result'),
      record_thinking: push('record_thinking'),
      end_session: push('end_session'),
    },
  };
  return { sdk: sdk as unknown as SynapSdkLike, stream, recordMessage };
}

type Hooks = ReturnType<typeof createSynapHooks>;

function callbackFor(hooks: Hooks, event: keyof Hooks) {
  const cb = hooks[event]?.[0]?.hooks?.[0];
  if (!cb) throw new Error(`no hook installed for ${String(event)}`);
  return cb;
}

const BASE = { session_id: 'sess-1', transcript_path: '/tmp/t.jsonl', cwd: '/tmp' };

/** Invoke a hook the way the agent SDK does: input, tool-use id, context. */
async function fire(hooks: Hooks, event: keyof Hooks, input: object) {
  const cb = callbackFor(hooks, event);
  return cb({ ...BASE, ...input } as never, undefined, { signal: new AbortController().signal });
}

function hooksFor(sdk: SynapSdkLike, extra: object = {}) {
  return createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1', ...extra });
}

function sent(stream: StreamCall[], method: string): Record<string, unknown>[] {
  return stream.filter((c) => c.method === method)
    .map((c) => c.options as Record<string, unknown>);
}

describe('every event the agent SDK can give us is installed', () => {
  it('installs a hook for each of the six events', () => {
    const { sdk } = makeSdk();
    const hooks = hooksFor(sdk);
    for (const event of ['UserPromptSubmit', 'PreToolUse', 'PostToolUse',
      'PostToolUseFailure', 'Stop', 'SessionEnd'] as const) {
      expect(callbackFor(hooks, event), `${event} is not installed`).toBeTypeOf('function');
    }
  });
});

describe('the user turn goes over the stream, not over both', () => {
  it('reports the prompt on the stream when one is listening', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: 'hello',
    });

    expect(sent(stream, 'send_message')[0]).toMatchObject({
      content: 'hello', role: 'user', event_type: 'user_message',
      conversation_id: 'conv-1', user_id: 'u1',
    });
  });

  it('and does NOT also write it over REST', async () => {
    // The double write. The server persists a stream turn by itself.
    const { sdk, recordMessage } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: 'hello',
    });
    expect(recordMessage).not.toHaveBeenCalled();
  });

  it('falls back to REST when there is no stream', async () => {
    const { sdk, recordMessage, stream } = makeSdk({ listening: false });
    await fire(hooksFor(sdk), 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: 'hello',
    });
    expect(recordMessage).toHaveBeenCalledOnce();
    expect(stream).toEqual([]);
  });

  it('falls back to REST when the stream report fails', async () => {
    // A stream that is listening but throws must not silently lose the turn.
    const { sdk, recordMessage } = makeSdk({ listening: true });
    (sdk as unknown as { instance: { send_message: unknown } }).instance.send_message =
      async () => { throw new Error('stream is having a bad day'); };
    await fire(hooksFor(sdk), 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: 'hello',
    });
    expect(recordMessage).toHaveBeenCalledOnce();
  });

  it('records nothing on either path when recording is off', async () => {
    const { sdk, recordMessage, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk, { recordUserPrompts: false }), 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: 'hello',
    });
    expect(recordMessage).not.toHaveBeenCalled();
    expect(stream).toEqual([]);
  });
});

describe('the assistant turn, which is what wakes anticipation', () => {
  it('reports last_assistant_message as an assistant_message', async () => {
    // ⚠ Without this event the stream carries the whole turn and the
    // anticipation agent never runs on any of it.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false,
      last_assistant_message: 'here is the answer',
    });

    expect(sent(stream, 'send_message')[0]).toMatchObject({
      content: 'here is the answer',
      role: 'assistant',
      event_type: 'assistant_message',
    });
  });

  it('sends nothing when the SDK gave no assistant text', async () => {
    // The field is optional in the SDK's own types. An empty assistant turn is
    // a worse signal than no turn.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false,
    });
    expect(stream).toEqual([]);
  });

  it('sends nothing for whitespace-only assistant text', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false,
      last_assistant_message: '   \n ',
    });
    expect(stream).toEqual([]);
  });

  it('never writes the assistant turn over REST', async () => {
    // There was no REST write for the assistant before. Adding one would start
    // persisting data an existing caller never asked for.
    const { sdk, recordMessage } = makeSdk({ listening: false });
    await fire(hooksFor(sdk), 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false,
      last_assistant_message: 'here is the answer',
    });
    expect(recordMessage).not.toHaveBeenCalled();
  });

  it('honours recordUserPrompts: false', async () => {
    // Both halves of the conversation are persisted server-side, so the one
    // switch governs both. Recording one and not the other would be strange.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk, { recordUserPrompts: false }), 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false,
      last_assistant_message: 'here is the answer',
    });
    expect(stream).toEqual([]);
  });
});

describe('tool calls and their results', () => {
  const PRE = {
    hook_event_name: 'PreToolUse', tool_name: 'Read',
    tool_input: { file_path: '/a.txt' }, tool_use_id: 'toolu_1',
  };
  const POST = {
    hook_event_name: 'PostToolUse', tool_name: 'Read',
    tool_input: { file_path: '/a.txt' }, tool_response: 'file contents',
    tool_use_id: 'toolu_1',
  };

  it('reports the call with its arguments', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PreToolUse', PRE);
    expect(sent(stream, 'record_tool_call')[0]).toMatchObject({
      tool_name: 'Read', tool_args: { file_path: '/a.txt' },
    });
  });

  it('reports the result', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PostToolUse', POST);
    expect(sent(stream, 'record_tool_result')[0]).toMatchObject({
      result: 'file contents', tool_name: 'Read',
    });
  });

  it('the call and the result carry the SDK\'s own id, so they pair', async () => {
    // Both read `tool_use_id`, so the pair cannot drift. A minted id pairs
    // nothing; two calls with id '' pair with each other's results.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PreToolUse', PRE);
    await fire(hooksFor(sdk), 'PostToolUse', POST);
    expect(sent(stream, 'record_tool_call')[0]?.['tool_call_id']).toBe('toolu_1');
    expect(sent(stream, 'record_tool_result')[0]?.['tool_call_id']).toBe('toolu_1');
  });

  it('a failed tool is still a result', async () => {
    // A tool that failed told the agent something, and it is usually why the
    // next turn goes the way it does.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PostToolUseFailure', {
      hook_event_name: 'PostToolUseFailure', tool_name: 'Read',
      tool_input: {}, tool_use_id: 'toolu_1', error: 'no such file',
    });
    expect(sent(stream, 'record_tool_result')[0]).toMatchObject({
      result: { error: 'no such file', is_interrupt: false },
      tool_call_id: 'toolu_1',
    });
  });

  it('an interrupt is marked as one', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PostToolUseFailure', {
      hook_event_name: 'PostToolUseFailure', tool_name: 'Bash',
      tool_input: {}, tool_use_id: 'toolu_2', error: 'cancelled',
      is_interrupt: true,
    });
    expect((sent(stream, 'record_tool_result')[0]?.['result'] as Record<string, unknown>))
      .toMatchObject({ is_interrupt: true });
  });

  it('tool events are silent with no stream, and never touch REST', async () => {
    const { sdk, stream, recordMessage } = makeSdk({ listening: false });
    const hooks = hooksFor(sdk);
    await fire(hooks, 'PreToolUse', PRE);
    await fire(hooks, 'PostToolUse', POST);
    expect(stream).toEqual([]);
    expect(recordMessage).not.toHaveBeenCalled();
  });

  it('tool events are NOT gated by recordUserPrompts', async () => {
    // They are anticipation-only and never persisted, so the switch that
    // governs stored conversation has nothing to say about them.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk, { recordUserPrompts: false }), 'PreToolUse', PRE);
    expect(sent(stream, 'record_tool_call')).toHaveLength(1);
  });
});

describe('the end of a session', () => {
  it('closes the conversation', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'SessionEnd', {
      hook_event_name: 'SessionEnd', reason: 'clear',
    });
    expect(stream).toEqual([{ method: 'end_session', options: 'conv-1' }]);
  });

  it('is silent with no stream', async () => {
    const { sdk, stream } = makeSdk({ listening: false });
    await fire(hooksFor(sdk), 'SessionEnd', {
      hook_event_name: 'SessionEnd', reason: 'clear',
    });
    expect(stream).toEqual([]);
  });
});

describe('the conversation id', () => {
  it('falls back to the agent session id when none was configured', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    await fire(hooks, 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read',
      tool_input: {}, tool_use_id: 't1',
    });
    expect(sent(stream, 'record_tool_call')[0]).toMatchObject({
      conversation_id: 'sess-1',
    });
  });

  it('a configured id wins over the session id', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read',
      tool_input: {}, tool_use_id: 't1',
    });
    expect(sent(stream, 'record_tool_call')[0]).toMatchObject({
      conversation_id: 'conv-1',
    });
  });

  it('no customer_id key at all when there is none', async () => {
    // A B2C instance refuses a call carrying one, so '' turns every ordinary
    // turn into a refusal.
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk), 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read',
      tool_input: {}, tool_use_id: 't1',
    });
    expect(sent(stream, 'record_tool_call')[0]).not.toHaveProperty('customer_id');
  });

  it('carries a customer_id when there is one', async () => {
    const { sdk, stream } = makeSdk({ listening: true });
    await fire(hooksFor(sdk, { customerId: 'cust-9' }), 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read',
      tool_input: {}, tool_use_id: 't1',
    });
    expect(sent(stream, 'record_tool_call')[0]).toMatchObject({
      customer_id: 'cust-9',
    });
  });
});

describe('a hook never breaks the agent run', () => {
  it('survives a stream namespace that throws on every call', async () => {
    const { sdk } = makeSdk({ listening: true });
    const inst = (sdk as unknown as { instance: Record<string, unknown> }).instance;
    for (const k of ['send_message', 'record_tool_call', 'record_tool_result',
      'end_session']) {
      inst[k] = async () => { throw new Error('nope'); };
    }
    const hooks = hooksFor(sdk);
    await expect(fire(hooks, 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'R', tool_input: {}, tool_use_id: 't',
    })).resolves.toEqual({});
    await expect(fire(hooks, 'PostToolUse', {
      hook_event_name: 'PostToolUse', tool_name: 'R', tool_input: {},
      tool_response: 'x', tool_use_id: 't',
    })).resolves.toEqual({});
    await expect(fire(hooks, 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false, last_assistant_message: 'a',
    })).resolves.toEqual({});
    await expect(fire(hooks, 'SessionEnd', {
      hook_event_name: 'SessionEnd', reason: 'clear',
    })).resolves.toEqual({});
  });
});
