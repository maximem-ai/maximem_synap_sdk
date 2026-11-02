/**
 * scopePath on the hooks: the fetch, the user prompt and the assistant reply
 * carry it; tool calls, tool results and the session end do not (the SDK takes
 * none there and the server reads the path from turns only). A conversation-only
 * read (the short-term hook) still works with a package configured with a path.
 */
import { describe, expect, it, vi } from 'vitest';

import { createSynapHooks } from '../hooks.js';
import { createSynapShortTermHook } from '../short_term.js';
import type { SynapSdkLike } from '../types.js';

const PATH = { customer: 'acme', team: 'eng', user: 'u1' };
const BASE = { session_id: 'sess-1', transcript_path: '/tmp/t.jsonl', cwd: '/tmp' };

interface StreamCall { method: string; options: unknown }

function makeSdk(opts: { listening: boolean }) {
  const stream: StreamCall[] = [];
  const record = vi.fn(async (_o: Record<string, unknown>) => ({}));
  const push = (method: string) => async (options: unknown) => { stream.push({ method, options }); };
  const sdk = {
    // The marker of an SDK that carries `scope_path` (>= 0.7.1); see scope-guard.ts.
    get_scope_ladder: vi.fn(),
    fetch: vi.fn(async (_o: Record<string, unknown>) => ({ formatted_context: 'ctx' })),
    conversation: {
      record_message: record,
      context: { get_context_for_prompt: vi.fn(async (_o: Record<string, unknown>) => ({ formatted_context: 'st', available: true })) },
    },
    memories: { create: vi.fn(async () => ({ ingestion_id: 'i' })) },
    instance: {
      isListening: () => opts.listening,
      send_message: push('send_message'),
      record_tool_call: push('record_tool_call'),
      record_tool_result: push('record_tool_result'),
      record_thinking: push('record_thinking'),
      end_session: push('end_session'),
    },
  };
  return { sdk: sdk as unknown as SynapSdkLike, raw: sdk, stream, record };
}

type Hooks = ReturnType<typeof createSynapHooks>;
async function fire(hooks: Hooks, event: keyof Hooks, input: object) {
  const cb = hooks[event]![0]!.hooks[0]!;
  return cb({ ...BASE, ...input } as never, undefined, { signal: new AbortController().signal });
}
const keys = (o: unknown) => Object.keys(o as object);

async function wholeTurn(hooks: Hooks) {
  await fire(hooks, 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', prompt: 'hello' });
  await fire(hooks, 'PreToolUse', { hook_event_name: 'PreToolUse', tool_name: 't', tool_input: {}, tool_use_id: 'a' });
  await fire(hooks, 'PostToolUse', { hook_event_name: 'PostToolUse', tool_name: 't', tool_input: {}, tool_response: 'r', tool_use_id: 'a' });
  await fire(hooks, 'Stop', { hook_event_name: 'Stop', last_assistant_message: 'done', stop_hook_active: false });
  await fire(hooks, 'SessionEnd', { hook_event_name: 'SessionEnd', reason: 'other' });
}

describe('createSynapHooks scopePath, stream open', () => {
  it('fetch and both turns carry it; tool call, tool result and end do not', async () => {
    const { sdk, raw, stream, record } = makeSdk({ listening: true });
    await wholeTurn(createSynapHooks({ sdk, userId: 'u1', customerId: 'acme', scopePath: PATH }));
    expect((raw.fetch.mock.calls[0]![0])['scope_path']).toEqual(PATH);
    expect(stream.map((c) => c.method)).toEqual(
      ['send_message', 'record_tool_call', 'record_tool_result', 'send_message', 'end_session']);
    expect((stream[0]!.options as Record<string, unknown>)['scope_path']).toEqual(PATH);
    expect((stream[3]!.options as Record<string, unknown>)['scope_path']).toEqual(PATH);
    expect(keys(stream[1]!.options)).not.toContain('scope_path');
    expect(keys(stream[2]!.options)).not.toContain('scope_path');
    expect(record).not.toHaveBeenCalled();
  });

  it('without a path nothing carries one', async () => {
    for (const scopePath of [undefined, {}]) {
      const { sdk, raw, stream } = makeSdk({ listening: true });
      await wholeTurn(createSynapHooks({ sdk, userId: 'u1', scopePath }));
      expect(keys(raw.fetch.mock.calls[0]![0])).not.toContain('scope_path');
      for (const c of stream) expect(keys(c.options)).not.toContain('scope_path');
    }
  });
});

describe('createSynapHooks scopePath, no stream (REST)', () => {
  it('the prompt and the reply reach record_message with scope_path', async () => {
    const { sdk, record } = makeSdk({ listening: false });
    process.env['SYNAP_AUTO_LISTEN'] = 'false';
    try {
      await wholeTurn(createSynapHooks({ sdk, userId: 'u1', customerId: 'acme', scopePath: PATH }));
    } finally { delete process.env['SYNAP_AUTO_LISTEN']; }
    expect(record.mock.calls.map((c) => [c[0]['role'], c[0]['scope_path']])).toEqual([
      ['user', PATH], ['assistant', PATH],
    ]);
  });

  it('without a path the REST payloads are exactly what they always were', async () => {
    const { sdk, record } = makeSdk({ listening: false });
    process.env['SYNAP_AUTO_LISTEN'] = 'false';
    try {
      await wholeTurn(createSynapHooks({ sdk, userId: 'u1' }));
    } finally { delete process.env['SYNAP_AUTO_LISTEN']; }
    expect(record.mock.calls.map((c) => c[0])).toEqual([
      { conversation_id: 'sess-1', role: 'user', content: 'hello', user_id: 'u1' },
      { conversation_id: 'sess-1', role: 'assistant', content: 'done', user_id: 'u1' },
    ]);
  });
});

describe('a conversation-only read beside a configured path', () => {
  it('the short-term hook asks for the conversation id (and a style) alone', async () => {
    const { sdk, raw } = makeSdk({ listening: false });
    createSynapHooks({ sdk, userId: 'u1', customerId: 'acme', scopePath: PATH });
    const hook = createSynapShortTermHook({ sdk, conversationId: 'c1' });
    const cb = (hook as Record<string, Array<{ hooks: Array<(...a: never[]) => Promise<unknown>> }>>)['UserPromptSubmit']![0]!.hooks[0]!;
    await cb({ ...BASE, hook_event_name: 'UserPromptSubmit', prompt: 'hi' } as never, undefined as never, { signal: new AbortController().signal } as never);
    const arg = raw.conversation.context.get_context_for_prompt.mock.calls[0]![0];
    expect(keys(arg).filter((k) => /scope/i.test(k))).toEqual([]);
    expect(arg['conversation_id']).toBe('c1');
  });
});
