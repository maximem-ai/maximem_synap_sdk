/**
 * The default path: no `listen()` from the caller, and the stream is used.
 *
 * Every other hook test hands the hooks an SDK that either has a stream or
 * cannot make one. This one hands them the common case: a stream that is closed
 * until somebody calls `listen()`, the way the real SDK behaves. The first hook
 * that fires must open it, and everything after it must travel on it.
 *
 * Also here: Synap's compaction reaching Claude Code. The CLI owns the
 * transcript, so the one place Synap's compaction can enter is a SessionStart
 * hook's `additionalContext`.
 */
import { describe, expect, it, vi } from 'vitest';

import { createSynapHooks } from '../hooks.js';
import { setAutoListen } from '../stream-events.js';
import type { SynapSdkLike } from '../types.js';

interface Call { method: string; options: unknown }

function closedSdk(context: unknown = { context_block: null }) {
  const calls: Call[] = [];
  const state = { listening: false, listens: 0 };
  const push = (method: string) => async (options: unknown) => { calls.push({ method, options }); };
  const recordMessage = vi.fn(async () => ({}));
  const sdk = {
    fetch: vi.fn(async () => ({ formatted_context: '' })),
    conversation: {
      record_message: recordMessage,
      context: { get_context_for_prompt: vi.fn(async () => context) },
    },
    memories: { create: vi.fn(async () => ({ ingestion_id: 'i' })) },
    instance: {
      get is_listening() { return state.listening; },
      async listen() {
        if (state.listening) throw new Error('already active');
        state.listens += 1;
        state.listening = true;
      },
      send_message: push('send_message'),
      record_tool_call: push('record_tool_call'),
      record_tool_result: push('record_tool_result'),
      end_session: push('end_session'),
    },
  };
  return { sdk: sdk as unknown as SynapSdkLike, calls, state, recordMessage };
}

const BASE = { session_id: 'sess-1', transcript_path: '/tmp/t.jsonl', cwd: '/tmp' };

type Hooks = ReturnType<typeof createSynapHooks>;
async function fire(hooks: Hooks, event: keyof Hooks, input: object) {
  const cb = hooks[event]?.[0]?.hooks?.[0];
  if (!cb) throw new Error(`no hook for ${String(event)}`);
  return cb({ ...BASE, ...input } as never, undefined, { signal: new AbortController().signal });
}

describe('no listen() from the caller', () => {
  it('the first prompt opens the stream and the turn goes on it', async () => {
    const { sdk, calls, state, recordMessage } = closedSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', prompt: 'hello' });
    expect(state.listens).toBe(1);
    expect(calls.map((c) => c.method)).toEqual(['send_message']);
    expect(recordMessage).not.toHaveBeenCalled();
  });

  it('the whole turn travels on that one stream', async () => {
    const { sdk, calls, state, recordMessage } = closedSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', prompt: 'hello' });
    await fire(hooks, 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read', tool_input: {}, tool_use_id: 't1',
    });
    await fire(hooks, 'PostToolUse', {
      hook_event_name: 'PostToolUse', tool_name: 'Read', tool_input: {}, tool_response: 'x', tool_use_id: 't1',
    });
    await fire(hooks, 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false, last_assistant_message: 'done',
    });
    await fire(hooks, 'SessionEnd', { hook_event_name: 'SessionEnd', reason: 'clear' });
    expect(state.listens).toBe(1);
    expect(calls.map((c) => c.method)).toEqual([
      'send_message', 'record_tool_call', 'record_tool_result', 'send_message', 'end_session',
    ]);
    expect(recordMessage).not.toHaveBeenCalled();
  });

  it('a first event that is a tool call (no prompt hook ran) still opens it', async () => {
    const { sdk, calls, state } = closedSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'PreToolUse', {
      hook_event_name: 'PreToolUse', tool_name: 'Read', tool_input: {}, tool_use_id: 't1',
    });
    expect(state.listens).toBe(1);
    expect(calls[0]?.method).toBe('record_tool_call');
  });

  it('SessionEnd on its own does not open a stream', async () => {
    const { sdk, state } = closedSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'SessionEnd', { hook_event_name: 'SessionEnd', reason: 'clear' });
    expect(state.listens).toBe(0);
  });

  it('setAutoListen(sdk, false) keeps the turn on REST', async () => {
    const { sdk, calls, state, recordMessage } = closedSdk();
    setAutoListen(sdk, false);
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', prompt: 'hello' });
    await fire(hooks, 'Stop', {
      hook_event_name: 'Stop', stop_hook_active: false, last_assistant_message: 'done',
    });
    expect(state.listens).toBe(0);
    expect(calls).toEqual([]);
    expect(recordMessage).toHaveBeenCalledTimes(2);
  });
});

describe('Synap\'s compaction reaches Claude Code', () => {
  const COMPACTED = {
    context_block: '## Conversation so far (compacted at 5)\nuser likes tea',
    compaction_id: 'cmp-1',
    recent_messages: [{ role: 'user', content: 'and coffee?' }, { role: 'assistant', content: 'also fine' }],
  };
  const PREAMBLE =
    '<synap_conversation_history>\n## Conversation so far (compacted at 5)\nuser likes tea\n</synap_conversation_history>';
  type Out = { hookSpecificOutput?: { hookEventName?: string; additionalContext?: string } };
  const prompt = (hooks: Hooks) =>
    fire(hooks, 'UserPromptSubmit', { hook_event_name: 'UserPromptSubmit', prompt: 'hello' }) as Promise<Out>;

  it('installs a SessionStart hook', () => {
    const { sdk } = closedSdk();
    expect(createSynapHooks({ sdk, userId: 'u1' }).SessionStart).toHaveLength(1);
  });

  it('the first prompt gets the compaction block, and not the turns after it', async () => {
    const { sdk } = closedSdk(COMPACTED);
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    const out = await prompt(hooks);
    expect(out.hookSpecificOutput?.additionalContext).toBe(PREAMBLE);
  });

  it('sits beside the long-term context, compaction first', async () => {
    const { sdk } = closedSdk(COMPACTED);
    (sdk.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ formatted_context: 'User is a VIP.' });
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    const ctx = (await prompt(hooks)).hookSpecificOutput?.additionalContext ?? '';
    expect(ctx.startsWith(PREAMBLE)).toBe(true);
    expect(ctx).toContain('<synap_memory>');
  });

  it('is injected once per compaction, not on every prompt', async () => {
    const { sdk } = closedSdk(COMPACTED);
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    expect((await prompt(hooks)).hookSpecificOutput).toBeDefined();
    expect(await prompt(hooks)).toEqual({});
    (sdk.conversation.context!.get_context_for_prompt as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ ...COMPACTED, compaction_id: 'cmp-2' });
    expect((await prompt(hooks)).hookSpecificOutput?.additionalContext).toBe(PREAMBLE);
  });

  it('SessionStart(compact) adds it again, because the CLI\'s compaction discards what was added', async () => {
    const { sdk } = closedSdk(COMPACTED);
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await prompt(hooks);
    const out = (await fire(hooks, 'SessionStart', { hook_event_name: 'SessionStart', source: 'compact' })) as Out;
    expect(out.hookSpecificOutput?.hookEventName).toBe('SessionStart');
    expect(out.hookSpecificOutput?.additionalContext).toBe(PREAMBLE);
    expect(out.hookSpecificOutput?.additionalContext).not.toContain('and coffee?');
  });

  it('SessionStart for every other source adds nothing', async () => {
    const { sdk } = closedSdk(COMPACTED);
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    for (const source of ['startup', 'resume', 'clear']) {
      expect(await fire(hooks, 'SessionStart', { hook_event_name: 'SessionStart', source })).toEqual({});
    }
  });

  it('adds nothing when Synap has no compaction yet', async () => {
    const { sdk } = closedSdk({ context_block: null });
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    expect(await prompt(hooks)).toEqual({});
    expect(await fire(hooks, 'SessionStart', { hook_event_name: 'SessionStart', source: 'compact' })).toEqual({});
  });

  it('adds nothing, and does not throw, when the read fails', async () => {
    const { sdk } = closedSdk();
    (sdk.conversation.context!.get_context_for_prompt as ReturnType<typeof vi.fn>)
      .mockRejectedValue(new Error('500'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    expect(await prompt(hooks)).toEqual({});
    expect(await fire(hooks, 'SessionStart', { hook_event_name: 'SessionStart', source: 'compact' })).toEqual({});
    vi.restoreAllMocks();
  });
});

describe('a prompt that is Synap\'s own history is not a turn', () => {
  it('is not recorded', async () => {
    const { sdk, calls, recordMessage } = closedSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    await fire(hooks, 'UserPromptSubmit', {
      hook_event_name: 'UserPromptSubmit', prompt: '<synap_conversation_history>\nx\n</synap_conversation_history>',
    });
    expect(calls).toEqual([]);
    expect(recordMessage).not.toHaveBeenCalled();
  });
});
