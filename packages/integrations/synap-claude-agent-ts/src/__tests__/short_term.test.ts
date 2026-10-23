/**
 * Tests for createSynapShortTermHook (src/short_term.ts)
 *
 * Strategy: Build duck-typed SynapSdkLike mocks per test; extract and invoke
 * the inner HookCallback directly. No live network, no Claude/Anthropic calls.
 */

import { describe, it, expect, vi } from 'vitest';
import { createSynapShortTermHook } from '../short_term.js';
import type { SynapSdkLike, SynapShortTermResponseLike } from '../types.js';

// ─── helpers ─────────────────────────────────────────────────────────────────

function makeSdk(
  stResponse: SynapShortTermResponseLike = {
    available: true,
    formatted_context: 'Summary: user likes TypeScript.',
  },
  shouldThrow: Error | null = null,
): SynapSdkLike {
  const getContextForPrompt = shouldThrow
    ? vi.fn(async () => {
        throw shouldThrow;
      })
    : vi.fn(async () => stResponse);

  return {
    fetch: vi.fn(async () => ({})),
    conversation: {
      record_message: vi.fn(async () => ({})),
      context: {
        get_context_for_prompt: getContextForPrompt,
      },
    },
    memories: {
      create: vi.fn(async () => ({ ingestion_id: 'x' })),
    },
  };
}

function extractCallback(hooks: ReturnType<typeof createSynapShortTermHook>) {
  const matchers = hooks.UserPromptSubmit;
  if (!matchers || matchers.length === 0) throw new Error('no UserPromptSubmit matchers');
  return matchers[0].hooks[0];
}

const dummyInput = {
  hook_event_name: 'UserPromptSubmit' as const,
  prompt: 'hello',
  session_id: 'sess-st-001',
  transcript_path: '/tmp/t.jsonl',
  cwd: '/tmp',
};
const signal = new AbortController().signal;

// ─── createSynapShortTermHook — construction guards ──────────────────────────

describe('createSynapShortTermHook — construction', () => {
  it('throws when sdk is null/falsy', () => {
    expect(() =>
      createSynapShortTermHook({
        sdk: null as unknown as SynapSdkLike,
        conversationId: 'conv-1',
      }),
    ).toThrow(/non-null sdk/);
  });

  it('throws when conversationId is empty', () => {
    const sdk = makeSdk();
    expect(() => createSynapShortTermHook({ sdk, conversationId: '' })).toThrow(
      /non-empty conversationId/,
    );
  });

  it('throws when conversationId is whitespace-only', () => {
    const sdk = makeSdk();
    expect(() => createSynapShortTermHook({ sdk, conversationId: '   ' })).toThrow(
      /non-empty conversationId/,
    );
  });

  it('throws on unsupported style', () => {
    const sdk = makeSdk();
    expect(() =>
      createSynapShortTermHook({
        sdk,
        conversationId: 'conv-1',
        style: 'timeline' as 'narrative',
      }),
    ).toThrow(/unsupported style/);
  });

  it('throws when sdk.conversation.context is missing', () => {
    const sdkNoContext: SynapSdkLike = {
      fetch: vi.fn(async () => ({})),
      conversation: {
        record_message: vi.fn(async () => ({})),
        // context intentionally absent
      },
      memories: { create: vi.fn(async () => ({ ingestion_id: 'x' })) },
    };
    expect(() =>
      createSynapShortTermHook({ sdk: sdkNoContext, conversationId: 'conv-1' }),
    ).toThrow(/get_context_for_prompt is missing/);
  });

  it('returns object with UserPromptSubmit key', () => {
    const sdk = makeSdk();
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    expect(hooks).toHaveProperty('UserPromptSubmit');
    expect(Array.isArray(hooks.UserPromptSubmit)).toBe(true);
  });
});

// ─── createSynapShortTermHook — happy path ────────────────────────────────────

/** A response from a conversation Synap has compacted. */
const COMPACTED: SynapShortTermResponseLike = {
  available: true,
  // Carries the recent turns too: the hook must NOT inject this.
  formatted_context: 'Summary: user likes TypeScript.\nuser: hello\nassistant: hi',
  context_block: '## Conversation so far (compacted at 5)\nuser likes TypeScript',
  history_block: 'earlier: asked about generics',
  recent_messages: [{ role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }],
  compaction_id: 'cmp-1',
};

type Out = { hookSpecificOutput?: { additionalContext?: string; hookEventName?: string } };

describe('createSynapShortTermHook — happy path', () => {
  it('calls get_context_for_prompt with correct conversationId and default style', async () => {
    const sdk = makeSdk();
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-abc' });
    const cb = extractCallback(hooks);
    await cb(dummyInput, undefined, { signal });

    expect(sdk.conversation.context!.get_context_for_prompt).toHaveBeenCalledOnce();
    const callArgs = (
      sdk.conversation.context!.get_context_for_prompt as ReturnType<typeof vi.fn>
    ).mock.calls[0][0];
    expect(callArgs.conversation_id).toBe('conv-abc');
    expect(callArgs.style).toBe('narrative'); // default
  });

  it('uses supplied style', async () => {
    const sdk = makeSdk();
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-abc',
      style: 'bullet_points',
    });
    const cb = extractCallback(hooks);
    await cb(dummyInput, undefined, { signal });

    const callArgs = (
      sdk.conversation.context!.get_context_for_prompt as ReturnType<typeof vi.fn>
    ).mock.calls[0][0];
    expect(callArgs.style).toBe('bullet_points');
  });

  it('injects the compaction and the earlier-turn excerpts', async () => {
    const sdk = makeSdk(COMPACTED);
    const cb = extractCallback(createSynapShortTermHook({ sdk, conversationId: 'conv-abc' }));
    const ctx = ((await cb(dummyInput, undefined, { signal })) as Out)
      .hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).toBe(
      '<synap_conversation_history>\n## Conversation so far (compacted at 5)\nuser likes TypeScript'
      + '\n\nearlier: asked about generics\n</synap_conversation_history>',
    );
  });

  it('never injects the recent turns: Claude Code\'s transcript already has them', async () => {
    const sdk = makeSdk(COMPACTED);
    const cb = extractCallback(createSynapShortTermHook({ sdk, conversationId: 'conv-abc' }));
    const ctx = ((await cb(dummyInput, undefined, { signal })) as Out)
      .hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).not.toContain('user: hello');
    expect(ctx).not.toContain('assistant: hi');
    expect(ctx).not.toContain('Summary: user likes TypeScript.');
  });

  it('injects nothing before the first compaction, however much formatted_context there is', async () => {
    // formatted_context is just recent turns until a compaction exists.
    const sdk = makeSdk({ available: true, formatted_context: 'user: hello\nassistant: hi' });
    const cb = extractCallback(createSynapShortTermHook({ sdk, conversationId: 'conv-abc' }));
    expect(await cb(dummyInput, undefined, { signal })).toEqual({});
  });

  it('injects a given compaction once, not on every prompt', async () => {
    const sdk = makeSdk(COMPACTED);
    const cb = extractCallback(createSynapShortTermHook({ sdk, conversationId: 'conv-abc' }));
    expect(((await cb(dummyInput, undefined, { signal })) as Out).hookSpecificOutput).toBeDefined();
    expect(await cb(dummyInput, undefined, { signal })).toEqual({});
  });

  it('injects again when the compaction changes', async () => {
    const sdk = makeSdk(COMPACTED);
    const cb = extractCallback(createSynapShortTermHook({ sdk, conversationId: 'conv-abc' }));
    await cb(dummyInput, undefined, { signal });
    (sdk.conversation.context!.get_context_for_prompt as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ ...COMPACTED, context_block: '## Conversation so far (compacted at 9)\nnew' });
    const out = (await cb(dummyInput, undefined, { signal })) as Out;
    expect(out.hookSpecificOutput?.additionalContext).toContain('compacted at 9');
  });

  it('wraps in custom tags when both are given', async () => {
    const sdk = makeSdk(COMPACTED);
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-abc',
      preambleOpen: '[OPEN]',
      preambleClose: '[CLOSE]',
    });
    const ctx = ((await extractCallback(hooks)(dummyInput, undefined, { signal })) as Out)
      .hookSpecificOutput?.additionalContext ?? '';
    expect(ctx.startsWith('[OPEN]\n<synap_conversation_history>')).toBe(true);
    expect(ctx.endsWith('</synap_conversation_history>\n[CLOSE]')).toBe(true);
  });

  it('supports structured style', async () => {
    const sdk = makeSdk(COMPACTED);
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-1',
      style: 'structured',
    });
    const out = (await extractCallback(hooks)(dummyInput, undefined, { signal })) as Out;
    expect(out.hookSpecificOutput?.additionalContext).toContain('user likes TypeScript');
  });

  it('hookSpecificOutput.hookEventName is UserPromptSubmit', async () => {
    const sdk = makeSdk(COMPACTED);
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const out = (await extractCallback(hooks)(dummyInput, undefined, { signal })) as Out;
    expect(out.hookSpecificOutput?.hookEventName).toBe('UserPromptSubmit');
  });
});

// ─── createSynapShortTermHook — no-op / empty cases ──────────────────────────

describe('createSynapShortTermHook — no-op paths', () => {
  it('returns {} when available is false', async () => {
    const sdk = makeSdk({ available: false, formatted_context: null });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    expect(await extractCallback(hooks)(dummyInput, undefined, { signal })).toEqual({});
  });

  it('returns {} when the compaction block is empty or whitespace', async () => {
    for (const block of ['', '   ', null]) {
      const sdk = makeSdk({ available: true, formatted_context: 'x', context_block: block });
      const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
      expect(await extractCallback(hooks)(dummyInput, undefined, { signal })).toEqual({});
    }
  });
});

// ─── createSynapShortTermHook — failure / degradation ────────────────────────

describe('createSynapShortTermHook — failure degradation', () => {
  it('SDK failure in fallback mode → returns {} without throwing', async () => {
    const sdk = makeSdk({}, new Error('SDK exploded'));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1', onError: 'fallback' });
    const cb = extractCallback(hooks);

    const result = await cb(dummyInput, undefined, { signal });

    expect(result).toEqual({});
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('SDK failure in raise mode → re-throws the error', async () => {
    const sdk = makeSdk({}, new Error('hard failure'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1', onError: 'raise' });
    const cb = extractCallback(hooks);

    await expect(cb(dummyInput, undefined, { signal })).rejects.toThrow('hard failure');
    vi.restoreAllMocks();
  });

  it('default onError is fallback (no throw)', async () => {
    const sdk = makeSdk({}, new Error('default mode error'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    // no onError specified → defaults to 'fallback'
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const cb = extractCallback(hooks);

    await expect(cb(dummyInput, undefined, { signal })).resolves.toEqual({});
    vi.restoreAllMocks();
  });

  it('non-2xx-like transport error (thrown) in fallback mode → {} (graceful)', async () => {
    const transportErr = Object.assign(new Error('HTTP 502'), { status: 502 });
    const sdk = makeSdk({}, transportErr);
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1', onError: 'fallback' });
    const cb = extractCallback(hooks);

    const result = await cb(dummyInput, undefined, { signal });
    expect(result).toEqual({});
    vi.restoreAllMocks();
  });
});
