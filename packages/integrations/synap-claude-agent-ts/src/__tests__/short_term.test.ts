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

  it('wraps content in default XML tags', async () => {
    const sdk = makeSdk({ available: true, formatted_context: 'ST content here' });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-abc' });
    const cb = extractCallback(hooks);
    const result = (await cb(dummyInput, undefined, { signal })) as {
      hookSpecificOutput?: { additionalContext?: string };
    };

    const ctx = result.hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).toContain('<synap_short_term_context>');
    expect(ctx).toContain('</synap_short_term_context>');
    expect(ctx).toContain('ST content here');
  });

  it('uses custom preamble tags', async () => {
    const sdk = makeSdk({ available: true, formatted_context: 'data' });
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-abc',
      preambleOpen: '[OPEN]',
      preambleClose: '[CLOSE]',
    });
    const cb = extractCallback(hooks);
    const result = (await cb(dummyInput, undefined, { signal })) as {
      hookSpecificOutput?: { additionalContext?: string };
    };

    const ctx = result.hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).toContain('[OPEN]');
    expect(ctx).toContain('[CLOSE]');
  });

  it('emits raw content when both preamble tags are null', async () => {
    const sdk = makeSdk({ available: true, formatted_context: 'raw block' });
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-abc',
      preambleOpen: null,
      preambleClose: null,
    });
    const cb = extractCallback(hooks);
    const result = (await cb(dummyInput, undefined, { signal })) as {
      hookSpecificOutput?: { additionalContext?: string };
    };

    expect(result.hookSpecificOutput?.additionalContext).toBe('raw block');
  });

  it('supports structured style', async () => {
    const sdk = makeSdk({ available: true, formatted_context: 'structured data' });
    const hooks = createSynapShortTermHook({
      sdk,
      conversationId: 'conv-1',
      style: 'structured',
    });
    const cb = extractCallback(hooks);
    const result = (await cb(dummyInput, undefined, { signal })) as {
      hookSpecificOutput?: { additionalContext?: string };
    };

    expect(result.hookSpecificOutput?.additionalContext).toContain('structured data');
  });

  it('hookSpecificOutput.hookEventName is UserPromptSubmit', async () => {
    const sdk = makeSdk({ available: true, formatted_context: 'ctx' });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const cb = extractCallback(hooks);
    const result = (await cb(dummyInput, undefined, { signal })) as {
      hookSpecificOutput?: { hookEventName?: string };
    };

    expect(result.hookSpecificOutput?.hookEventName).toBe('UserPromptSubmit');
  });
});

// ─── createSynapShortTermHook — no-op / empty cases ──────────────────────────

describe('createSynapShortTermHook — no-op paths', () => {
  it('returns {} when available is false', async () => {
    const sdk = makeSdk({ available: false, formatted_context: null });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const cb = extractCallback(hooks);
    const result = await cb(dummyInput, undefined, { signal });

    expect(result).toEqual({});
  });

  it('returns {} when formatted_context is empty string even if available=true', async () => {
    const sdk = makeSdk({ available: true, formatted_context: '' });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const cb = extractCallback(hooks);
    const result = await cb(dummyInput, undefined, { signal });

    expect(result).toEqual({});
  });

  it('returns {} when formatted_context is whitespace only', async () => {
    const sdk = makeSdk({ available: true, formatted_context: '   ' });
    const hooks = createSynapShortTermHook({ sdk, conversationId: 'conv-1' });
    const cb = extractCallback(hooks);
    const result = await cb(dummyInput, undefined, { signal });

    expect(result).toEqual({});
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
