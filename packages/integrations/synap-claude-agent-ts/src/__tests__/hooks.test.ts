/**
 * Tests for createSynapHooks (src/hooks.ts)
 *
 * Strategy: build a minimal duck-typed SynapSdkLike mock per test; invoke the
 * inner HookCallback directly (extracted from the returned matcher array).
 * No live network, no Claude/Anthropic calls.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createSynapHooks } from '../hooks.js';
import type { SynapSdkLike } from '../types.js';

// ─── helpers ─────────────────────────────────────────────────────────────────

function makeSdk(overrides: Partial<SynapSdkLike> = {}): SynapSdkLike {
  return {
    fetch: vi.fn(async () => ({ formatted_context: 'User is a VIP.', facts: [] })),
    conversation: {
      record_message: vi.fn(async () => ({})),
    },
    memories: {
      create: vi.fn(async () => ({ ingestion_id: 'ingestion-abc' })),
    },
    ...overrides,
  };
}

/**
 * Pull the first HookCallback out of the returned hooks dict so we can invoke
 * it directly without spinning up the Claude Agent SDK runtime.
 */
function extractCallback(hooks: ReturnType<typeof createSynapHooks>) {
  const matchers = hooks.UserPromptSubmit;
  if (!matchers || matchers.length === 0) throw new Error('no UserPromptSubmit matchers');
  const [firstMatcher] = matchers;
  if (!firstMatcher.hooks || firstMatcher.hooks.length === 0)
    throw new Error('no hooks in first matcher');
  return firstMatcher.hooks[0];
}

/** Minimal hook input that satisfies UserPromptSubmitHookInput */
function makeInput(prompt: string, session_id = 'sess-001') {
  return {
    hook_event_name: 'UserPromptSubmit' as const,
    prompt,
    session_id,
    transcript_path: '/tmp/t.jsonl',
    cwd: '/tmp',
  };
}

// ─── createSynapHooks — construction guards ───────────────────────────────────

describe('createSynapHooks — construction', () => {
  it('throws when sdk is null/falsy', () => {
    expect(() =>
      createSynapHooks({ sdk: null as unknown as SynapSdkLike, userId: 'u1' }),
    ).toThrow(/non-null sdk/);
  });

  it('throws when userId is empty', () => {
    const sdk = makeSdk();
    expect(() => createSynapHooks({ sdk, userId: '' })).toThrow(/non-empty userId/);
  });

  it('returns an object with a UserPromptSubmit key', () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    expect(hooks).toHaveProperty('UserPromptSubmit');
    expect(Array.isArray(hooks.UserPromptSubmit)).toBe(true);
  });

  it('each matcher has a hooks array with one callback', () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const [matcher] = hooks.UserPromptSubmit!;
    expect(Array.isArray(matcher.hooks)).toBe(true);
    expect(matcher.hooks.length).toBe(1);
    expect(typeof matcher.hooks[0]).toBe('function');
  });
});

// ─── createSynapHooks — happy path ───────────────────────────────────────────

describe('createSynapHooks — happy path', () => {
  it('calls sdk.fetch with correct arguments and injects formatted context', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', customerId: 'cust-A' });
    const cb = extractCallback(hooks);

    const result = await cb(makeInput('What is my name?', 'sess-xyz'), undefined, {
      signal: new AbortController().signal,
    });

    expect(sdk.fetch).toHaveBeenCalledOnce();
    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.user_id).toBe('u1');
    expect(fetchArgs.customer_id).toBe('cust-A');
    expect(fetchArgs.search_query).toEqual(['What is my name?']);
    expect(fetchArgs.mode).toBe('accurate');

    // result must contain hookSpecificOutput with additionalContext
    const r = result as {
      hookSpecificOutput?: { hookEventName: string; additionalContext?: string };
    };
    expect(r.hookSpecificOutput?.hookEventName).toBe('UserPromptSubmit');
    expect(r.hookSpecificOutput?.additionalContext).toContain('User is a VIP.');
  });

  it('wraps context in default preamble tags', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    const result = (await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    })) as { hookSpecificOutput?: { additionalContext?: string } };

    const ctx = result.hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).toContain('<synap_memory>');
    expect(ctx).toContain('</synap_memory>');
  });

  it('uses custom contextPreamble when supplied', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({
      sdk,
      userId: 'u1',
      contextPreamble: '[START]{body}[END]',
    });
    const cb = extractCallback(hooks);
    const result = (await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    })) as { hookSpecificOutput?: { additionalContext?: string } };

    const ctx = result.hookSpecificOutput?.additionalContext ?? '';
    expect(ctx).toContain('[START]');
    expect(ctx).toContain('[END]');
    expect(ctx).toContain('User is a VIP.');
  });

  it('uses the static conversationId when supplied (overrides session_id)', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-static' });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello', 'sess-dynamic'), undefined, {
      signal: new AbortController().signal,
    });

    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.conversation_id).toBe('conv-static');
  });

  it('falls back to session_id from input when conversationId is not supplied', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello', 'fallback-sess'), undefined, {
      signal: new AbortController().signal,
    });

    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.conversation_id).toBe('fallback-sess');
  });

  it('records user prompt when recordUserPrompts is true (default)', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-001' });
    const cb = extractCallback(hooks);
    await cb(makeInput('Remember me!'), undefined, {
      signal: new AbortController().signal,
    });

    expect(sdk.conversation.record_message).toHaveBeenCalledOnce();
    const recArgs = (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(recArgs.role).toBe('user');
    expect(recArgs.content).toBe('Remember me!');
    expect(recArgs.conversation_id).toBe('conv-001');
    expect(recArgs.user_id).toBe('u1');
  });

  it('does NOT record user prompt when recordUserPrompts is false', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({
      sdk,
      userId: 'u1',
      conversationId: 'conv-001',
      recordUserPrompts: false,
    });
    const cb = extractCallback(hooks);
    await cb(makeInput('silent'), undefined, { signal: new AbortController().signal });

    expect(sdk.conversation.record_message).not.toHaveBeenCalled();
  });

  it('respects maxResults option', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', maxResults: 5 });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello'), undefined, { signal: new AbortController().signal });

    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.max_results).toBe(5);
  });

  it('respects mode option', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1', mode: 'fast' });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello'), undefined, { signal: new AbortController().signal });

    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.mode).toBe('fast');
  });
});

// ─── createSynapHooks — empty / no-op cases ──────────────────────────────────

describe('createSynapHooks — no-op paths', () => {
  it('returns {} for empty prompt (whitespace-only)', async () => {
    const sdk = makeSdk();
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    const result = await cb(makeInput('   '), undefined, {
      signal: new AbortController().signal,
    });

    expect(result).toEqual({});
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it('returns {} (no hookSpecificOutput) when formatted_context is empty', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => ({ formatted_context: '' })),
    });
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    const result = await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    });

    expect(result).toEqual({});
  });

  it('returns {} when formatted_context is null', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => ({ formatted_context: null })),
    });
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    const result = await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    });

    expect(result).toEqual({});
  });

  it('does NOT record message when no conversationId is available', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => ({ formatted_context: '' })),
    });
    // No conversationId supplied, and session_id in input is empty string
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello', ''), undefined, { signal: new AbortController().signal });

    // No conv id available → record_message must NOT be called
    expect(sdk.conversation.record_message).not.toHaveBeenCalled();
  });
});

// ─── createSynapHooks — failure / degradation ────────────────────────────────

describe('createSynapHooks — failure degradation', () => {
  it('sdk.fetch failure → hook returns {} (no throw, no crash)', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => {
        throw new Error('network failure');
      }),
    });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    const cb = extractCallback(hooks);

    const result = await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    });

    expect(result).toEqual({});
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  it('sdk.fetch failure does not prevent record_message from being called', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => {
        throw new Error('fetch down');
      }),
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    const cb = extractCallback(hooks);
    await cb(makeInput('hello'), undefined, { signal: new AbortController().signal });

    // record_message should still be attempted even after fetch failure
    expect(sdk.conversation.record_message).toHaveBeenCalledOnce();
    vi.restoreAllMocks();
  });

  it('record_message failure → hook returns result normally (no throw)', async () => {
    const sdk = makeSdk({
      conversation: {
        record_message: vi.fn(async () => {
          throw new Error('write failure');
        }),
      },
    });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const hooks = createSynapHooks({ sdk, userId: 'u1', conversationId: 'conv-1' });
    const cb = extractCallback(hooks);

    const result = await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    });

    // Should still return context from successful fetch
    const r = result as { hookSpecificOutput?: { additionalContext?: string } };
    expect(r.hookSpecificOutput?.additionalContext).toContain('User is a VIP.');
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  it('non-2xx sdk.fetch (simulated via thrown error) → {} (graceful)', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => {
        const err = Object.assign(new Error('HTTP 503 Service Unavailable'), {
          status: 503,
        });
        throw err;
      }),
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const hooks = createSynapHooks({ sdk, userId: 'u1' });
    const cb = extractCallback(hooks);
    const result = await cb(makeInput('hello'), undefined, {
      signal: new AbortController().signal,
    });

    expect(result).toEqual({});
    vi.restoreAllMocks();
  });
});
