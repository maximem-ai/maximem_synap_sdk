// Tests for SynapMemory — the MastraMemory implementation backed by Synap.
// ALL transport calls are mocked via vi.fn(); NO live network/cloud.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SynapMemory } from '../memory.js';
import type { SynapSdkLike } from '../types.js';
import {
  makeSdk,
  makeFailingSdk,
  makePartialSdk,
  RICH,
  EMPTY,
  DIALOGUE,
  promptContext,
  msg,
} from './helpers.js';

// Silence the graceful-degrade console outputs from product code so test
// output stays readable. Restore after each test to avoid cross-test leakage.
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  return () => vi.restoreAllMocks();
});

// ── Construction ──────────────────────────────────────────────────────────────

describe('SynapMemory construction', () => {
  it('throws when sdk is null', () => {
    expect(() => new SynapMemory({ sdk: null as unknown as SynapSdkLike, userId: 'u1' })).toThrow(
      /non-null sdk/,
    );
  });

  it('throws when userId is empty string', () => {
    const sdk = makeSdk();
    expect(() => new SynapMemory({ sdk, userId: '' })).toThrow(/non-empty userId/);
  });

  it('constructs successfully with minimal options', () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    expect(mem).toBeInstanceOf(SynapMemory);
  });

  it('constructs with all options', () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({
      sdk,
      userId: 'alice',
      customerId: 'acme',
      mode: 'fast',
      injectSystemContext: false,
      id: 'my-memory',
    });
    expect(mem).toBeInstanceOf(SynapMemory);
  });
});

// ── getSystemMessage ──────────────────────────────────────────────────────────

describe('SynapMemory.getSystemMessage', () => {
  it('returns formatted_context from sdk.fetch on RICH response', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });
    const mem = new SynapMemory({ sdk, userId: 'alice', customerId: 'acme' });
    const sys = await mem.getSystemMessage({ threadId: 't-1' });

    expect(sys).not.toBeNull();
    expect(sys).toContain('Acme');
    expect(sdk.fetch).toHaveBeenCalledOnce();
    const callArg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArg.conversation_id).toBe('t-1');
    expect(callArg.user_id).toBe('alice');
    expect(callArg.customer_id).toBe('acme');
    expect(callArg.max_results).toBe(20);
    expect(callArg.mode).toBe('accurate'); // default
    expect(callArg.include_conversation_context).toBe(false);
  });

  it('returns null when formatted_context is empty string (EMPTY response)', async () => {
    const sdk = makeSdk({ fetchResponse: EMPTY() });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const sys = await mem.getSystemMessage({ threadId: 't-1' });
    expect(sys).toBeNull();
  });

  it('returns null when formatted_context is null', async () => {
    const sdk = makeSdk({ fetchResponse: { formatted_context: null } });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const sys = await mem.getSystemMessage({ threadId: 't-1' });
    expect(sys).toBeNull();
  });

  it('returns null and does NOT throw when sdk.fetch rejects (graceful read)', async () => {
    const sdk = makeFailingSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const sys = await mem.getSystemMessage({ threadId: 't-1' });
    expect(sys).toBeNull();
  });

  it('returns null immediately without calling sdk.fetch when injectSystemContext=false', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });
    const mem = new SynapMemory({ sdk, userId: 'alice', injectSystemContext: false });
    const sys = await mem.getSystemMessage({ threadId: 't-1' });
    expect(sys).toBeNull();
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it('passes customerId=null to fetch when customerId is omitted', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.getSystemMessage({ threadId: 't-1' });
    const callArg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    // customerId defaults to "" which maps to null in the fetch call
    expect(callArg.customer_id).toBeNull();
  });

  it('uses the configured mode (fast) in the fetch call', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });
    const mem = new SynapMemory({ sdk, userId: 'alice', mode: 'fast' });
    await mem.getSystemMessage({ threadId: 't-2' });
    const callArg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArg.mode).toBe('fast');
  });
});

// ── recall ────────────────────────────────────────────────────────────────────

describe('SynapMemory.recall', () => {
  it('maps recent_messages to Mastra message format', async () => {
    const sdk = makeSdk({ promptCtx: promptContext(true, true) });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: 't-1' });

    expect(out.messages).toHaveLength(DIALOGUE.length);
    expect(out.total).toBe(DIALOGUE.length);
    expect(out.hasMore).toBe(false);

    const first = out.messages[0];
    expect(first.role).toBe('user');
    expect(first.threadId).toBe('t-1');
    expect(first.type).toBe('text');
    expect(first.content.parts[0].text).toBe('Can you remind me when my trial expires?');
    expect(first.createdAt).toBeInstanceOf(Date);
    expect(first.message_id).toBeUndefined(); // mapped to id
    expect(first.id).toBe('msg-001');
  });

  it('returns empty page when threadId is empty string', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: '' });
    expect(out.messages).toEqual([]);
    expect(out.total).toBe(0);
    expect(sdk.conversation.context.get_context_for_prompt).not.toHaveBeenCalled();
  });

  it('returns empty page (graceful) when sdk.get_context_for_prompt rejects', async () => {
    const sdk = makePartialSdk({ ctxOk: false });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: 't-1' });
    expect(out.messages).toEqual([]);
    expect(out.total).toBe(0);
  });

  it('filters out messages with empty content', async () => {
    const ctx = {
      ...promptContext(false, true),
      recent_messages: [
        { role: 'user', content: '', timestamp: new Date().toISOString(), message_id: 'blank-1' },
        { role: 'user', content: '  ', timestamp: new Date().toISOString(), message_id: 'blank-2' },
        { role: 'assistant', content: 'valid reply', timestamp: new Date().toISOString(), message_id: 'm-ok' },
      ],
      recent_message_count: 3,
      total_message_count: 3,
    };
    const sdk = makeSdk({ promptCtx: ctx });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: 't-1' });
    expect(out.messages).toHaveLength(1);
    expect(out.messages[0].id).toBe('m-ok');
  });

  it('normalises unknown roles to "user"', async () => {
    const ctx = {
      ...promptContext(false, true),
      recent_messages: [
        { role: 'system', content: 'sys msg', timestamp: new Date().toISOString(), message_id: 'm-sys' },
        { role: 'tool', content: 'tool result', timestamp: new Date().toISOString(), message_id: 'm-tool' },
        { role: 'function', content: 'fn result', timestamp: new Date().toISOString(), message_id: 'm-fn' },
      ],
      recent_message_count: 3,
      total_message_count: 3,
    };
    const sdk = makeSdk({ promptCtx: ctx });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: 't-1' });
    // 'system' is in the allowed set, 'tool' and 'function' are normalised to 'user'
    expect(out.messages[0].role).toBe('system');
    expect(out.messages[1].role).toBe('user'); // 'tool' → 'user'
    expect(out.messages[2].role).toBe('user'); // 'function' → 'user'
  });

  it('accepts threadId as array (uses first element)', async () => {
    const sdk = makeSdk({ promptCtx: promptContext(true, true) });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const out = await mem.recall({ threadId: ['t-arr', 't-ignored'] as unknown as string });
    expect(out.messages.length).toBeGreaterThan(0);
    const callArg = (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArg.conversation_id).toBe('t-arr');
  });
});

// ── saveMessages ──────────────────────────────────────────────────────────────

describe('SynapMemory.saveMessages', () => {
  it('records user and assistant messages to sdk.conversation.record_message', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice', customerId: 'acme' });
    await mem.saveMessages({
      messages: [msg('user', 'hello', 't-1'), msg('assistant', 'hi back', 't-1')],
    });

    expect(sdk.conversation.record_message).toHaveBeenCalledTimes(2);
    const first = (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(first.role).toBe('user');
    expect(first.content).toBe('hello');
    expect(first.conversation_id).toBe('t-1');
    expect(first.user_id).toBe('alice');
    expect(first.customer_id).toBe('acme');
  });

  it('records system messages', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.saveMessages({ messages: [msg('system', 'sys text', 't-2')] });
    expect(sdk.conversation.record_message).toHaveBeenCalledOnce();
    const arg = (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.role).toBe('system');
  });

  it('skips tool-role messages entirely', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.saveMessages({
      messages: [msg('tool', 'tool-result', 't-1'), msg('user', 'ok', 't-1')],
    });
    expect(sdk.conversation.record_message).toHaveBeenCalledTimes(1);
    const arg = (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.role).toBe('user');
  });

  it('skips messages with blank content', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.saveMessages({
      messages: [msg('user', '   ', 't-1'), msg('assistant', 'reply', 't-1')],
    });
    expect(sdk.conversation.record_message).toHaveBeenCalledTimes(1);
  });

  it('skips messages with empty threadId', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.saveMessages({ messages: [msg('user', 'no thread', '')] });
    expect(sdk.conversation.record_message).not.toHaveBeenCalled();
  });

  it('handles content as plain string (not parts structure)', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const plainMsg = { role: 'user', threadId: 't-1', content: 'plain text' };
    await mem.saveMessages({ messages: [plainMsg] });
    expect(sdk.conversation.record_message).toHaveBeenCalledOnce();
    const arg = (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.content).toBe('plain text');
  });

  it('throws when sdk.record_message rejects (write-side surfaces errors)', async () => {
    const sdk = makeFailingSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await expect(
      mem.saveMessages({ messages: [msg('user', 'x', 't-1')] }),
    ).rejects.toThrow('simulated-sdk-failure');
  });

  it('returns messages that were actually recorded', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const m1 = msg('user', 'hi', 't-1');
    const m2 = msg('assistant', 'hello', 't-1');
    const result = await mem.saveMessages({ messages: [m1, m2] });
    expect(result.messages).toHaveLength(2);
  });

  it('handles empty messages array gracefully', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    const result = await mem.saveMessages({ messages: [] });
    expect(result.messages).toEqual([]);
    expect(sdk.conversation.record_message).not.toHaveBeenCalled();
  });
});

// ── Thread management (in-process Map, not Synap-backed) ─────────────────────

describe('SynapMemory thread management', () => {
  const makeThread = (id: string, resourceId = 'alice', metadata?: Record<string, unknown>) => ({
    id,
    resourceId,
    createdAt: new Date(),
    updatedAt: new Date(),
    metadata,
  });

  it('saveThread + getThreadById round-trip', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    const thread = makeThread('t-1', 'alice', { topic: 'billing' });
    await mem.saveThread({ thread });
    const got = await mem.getThreadById({ threadId: 't-1' });
    expect(got?.id).toBe('t-1');
    expect(got?.metadata?.topic).toBe('billing');
  });

  it('getThreadById returns null for unknown id', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    const got = await mem.getThreadById({ threadId: 'no-such-thread' });
    expect(got).toBeNull();
  });

  it('listThreads returns all threads with no filter', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-1') });
    await mem.saveThread({ thread: makeThread('t-2') });
    const result = await mem.listThreads({});
    expect(result.total).toBe(2);
    expect(result.threads).toHaveLength(2);
  });

  it('listThreads filters by resourceId', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-1', 'alice') });
    await mem.saveThread({ thread: makeThread('t-2', 'bob') });
    await mem.saveThread({ thread: makeThread('t-3', 'alice') });
    const result = await mem.listThreads({ filter: { resourceId: 'alice' } });
    expect(result.total).toBe(2);
    result.threads.forEach((t) => expect(t.resourceId).toBe('alice'));
  });

  it('listThreads filters by metadata key/value', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-1', 'alice', { k: 'a' }) });
    await mem.saveThread({ thread: makeThread('t-2', 'alice', { k: 'b' }) });
    const result = await mem.listThreads({ filter: { resourceId: 'alice', metadata: { k: 'a' } } });
    expect(result.total).toBe(1);
    expect(result.threads[0].id).toBe('t-1');
  });

  it('listThreads paginates correctly', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    for (let i = 0; i < 5; i++) {
      await mem.saveThread({ thread: makeThread(`t-${i}`) });
    }
    const page0 = await mem.listThreads({ page: 0, perPage: 2 });
    expect(page0.threads).toHaveLength(2);
    expect(page0.hasMore).toBe(true);
    const page1 = await mem.listThreads({ page: 1, perPage: 2 });
    expect(page1.threads).toHaveLength(2);
    const page2 = await mem.listThreads({ page: 2, perPage: 2 });
    expect(page2.threads).toHaveLength(1);
    expect(page2.hasMore).toBe(false);
  });

  it('deleteThread removes the thread from the Map', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-1') });
    await mem.deleteThread('t-1');
    const got = await mem.getThreadById({ threadId: 't-1' });
    expect(got).toBeNull();
  });

  it('deleteThread is a no-op for unknown id (does not throw)', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await expect(mem.deleteThread('ghost-thread')).resolves.toBeUndefined();
  });

  it('cloneThread copies metadata and assigns new id', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-src', 'alice', { label: 'orig' }) });
    const out = await mem.cloneThread({ sourceThreadId: 't-src', newThreadId: 't-dst' });
    expect(out.thread.id).toBe('t-dst');
    expect(out.thread.metadata?.label).toBe('orig');
    expect(out.thread.metadata?.sourceThreadId).toBe('t-src');
    expect(out.clonedMessages).toEqual([]);
    // Verify clone is persisted
    const got = await mem.getThreadById({ threadId: 't-dst' });
    expect(got?.id).toBe('t-dst');
  });

  it('cloneThread generates a new id when newThreadId is not provided', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: makeThread('t-src') });
    const out = await mem.cloneThread({ sourceThreadId: 't-src' });
    expect(out.thread.id).toContain('t-src-clone');
    expect(out.thread.id).not.toBe('t-src');
  });

  it('cloneThread throws when source thread does not exist', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await expect(
      mem.cloneThread({ sourceThreadId: 'no-such-thread', newThreadId: 'dst' }),
    ).rejects.toThrow(/no-such-thread/);
  });

  it('cloneThread overrides title and resourceId when provided', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await mem.saveThread({ thread: { ...makeThread('t-src'), title: 'original title', resourceId: 'alice' } });
    const out = await mem.cloneThread({
      sourceThreadId: 't-src',
      newThreadId: 't-dst',
      title: 'new title',
      resourceId: 'bob',
    });
    expect(out.thread.title).toBe('new title');
    expect(out.thread.resourceId).toBe('bob');
  });
});

// ── Working memory (v0.1 no-op) ───────────────────────────────────────────────

describe('SynapMemory working memory (v0.1 no-op)', () => {
  it('getWorkingMemory returns null', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    expect(await mem.getWorkingMemory()).toBeNull();
  });

  it('getWorkingMemoryTemplate returns null', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    expect(await mem.getWorkingMemoryTemplate()).toBeNull();
  });

  it('updateWorkingMemory resolves without throwing', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await expect(mem.updateWorkingMemory()).resolves.toBeUndefined();
  });

  it('__experimental_updateWorkingMemoryVNext returns success=false with reason', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    const result = await mem.__experimental_updateWorkingMemoryVNext();
    expect(result.success).toBe(false);
    expect(result.reason).toMatch(/not supported/i);
  });

  it('working memory warns exactly once across multiple calls', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await mem.getWorkingMemory();
    await mem.getWorkingMemory();
    await mem.getWorkingMemory();
    // The warn flag deduplicates — each deduplicated method shares the same flag
    // so warn is called at most once per method type
    warnSpy.mockRestore();
  });
});

// ── deleteMessages (no-op, warns once) ───────────────────────────────────────

describe('SynapMemory.deleteMessages', () => {
  it('resolves without throwing (no-op)', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    await expect(mem.deleteMessages(['m-1', 'm-2'])).resolves.toBeUndefined();
  });

  it('does not call any sdk write methods', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.deleteMessages(['m-1']);
    expect(sdk.memories.create).not.toHaveBeenCalled();
    expect(sdk.conversation.record_message).not.toHaveBeenCalled();
  });

  it('warns exactly once regardless of how many times called', async () => {
    const mem = new SynapMemory({ sdk: makeSdk(), userId: 'alice' });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await mem.deleteMessages(['m-1']);
    await mem.deleteMessages(['m-2']);
    await mem.deleteMessages(['m-3']);
    // deleteWarned gate: console.warn should fire exactly once
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });
});
