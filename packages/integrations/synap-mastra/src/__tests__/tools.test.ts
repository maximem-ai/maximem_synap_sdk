// Tests for synapSearchTool and synapStoreTool.
// ALL transport calls are mocked via vi.fn(); NO live network/cloud.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { synapSearchTool, synapStoreTool } from '../tools.js';
import type { SynapSdkLike } from '../types.js';
import { makeSdk, makePartialSdk, RICH } from './helpers.js';

// Silence the console.error logged by graceful-degrade paths.
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  return () => vi.restoreAllMocks();
});

// Helper: call the tool's execute function directly.
// createTool returns a Tool object whose .execute property holds the handler.
async function execSearch(
  sdk: SynapSdkLike,
  input: { query?: string; maxResults?: number },
  opts: { userId?: string; customerId?: string; conversationId?: string; mode?: string } = {},
) {
  const tool = synapSearchTool({
    sdk,
    userId: opts.userId ?? 'alice',
    customerId: opts.customerId,
    conversationId: opts.conversationId,
    mode: opts.mode,
  });
  return tool.execute!(input);
}

async function execStore(
  sdk: SynapSdkLike,
  input: { content?: string; metadata?: Record<string, unknown> },
  opts: { userId?: string; customerId?: string; conversationId?: string } = {},
) {
  const tool = synapStoreTool({
    sdk,
    userId: opts.userId ?? 'alice',
    customerId: opts.customerId,
    conversationId: opts.conversationId,
  });
  return tool.execute!(input);
}

// ── synapSearchTool — construction guards ─────────────────────────────────────

describe('synapSearchTool construction guards', () => {
  it('throws when sdk is null', () => {
    expect(() =>
      synapSearchTool({ sdk: null as unknown as SynapSdkLike, userId: 'alice' }),
    ).toThrow(/non-null sdk/);
  });

  it('throws when userId is empty string', () => {
    const sdk = makeSdk();
    expect(() => synapSearchTool({ sdk, userId: '' })).toThrow(/non-empty userId/);
  });

  it('constructs successfully with minimal options', () => {
    const sdk = makeSdk();
    const tool = synapSearchTool({ sdk, userId: 'alice' });
    expect(tool).toBeDefined();
    expect(tool.id).toBe('synap_search');
  });

  it('exposes correct id and description', () => {
    const sdk = makeSdk();
    const tool = synapSearchTool({ sdk, userId: 'alice' });
    expect(tool.id).toBe('synap_search');
    expect(tool.description).toMatch(/Synap memory/i);
  });

  it('has inputSchema and outputSchema defined', () => {
    const sdk = makeSdk();
    const tool = synapSearchTool({ sdk, userId: 'alice' });
    expect(tool.inputSchema).toBeDefined();
    expect(tool.outputSchema).toBeDefined();
  });
});

// ── synapSearchTool — execute happy-path ─────────────────────────────────────

describe('synapSearchTool execute happy-path', () => {
  it('returns formattedContext and available=true for a rich response', async () => {
    const richResp = RICH();
    richResp.formatted_context = 'Senior engineer at Acme.';
    const sdk = makeSdk({ fetchResponse: richResp });

    const result = await execSearch(sdk, { query: 'job role' });

    expect(result.available).toBe(true);
    expect(result.formattedContext).toBe('Senior engineer at Acme.');
  });

  it('calls sdk.fetch with the correct arguments', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });

    await execSearch(sdk, { query: 'preferences', maxResults: 5 }, {
      userId: 'bob',
      customerId: 'corp',
      conversationId: 'conv-xyz',
      mode: 'fast',
    });

    expect(sdk.fetch).toHaveBeenCalledOnce();
    const arg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.search_query).toEqual(['preferences']);
    expect(arg.max_results).toBe(5);
    expect(arg.user_id).toBe('bob');
    expect(arg.customer_id).toBe('corp');
    expect(arg.conversation_id).toBe('conv-xyz');
    expect(arg.mode).toBe('fast');
    expect(arg.include_conversation_context).toBe(false);
  });

  it('defaults maxResults to 10 when not provided', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });

    await execSearch(sdk, { query: 'timezone' });

    const arg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.max_results).toBe(10);
  });

  it('passes null for conversation_id when conversationId is not set', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });

    await execSearch(sdk, { query: 'something' }, { userId: 'alice' });

    const arg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.conversation_id).toBeNull();
  });

  it('passes null for customer_id when customerId is not set (empty string)', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });

    await execSearch(sdk, { query: 'something' }, { userId: 'alice' });

    const arg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.customer_id).toBeNull();
  });

  it('trims and returns empty formattedContext when sdk returns null', async () => {
    const sdk = makeSdk({ fetchResponse: { formatted_context: null } });

    const result = await execSearch(sdk, { query: 'nothing' });

    expect(result.formattedContext).toBe('');
    expect(result.available).toBe(false);
  });

  it('returns available=false when formatted_context is empty string', async () => {
    const sdk = makeSdk({ fetchResponse: { formatted_context: '' } });

    const result = await execSearch(sdk, { query: 'empty' });

    expect(result.available).toBe(false);
    expect(result.formattedContext).toBe('');
  });
});

// ── synapSearchTool — execute edge cases ─────────────────────────────────────

describe('synapSearchTool execute edge cases', () => {
  it('returns { formattedContext: "", available: false } when query is empty', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });

    const result = await execSearch(sdk, { query: '' });

    // Early-return for empty query; sdk.fetch must NOT be called
    expect(sdk.fetch).not.toHaveBeenCalled();
    expect(result.available).toBe(false);
    expect(result.formattedContext).toBe('');
  });

  it('does not call sdk.fetch when query is undefined (schema validation blocks it)', async () => {
    // Mastra's Tool wrapper validates input against inputSchema (Zod) before
    // calling execute. query: z.string() is required, so {} fails schema
    // validation — sdk.fetch is never reached. The product's in-code early
    // return is a second line of defence for callers who bypass Mastra.
    const sdk = makeSdk({ fetchResponse: RICH() });

    await execSearch(sdk, {});

    expect(sdk.fetch).not.toHaveBeenCalled();
  });
});

// ── synapSearchTool — failure path ───────────────────────────────────────────

describe('synapSearchTool execute failure-path', () => {
  it('returns { formattedContext: "", available: false } when sdk.fetch rejects (graceful)', async () => {
    const sdk = makePartialSdk({ fetchOk: false });

    const result = await execSearch(sdk, { query: 'anything' });

    expect(result.available).toBe(false);
    expect(result.formattedContext).toBe('');
  });

  it('logs console.error when sdk.fetch rejects', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const sdk = makePartialSdk({ fetchOk: false });

    await execSearch(sdk, { query: 'fail-query' });

    expect(errSpy).toHaveBeenCalledOnce();
    errSpy.mockRestore();
  });
});

// ── synapStoreTool — construction guards ─────────────────────────────────────

describe('synapStoreTool construction guards', () => {
  it('throws when sdk is null', () => {
    expect(() =>
      synapStoreTool({ sdk: null as unknown as SynapSdkLike, userId: 'alice' }),
    ).toThrow(/non-null sdk/);
  });

  it('throws when userId is empty string', () => {
    const sdk = makeSdk();
    expect(() => synapStoreTool({ sdk, userId: '' })).toThrow(/non-empty userId/);
  });

  it('constructs successfully with minimal options', () => {
    const sdk = makeSdk();
    const tool = synapStoreTool({ sdk, userId: 'alice' });
    expect(tool).toBeDefined();
    expect(tool.id).toBe('synap_store');
  });

  it('exposes correct id and description', () => {
    const sdk = makeSdk();
    const tool = synapStoreTool({ sdk, userId: 'alice' });
    expect(tool.id).toBe('synap_store');
    expect(tool.description).toMatch(/Synap memory/i);
  });

  it('has inputSchema and outputSchema defined', () => {
    const sdk = makeSdk();
    const tool = synapStoreTool({ sdk, userId: 'alice' });
    expect(tool.inputSchema).toBeDefined();
    expect(tool.outputSchema).toBeDefined();
  });
});

// ── synapStoreTool — execute happy-path ──────────────────────────────────────

describe('synapStoreTool execute happy-path', () => {
  it('returns { recorded: true, ingestionId } on success', async () => {
    const sdk = makeSdk({ ingestionId: 'ing-abc-123' });

    const result = await execStore(sdk, { content: 'User prefers dark mode.' });

    expect(result.recorded).toBe(true);
    expect(result.ingestionId).toBe('ing-abc-123');
  });

  it('calls sdk.memories.create with the correct arguments', async () => {
    const sdk = makeSdk();

    await execStore(sdk, { content: 'Fact: loves hiking.' }, {
      userId: 'carol',
      customerId: 'bigcorp',
    });

    expect(sdk.memories.create).toHaveBeenCalledOnce();
    const arg = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.document).toBe('Fact: loves hiking.');
    expect(arg.user_id).toBe('carol');
    expect(arg.customer_id).toBe('bigcorp');
  });

  it('injects source=mastra in metadata when metadata.source is not set', async () => {
    const sdk = makeSdk();

    await execStore(sdk, { content: 'Some fact.' });

    const arg = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.metadata?.source).toBe('mastra');
  });

  it('does NOT override source when caller already sets it', async () => {
    const sdk = makeSdk();

    await execStore(sdk, { content: 'Custom source.', metadata: { source: 'my-app' } });

    const arg = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.metadata?.source).toBe('my-app');
  });

  it('merges caller metadata alongside source=mastra', async () => {
    const sdk = makeSdk();

    await execStore(sdk, {
      content: 'With extra meta.',
      metadata: { category: 'preference', confidence: 0.95 },
    });

    const arg = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.metadata?.category).toBe('preference');
    expect(arg.metadata?.confidence).toBe(0.95);
    expect(arg.metadata?.source).toBe('mastra');
  });

  it('passes null for customer_id when customerId is empty / not set', async () => {
    const sdk = makeSdk();

    await execStore(sdk, { content: 'No customer.' }, { userId: 'alice' });

    const arg = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.customer_id).toBeNull();
  });
});

// ── synapStoreTool — execute edge cases ──────────────────────────────────────

describe('synapStoreTool execute edge cases', () => {
  it('returns { recorded: false, error: "missing content" } when content is empty string', async () => {
    const sdk = makeSdk();

    const result = await execStore(sdk, { content: '' });

    expect(result.recorded).toBe(false);
    expect(result.error).toMatch(/missing content/);
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it('returns { recorded: false, error: "missing content" } when content is whitespace-only', async () => {
    const sdk = makeSdk();

    const result = await execStore(sdk, { content: '   ' });

    expect(result.recorded).toBe(false);
    expect(result.error).toMatch(/missing content/);
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it('does not call sdk.memories.create when content is undefined (schema validation blocks it)', async () => {
    // Mastra's Tool wrapper validates input against inputSchema (Zod) before
    // calling execute. content: z.string() is required, so {} fails schema
    // validation — sdk.memories.create is never reached. The product's in-code
    // early return is a second line of defence for callers that bypass Mastra.
    const sdk = makeSdk();

    await execStore(sdk, {});

    expect(sdk.memories.create).not.toHaveBeenCalled();
  });
});

// ── synapStoreTool — failure path (write-side throws) ────────────────────────

describe('synapStoreTool execute failure-path', () => {
  it('throws (re-wrapped) when sdk.memories.create rejects', async () => {
    const sdk = makePartialSdk({ createOk: false });

    await expect(
      execStore(sdk, { content: 'Important fact.' }),
    ).rejects.toThrow(/synap_store/);
  });

  it('includes the original error message in the rethrown error', async () => {
    const sdk = makePartialSdk({ createOk: false });

    await expect(
      execStore(sdk, { content: 'A fact.' }),
    ).rejects.toThrow('endpoint-down');
  });

  it('logs console.error when sdk.memories.create rejects', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const sdk = makePartialSdk({ createOk: false });

    try {
      await execStore(sdk, { content: 'will fail' });
    } catch {
      // expected
    }

    expect(errSpy).toHaveBeenCalledOnce();
    errSpy.mockRestore();
  });
});
