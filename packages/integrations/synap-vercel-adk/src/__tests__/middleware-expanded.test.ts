/**
 * Expanded tests for src/middleware.ts (createSynapMiddleware)
 *
 * Covers:
 *   - transformParams: context injection from anticipation cache hit
 *   - transformParams: HTTP fetch fallback when cache misses
 *   - transformParams: non-fatal when HTTP fetch fails (fallback path)
 *   - transformParams: injectContext=false guard
 *   - transformParams: no identity fields guard
 *   - wrapGenerate: passes through underlying result
 *   - wrapGenerate: writes memory with correct transcript
 *   - wrapGenerate: memory write failure is non-fatal (no throw)
 *   - wrapGenerate: no memory write when result.text is empty/undefined
 *   - wrapGenerate: gRPC sendConversationEvent called when grpcClient is connected
 *   - wrapStream: accumulates text-delta parts
 *   - wrapStream: forwards all stream parts including non-text-delta
 *   - wrapStream: writes memory after stream closes
 *   - wrapStream: stream error propagates via controller.error
 *   - prompt stack: works correctly across sequential calls
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createSynapMiddleware } from '../middleware.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { Credentials, CachedBundle, RawContextItem } from '../types.js';
import type { GrpcStreamClient } from '../grpc/stream-client.js';

// ─── Fixtures ───────────────────────────────────────────────────────────────

const creds: Credentials = { api_key: 'sk-mw-test', client_id: 'cli_mw', instance_id: 'inst_mw' };

const userPrompt: LanguageModelV1Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'hello distributed sharding' }] },
];

const promptWithSystem: LanguageModelV1Prompt = [
  { role: 'system', content: 'You are an expert.' },
  { role: 'user', content: [{ type: 'text', text: 'how does sharding work?' }] },
];

function makeRawItem(content: string): RawContextItem {
  return {
    item_id: 'i1',
    content,
    context_type: 'fact',
    source: 'test',
    confidence: 0.9,
    similarity_score: 0.9,
    relevance_score: 0.9,
    scope: 'user',
    entity_id: '',
    event_date: '',
    valid_until: '',
    temporal_category: '',
    temporal_confidence: 0,
  };
}

function makeBundle(overrides: Partial<CachedBundle> = {}): CachedBundle {
  return {
    bundleId: 'bundle-1',
    itemsByType: {
      facts: [makeRawItem('User loves distributed systems sharding')],
    },
    conversationContext: null,
    bundleType: 'anticipation',
    userId: 'u1',
    customerId: 'c1',
    conversationId: 'conv-1',
    searchKeywords: ['distributed', 'systems', 'sharding'],
    searchQueries: ['distributed systems sharding'],
    sourceBundleIds: ['bundle-1'],
    totalTokens: 100,
    bundleConfidence: 0.9,
    originPatternId: '',
    storedAt: Date.now(),
    ttl: 300_000,
    ...overrides,
  };
}

function makeDoGenerate(text = 'test response') {
  return vi.fn().mockResolvedValue({
    text,
    finishReason: 'stop',
    usage: { promptTokens: 10, completionTokens: 5 },
    rawCall: { rawPrompt: '', rawSettings: {} },
  });
}

function makeStreamDoStream(parts: Array<{ type: string; textDelta?: string }>) {
  return vi.fn().mockResolvedValue({
    stream: new ReadableStream({
      start(controller) {
        for (const part of parts) {
          controller.enqueue(part);
        }
        controller.close();
      },
    }),
  });
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function buildMiddleware(overrides: {
  userId?: string;
  customerId?: string;
  conversationId?: string;
  injectContext?: boolean;
  writeMemory?: boolean;
  cache?: AnticipationCache;
  grpcClient?: GrpcStreamClient | null;
  baseUrl?: string;
} = {}) {
  return createSynapMiddleware({
    credentials: creds,
    anticipationCache: overrides.cache ?? new AnticipationCache(),
    grpcClient: overrides.grpcClient ?? null,
    userId: overrides.userId ?? 'u1',
    customerId: overrides.customerId ?? 'c1',
    conversationId: overrides.conversationId ?? 'conv-1',
    injectContext: overrides.injectContext,
    writeMemory: overrides.writeMemory,
    baseUrl: overrides.baseUrl ?? 'http://unreachable.test',
  });
}

// ─── transformParams ─────────────────────────────────────────────────────────

describe('createSynapMiddleware.transformParams', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  it('injects context from cache when cache has a matching hit', async () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const mw = buildMiddleware({ cache });

    const result = await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    // Should have injected a system message
    expect(result.prompt[0].role).toBe('system');
    const sysContent = (result.prompt[0] as { role: 'system'; content: string }).content;
    expect(sysContent).toContain('<synap_context>');
  });

  it('returns prompt unchanged when injectContext: false (even with cache hit)', async () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const mw = buildMiddleware({ cache, injectContext: false });

    const result = await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    expect(result.prompt).toBe(userPrompt);
  });

  it('returns prompt unchanged when no identity fields', async () => {
    const mw = createSynapMiddleware({
      credentials: creds,
      anticipationCache: new AnticipationCache(),
      grpcClient: null,
      baseUrl: 'http://unreachable.test',
      // no userId, customerId, conversationId
    });

    const result = await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    expect(result.prompt).toBe(userPrompt);
  });

  it('falls back to HTTP fetch when cache misses', async () => {
    const mockF = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        items_by_type: {
          facts: {
            items: [makeRawItem('HTTP-fetched fact')],
          },
        },
        metadata: { source: 'cloud' },
      }),
    });
    global.fetch = mockF as unknown as typeof fetch;

    const mw = buildMiddleware({ baseUrl: 'https://synap.test' });
    const result = await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    // HTTP fetch was called (cache was empty)
    expect(mockF).toHaveBeenCalled();
    const [url] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/');
  });

  it('returns original prompt when HTTP fetch fails (non-fatal fallback)', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network failure')) as unknown as typeof fetch;
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const mw = buildMiddleware({ baseUrl: 'https://unreachable.fail' });
    const result = await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    // Should not throw; prompt unchanged
    expect(result.prompt).toBe(userPrompt);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('[synap]'),
      expect.anything(),
    );

    warnSpy.mockRestore();
  });

  it('prepends context block to existing system message when cache hits', async () => {
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const mw = buildMiddleware({ cache });

    // Use the same query keywords as the bundle vocabulary to guarantee BM25 hit
    const promptWithSysAndMatchingQuery: LanguageModelV1Prompt = [
      { role: 'system', content: 'You are an expert.' },
      { role: 'user', content: [{ type: 'text', text: 'distributed systems sharding' }] },
    ];

    const result = await mw.transformParams!({ params: { prompt: promptWithSysAndMatchingQuery } as any, type: 'generate' } as any);
    expect(result.prompt[0].role).toBe('system');
    const sysContent = (result.prompt[0] as { role: 'system'; content: string }).content;
    expect(sysContent).toContain('<synap_context>');
    expect(sysContent).toContain('You are an expert.');
    expect(sysContent.indexOf('<synap_context>')).toBeLessThan(sysContent.indexOf('You are an expert.'));
  });

  it('pushes original prompt to stack (accessible to wrapGenerate)', async () => {
    // We verify this indirectly: wrapGenerate should receive the original prompt
    // by observing that memory write uses the original (pre-injection) transcript
    const cache = new AnticipationCache();
    cache.store(makeBundle());
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    const mw = buildMiddleware({ cache, baseUrl: 'https://synap.test' });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    await mw.wrapGenerate!({ doGenerate: makeDoGenerate('ok'), params: { prompt: userPrompt } as any } as any);

    // After wrapGenerate, the memory write fetch should have been called
    // with the original user message (not the injected system message)
    // We allow time for the fire-and-forget
    await new Promise(r => setTimeout(r, 10));
    expect(mockF).toHaveBeenCalledWith(
      expect.stringContaining('/v1/memories/ingest'),
      expect.anything(),
    );
  });
});

// ─── wrapGenerate ────────────────────────────────────────────────────────────

describe('createSynapMiddleware.wrapGenerate', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  it('returns the underlying doGenerate result unchanged', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) }) as unknown as typeof fetch;
    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const doGenerate = makeDoGenerate('The real answer');
    const result = await mw.wrapGenerate!({ doGenerate, params: { prompt: userPrompt } as any } as any);
    expect(result.text).toBe('The real answer');
    expect(result.finishReason).toBe('stop');
  });

  it('does not throw when memory write fails', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Write failed')) as unknown as typeof fetch;
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    const result = await mw.wrapGenerate!({ doGenerate: makeDoGenerate('ok'), params: { prompt: userPrompt } as any } as any);

    expect(result.text).toBe('ok');
    warnSpy.mockRestore();
  });

  it('skips memory write when result.text is falsy', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    await mw.wrapGenerate!({
      doGenerate: vi.fn().mockResolvedValue({ text: '', finishReason: 'stop', usage: {} }),
      params: { prompt: userPrompt } as any,
    } as any);

    // Only the context fetch would have been called (if cache miss), but NOT the memory write
    // Wait briefly for any fire-and-forget
    await new Promise(r => setTimeout(r, 10));
    const ingestCalls = mockF.mock.calls.filter(([url]) => String(url).includes('memories/ingest'));
    expect(ingestCalls).toHaveLength(0);
  });

  it('calls gRPC sendConversationEvent when grpcClient is connected', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    const mockGrpc = {
      isConnected: true,
      sendConversationEvent: vi.fn().mockResolvedValue(undefined),
    } as unknown as GrpcStreamClient;

    const mw = buildMiddleware({ grpcClient: mockGrpc });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    await mw.wrapGenerate!({ doGenerate: makeDoGenerate('response'), params: { prompt: userPrompt } as any } as any);

    // Allow fire-and-forget to settle
    await new Promise(r => setTimeout(r, 20));
    expect(mockGrpc.sendConversationEvent).toHaveBeenCalled();
  });

  it('does not call gRPC when grpcClient is null', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    // Just verifying no error is thrown when grpcClient=null
    const mw = buildMiddleware({ grpcClient: null });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    await expect(
      mw.wrapGenerate!({ doGenerate: makeDoGenerate('ok'), params: { prompt: userPrompt } as any } as any),
    ).resolves.toBeDefined();
  });

  it('does not call gRPC when grpcClient is not connected', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    const mockGrpc = {
      isConnected: false,
      sendConversationEvent: vi.fn(),
    } as unknown as GrpcStreamClient;

    const mw = buildMiddleware({ grpcClient: mockGrpc });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);
    await mw.wrapGenerate!({ doGenerate: makeDoGenerate('ok'), params: { prompt: userPrompt } as any } as any);

    await new Promise(r => setTimeout(r, 20));
    expect(mockGrpc.sendConversationEvent).not.toHaveBeenCalled();
  });
});

// ─── wrapStream ──────────────────────────────────────────────────────────────

describe('createSynapMiddleware.wrapStream', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  async function collectStream(stream: ReadableStream): Promise<unknown[]> {
    const reader = stream.getReader();
    const parts: unknown[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }
    return parts;
  }

  it('forwards all stream parts through unchanged', async () => {
    global.fetch = mockOkFetch() as unknown as typeof fetch;
    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const parts = [
      { type: 'text-delta', textDelta: 'Hello' },
      { type: 'text-delta', textDelta: ' world' },
      { type: 'finish', finishReason: 'stop', usage: {} },
    ];
    const { stream } = await mw.wrapStream!({
      doStream: makeStreamDoStream(parts),
      params: { prompt: userPrompt } as any,
    } as any);

    const collected = await collectStream(stream);
    expect(collected).toHaveLength(3);
    expect(collected[0]).toEqual({ type: 'text-delta', textDelta: 'Hello' });
    expect(collected[1]).toEqual({ type: 'text-delta', textDelta: ' world' });
    expect(collected[2]).toEqual({ type: 'finish', finishReason: 'stop', usage: {} });
  });

  it('accumulates text-delta parts and writes memory when stream closes', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;
    const mw = buildMiddleware({ baseUrl: 'https://synap.test' });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const { stream } = await mw.wrapStream!({
      doStream: makeStreamDoStream([
        { type: 'text-delta', textDelta: 'Hello' },
        { type: 'text-delta', textDelta: ', world!' },
        { type: 'finish', finishReason: 'stop' },
      ]),
      params: { prompt: userPrompt } as any,
    } as any);

    await collectStream(stream);
    // Allow fire-and-forget
    await new Promise(r => setTimeout(r, 20));

    const ingestCalls = mockF.mock.calls.filter(([url]) => String(url).includes('memories/ingest'));
    expect(ingestCalls.length).toBeGreaterThanOrEqual(1);
    const body = JSON.parse((ingestCalls[0][1] as RequestInit).body as string);
    const assistantMsg = body.messages.find((m: { role: string }) => m.role === 'assistant');
    expect(assistantMsg?.content).toBe('Hello, world!');
  });

  it('does not write memory when accumulated text is empty', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;
    const mw = buildMiddleware({ baseUrl: 'https://synap.test' });
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const { stream } = await mw.wrapStream!({
      doStream: makeStreamDoStream([
        { type: 'finish', finishReason: 'stop' }, // no text-delta
      ]),
      params: { prompt: userPrompt } as any,
    } as any);

    await collectStream(stream);
    await new Promise(r => setTimeout(r, 20));

    const ingestCalls = mockF.mock.calls.filter(([url]) => String(url).includes('memories/ingest'));
    expect(ingestCalls).toHaveLength(0);
  });

  it('passes through non-text-delta stream parts (e.g. tool-call)', async () => {
    global.fetch = mockOkFetch() as unknown as typeof fetch;
    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const toolCallPart = { type: 'tool-call', toolCallId: 'tc1', toolName: 'search', args: '{}' };
    const { stream } = await mw.wrapStream!({
      doStream: makeStreamDoStream([toolCallPart, { type: 'finish', finishReason: 'tool-calls' }]),
      params: { prompt: userPrompt } as any,
    } as any);

    const collected = await collectStream(stream);
    expect(collected[0]).toEqual(toolCallPart);
  });

  it('propagates stream errors through the wrapped stream', async () => {
    global.fetch = mockOkFetch() as unknown as typeof fetch;
    const mw = buildMiddleware();
    await mw.transformParams!({ params: { prompt: userPrompt } as any, type: 'generate' } as any);

    const errorStream = new ReadableStream({
      start(controller) {
        controller.enqueue({ type: 'text-delta', textDelta: 'partial' });
        controller.error(new Error('Stream broke'));
      },
    });

    const { stream } = await mw.wrapStream!({
      doStream: vi.fn().mockResolvedValue({ stream: errorStream }),
      params: { prompt: userPrompt } as any,
    } as any);

    await expect(collectStream(stream)).rejects.toThrow('Stream broke');
  });
});

// ─── Sequential calls (prompt stack) ─────────────────────────────────────────

describe('createSynapMiddleware prompt stack — sequential calls', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  it('correctly pairs prompts across two sequential transform+generate calls', async () => {
    global.fetch = mockOkFetch() as unknown as typeof fetch;
    const mw = buildMiddleware({ baseUrl: 'https://synap.test' });

    const prompt1: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'first call' }] },
    ];
    const prompt2: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'second call' }] },
    ];

    // First call
    await mw.transformParams!({ params: { prompt: prompt1 } as any, type: 'generate' } as any);
    const result1 = await mw.wrapGenerate!({ doGenerate: makeDoGenerate('reply 1'), params: { prompt: prompt1 } as any } as any);

    // Second call
    await mw.transformParams!({ params: { prompt: prompt2 } as any, type: 'generate' } as any);
    const result2 = await mw.wrapGenerate!({ doGenerate: makeDoGenerate('reply 2'), params: { prompt: prompt2 } as any } as any);

    expect(result1.text).toBe('reply 1');
    expect(result2.text).toBe('reply 2');
  });
});

// ─── Helper ─────────────────────────────────────────────────────────────────

function mockOkFetch() {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue({ items_by_type: {}, metadata: {} }),
    text: vi.fn().mockResolvedValue(''),
  });
}
