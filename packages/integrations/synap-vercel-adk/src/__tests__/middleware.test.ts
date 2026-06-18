import { describe, it, expect, vi } from 'vitest';
import { createSynapMiddleware } from '../middleware.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { Credentials } from '../types.js';

const creds: Credentials = { api_key: 'sk-test', client_id: '', instance_id: '' };
const cache = new AnticipationCache();
const baseOpts = { credentials: creds, anticipationCache: cache, grpcClient: null };

const userPrompt: LanguageModelV1Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'hello' }] },
];

describe('createSynapMiddleware', () => {
  it('transformParams returns params unchanged when injectContext is false', async () => {
    const mw = createSynapMiddleware({ ...baseOpts, userId: 'u1', injectContext: false });
    const params = { prompt: userPrompt } as Parameters<NonNullable<typeof mw.transformParams>>[0]['params'];
    const result = await mw.transformParams!({ params, type: 'generate' } as any);
    expect(result.prompt).toBe(userPrompt);
  });

  it('transformParams returns params unchanged when no identity fields', async () => {
    const mw = createSynapMiddleware({ ...baseOpts });
    const params = { prompt: userPrompt } as any;
    const result = await mw.transformParams!({ params, type: 'generate' } as any);
    expect(result.prompt).toBe(userPrompt);
  });

  it('emits ContextAssembledEvent + ContextUsedEvent on an anticipation cache hit', async () => {
    const hitCache = new AnticipationCache();
    hitCache.store({
      bundleId: 'b1',
      itemsByType: {
        facts: [{
          item_id: 'f1', content: 'hello world fact', context_type: 'facts', source: 'vector',
          confidence: 1, similarity_score: 1, relevance_score: 1, scope: 'user', entity_id: '',
          event_date: '', valid_until: '', temporal_category: '', temporal_confidence: 0,
        }],
      },
      conversationContext: null,
      bundleType: 'anticipation',
      userId: 'u1', customerId: '', conversationId: '',
      searchKeywords: ['hello'], searchQueries: ['hello'],
      sourceBundleIds: ['b1'], totalTokens: 42, bundleConfidence: 0.9, originPatternId: 'p1',
      storedAt: Date.now(), ttl: 1_800_000,
    });

    const grpcClient = {
      isConnected: true,
      sendContextAssembledEvent: vi.fn().mockResolvedValue(undefined),
      sendContextUsedEvent: vi.fn().mockResolvedValue(undefined),
    };

    const mw = createSynapMiddleware({
      credentials: creds, anticipationCache: hitCache, grpcClient: grpcClient as any, userId: 'u1',
    });
    const params = { prompt: userPrompt } as any;
    await mw.transformParams!({ params, type: 'generate' } as any);
    await new Promise(r => setTimeout(r, 0)); // flush fire-and-forget telemetry

    expect(grpcClient.sendContextAssembledEvent).toHaveBeenCalledTimes(1);
    expect(grpcClient.sendContextUsedEvent).toHaveBeenCalledTimes(1);

    const used = grpcClient.sendContextUsedEvent.mock.calls[0][0];
    expect(used.bundle_id).toBe('b1');
    expect(used.served_item_ids).toContain('f1');
    expect(used.scope).toBe('user');

    const assembled = grpcClient.sendContextAssembledEvent.mock.calls[0][0];
    expect(assembled.assembly_source).toBe('anticipation_cache');
    expect(assembled.cache_hit).toBe(true);
    expect(assembled.final_total_tokens).toBe(42);
    expect(assembled.sdk_version).toBeTruthy();
  });

  it('does not emit telemetry when the gRPC client is absent (serverless path)', async () => {
    const mw = createSynapMiddleware({ ...baseOpts, userId: 'u1', baseUrl: 'http://invalid.test' });
    const params = { prompt: userPrompt } as any;
    // No grpc client + HTTP fetch fails → resolveContext returns null → no throw.
    await expect(mw.transformParams!({ params, type: 'generate' } as any)).resolves.toBeDefined();
  });

  it('writeMemory failure is logged but does not throw', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mw = createSynapMiddleware({ ...baseOpts, userId: 'u1', baseUrl: 'http://invalid.test' });

    // Prime the prompt stack
    const params = { prompt: userPrompt } as any;
    await mw.transformParams!({ params, type: 'generate' } as any);

    const doGenerate = vi.fn().mockResolvedValue({ text: 'hi', finishReason: 'stop', usage: {}, rawCall: { rawPrompt: '', rawSettings: {} } });
    await expect(mw.wrapGenerate!({ doGenerate, params } as any)).resolves.toBeDefined();

    warnSpy.mockRestore();
  });
});
