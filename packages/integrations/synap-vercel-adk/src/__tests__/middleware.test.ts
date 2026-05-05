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
