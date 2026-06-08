import { describe, it, expect, vi } from 'vitest';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { Credentials } from '../types.js';
import {
  fetchShortTermContext,
  buildShortTermSystemBlock,
  injectShortTermIntoPrompt,
  type SynapShortTermResponse,
} from '../short_term.js';

const creds: Credentials = { api_key: 'sk-test', client_id: '', instance_id: '' };

function mockOk(body: Record<string, unknown>): typeof fetch {
  return vi.fn(async () => ({
    ok: true,
    status: 200,
    text: async () => JSON.stringify(body),
    json: async () => body,
  } as unknown as Response)) as unknown as typeof fetch;
}

function mockFail(status = 500, text = 'boom'): typeof fetch {
  return vi.fn(async () => ({
    ok: false,
    status,
    text: async () => text,
    json: async () => ({}),
  } as unknown as Response)) as unknown as typeof fetch;
}

// ─── fetchShortTermContext ────────────────────────────────────────────────────

describe('fetchShortTermContext', () => {
  it('throws on empty conversationId', async () => {
    await expect(
      fetchShortTermContext({
        credentials: creds,
        conversationId: '',
      }),
    ).rejects.toThrow(/non-empty conversationId/);
  });

  it('throws on unknown style', async () => {
    await expect(
      fetchShortTermContext({
        credentials: creds,
        conversationId: 'conv_abc',
        style: 'wat' as any,
      }),
    ).rejects.toThrow(/unsupported style/);
  });

  it('returns formattedContext + available on a populated response', async () => {
    const fetchImpl = mockOk({
      available: true,
      formatted_context: '## Summary\nUser is VIP.',
    });
    const out = await fetchShortTermContext({
      credentials: creds,
      conversationId: 'conv_abc',
      fetchImpl,
    });
    expect(out.available).toBe(true);
    expect(out.formattedContext).toContain('User is VIP.');
  });

  it('returns empty when server reports available=false', async () => {
    const fetchImpl = mockOk({ available: false, formatted_context: null });
    const out = await fetchShortTermContext({
      credentials: creds,
      conversationId: 'conv_abc',
      fetchImpl,
    });
    expect(out.available).toBe(false);
    expect(out.formattedContext).toBe('');
  });

  it('fallback returns empty on HTTP failure', async () => {
    const fetchImpl = mockFail(500, 'nope');
    const out = await fetchShortTermContext({
      credentials: creds,
      conversationId: 'conv_abc',
      fetchImpl,
    });
    expect(out.available).toBe(false);
    expect(out.formattedContext).toBe('');
  });

  it('raise mode throws on HTTP failure', async () => {
    const fetchImpl = mockFail(500, 'nope');
    await expect(
      fetchShortTermContext({
        credentials: creds,
        conversationId: 'conv_abc',
        onError: 'raise',
        fetchImpl,
      }),
    ).rejects.toThrow(/HTTP 500/);
  });

  it('passes style + auth headers correctly', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      status: 200,
      text: async () => '',
      json: async () => ({ available: true, formatted_context: 'x' }),
    } as unknown as Response)) as unknown as typeof fetch;
    await fetchShortTermContext({
      credentials: { api_key: 'sk-1', client_id: 'c1', instance_id: 'i1' },
      conversationId: 'conv_abc',
      style: 'bullet_points',
      fetchImpl,
    });
    const callArg = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    const url = callArg[0] as string;
    const init = callArg[1] as RequestInit;
    expect(url).toContain('/v1/conversations/conv_abc/context-for-prompt');
    expect(url).toContain('style=bullet_points');
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer sk-1');
    expect((init.headers as Record<string, string>)['X-Synap-Instance-Id']).toBe('i1');
    expect((init.headers as Record<string, string>)['X-Synap-Client-Id']).toBe('c1');
  });
});

// ─── buildShortTermSystemBlock ────────────────────────────────────────────────

describe('buildShortTermSystemBlock', () => {
  it('wraps non-empty content in default tags', () => {
    const block = buildShortTermSystemBlock({
      formattedContext: 'hello',
      available: true,
    });
    expect(block).toBe('<synap_short_term_context>\nhello\n</synap_short_term_context>');
  });

  it('returns empty string for empty content', () => {
    expect(
      buildShortTermSystemBlock({ formattedContext: '', available: false }),
    ).toBe('');
    expect(
      buildShortTermSystemBlock({ formattedContext: '   ', available: true }),
    ).toBe('');
  });

  it('honours custom preamble', () => {
    const block = buildShortTermSystemBlock(
      { formattedContext: 'X', available: true },
      { preambleOpen: '[B]', preambleClose: '[E]' },
    );
    expect(block).toBe('[B]\nX\n[E]');
  });

  it('emits raw concat when preamble set to null', () => {
    const block = buildShortTermSystemBlock(
      { formattedContext: 'raw', available: true },
      { preambleOpen: null, preambleClose: null },
    );
    expect(block).toBe('raw');
  });
});

// ─── injectShortTermIntoPrompt ────────────────────────────────────────────────

describe('injectShortTermIntoPrompt', () => {
  const userMsg: LanguageModelV1Prompt = [
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
  ];
  const withSystem: LanguageModelV1Prompt = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
  ];

  it('inserts SystemMessage when none exists', () => {
    const out = injectShortTermIntoPrompt(userMsg, {
      formattedContext: 'ST!',
      available: true,
    });
    expect(out[0]).toEqual({
      role: 'system',
      content: '<synap_short_term_context>\nST!\n</synap_short_term_context>',
    });
    expect(out[1].role).toBe('user');
  });

  it('prepends ST above existing system text', () => {
    const out = injectShortTermIntoPrompt(withSystem, {
      formattedContext: 'ST!',
      available: true,
    });
    expect(out[0].role).toBe('system');
    const content = (out[0] as any).content as string;
    expect(content).toContain('ST!');
    expect(content).toContain('You are helpful.');
    expect(content.indexOf('ST!')).toBeLessThan(content.indexOf('You are helpful.'));
  });

  it('no-op when response is empty', () => {
    const out = injectShortTermIntoPrompt(withSystem, {
      formattedContext: '',
      available: false,
    });
    expect(out).toEqual(withSystem);
  });
});
