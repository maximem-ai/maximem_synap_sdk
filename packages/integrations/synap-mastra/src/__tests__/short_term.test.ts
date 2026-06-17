// Tests for fetchSynapShortTerm, buildSynapShortTermSystem, synapShortTermInstructions.
// ALL transport calls are mocked via vi.fn(); NO live network/cloud.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  fetchSynapShortTerm,
  buildSynapShortTermSystem,
  synapShortTermInstructions,
} from '../short_term.js';
import type { SynapShortTermResult } from '../short_term.js';
import type { SynapSdkLike } from '../types.js';
import { makeSdk, makePartialSdk, promptContext } from './helpers.js';

// Silence console.warn from graceful-fallback paths.
beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
  return () => vi.restoreAllMocks();
});

// ── fetchSynapShortTerm — guard-clause validation ─────────────────────────────

describe('fetchSynapShortTerm validation', () => {
  it('throws TypeError when sdk is null', async () => {
    await expect(
      fetchSynapShortTerm({
        sdk: null as unknown as SynapSdkLike,
        conversationId: 'conv-1',
      }),
    ).rejects.toThrow(/non-null sdk/);
  });

  it('throws TypeError when conversationId is empty string', async () => {
    const sdk = makeSdk();
    await expect(
      fetchSynapShortTerm({ sdk, conversationId: '' }),
    ).rejects.toThrow(/non-empty conversationId/);
  });

  it('throws TypeError when conversationId is whitespace-only', async () => {
    const sdk = makeSdk();
    await expect(
      fetchSynapShortTerm({ sdk, conversationId: '   ' }),
    ).rejects.toThrow(/non-empty conversationId/);
  });

  it('throws TypeError for an unsupported style', async () => {
    const sdk = makeSdk();
    await expect(
      fetchSynapShortTerm({
        sdk,
        conversationId: 'c-1',
        style: 'unknown_style' as never,
      }),
    ).rejects.toThrow(/unsupported style/);
  });
});

// ── fetchSynapShortTerm — happy paths ────────────────────────────────────────

describe('fetchSynapShortTerm happy-path', () => {
  it('returns available=true and formatted context when server returns content', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'User prefers dark mode.';
    const sdk = makeSdk({ promptCtx: ctx });

    const result = await fetchSynapShortTerm({ sdk, conversationId: 'conv-abc' });

    expect(result.available).toBe(true);
    expect(result.formattedContext).toBe('User prefers dark mode.');
    expect(sdk.conversation.context.get_context_for_prompt).toHaveBeenCalledOnce();
    const callArg = (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(callArg.conversation_id).toBe('conv-abc');
  });

  it('passes style to the SDK call', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'Bullet list';
    const sdk = makeSdk({ promptCtx: ctx });

    await fetchSynapShortTerm({ sdk, conversationId: 'c-1', style: 'bullet_points' });

    const callArg = (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(callArg.style).toBe('bullet_points');
  });

  it('defaults style to "narrative" when not specified', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'narrative text';
    const sdk = makeSdk({ promptCtx: ctx });

    await fetchSynapShortTerm({ sdk, conversationId: 'c-1' });

    // style defaults to narrative — SDK is called with style: 'narrative'
    const callArg = (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(callArg.style).toBe('narrative');
  });

  it('accepts "structured" as a valid style', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'Structured content';
    const sdk = makeSdk({ promptCtx: ctx });

    const result = await fetchSynapShortTerm({ sdk, conversationId: 'c-1', style: 'structured' });
    expect(result.available).toBe(true);
  });

  it('returns available=false and empty formattedContext when server reports unavailable', async () => {
    const ctx = promptContext(false, false);
    const sdk = makeSdk({ promptCtx: ctx });

    const result = await fetchSynapShortTerm({ sdk, conversationId: 'conv-no-ctx' });

    expect(result.available).toBe(false);
    expect(result.formattedContext).toBe('');
  });

  it('trims whitespace from formatted_context', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = '   trimmed content   ';
    const sdk = makeSdk({ promptCtx: ctx });

    const result = await fetchSynapShortTerm({ sdk, conversationId: 'c-1' });
    expect(result.formattedContext).toBe('trimmed content');
  });
});

// ── fetchSynapShortTerm — failure paths ──────────────────────────────────────

describe('fetchSynapShortTerm failure-path', () => {
  it('returns fallback { formattedContext: "", available: false } when SDK rejects (default onError=fallback)', async () => {
    const sdk = makePartialSdk({ ctxOk: false });

    const result = await fetchSynapShortTerm({ sdk, conversationId: 'conv-fail' });

    expect(result.available).toBe(false);
    expect(result.formattedContext).toBe('');
  });

  it('warns to console when SDK rejects and onError=fallback', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const sdk = makePartialSdk({ ctxOk: false });

    await fetchSynapShortTerm({ sdk, conversationId: 'conv-fail' });

    expect(warnSpy).toHaveBeenCalledOnce();
    warnSpy.mockRestore();
  });

  it('rethrows the SDK error when onError="raise"', async () => {
    const sdk = makePartialSdk({ ctxOk: false });

    await expect(
      fetchSynapShortTerm({ sdk, conversationId: 'conv-fail', onError: 'raise' }),
    ).rejects.toThrow('endpoint-down');
  });

  it('does NOT warn when onError="raise" and SDK rejects (the error propagates)', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const sdk = makePartialSdk({ ctxOk: false });

    try {
      await fetchSynapShortTerm({ sdk, conversationId: 'c-1', onError: 'raise' });
    } catch {
      // expected
    }

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

// ── buildSynapShortTermSystem ─────────────────────────────────────────────────

describe('buildSynapShortTermSystem', () => {
  const available: SynapShortTermResult = {
    formattedContext: 'User is a senior engineer.',
    available: true,
  };
  const unavailable: SynapShortTermResult = {
    formattedContext: '',
    available: false,
  };

  it('wraps ST content in default preamble tags when available', () => {
    const result = buildSynapShortTermSystem(available);
    expect(result).toContain('<synap_short_term_context>');
    expect(result).toContain('</synap_short_term_context>');
    expect(result).toContain('User is a senior engineer.');
  });

  it('preserves the user system prompt below ST content', () => {
    const result = buildSynapShortTermSystem(available, { system: 'You are a helpful assistant.' });
    expect(result).toContain('User is a senior engineer.');
    expect(result).toContain('You are a helpful assistant.');
    // ST should come before the user system prompt
    const stIdx = result.indexOf('<synap_short_term_context>');
    const sysIdx = result.indexOf('You are a helpful assistant.');
    expect(stIdx).toBeLessThan(sysIdx);
  });

  it('returns only the user system prompt when ST is unavailable', () => {
    const result = buildSynapShortTermSystem(unavailable, { system: 'My instructions.' });
    expect(result).toBe('My instructions.');
    expect(result).not.toContain('<synap_short_term_context>');
  });

  it('returns empty string when both ST and system are empty', () => {
    const result = buildSynapShortTermSystem(unavailable);
    expect(result).toBe('');
  });

  it('uses custom preambleOpen/preambleClose tags when provided', () => {
    const result = buildSynapShortTermSystem(available, {
      preambleOpen: '<ctx>',
      preambleClose: '</ctx>',
    });
    expect(result).toContain('<ctx>');
    expect(result).toContain('</ctx>');
    expect(result).not.toContain('<synap_short_term_context>');
  });

  it('emits raw ST body (no wrapper) when preambleOpen and preambleClose are both null', () => {
    const result = buildSynapShortTermSystem(available, {
      preambleOpen: null,
      preambleClose: null,
    });
    expect(result).toContain('User is a senior engineer.');
    expect(result).not.toContain('<synap_short_term_context>');
    expect(result).not.toContain('</synap_short_term_context>');
  });

  it('trims whitespace from formattedContext before composing', () => {
    const padded: SynapShortTermResult = { formattedContext: '   padded   ', available: true };
    const result = buildSynapShortTermSystem(padded, { preambleOpen: null, preambleClose: null });
    expect(result.startsWith(' ')).toBe(false);
    expect(result).toContain('padded');
  });

  it('trims user system text before composing', () => {
    const result = buildSynapShortTermSystem(unavailable, { system: '   spaced   ' });
    expect(result).toBe('spaced');
  });

  it('separates ST block and user system with double newline', () => {
    const result = buildSynapShortTermSystem(available, { system: 'My system.' });
    expect(result).toContain('\n\n');
  });
});

// ── synapShortTermInstructions (convenience wrapper) ─────────────────────────

describe('synapShortTermInstructions', () => {
  it('returns combined system string on success', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'Prior context.';
    const sdk = makeSdk({ promptCtx: ctx });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
      system: 'You are helpful.',
    });

    expect(instructions).toContain('Prior context.');
    expect(instructions).toContain('You are helpful.');
    expect(instructions).toContain('<synap_short_term_context>');
  });

  it('returns only user system text when ST unavailable', async () => {
    const ctx = promptContext(false, false);
    const sdk = makeSdk({ promptCtx: ctx });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
      system: 'My static instructions.',
    });

    expect(instructions).toBe('My static instructions.');
  });

  it('returns empty string when ST unavailable AND no system provided', async () => {
    const ctx = promptContext(false, false);
    const sdk = makeSdk({ promptCtx: ctx });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
    });

    expect(instructions).toBe('');
  });

  it('falls back gracefully (empty string) when SDK rejects and onError=fallback (default)', async () => {
    const sdk = makePartialSdk({ ctxOk: false });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-fail',
      system: 'Static.',
      onError: 'fallback',
    });

    // fallback: ST is empty, system is preserved
    expect(instructions).toBe('Static.');
  });

  it('rethrows when onError="raise" and SDK rejects', async () => {
    const sdk = makePartialSdk({ ctxOk: false });

    await expect(
      synapShortTermInstructions({
        sdk,
        conversationId: 'conv-fail',
        onError: 'raise',
      }),
    ).rejects.toThrow('endpoint-down');
  });

  it('passes style through to fetchSynapShortTerm', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'content';
    const sdk = makeSdk({ promptCtx: ctx });

    await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
      style: 'structured',
    });

    const callArg = (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(callArg.style).toBe('structured');
  });

  it('uses custom preamble tags when provided', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'body';
    const sdk = makeSdk({ promptCtx: ctx });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
      preambleOpen: '[[START]]',
      preambleClose: '[[END]]',
    });

    expect(instructions).toContain('[[START]]');
    expect(instructions).toContain('[[END]]');
  });

  it('passes null preamble options through to buildSynapShortTermSystem', async () => {
    const ctx = promptContext(false, true);
    ctx.formatted_context = 'raw body';
    const sdk = makeSdk({ promptCtx: ctx });

    const instructions = await synapShortTermInstructions({
      sdk,
      conversationId: 'conv-1',
      preambleOpen: null,
      preambleClose: null,
    });

    expect(instructions).toBe('raw body');
    expect(instructions).not.toContain('<synap_short_term_context>');
  });
});
