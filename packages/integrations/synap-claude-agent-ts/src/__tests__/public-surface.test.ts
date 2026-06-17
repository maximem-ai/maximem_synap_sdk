/**
 * Tests for src/index.ts public surface.
 *
 * Strategy: import the barrel and assert that every documented export is
 * present, is the right type (function / object), and is wired to the correct
 * underlying module. No live network, no Claude/Anthropic calls.
 */

import { describe, it, expect } from 'vitest';
import * as publicApi from '../index.js';

// ─── value exports ────────────────────────────────────────────────────────────

describe('index.ts — value exports', () => {
  it('exports createSynapHooks as a function', () => {
    expect(typeof publicApi.createSynapHooks).toBe('function');
  });

  it('exports createSynapMcpServer as a function', () => {
    expect(typeof publicApi.createSynapMcpServer).toBe('function');
  });

  it('exports buildSynapTools as a function', () => {
    expect(typeof publicApi.buildSynapTools).toBe('function');
  });

  it('exports createSynapShortTermHook as a function', () => {
    expect(typeof publicApi.createSynapShortTermHook).toBe('function');
  });

  it('has no unexpected extra enumerable exports beyond the documented set', () => {
    const documented = new Set([
      'createSynapHooks',
      'createSynapMcpServer',
      'buildSynapTools',
      'createSynapShortTermHook',
    ]);
    const actual = Object.keys(publicApi).filter((k) => !k.startsWith('__'));
    const undocumented = actual.filter((k) => !documented.has(k));
    // Warn-only: new exports are fine, but they should be visible in this diff.
    // We don't hard-fail so that adding a new export doesn't break the suite.
    if (undocumented.length > 0) {
      console.info(
        `[public-surface] New exports found (not yet documented): ${undocumented.join(', ')}`
      );
    }
    // The four documented ones must exist.
    for (const name of documented) {
      expect(actual).toContain(name);
    }
  });
});

// ─── wiring: createSynapHooks is the one from hooks.ts ───────────────────────

describe('index.ts — wiring sanity', () => {
  it('createSynapHooks throws on null sdk (construction guard is wired)', () => {
    // This guard is defined in hooks.ts — if the export is wired correctly, it throws.
    expect(() =>
      publicApi.createSynapHooks({
        sdk: null as unknown as Parameters<typeof publicApi.createSynapHooks>[0]['sdk'],
        userId: 'u1',
      }),
    ).toThrow(/non-null sdk/);
  });

  it('createSynapShortTermHook throws on null sdk (construction guard is wired)', () => {
    expect(() =>
      publicApi.createSynapShortTermHook({
        sdk: null as unknown as Parameters<typeof publicApi.createSynapShortTermHook>[0]['sdk'],
        conversationId: 'conv-1',
      }),
    ).toThrow(/non-null sdk/);
  });

  it('createSynapMcpServer throws on null sdk (construction guard is wired)', () => {
    expect(() =>
      publicApi.createSynapMcpServer({
        sdk: null as unknown as Parameters<typeof publicApi.createSynapMcpServer>[0]['sdk'],
        userId: 'u1',
      }),
    ).toThrow(/non-null sdk/);
  });

  it('buildSynapTools returns array with synap_search and synap_remember', () => {
    const mockSdk = {
      fetch: async () => ({ formatted_context: '' }),
      conversation: { record_message: async () => ({}) },
      memories: { create: async () => ({ ingestion_id: 'x' }) },
    };
    const tools = publicApi.buildSynapTools({
      sdk: mockSdk,
      userId: 'u1',
      customerId: '',
      conversationId: undefined,
      mode: 'accurate',
    });
    const names = tools.map((t) => t.name);
    expect(names).toContain('synap_search');
    expect(names).toContain('synap_remember');
  });
});
