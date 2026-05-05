import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SynapProvider, createSynap } from '../provider.js';
import type { LanguageModelV1 } from 'ai';

const fakeModel = {} as LanguageModelV1;

describe('SynapProvider', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_API_KEY: 'sk-test' };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('wrap() throws if init() was not called', () => {
    const provider = new SynapProvider({});
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).toThrow('not initialized');
  });

  it('wrap() throws if no identity field is provided', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, {})).toThrow('requires at least one of');
  });

  it('wrap() succeeds with userId', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });

  it('wrap() succeeds with conversationId only', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, { conversationId: 'conv-1' })).not.toThrow();
  });

  it('createSynap() returns an initialized provider', async () => {
    const provider = await createSynap({});
    expect(provider).toBeInstanceOf(SynapProvider);
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });
});
