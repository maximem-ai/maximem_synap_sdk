/**
 * Expanded tests for src/provider.ts (SynapProvider, createSynap)
 *
 * Covers:
 *   - init(): loads credentials from apiKey option
 *   - init(): loads credentials from SYNAP_API_KEY env var
 *   - init(): throws when no API key available
 *   - wrap(): throws before init()
 *   - wrap(): throws when no identity fields
 *   - wrap(): succeeds with userId, customerId, conversationId
 *   - wrap(): returns a wrapLanguageModel-wrapped model (has .doGenerate-compatible interface)
 *   - isListening: false before listen()
 *   - cacheSize: starts at 0
 *   - listen(): no-op in non-Node.js environment (grpc import failure → graceful degradation)
 *   - stopListening(): does not throw when not listening
 *   - createSynap(): returns initialized SynapProvider
 *   - createSynap(): throws when no API key
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SynapProvider, createSynap } from '../provider.js';
import type { LanguageModelV1 } from 'ai';

// ─── Minimal fake LanguageModelV1 ─────────────────────────────────────────────
const fakeModel = {
  specificationVersion: 'v1',
  provider: 'fake',
  modelId: 'fake-model',
  defaultObjectGenerationMode: undefined,
  doGenerate: vi.fn().mockResolvedValue({
    text: 'ok',
    finishReason: 'stop',
    usage: { promptTokens: 1, completionTokens: 1 },
    rawCall: { rawPrompt: '', rawSettings: {} },
  }),
  doStream: vi.fn().mockResolvedValue({
    stream: new ReadableStream({ start(c) { c.close(); } }),
    rawCall: { rawPrompt: '', rawSettings: {} },
  }),
} as unknown as LanguageModelV1;

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('SynapProvider.init', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_API_KEY'];
    delete process.env['SYNAP_CLIENT_ID'];
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  it('initializes successfully with explicit apiKey', async () => {
    const provider = new SynapProvider({ apiKey: 'sk-explicit' });
    await provider.init();
    // After init, wrap should succeed
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });

  it('initializes successfully from SYNAP_API_KEY env var', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-from-env';
    const provider = new SynapProvider({});
    await provider.init();
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });

  it('throws when no API key is available anywhere', async () => {
    const provider = new SynapProvider({});
    await expect(provider.init()).rejects.toThrow(/No Synap API key/);
  });

  it('returns `this` from init() enabling chaining', async () => {
    const provider = new SynapProvider({ apiKey: 'sk-chain' });
    const returned = await provider.init();
    expect(returned).toBe(provider);
  });
});

describe('SynapProvider.wrap', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_API_KEY: 'sk-test' };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('throws if init() was not called', () => {
    const provider = new SynapProvider({});
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).toThrow(/not initialized/);
  });

  it('throws if no identity field provided', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, {})).toThrow(/requires at least one of/);
  });

  it('succeeds with userId only', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });

  it('succeeds with customerId only', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, { customerId: 'cust-1' })).not.toThrow();
  });

  it('succeeds with conversationId only', async () => {
    const provider = await new SynapProvider({}).init();
    expect(() => provider.wrap(fakeModel, { conversationId: 'conv-1' })).not.toThrow();
  });

  it('returns a LanguageModelV1-compatible object', async () => {
    const provider = await new SynapProvider({}).init();
    const wrapped = provider.wrap(fakeModel, { userId: 'u1' });
    // Must have doGenerate and doStream (interface requirement)
    expect(typeof wrapped.doGenerate).toBe('function');
    expect(typeof wrapped.doStream).toBe('function');
  });

  it('can be called multiple times with different model options', async () => {
    const provider = await new SynapProvider({}).init();
    const m1 = provider.wrap(fakeModel, { userId: 'u1' });
    const m2 = provider.wrap(fakeModel, { userId: 'u2' });
    expect(m1).not.toBe(m2);
  });
});

describe('SynapProvider.isListening + cacheSize', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_API_KEY: 'sk-test' };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('isListening is false before listen()', async () => {
    const provider = await new SynapProvider({}).init();
    expect(provider.isListening).toBe(false);
  });

  it('cacheSize starts at 0', async () => {
    const provider = await new SynapProvider({}).init();
    expect(provider.cacheSize).toBe(0);
  });

  it('stopListening() does not throw when not listening', async () => {
    const provider = await new SynapProvider({}).init();
    await expect(provider.stopListening()).resolves.toBeUndefined();
  });

  it('isListening remains false after stopListening() when never started', async () => {
    const provider = await new SynapProvider({}).init();
    await provider.stopListening();
    expect(provider.isListening).toBe(false);
  });
});

describe('SynapProvider.listen (graceful degradation)', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_API_KEY: 'sk-test' };
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  it('throws if called before init()', async () => {
    const provider = new SynapProvider({});
    await expect(provider.listen()).rejects.toThrow(/init\(\)/);
  });

  it('silently degrades when gRPC import fails (connect error)', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const provider = await new SynapProvider({
      grpcHost: '127.0.0.1',
      grpcPort: 1, // unreachable
      grpcUseTls: false,
    }).init();

    // listen() should not throw even if gRPC connect fails
    await expect(provider.listen()).resolves.toBeUndefined();
    // isListening may be false due to connection failure
    // (grpcClient set to null in catch block)
    warnSpy.mockRestore();
  });
});

describe('createSynap', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_API_KEY'];
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('returns an initialized SynapProvider', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-create-synap';
    const provider = await createSynap({});
    expect(provider).toBeInstanceOf(SynapProvider);
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });

  it('accepts explicit apiKey', async () => {
    const provider = await createSynap({ apiKey: 'sk-explicit-create' });
    expect(provider).toBeInstanceOf(SynapProvider);
  });

  it('throws when no API key available', async () => {
    await expect(createSynap({})).rejects.toThrow(/No Synap API key/);
  });

  it('forwards baseUrl to the credential manager', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-baseurl';
    const provider = await createSynap({ baseUrl: 'https://custom.synap.test' });
    // Should succeed and be initialized
    expect(() => provider.wrap(fakeModel, { userId: 'u1' })).not.toThrow();
  });
});
