/**
 * Tests for src/memory/writer.ts
 *
 * Covers:
 *   - writeMemory: happy path — POSTs conversation turn to /v1/memories/ingest
 *   - writeMemory: includes auth headers (Authorization, X-Client-ID, X-Instance-ID, X-Correlation-ID)
 *   - writeMemory: request body contains messages + assistantResponse appended
 *   - writeMemory: no-op when writeMemory: false
 *   - writeMemory: no-op when no identity fields (userId, conversationId, customerId)
 *   - writeMemory: non-fatal on network error (resolves, does not throw)
 *   - writeMemory: non-fatal on HTTP 500 (resolves, does not throw)
 *   - writeMemory: uses correct baseUrl
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { writeMemory } from '../memory/writer.js';
import type { Credentials, SynapModelOptions } from '../types.js';

// ─── Helpers ────────────────────────────────────────────────────────────────

const creds: Credentials = {
  api_key: 'sk-write-test',
  client_id: 'cli_write',
  instance_id: 'inst_write',
};

const baseMessages = [
  { role: 'user', content: 'What is sharding?' },
  { role: 'assistant', content: 'Sharding splits data.' },
];

function makeOpts(overrides: Partial<SynapModelOptions> = {}): SynapModelOptions {
  return {
    userId: 'u1',
    customerId: 'c1',
    conversationId: 'conv-1',
    ...overrides,
  };
}

function mockOkFetch(): ReturnType<typeof vi.fn> {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue({}),
  });
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('writeMemory', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  // ── Happy path ──────────────────────────────────────────────────────────────

  it('POSTs to /v1/memories/ingest on success', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: baseMessages,
      assistantResponse: 'Final answer',
      baseUrl: 'https://synap.test',
    });

    expect(mockF).toHaveBeenCalledOnce();
    const [url] = mockF.mock.calls[0];
    expect(url).toBe('https://synap.test/v1/memories/ingest');
  });

  it('sends POST method', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: baseMessages,
      assistantResponse: 'Answer',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    expect((init as RequestInit).method).toBe('POST');
  });

  it('includes Authorization header with Bearer token', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: { api_key: 'sk-secret', client_id: 'c', instance_id: 'i' },
      modelOptions: makeOpts(),
      messages: [],
      assistantResponse: 'x',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer sk-secret');
  });

  it('includes X-Client-ID and X-Instance-ID headers', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: { api_key: 'sk-x', client_id: 'cli_abc', instance_id: 'inst_xyz' },
      modelOptions: makeOpts(),
      messages: [],
      assistantResponse: 'x',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers['X-Client-ID']).toBe('cli_abc');
    expect(headers['X-Instance-ID']).toBe('inst_xyz');
  });

  it('includes X-Correlation-ID header', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: [],
      assistantResponse: 'x',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers['X-Correlation-ID']).toBeTruthy();
  });

  it('appends assistant response to messages in request body', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: [{ role: 'user', content: 'Hello' }],
      assistantResponse: 'World',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0]).toEqual({ role: 'user', content: 'Hello' });
    expect(body.messages[1]).toEqual({ role: 'assistant', content: 'World' });
  });

  it('sends userId, customerId, conversationId in body', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: { userId: 'u99', customerId: 'cust-99', conversationId: 'conv-99' },
      messages: [],
      assistantResponse: 'ok',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.user_id).toBe('u99');
    expect(body.customer_id).toBe('cust-99');
    expect(body.conversation_id).toBe('conv-99');
  });

  it('sends source = "vercel_ai_sdk" in body', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: [],
      assistantResponse: 'ok',
      baseUrl: 'https://synap.test',
    });

    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.source).toBe('vercel_ai_sdk');
  });

  it('uses the default base URL when baseUrl is not specified', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts(),
      messages: [],
      assistantResponse: 'ok',
    });

    const [url] = mockF.mock.calls[0];
    expect(url).toContain('synap-cloud-prod.maximem.ai');
    expect(url).toContain('/v1/memories/ingest');
  });

  // ── No-op conditions ──────────────────────────────────────────────────────

  it('does not call fetch when writeMemory: false', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: makeOpts({ writeMemory: false }),
      messages: baseMessages,
      assistantResponse: 'answer',
      baseUrl: 'https://synap.test',
    });

    expect(mockF).not.toHaveBeenCalled();
  });

  it('does not call fetch when no identity fields are provided', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: {}, // no userId, customerId, conversationId
      messages: baseMessages,
      assistantResponse: 'answer',
      baseUrl: 'https://synap.test',
    });

    expect(mockF).not.toHaveBeenCalled();
  });

  // ── Non-fatal failure ─────────────────────────────────────────────────────

  it('resolves (does not throw) on network error', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network down')) as unknown as typeof fetch;

    await expect(
      writeMemory({
        credentials: creds,
        modelOptions: makeOpts(),
        messages: baseMessages,
        assistantResponse: 'ok',
        baseUrl: 'https://synap.test',
      }),
    ).resolves.toBeUndefined();
  });

  it('resolves (does not throw) on HTTP 500', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: vi.fn().mockResolvedValue({}),
    }) as unknown as typeof fetch;

    await expect(
      writeMemory({
        credentials: creds,
        modelOptions: makeOpts(),
        messages: baseMessages,
        assistantResponse: 'ok',
        baseUrl: 'https://synap.test',
      }),
    ).resolves.toBeUndefined();
  });

  // ── userId only (no conversationId) ───────────────────────────────────────

  it('fires write with only userId (no conversationId)', async () => {
    const mockF = mockOkFetch();
    global.fetch = mockF as unknown as typeof fetch;

    await writeMemory({
      credentials: creds,
      modelOptions: { userId: 'u-only' },
      messages: [{ role: 'user', content: 'hi' }],
      assistantResponse: 'hello',
      baseUrl: 'https://synap.test',
    });

    expect(mockF).toHaveBeenCalledOnce();
    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.user_id).toBe('u-only');
    expect(body.conversation_id).toBe('');
  });
});
