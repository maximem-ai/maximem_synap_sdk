/**
 * Tests for src/context/http-fetcher.ts
 *
 * Covers:
 *   - fetchContext: happy path with full response body
 *   - fetchContext: correct endpoint selection per scope (conversation > user > customer > client)
 *   - fetchContext: correct auth headers
 *   - fetchContext: throws on non-2xx HTTP response
 *   - fetchContext: network error propagates
 *   - fetchContext: empty items_by_type handled gracefully
 *   - fetchContext: conversation_context parsed correctly
 *   - fetchContext: source "cache" vs "cloud" from metadata
 *   - parseContextResponse: malformed JSON in current_state_json handled
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchContext } from '../context/http-fetcher.js';
import type { Credentials, SynapModelOptions } from '../types.js';

// ─── Helpers ────────────────────────────────────────────────────────────────

const creds: Credentials = {
  api_key: 'sk-test-123',
  client_id: 'cli_abc',
  instance_id: 'inst_xyz',
};

function makeModelOpts(overrides: Partial<SynapModelOptions> = {}): SynapModelOptions {
  return {
    userId: 'u1',
    customerId: 'c1',
    conversationId: undefined,
    ...overrides,
  };
}

function mockFetch(status: number, body: unknown): ReturnType<typeof vi.fn> {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    text: vi.fn().mockResolvedValue(typeof body === 'string' ? body : JSON.stringify(body)),
    json: vi.fn().mockResolvedValue(body),
  });
}

// Full valid context response
function makeContextBody(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    items_by_type: {
      facts: {
        items: [
          {
            item_id: 'f1',
            content: 'User is a senior engineer',
            context_type: 'fact',
            source: 'profile',
            confidence: 0.9,
            similarity_score: 0.85,
            relevance_score: 0.8,
            scope: 'user',
            entity_id: '',
            event_date: '2026-01-01',
            valid_until: '2026-12-31',
            temporal_category: '',
            temporal_confidence: 0,
          },
        ],
      },
      preferences: { items: [] },
      episodes: { items: [] },
      emotions: { items: [] },
      temporal_events: { items: [] },
    },
    conversation_context: null,
    metadata: { source: 'cloud' },
    ...overrides,
  };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('fetchContext', () => {
  let globalFetch: typeof fetch;

  beforeEach(() => {
    globalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = globalFetch;
    vi.restoreAllMocks();
  });

  // ── Happy path ──────────────────────────────────────────────────────────────

  it('returns a FetchedContext with facts on success', async () => {
    global.fetch = mockFetch(200, makeContextBody()) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1' }),
      searchQuery: ['engineer'],
      baseUrl: 'https://synap.test',
    });
    expect(result.facts).toHaveLength(1);
    expect(result.facts[0].content).toBe('User is a senior engineer');
    expect(result.facts[0].id).toBe('f1');
    expect(result.facts[0].confidence).toBe(0.9);
  });

  it('maps eventDate and validUntil on fact items', async () => {
    global.fetch = mockFetch(200, makeContextBody()) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1' }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.facts[0].eventDate).toBe('2026-01-01');
    expect(result.facts[0].validUntil).toBe('2026-12-31');
  });

  it('sets source to "cloud" when metadata.source is not "cache"', async () => {
    global.fetch = mockFetch(200, makeContextBody({ metadata: { source: 'cloud' } })) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1' }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.source).toBe('cloud');
  });

  it('sets source to "cache" when metadata.source is "cache"', async () => {
    global.fetch = mockFetch(200, makeContextBody({ metadata: { source: 'cache' } })) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1' }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.source).toBe('cache');
  });

  it('returns a non-empty correlationId (UUID format)', async () => {
    global.fetch = mockFetch(200, makeContextBody()) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1' }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.correlationId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    );
  });

  // ── Scope-based endpoint selection ────────────────────────────────────────

  it('uses /v1/context/conversation/fetch when conversationId is set', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ conversationId: 'conv-1', userId: 'u1' }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [url, init] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/conversation/fetch');
    const body = JSON.parse(init.body as string);
    expect(body.conversation_id).toBe('conv-1');
  });

  it('uses /v1/context/user/fetch when only userId is set', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: makeModelOpts({ userId: 'u1', conversationId: undefined }),
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [url, init] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/user/fetch');
    const body = JSON.parse(init.body as string);
    expect(body.user_id).toBe('u1');
  });

  it('uses /v1/context/customer/fetch when only customerId is set', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: { customerId: 'cust-1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [url, init] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/customer/fetch');
    const body = JSON.parse(init.body as string);
    expect(body.customer_id).toBe('cust-1');
  });

  it('uses /v1/context/client/fetch when no scope fields are set', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: {},
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [url] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/client/fetch');
  });

  it('prefers conversation scope over user scope', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: { conversationId: 'conv-1', userId: 'u1', customerId: 'c1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [url] = mockF.mock.calls[0];
    expect(url).toContain('/v1/context/conversation/fetch');
  });

  // ── Auth headers ──────────────────────────────────────────────────────────

  it('sends correct Authorization and client/instance headers', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: { api_key: 'sk-abc', client_id: 'cli_1', instance_id: 'inst_1' },
      modelOptions: { userId: 'u1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [, init] = mockF.mock.calls[0];
    const headers = init.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer sk-abc');
    expect(headers['X-Client-ID']).toBe('cli_1');
    expect(headers['X-Instance-ID']).toBe('inst_1');
    expect(headers['Content-Type']).toBe('application/json');
    expect(headers['X-Correlation-ID']).toBeTruthy();
  });

  // ── Request body ───────────────────────────────────────────────────────────

  it('sends max_results from modelOptions', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: { userId: 'u1', maxContextResults: 25 },
      searchQuery: ['test'],
      baseUrl: 'https://synap.test',
    });
    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body.max_results).toBe(25);
  });

  it('defaults max_results to 10', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: { userId: 'u1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body.max_results).toBe(10);
  });

  it('sends search_query array in request body', async () => {
    const mockF = mockFetch(200, makeContextBody());
    global.fetch = mockF as unknown as typeof fetch;
    await fetchContext({
      credentials: creds,
      modelOptions: { userId: 'u1' },
      searchQuery: ['what is sharding?'],
      baseUrl: 'https://synap.test',
    });
    const [, init] = mockF.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body.search_query).toEqual(['what is sharding?']);
  });

  // ── Failure paths ─────────────────────────────────────────────────────────

  it('throws on HTTP 401', async () => {
    global.fetch = mockFetch(401, 'Unauthorized') as unknown as typeof fetch;
    await expect(
      fetchContext({
        credentials: creds,
        modelOptions: { userId: 'u1' },
        searchQuery: [],
        baseUrl: 'https://synap.test',
      }),
    ).rejects.toThrow(/HTTP 401/);
  });

  it('throws on HTTP 500', async () => {
    global.fetch = mockFetch(500, 'Internal Server Error') as unknown as typeof fetch;
    await expect(
      fetchContext({
        credentials: creds,
        modelOptions: { userId: 'u1' },
        searchQuery: [],
        baseUrl: 'https://synap.test',
      }),
    ).rejects.toThrow(/HTTP 500/);
  });

  it('throws on network error (fetch rejects)', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED')) as unknown as typeof fetch;
    await expect(
      fetchContext({
        credentials: creds,
        modelOptions: { userId: 'u1' },
        searchQuery: [],
        baseUrl: 'https://synap.test',
      }),
    ).rejects.toThrow('ECONNREFUSED');
  });

  // ── Empty / partial responses ──────────────────────────────────────────────

  it('handles empty items_by_type gracefully', async () => {
    global.fetch = mockFetch(200, { items_by_type: {}, metadata: {} }) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: { userId: 'u1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.facts).toEqual([]);
    expect(result.preferences).toEqual([]);
    expect(result.episodes).toEqual([]);
    expect(result.emotions).toEqual([]);
    expect(result.temporalEvents).toEqual([]);
    expect(result.conversationContext).toBeNull();
  });

  it('handles missing items_by_type key gracefully', async () => {
    global.fetch = mockFetch(200, { metadata: {} }) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: { userId: 'u1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.facts).toEqual([]);
  });

  // ── conversation_context parsing ──────────────────────────────────────────

  it('parses conversation_context from response', async () => {
    const body = makeContextBody({
      conversation_context: {
        summary: 'Context summary here',
        current_state_json: '{"step":3}',
        key_extractions_json: '{"intent":"deploy"}',
        recent_turns: [
          { role: 'user', content: 'How?', timestamp: '2026-01-01T00:00:00Z' },
        ],
        compaction_id: 'comp-42',
        compacted_at: '2026-01-01T00:00:00Z',
        conversation_id: 'conv-parse',
      },
    });
    global.fetch = mockFetch(200, body) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: { conversationId: 'conv-parse', userId: 'u1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.conversationContext).not.toBeNull();
    expect(result.conversationContext!.summary).toBe('Context summary here');
    expect(result.conversationContext!.currentState).toEqual({ step: 3 });
    expect(result.conversationContext!.keyExtractions).toEqual({ intent: 'deploy' });
    expect(result.conversationContext!.compactionId).toBe('comp-42');
    expect(result.conversationContext!.conversationId).toBe('conv-parse');
    expect(result.conversationContext!.recentTurns).toHaveLength(1);
  });

  it('handles invalid JSON in current_state_json gracefully', async () => {
    const body = makeContextBody({
      conversation_context: {
        summary: 'test',
        current_state_json: '{ bad json >>>',
        key_extractions_json: '{}',
        recent_turns: [],
        compaction_id: '',
        compacted_at: '',
        conversation_id: 'conv-1',
      },
    });
    global.fetch = mockFetch(200, body) as unknown as typeof fetch;
    const result = await fetchContext({
      credentials: creds,
      modelOptions: { conversationId: 'conv-1' },
      searchQuery: [],
      baseUrl: 'https://synap.test',
    });
    expect(result.conversationContext!.currentState).toEqual({});
  });
});
