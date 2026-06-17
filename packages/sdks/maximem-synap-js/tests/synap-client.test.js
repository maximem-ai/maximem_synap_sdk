/**
 * SynapClient unit tests — NO real Python spawn, NO cloud.
 *
 * Strategy: SynapClient constructs a BridgeManager internally.  We intercept
 * at the BridgeManager level by replacing `bridge.call` and `bridge.ensureStarted`
 * with vi.fn() mocks after construction, before any real I/O can occur.
 *
 * This avoids any child-process spawn or network activity while still exercising
 * the full public surface of SynapClient: argument validation, JSON-RPC param
 * serialisation, and response normalisation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// SynapClient is CJS — vitest handles CJS→ESM interop via its module transformer.
import { SynapClient } from '../src/synap-client.js';

// ---------------------------------------------------------------------------
// Factory: build a SynapClient whose bridge.call is mocked
// ---------------------------------------------------------------------------

function makeClient(bridgeResult = {}) {
  const client = new SynapClient({
    // Dummy values to avoid env-var lookups inside BridgeManager init
    instanceId: 'test-instance',
    apiKey: 'test-api-key',
  });

  // Prevent any real bridge startup
  client.bridge.ensureStarted = vi.fn().mockResolvedValue(undefined);
  // Default: bridge.call resolves with bridgeResult
  client.bridge.call = vi.fn().mockResolvedValue(bridgeResult);

  return client;
}

// ---------------------------------------------------------------------------
// init()
// ---------------------------------------------------------------------------

describe('SynapClient.init', () => {
  it('calls bridge.ensureStarted', async () => {
    const client = makeClient();
    await client.init();
    expect(client.bridge.ensureStarted).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// addMemory()
// ---------------------------------------------------------------------------

describe('SynapClient.addMemory', () => {
  it('sends add_memory with required params', async () => {
    const client = makeClient({ success: true });

    await client.addMemory({
      userId: 'u1',
      customerId: 'c1',
      messages: [{ role: 'user', content: 'hello' }],
    });

    expect(client.bridge.call).toHaveBeenCalledOnce();
    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('add_memory');
    expect(params.user_id).toBe('u1');
    expect(params.customer_id).toBe('c1');
    expect(params.messages).toEqual([{ role: 'user', content: 'hello' }]);
  });

  it('sends optional params when provided', async () => {
    const client = makeClient({ success: true });

    await client.addMemory({
      userId: 'u1',
      customerId: 'c1',
      messages: [],
      conversationId: 'conv-1',
      sessionId: 'sess-1',
      mode: 'fast',
      documentType: 'article',
      documentId: 'doc-1',
      documentCreatedAt: '2024-01-01',
      metadata: { tag: 'test' },
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.conversation_id).toBe('conv-1');
    expect(params.session_id).toBe('sess-1');
    expect(params.mode).toBe('fast');
    expect(params.document_type).toBe('article');
    expect(params.document_id).toBe('doc-1');
    expect(params.document_created_at).toBe('2024-01-01');
    expect(params.metadata).toEqual({ tag: 'test' });
  });

  it('throws when userId is missing', async () => {
    const client = makeClient();
    await expect(client.addMemory({ customerId: 'c1', messages: [] })).rejects.toThrow('userId is required');
  });

  it('throws when customerId is missing', async () => {
    const client = makeClient();
    await expect(client.addMemory({ userId: 'u1', messages: [] })).rejects.toThrow('customerId is required');
  });

  it('throws when messages is not an array', async () => {
    const client = makeClient();
    await expect(
      client.addMemory({ userId: 'u1', customerId: 'c1', messages: 'bad' })
    ).rejects.toThrow('messages must be an array');
  });

  it('uses ingestTimeoutMs for bridge.call timeout', async () => {
    const client = makeClient({ success: true });
    const defaultIngestMs = client.options.ingestTimeoutMs;

    await client.addMemory({ userId: 'u1', customerId: 'c1', messages: [] });

    const [, , timeoutMs] = client.bridge.call.mock.calls[0];
    expect(timeoutMs).toBe(defaultIngestMs);
  });

  it('propagates bridge error', async () => {
    const client = makeClient();
    client.bridge.call = vi.fn().mockRejectedValue(new Error('Bridge exploded'));

    await expect(
      client.addMemory({ userId: 'u1', customerId: 'c1', messages: [] })
    ).rejects.toThrow('Bridge exploded');
  });
});

// ---------------------------------------------------------------------------
// searchMemory()
// ---------------------------------------------------------------------------

describe('SynapClient.searchMemory', () => {
  it('sends search_memory with required params', async () => {
    const client = makeClient({
      success: true,
      results: [],
      latencyMs: 5,
    });

    await client.searchMemory({ userId: 'u1', query: 'cats' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('search_memory');
    expect(params.user_id).toBe('u1');
    expect(params.query).toBe('cats');
    expect(params.max_results).toBe(10); // default
  });

  it('sends optional params when provided', async () => {
    const client = makeClient({ success: true, results: [] });

    await client.searchMemory({
      userId: 'u1',
      query: 'q',
      customerId: 'c1',
      maxResults: 5,
      mode: 'deep',
      conversationId: 'conv-1',
      types: ['fact', 'preference'],
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.customer_id).toBe('c1');
    expect(params.max_results).toBe(5);
    expect(params.mode).toBe('deep');
    expect(params.conversation_id).toBe('conv-1');
    expect(params.types).toEqual(['fact', 'preference']);
  });

  it('throws when userId is missing', async () => {
    const client = makeClient();
    await expect(client.searchMemory({ query: 'x' })).rejects.toThrow('userId is required');
  });

  it('throws when query is missing', async () => {
    const client = makeClient();
    await expect(client.searchMemory({ userId: 'u1' })).rejects.toThrow('query is required');
  });

  it('throws when types is not an array', async () => {
    const client = makeClient();
    await expect(
      client.searchMemory({ userId: 'u1', query: 'x', types: 'fact' })
    ).rejects.toThrow('types must be an array when provided');
  });

  it('normalises the bridge result into searchMemoryResult shape', async () => {
    const client = makeClient({
      success: true,
      latencyMs: 12,
      results: [
        { id: 'm1', memory: 'I like cats', score: 0.9, source: 'user', metadata: {} },
      ],
    });

    const result = await client.searchMemory({ userId: 'u1', query: 'cats' });

    expect(result.success).toBe(true);
    expect(result.latencyMs).toBe(12);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe('m1');
    expect(result.results[0].memory).toBe('I like cats');
    expect(result.results[0].score).toBe(0.9);
  });

  it('normalises snake_case content field to memory', async () => {
    const client = makeClient({
      success: true,
      results: [{ id: 'm2', content: 'alt field', score: 0.5 }],
    });

    const result = await client.searchMemory({ userId: 'u1', query: 'x' });
    expect(result.results[0].memory).toBe('alt field');
  });

  it('propagates bridge error', async () => {
    const client = makeClient();
    client.bridge.call = vi.fn().mockRejectedValue(new Error('timeout'));
    await expect(client.searchMemory({ userId: 'u1', query: 'x' })).rejects.toThrow('timeout');
  });
});

// ---------------------------------------------------------------------------
// getMemories()
// ---------------------------------------------------------------------------

describe('SynapClient.getMemories', () => {
  it('sends get_memories with required params', async () => {
    const client = makeClient({ success: true, memories: [] });

    await client.getMemories({ userId: 'u1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('get_memories');
    expect(params.user_id).toBe('u1');
  });

  it('sends optional params when provided', async () => {
    const client = makeClient({ success: true, memories: [] });

    await client.getMemories({
      userId: 'u1',
      customerId: 'c1',
      mode: 'all',
      conversationId: 'conv-1',
      maxResults: 20,
      types: ['episode'],
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.customer_id).toBe('c1');
    expect(params.mode).toBe('all');
    expect(params.conversation_id).toBe('conv-1');
    expect(params.max_results).toBe(20);
    expect(params.types).toEqual(['episode']);
  });

  it('throws when userId is missing', async () => {
    const client = makeClient();
    await expect(client.getMemories({})).rejects.toThrow('userId is required');
  });

  it('throws when types is not an array', async () => {
    const client = makeClient();
    await expect(
      client.getMemories({ userId: 'u1', types: 'episode' })
    ).rejects.toThrow('types must be an array when provided');
  });

  it('normalises to getMemoriesResult shape', async () => {
    const client = makeClient({
      success: true,
      latencyMs: 3,
      memories: [{ id: 'mem1', memory: 'first memory', score: 1 }],
    });

    const result = await client.getMemories({ userId: 'u1' });
    expect(result.success).toBe(true);
    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].id).toBe('mem1');
  });
});

// ---------------------------------------------------------------------------
// fetchUserContext()
// ---------------------------------------------------------------------------

describe('SynapClient.fetchUserContext', () => {
  const minimalContextResult = {
    context: {
      facts: [],
      preferences: [],
      episodes: [],
      emotions: [],
      temporal_events: [],
      conversation_context: null,
      metadata: {},
    },
  };

  it('sends fetch_user_context with userId', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchUserContext({ userId: 'u1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('fetch_user_context');
    expect(params.user_id).toBe('u1');
    expect(params.max_results).toBe(10);
  });

  it('sends optional params', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchUserContext({
      userId: 'u1',
      customerId: 'c1',
      conversationId: 'conv-1',
      searchQuery: ['a', 'b'],
      types: ['fact'],
      mode: 'fast',
      maxResults: 5,
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.customer_id).toBe('c1');
    expect(params.conversation_id).toBe('conv-1');
    expect(params.search_query).toEqual(['a', 'b']);
    expect(params.types).toEqual(['fact']);
    expect(params.mode).toBe('fast');
    expect(params.max_results).toBe(5);
  });

  it('throws when userId is missing', async () => {
    const client = makeClient();
    await expect(client.fetchUserContext({})).rejects.toThrow('userId is required');
  });

  it('throws when searchQuery is not an array', async () => {
    const client = makeClient();
    await expect(
      client.fetchUserContext({ userId: 'u1', searchQuery: 'bad' })
    ).rejects.toThrow('searchQuery must be an array when provided');
  });

  it('throws when types is not an array', async () => {
    const client = makeClient();
    await expect(
      client.fetchUserContext({ userId: 'u1', types: 'fact' })
    ).rejects.toThrow('types must be an array when provided');
  });

  it('normalises context response', async () => {
    const client = makeClient({
      context: {
        facts: [{ id: 'f1', content: 'test fact', confidence: 0.8 }],
        preferences: [],
        episodes: [],
        emotions: [],
        temporal_events: [],
        conversation_context: null,
        metadata: { correlation_id: 'corr-1', source: 'bridge' },
      },
    });

    const result = await client.fetchUserContext({ userId: 'u1' });
    expect(result.facts).toHaveLength(1);
    expect(result.facts[0].id).toBe('f1');
    expect(result.metadata.correlationId).toBe('corr-1');
  });
});

// ---------------------------------------------------------------------------
// fetchCustomerContext()
// ---------------------------------------------------------------------------

describe('SynapClient.fetchCustomerContext', () => {
  const minimalContextResult = {
    context: {
      facts: [],
      preferences: [],
      episodes: [],
      emotions: [],
      temporal_events: [],
      conversation_context: null,
      metadata: {},
    },
  };

  it('sends fetch_customer_context with customerId', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchCustomerContext({ customerId: 'c1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('fetch_customer_context');
    expect(params.customer_id).toBe('c1');
    expect(params.max_results).toBe(10);
  });

  it('throws when customerId is missing', async () => {
    const client = makeClient();
    await expect(client.fetchCustomerContext({})).rejects.toThrow('customerId is required');
  });

  it('throws when searchQuery is not an array', async () => {
    const client = makeClient();
    await expect(
      client.fetchCustomerContext({ customerId: 'c1', searchQuery: 'bad' })
    ).rejects.toThrow('searchQuery must be an array when provided');
  });

  it('sends optional params', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchCustomerContext({
      customerId: 'c1',
      conversationId: 'conv-1',
      searchQuery: ['q1'],
      types: ['preference'],
      mode: 'deep',
      maxResults: 15,
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.conversation_id).toBe('conv-1');
    expect(params.search_query).toEqual(['q1']);
    expect(params.types).toEqual(['preference']);
    expect(params.mode).toBe('deep');
    expect(params.max_results).toBe(15);
  });
});

// ---------------------------------------------------------------------------
// fetchClientContext()
// ---------------------------------------------------------------------------

describe('SynapClient.fetchClientContext', () => {
  const minimalContextResult = {
    context: {
      facts: [],
      preferences: [],
      episodes: [],
      emotions: [],
      temporal_events: [],
      conversation_context: null,
      metadata: {},
    },
  };

  it('sends fetch_client_context with default maxResults', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchClientContext();

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('fetch_client_context');
    expect(params.max_results).toBe(10);
  });

  it('sends optional params', async () => {
    const client = makeClient(minimalContextResult);

    await client.fetchClientContext({
      conversationId: 'conv-2',
      searchQuery: ['term'],
      types: ['episode'],
      mode: 'fast',
      maxResults: 3,
    });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.conversation_id).toBe('conv-2');
    expect(params.search_query).toEqual(['term']);
    expect(params.types).toEqual(['episode']);
    expect(params.mode).toBe('fast');
    expect(params.max_results).toBe(3);
  });

  it('throws when searchQuery is not an array', async () => {
    const client = makeClient();
    await expect(
      client.fetchClientContext({ searchQuery: 'bad' })
    ).rejects.toThrow('searchQuery must be an array when provided');
  });
});

// ---------------------------------------------------------------------------
// getContextForPrompt()
// ---------------------------------------------------------------------------

describe('SynapClient.getContextForPrompt', () => {
  const minimalPromptResult = {
    context_for_prompt: {
      formatted_context: 'Context: ...',
      available: true,
      is_stale: false,
      compression_ratio: null,
      validation_score: null,
      compaction_age_seconds: null,
      quality_warning: false,
      recent_messages: [],
      recent_message_count: 0,
      compacted_message_count: 0,
      total_message_count: 0,
    },
  };

  it('sends get_context_for_prompt with conversationId', async () => {
    const client = makeClient(minimalPromptResult);

    await client.getContextForPrompt({ conversationId: 'conv-1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('get_context_for_prompt');
    expect(params.conversation_id).toBe('conv-1');
  });

  it('sends style when provided', async () => {
    const client = makeClient(minimalPromptResult);

    await client.getContextForPrompt({ conversationId: 'conv-1', style: 'concise' });

    const [, params] = client.bridge.call.mock.calls[0];
    expect(params.style).toBe('concise');
  });

  it('throws when conversationId is missing', async () => {
    const client = makeClient();
    await expect(client.getContextForPrompt({})).rejects.toThrow('conversationId is required');
  });

  it('normalises the result into contextForPrompt shape', async () => {
    const client = makeClient({
      context_for_prompt: {
        formatted_context: 'Here is context',
        available: true,
        is_stale: false,
        recent_messages: [
          { role: 'user', content: 'hi', timestamp: null, message_id: 'msg-1' },
        ],
        recent_message_count: 1,
        compacted_message_count: 0,
        total_message_count: 1,
      },
    });

    const result = await client.getContextForPrompt({ conversationId: 'conv-1' });

    expect(result.formattedContext).toBe('Here is context');
    expect(result.available).toBe(true);
    expect(result.isStale).toBe(false);
    expect(result.recentMessages).toHaveLength(1);
    expect(result.recentMessages[0].messageId).toBe('msg-1');
    expect(result.recentMessageCount).toBe(1);
    expect(result.totalMessageCount).toBe(1);
  });

  it('normalises camelCase fields from bridge', async () => {
    // Bridge may return camelCase (JS side)
    const client = makeClient({
      contextForPrompt: {
        formattedContext: 'camelCase context',
        available: true,
        isStale: true,
        recentMessages: [],
        recentMessageCount: 0,
        compactedMessageCount: 2,
        totalMessageCount: 2,
      },
    });

    const result = await client.getContextForPrompt({ conversationId: 'conv-1' });
    expect(result.formattedContext).toBe('camelCase context');
    expect(result.isStale).toBe(true);
    expect(result.compactedMessageCount).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// deleteMemory()
// ---------------------------------------------------------------------------

describe('SynapClient.deleteMemory', () => {
  it('sends delete_memory with userId and memoryId', async () => {
    const client = makeClient({ success: true, deletedCount: 1 });

    await client.deleteMemory({ userId: 'u1', memoryId: 'mem-1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('delete_memory');
    expect(params.user_id).toBe('u1');
    expect(params.memory_id).toBe('mem-1');
  });

  it('sends delete_memory with userId and customerId (bulk delete)', async () => {
    const client = makeClient({ success: true, deletedCount: 3 });

    await client.deleteMemory({ userId: 'u1', customerId: 'c1' });

    const [method, params] = client.bridge.call.mock.calls[0];
    expect(method).toBe('delete_memory');
    expect(params.user_id).toBe('u1');
    expect(params.customer_id).toBe('c1');
    expect(params.memory_id).toBeNull();
  });

  it('throws when userId is missing', async () => {
    const client = makeClient();
    await expect(client.deleteMemory({ memoryId: 'm1' })).rejects.toThrow('userId is required');
  });

  it('throws when memoryId is null and customerId is missing', async () => {
    const client = makeClient();
    await expect(client.deleteMemory({ userId: 'u1' })).rejects.toThrow(
      'customerId is required when memoryId is not provided'
    );
  });

  it('normalises deleteMemoryResult shape', async () => {
    const client = makeClient({
      success: true,
      latencyMs: 2,
      deletedCount: 1,
      rawResponse: { deleted: 1 },
    });

    const result = await client.deleteMemory({ userId: 'u1', memoryId: 'mem-1' });
    expect(result.success).toBe(true);
    expect(result.latencyMs).toBe(2);
    expect(result.deletedCount).toBe(1);
  });

  it('normalises deletedCount from rawResponse.deleted when deletedCount not present', async () => {
    const client = makeClient({
      success: true,
      latencyMs: 1,
      rawResponse: { deleted: 5 },
    });

    const result = await client.deleteMemory({ userId: 'u1', memoryId: 'mem-1' });
    expect(result.deletedCount).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// shutdown()
// ---------------------------------------------------------------------------

describe('SynapClient.shutdown', () => {
  it('calls bridge.shutdown', async () => {
    const client = makeClient();
    client.bridge.shutdown = vi.fn().mockResolvedValue(undefined);

    await client.shutdown();

    expect(client.bridge.shutdown).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// Reduced surface — gRPC / streaming / batch NOT present
// ---------------------------------------------------------------------------

describe('SynapClient reduced surface (JS = REST-only subset)', () => {
  it('has no streamMemory method', () => {
    const client = makeClient();
    expect(client.streamMemory).toBeUndefined();
  });

  it('has no batchAddMemory method', () => {
    const client = makeClient();
    expect(client.batchAddMemory).toBeUndefined();
  });

  it('has no connectGrpc method', () => {
    const client = makeClient();
    expect(client.connectGrpc).toBeUndefined();
  });

  it('has no subscribe method', () => {
    const client = makeClient();
    expect(client.subscribe).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Context normalisation edge cases
// ---------------------------------------------------------------------------

describe('normalisation edge cases', () => {
  it('fetchUserContext handles temporal_events with both camelCase and snake_case', async () => {
    const client = makeClient({
      context: {
        facts: [],
        preferences: [],
        episodes: [],
        emotions: [],
        temporalEvents: [
          {
            id: 'te1',
            content: 'project deadline',
            event_date: '2024-06-01',
            valid_until: '2024-07-01',
            temporal_category: 'deadline',
            temporal_confidence: 0.95,
            confidence: 0.8,
            source: 'user',
          },
        ],
        conversation_context: null,
        metadata: {},
      },
    });

    const result = await client.fetchUserContext({ userId: 'u1' });
    expect(result.temporalEvents).toHaveLength(1);
    const te = result.temporalEvents[0];
    expect(te.id).toBe('te1');
    expect(te.content).toBe('project deadline');
    expect(te.eventDate).toBe('2024-06-01');
    expect(te.validUntil).toBe('2024-07-01');
    expect(te.temporalCategory).toBe('deadline');
    expect(te.temporalConfidence).toBe(0.95);
  });

  it('fetchUserContext handles conversationContext with snake_case fields', async () => {
    const client = makeClient({
      context: {
        facts: [],
        preferences: [],
        episodes: [],
        emotions: [],
        temporal_events: [],
        conversation_context: {
          summary: 'A chat about cats',
          current_state: { topic: 'cats' },
          key_extractions: { animal: 'cat' },
          recent_turns: ['Hello', 'Hi'],
          compaction_id: 'comp-1',
          compacted_at: '2024-01-01',
          conversation_id: 'conv-1',
        },
        metadata: {},
      },
    });

    const result = await client.fetchUserContext({ userId: 'u1' });
    const cc = result.conversationContext;
    expect(cc).not.toBeNull();
    expect(cc.summary).toBe('A chat about cats');
    expect(cc.currentState).toEqual({ topic: 'cats' });
    expect(cc.keyExtractions).toEqual({ animal: 'cat' });
    expect(cc.recentTurns).toEqual(['Hello', 'Hi']);
    expect(cc.compactionId).toBe('comp-1');
    expect(cc.conversationId).toBe('conv-1');
  });

  it('getMemories normalises memories with snake_case fields', async () => {
    const client = makeClient({
      success: true,
      memories: [
        {
          id: 'mem1',
          content: 'Uses snake_case',
          score: 0.7,
          source: 'ingestion',
          context_type: 'user',
          event_date: '2024-01-10',
          temporal_confidence: 0.6,
        },
      ],
    });

    const result = await client.getMemories({ userId: 'u1' });
    const mem = result.memories[0];
    expect(mem.memory).toBe('Uses snake_case');
    expect(mem.contextType).toBe('user');
    expect(mem.eventDate).toBe('2024-01-10');
    expect(mem.temporalConfidence).toBe(0.6);
  });

  it('searchMemory returns resultsCount equal to results.length when not provided', async () => {
    const client = makeClient({
      success: true,
      results: [{ id: 'm1', memory: 'a' }, { id: 'm2', memory: 'b' }],
    });

    const result = await client.searchMemory({ userId: 'u1', query: 'x' });
    // normalizeSearchMemoryResult: resultsCount = result.resultsCount ?? result.results?.length ?? 0
    expect(result.resultsCount).toBe(2);
  });
});
