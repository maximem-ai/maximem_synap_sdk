/**
 * Tests for buildSynapTools and createSynapMcpServer (src/mcp-server.ts)
 *
 * Strategy:
 *   - buildSynapTools: construct a ToolContext with a duck-typed SDK mock, call
 *     buildSynapTools(), then invoke each tool's handler directly. The SDK's
 *     `tool()` factory returns an object with a `handler` property — no MCP
 *     server runtime is required.
 *   - createSynapMcpServer: mock `@anthropic-ai/claude-agent-sdk` so that
 *     createSdkMcpServer returns a deterministic stub, then assert the function
 *     throws on bad options, passes the right name/version, and returns the
 *     server object from createSdkMcpServer.
 *
 * No live network, no Claude/Anthropic calls.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { SynapSdkLike } from '../types.js';

// ─── mock the entire claude-agent-sdk ────────────────────────────────────────
// We need createSdkMcpServer to be a controllable mock. The `tool` function
// must work normally (we use the real implementation) so we only replace
// createSdkMcpServer.

const mockMcpServerInstance = { name: 'stub-server', type: 'sdk' };
const mockCreateSdkMcpServer = vi.fn((_opts: unknown) => mockMcpServerInstance);

// The real `tool` function just packages args into an SdkMcpToolDefinition.
// We replicate its minimal behavior here so we don't need the live SDK in tests.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function realTool(name: string, description: string, inputSchema: unknown, handler: any) {
  return { name, description, inputSchema, handler };
}

vi.mock('@anthropic-ai/claude-agent-sdk', async (importOriginal) => {
  // Import original so zod types and other exports are unaffected
  const original = await importOriginal<typeof import('@anthropic-ai/claude-agent-sdk')>();
  return {
    ...original,
    createSdkMcpServer: mockCreateSdkMcpServer,
    tool: realTool,
  };
});

// Import AFTER setting up the mock
const { buildSynapTools, createSynapMcpServer } = await import('../mcp-server.js');

// ─── helpers ─────────────────────────────────────────────────────────────────

function makeSdk(overrides: Partial<SynapSdkLike> = {}): SynapSdkLike {
  return {
    fetch: vi.fn(async () => ({
      formatted_context: 'User likes TypeScript.',
      facts: [],
    })),
    conversation: {
      record_message: vi.fn(async () => ({})),
    },
    memories: {
      create: vi.fn(async () => ({ ingestion_id: 'mem-123' })),
    },
    ...overrides,
  };
}

function makeToolCtx(sdk: SynapSdkLike, overrides: Partial<{
  userId: string;
  customerId: string;
  conversationId: string;
  mode: string;
}> = {}) {
  return {
    sdk,
    userId: 'u-test',
    customerId: 'cust-test',
    conversationId: 'conv-test',
    mode: 'accurate',
    ...overrides,
  };
}

/** Extract handlers by name from buildSynapTools result */
function getHandler(
  tools: ReturnType<typeof buildSynapTools>,
  name: string,
): (args: Record<string, unknown>) => Promise<unknown> {
  const t = tools.find((x) => x.name === name);
  if (!t) throw new Error(`Tool "${name}" not found in buildSynapTools result`);
  return t.handler as (args: Record<string, unknown>) => Promise<unknown>;
}

// ─── buildSynapTools — tool list ─────────────────────────────────────────────

describe('buildSynapTools — tool list', () => {
  it('returns exactly two tools: synap_search and synap_remember', () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    expect(tools).toHaveLength(2);
    expect(tools[0].name).toBe('synap_search');
    expect(tools[1].name).toBe('synap_remember');
  });

  it('each tool has a non-empty description', () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    for (const t of tools) {
      expect(typeof t.description).toBe('string');
      expect(t.description.length).toBeGreaterThan(10);
    }
  });

  it('each tool has a handler function', () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    for (const t of tools) {
      expect(typeof t.handler).toBe('function');
    }
  });
});

// ─── synap_search — happy path ────────────────────────────────────────────────

describe('synap_search — happy path', () => {
  let sdk: SynapSdkLike;
  let handler: Awaited<ReturnType<typeof getHandler>>;

  beforeEach(() => {
    sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk, { mode: 'accurate' }));
    handler = getHandler(tools, 'synap_search');
  });

  it('calls sdk.fetch with correct arguments', async () => {
    await handler({ query: 'what does user like?' });
    expect(sdk.fetch).toHaveBeenCalledOnce();
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.user_id).toBe('u-test');
    expect(args.customer_id).toBe('cust-test');
    expect(args.search_query).toEqual(['what does user like?']);
    expect(args.mode).toBe('accurate');
    expect(args.max_results).toBe(10);
    expect(args.conversation_id).toBe('conv-test');
  });

  it('returns content array with the formatted context text', async () => {
    const result = (await handler({ query: 'hello' })) as {
      content: { type: string; text: string }[];
      isError?: boolean;
    };
    expect(result.content).toBeDefined();
    expect(result.content[0].type).toBe('text');
    expect(result.content[0].text).toContain('User likes TypeScript.');
    expect(result.isError).toBeFalsy();
  });

  it('respects custom max_results', async () => {
    await handler({ query: 'hello', max_results: 5 });
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.max_results).toBe(5);
  });

  it('defaults max_results to 10 when not supplied', async () => {
    await handler({ query: 'hello' });
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.max_results).toBe(10);
  });

  it('falls back to "no relevant context" message when formatted_context is empty', async () => {
    sdk = makeSdk({ fetch: vi.fn(async () => ({ formatted_context: '' })) });
    const tools = buildSynapTools(makeToolCtx(sdk));
    handler = getHandler(tools, 'synap_search');

    const result = (await handler({ query: 'hello' })) as {
      content: { text: string }[];
    };
    expect(result.content[0].text).toContain('no relevant context');
  });

  it('falls back when formatted_context is null', async () => {
    sdk = makeSdk({ fetch: vi.fn(async () => ({ formatted_context: null })) });
    const tools = buildSynapTools(makeToolCtx(sdk));
    handler = getHandler(tools, 'synap_search');

    const result = (await handler({ query: 'hello' })) as {
      content: { text: string }[];
    };
    expect(result.content[0].text).toContain('no relevant context');
  });

  it('uses null conversation_id when conversationId is undefined', async () => {
    const tools = buildSynapTools(makeToolCtx(sdk, { conversationId: undefined }));
    const h = getHandler(tools, 'synap_search');
    await h({ query: 'hello' });
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.conversation_id).toBeNull();
  });

  it('uses null customer_id when customerId is empty string', async () => {
    const tools = buildSynapTools(makeToolCtx(sdk, { customerId: '' }));
    const h = getHandler(tools, 'synap_search');
    await h({ query: 'hello' });
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.customer_id).toBeNull();
  });

  it('passes mode from context to sdk.fetch', async () => {
    const tools = buildSynapTools(makeToolCtx(sdk, { mode: 'fast' }));
    const h = getHandler(tools, 'synap_search');
    await h({ query: 'hello' });
    const args = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.mode).toBe('fast');
  });
});

// ─── synap_search — failure / degradation ────────────────────────────────────

describe('synap_search — failure paths', () => {
  it('missing query → returns isError=true without calling sdk.fetch', async () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_search');

    const result = (await handler({})) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('missing `query`');
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it('sdk.fetch throws → returns graceful error text (no re-throw)', async () => {
    const sdk = makeSdk({
      fetch: vi.fn(async () => {
        throw new Error('network down');
      }),
    });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_search');

    const result = (await handler({ query: 'hello' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBeFalsy(); // graceful, not isError
    expect(result.content[0].text).toContain('no context available');
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  it('sdk.fetch non-2xx-like error → graceful "no context available" response', async () => {
    const transportErr = Object.assign(new Error('HTTP 503'), { status: 503 });
    const sdk = makeSdk({ fetch: vi.fn(async () => { throw transportErr; }) });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_search');

    const result = (await handler({ query: 'hello' })) as {
      content: { text: string }[];
    };
    expect(result.content[0].text).toContain('no context available');
    vi.restoreAllMocks();
  });

  it('error message includes constructor name of thrown error', async () => {
    class CustomFetchError extends Error {
      constructor(msg: string) { super(msg); this.name = 'CustomFetchError'; }
    }
    const sdk = makeSdk({
      fetch: vi.fn(async () => { throw new CustomFetchError('boom'); }),
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_search');

    const result = (await handler({ query: 'hello' })) as {
      content: { text: string }[];
    };
    // The product reads err.constructor.name
    expect(result.content[0].text).toContain('CustomFetchError');
    vi.restoreAllMocks();
  });
});

// ─── synap_remember — happy path ─────────────────────────────────────────────

describe('synap_remember — happy path', () => {
  let sdk: SynapSdkLike;
  let handler: Awaited<ReturnType<typeof getHandler>>;

  beforeEach(() => {
    sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    handler = getHandler(tools, 'synap_remember');
  });

  it('calls sdk.memories.create with correct document and user_id', async () => {
    await handler({ content: 'User prefers dark mode.' });
    expect(sdk.memories.create).toHaveBeenCalledOnce();
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.document).toBe('User prefers dark mode.');
    expect(args.user_id).toBe('u-test');
  });

  it('returns success text with ingestion_id', async () => {
    const result = (await handler({ content: 'remember this' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.content[0].text).toContain('recorded');
    expect(result.content[0].text).toContain('mem-123');
    expect(result.isError).toBeFalsy();
  });

  it('adds source=claude_agent_sdk to metadata when not provided', async () => {
    await handler({ content: 'test fact' });
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.metadata?.source).toBe('claude_agent_sdk');
  });

  it('does NOT override source if caller provides it', async () => {
    await handler({ content: 'test', metadata: { source: 'my-app' } });
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.metadata?.source).toBe('my-app');
  });

  it('passes through extra metadata fields', async () => {
    await handler({ content: 'note', metadata: { tag: 'important', priority: 1 } });
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.metadata?.tag).toBe('important');
    expect(args.metadata?.priority).toBe(1);
  });

  it('sends customer_id from context', async () => {
    await handler({ content: 'remember' });
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.customer_id).toBe('cust-test');
  });

  it('sends null customer_id when customerId is empty', async () => {
    const tools = buildSynapTools(makeToolCtx(sdk, { customerId: '' }));
    const h = getHandler(tools, 'synap_remember');
    await h({ content: 'hello' });
    const args = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(args.customer_id).toBeNull();
  });

  it('empty ingestion_id is still a success (returns empty string for id)', async () => {
    sdk = makeSdk({
      memories: { create: vi.fn(async () => ({ ingestion_id: '' })) },
    });
    const tools = buildSynapTools(makeToolCtx(sdk));
    const h = getHandler(tools, 'synap_remember');
    const result = (await h({ content: 'test' })) as { content: { text: string }[] };
    expect(result.content[0].text).toContain('recorded');
  });
});

// ─── synap_remember — failure / degradation ──────────────────────────────────

describe('synap_remember — failure paths', () => {
  it('empty content → isError=true without calling sdk.memories.create', async () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_remember');

    const result = (await handler({ content: '' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('missing `content`');
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it('whitespace-only content → isError=true', async () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_remember');

    const result = (await handler({ content: '   ' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
  });

  it('missing content entirely → isError=true', async () => {
    const sdk = makeSdk();
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_remember');

    const result = (await handler({})) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
  });

  it('sdk.memories.create throws → returns isError=true with error message', async () => {
    const sdk = makeSdk({
      memories: {
        create: vi.fn(async () => { throw new Error('write failure'); }),
      },
    });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_remember');

    const result = (await handler({ content: 'test' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('ingestion failed');
    expect(result.content[0].text).toContain('write failure');
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  it('sdk.memories.create non-2xx-like error → isError=true', async () => {
    const transportErr = Object.assign(new Error('HTTP 502'), { status: 502 });
    const sdk = makeSdk({
      memories: { create: vi.fn(async () => { throw transportErr; }) },
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const tools = buildSynapTools(makeToolCtx(sdk));
    const handler = getHandler(tools, 'synap_remember');

    const result = (await handler({ content: 'test' })) as {
      content: { text: string }[];
      isError?: boolean;
    };
    expect(result.isError).toBe(true);
    vi.restoreAllMocks();
  });
});

// ─── createSynapMcpServer — construction guards ───────────────────────────────

describe('createSynapMcpServer — construction guards', () => {
  beforeEach(() => {
    mockCreateSdkMcpServer.mockClear();
  });

  it('throws when sdk is null', () => {
    expect(() =>
      createSynapMcpServer({
        sdk: null as unknown as SynapSdkLike,
        userId: 'u1',
      }),
    ).toThrow(/non-null sdk/);
  });

  it('throws when userId is empty', () => {
    const sdk = makeSdk();
    expect(() => createSynapMcpServer({ sdk, userId: '' })).toThrow(/non-empty userId/);
  });

  it('throws when userId is falsy (undefined cast)', () => {
    const sdk = makeSdk();
    expect(() =>
      createSynapMcpServer({ sdk, userId: undefined as unknown as string }),
    ).toThrow(/non-empty userId/);
  });
});

// ─── createSynapMcpServer — happy path ───────────────────────────────────────

describe('createSynapMcpServer — happy path', () => {
  beforeEach(() => {
    mockCreateSdkMcpServer.mockClear();
  });

  it('calls createSdkMcpServer and returns its result', () => {
    const sdk = makeSdk();
    const result = createSynapMcpServer({ sdk, userId: 'u1' });
    expect(mockCreateSdkMcpServer).toHaveBeenCalledOnce();
    expect(result).toBe(mockMcpServerInstance);
  });

  it('passes default name "synap" to createSdkMcpServer', () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as { name: string };
    expect(opts.name).toBe('synap');
  });

  it('passes custom name to createSdkMcpServer', () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1', name: 'my-synap' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as { name: string };
    expect(opts.name).toBe('my-synap');
  });

  it('passes default version "0.1.0" to createSdkMcpServer', () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as { version: string };
    expect(opts.version).toBe('0.1.0');
  });

  it('passes custom version to createSdkMcpServer', () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1', version: '2.3.4' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as { version: string };
    expect(opts.version).toBe('2.3.4');
  });

  it('passes a tools array with two entries', () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as { tools: unknown[] };
    expect(Array.isArray(opts.tools)).toBe(true);
    expect(opts.tools).toHaveLength(2);
  });

  it('default mode is "accurate" (propagated to tool handlers)', async () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1' });
    // Extract tool handlers from createSdkMcpServer call args
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as {
      tools: Array<{ name: string; handler: (args: Record<string, unknown>) => Promise<unknown> }>;
    };
    const searchTool = opts.tools.find((t) => t.name === 'synap_search')!;
    await searchTool.handler({ query: 'test' });
    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.mode).toBe('accurate');
  });

  it('custom mode is forwarded to tools', async () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1', mode: 'fast' });
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as {
      tools: Array<{ name: string; handler: (args: Record<string, unknown>) => Promise<unknown> }>;
    };
    const searchTool = opts.tools.find((t) => t.name === 'synap_search')!;
    await searchTool.handler({ query: 'test' });
    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.mode).toBe('fast');
  });

  it('customerId defaults to empty string → customer_id becomes null in fetch', async () => {
    const sdk = makeSdk();
    createSynapMcpServer({ sdk, userId: 'u1' }); // no customerId
    const opts = mockCreateSdkMcpServer.mock.calls[0][0] as {
      tools: Array<{ name: string; handler: (args: Record<string, unknown>) => Promise<unknown> }>;
    };
    const searchTool = opts.tools.find((t) => t.name === 'synap_search')!;
    await searchTool.handler({ query: 'hello' });
    const fetchArgs = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(fetchArgs.customer_id).toBeNull();
  });
});
