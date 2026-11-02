// A path fixed at construction must agree with the ids fixed beside it. The
// server lets the path's `user` and `customer` win over the flat ids, so a
// disagreement would address another identity while the caller believes it
// addressed its own. Refused once, at construction, before any SDK call.

import { describe, expect, it, vi } from 'vitest';

vi.mock('@anthropic-ai/claude-agent-sdk', () => ({
  tool: (name: string, description: string, inputSchema: object, handler: unknown) => ({
    name, description, inputSchema, handler,
  }),
  createSdkMcpServer: (opts: unknown) => opts,
}));

const { SynapScopeError } = await import('../index.js');
const { createSynapHooks } = await import('../hooks.js');
const { buildSynapTools, createSynapMcpServer } = await import('../mcp-server.js');

import type { SynapSdkLike } from '../types.js';

const PATH = { customer: 'acme', user: 'alice' };

function makeSdk(withMarker = true): SynapSdkLike {
  return {
    ...(withMarker ? { get_scope_ladder: vi.fn() } : {}),
    fetch: vi.fn(async () => ({ formatted_context: 'c' })),
    conversation: { record_message: vi.fn(async () => ({})) },
    memories: { create: vi.fn(async () => ({ ingestion_id: 'i' })) },
  };
}

const tools = (userId: string, customerId: string, sdk = makeSdk(), scopePath: Record<string, string> = PATH) =>
  buildSynapTools({ sdk, userId, customerId, mode: 'accurate', scopePath });

describe('ids vs path, at construction', () => {
  it('hooks, tools and the MCP server refuse another userId, and another customerId', () => {
    expect(() => createSynapHooks({ sdk: makeSdk(), userId: 'bob', customerId: 'acme', scopePath: PATH }))
      .toThrow(SynapScopeError);
    expect(() => createSynapHooks({ sdk: makeSdk(), userId: 'alice', customerId: 'globex', scopePath: PATH }))
      .toThrow(SynapScopeError);
    expect(() => tools('bob', 'acme')).toThrow(SynapScopeError);
    expect(() => tools('alice', 'globex')).toThrow(SynapScopeError);
    expect(() => createSynapMcpServer({ sdk: makeSdk(), userId: 'bob', customerId: 'acme', scopePath: PATH }))
      .toThrow(SynapScopeError);
  });

  it('the refusal names no raw id and no SDK call was made', () => {
    const sdk = makeSdk();
    let err: Error | undefined;
    try {
      createSynapHooks({ sdk, userId: 'bob-secret', customerId: 'acme', scopePath: PATH });
    } catch (e) {
      err = e as Error;
    }
    expect(err).toBeInstanceOf(SynapScopeError);
    for (const raw of ['bob-secret', 'alice', 'acme']) expect(err!.message).not.toContain(raw);
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it('matching ids pass; an empty customerId passes (the path supplies it)', () => {
    expect(() => createSynapHooks({ sdk: makeSdk(), userId: 'alice', customerId: 'acme', scopePath: PATH })).not.toThrow();
    expect(() => createSynapHooks({ sdk: makeSdk(), userId: 'alice', scopePath: PATH })).not.toThrow();
    expect(() => tools('alice', '')).not.toThrow();
  });

  it('B2C: customer == user passes with no customerId or an equal one', () => {
    const b2c = { customer: 'alice', user: 'alice' };
    expect(() => tools('alice', '', makeSdk(), b2c)).not.toThrow();
    expect(() => tools('alice', 'alice', makeSdk(), b2c)).not.toThrow();
    expect(() => tools('alice', 'rival', makeSdk(), b2c)).toThrow(SynapScopeError);
  });

  it('a path with no user / customer key accepts any ids', () => {
    expect(() => tools('bob', 'globex', makeSdk(), { team: 'eng' })).not.toThrow();
  });

  it('without a path nothing is checked', () => {
    expect(() => createSynapHooks({ sdk: makeSdk(), userId: 'bob', customerId: 'globex' })).not.toThrow();
    expect(() => tools('bob', 'globex', makeSdk(), {})).not.toThrow();
  });
});

describe('an SDK without get_scope_ladder is refused when a path is set', () => {
  it('hooks, tools and the MCP server throw at construction, naming 0.7.1', () => {
    expect(() => createSynapHooks({ sdk: makeSdk(false), userId: 'alice', scopePath: PATH })).toThrow(/>= 0\.7\.1/);
    expect(() => tools('alice', '', makeSdk(false))).toThrow(/>= 0\.7\.1/);
    expect(() => createSynapMcpServer({ sdk: makeSdk(false), userId: 'alice', scopePath: PATH })).toThrow(/>= 0\.7\.1/);
  });

  it('without a path (or an empty one) an old SDK still works', () => {
    expect(() => createSynapHooks({ sdk: makeSdk(false), userId: 'alice' })).not.toThrow();
    expect(() => tools('alice', '', makeSdk(false), {})).not.toThrow();
  });
});

describe('buildSynapTools snapshots its input', () => {
  it('mutating the ctx or its path after build changes nothing that is sent', async () => {
    const sdk = makeSdk();
    const ctx = { sdk, userId: 'alice', customerId: 'acme', mode: 'accurate', scopePath: { ...PATH } };
    const built = buildSynapTools(ctx);
    ctx.userId = 'bob';
    ctx.customerId = 'globex';
    ctx.scopePath.user = 'bob';
    const search = built.find((t) => t.name === 'synap_search')!.handler as (a: object) => Promise<unknown>;
    const remember = built.find((t) => t.name === 'synap_remember')!.handler as (a: object) => Promise<unknown>;
    await search({ query: 'q' });
    await remember({ content: 'n' });
    const f = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    const c = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(f).toMatchObject({ user_id: 'alice', customer_id: 'acme', scope_path: PATH });
    expect(c).toMatchObject({ user_id: 'alice', customer_id: 'acme', scope: PATH });
  });
});
