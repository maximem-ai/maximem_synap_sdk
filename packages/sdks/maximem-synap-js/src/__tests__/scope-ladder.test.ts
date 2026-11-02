/**
 * The SDK learns the account's scope ladder and addresses a level with it.
 * Mirror of synap/tests/sdk/test_scope_ladder.py; the rules are section 4.2 of
 * the scope-ladder design.
 */
import { describe, it, expect } from 'vitest';
import { SynapClient } from '../client.js';
import { InvalidInputError, ScopeLadderError } from '../errors.js';
import {
  LADDER_INVALIDATION_MIN_INTERVAL_MS, LADDER_RETRY_BACKOFF_MS, LADDER_TTL_MS, ScopeAddress, ScopeLadder, ladderFromWhoami,
  type ScopeLevel,
} from '../scope-ladder.js';
import { buildTool } from '../tool/as-tool.js';
import { carriesScopePath } from '../transport/http.js';
import { filePart, textPart } from '../memories/parts.js';

const L = (key: string, position: number, role: string | null): ScopeLevel => ({
  key, label: key, position, role, required: true,
});
const CUSTOMER = L('customer', 1, 'tenant_of_record');
const TEAM = L('team', 2, null);
const USER = L('user', 3, 'identity');
const USER2 = L('user', 2, 'identity');

const deep = (o: { ratified?: boolean; nested?: boolean; b2c?: boolean } = {}) =>
  new ScopeLadder({
    levels: [CUSTOMER, TEAM, USER], is_default: false,
    ladder_ratified: o.ratified ?? true, nested_scoping_enabled: o.nested ?? true,
    source: 'server', b2c: o.b2c ?? false,
  });
const dflt = (b2c = false) =>
  new ScopeLadder({
    levels: [CUSTOMER, USER2], is_default: true, ladder_ratified: false,
    nested_scoping_enabled: false, source: 'server', b2c,
  });

const DEEP_BLOCK = {
  levels: [
    { key: 'customer', label: 'Customer', position: 1, role: 'tenant_of_record', required: true },
    { key: 'team', label: 'Team', position: 2, role: null, required: true },
    { key: 'user', label: 'User', position: 3, role: 'identity', required: true },
  ],
  is_default: false, ladder_ratified: true, nested_scoping_enabled: true,
};
const DEFAULT_BLOCK = {
  levels: [
    { key: 'customer', label: 'Customer', position: 1, role: 'tenant_of_record', required: true },
    { key: 'user', label: 'User', position: 2, role: 'identity', required: true },
  ],
  is_default: true, ladder_ratified: false, nested_scoping_enabled: false,
};

describe('address(): default ladder', () => {
  it('ids only and never a path', () => {
    const a = dflt().address(['acme', 'dana']);
    expect([a.customer_id, a.user_id, a.scope_path]).toEqual(['acme', 'dana', null]);
    expect(a.read_kwargs()).toEqual({ user_id: 'dana', customer_id: 'acme' });
    expect('scope' in a.write_kwargs()).toBe(false);
  });
  it('a shorter list addresses a higher level', () => {
    const a = dflt().address(['acme']);
    expect([a.customer_id, a.user_id, a.scope_path]).toEqual(['acme', null, null]);
    expect(a.read_kwargs()).toEqual({ customer_id: 'acme' });
  });
  it('B2C is user_id alone', () => {
    const ladder = new ScopeLadder({
      levels: [L('user', 1, 'identity')], is_default: true, ladder_ratified: false,
      nested_scoping_enabled: false, source: 'inferred', b2c: true,
    });
    const a = ladder.address(['dana']);
    expect([a.user_id, a.customer_id, a.scope_path]).toEqual(['dana', null, null]);
  });
  it('B2C server block with a default ladder takes one value', () => {
    const a = dflt(true).address(['dana']);
    expect([a.user_id, a.customer_id, a.scope_path]).toEqual(['dana', null, null]);
    expect(() => dflt(true).address(['acme', 'dana'])).toThrow(ScopeLadderError);
  });
});

describe('address(): deeper ladder', () => {
  it('ids by role plus the full prefix path', () => {
    const a = deep().address(['acme', 'payments', 'dana']);
    expect(a.customer_id).toBe('acme');
    expect(a.user_id).toBe('dana');
    expect(a.scope_path).toEqual({ customer: 'acme', team: 'payments', user: 'dana' });
  });
  it('a prefix addresses that level', () => {
    expect(deep().address(['acme', 'payments']).scope_path).toEqual({ customer: 'acme', team: 'payments' });
    expect(deep().address(['acme', 'payments']).user_id).toBeNull();
    expect(deep().address(['acme']).scope_path).toEqual({ customer: 'acme' });
  });
  it.each([[false, true], [true, false], [false, false]])(
    'raises before any call when the server would refuse (ratified=%s nested=%s)', (nested, ratified) => {
      let message = '';
      try { deep({ ratified, nested }).address(['acme', 'payments', 'dana']); } catch (e) {
        expect(e).toBeInstanceOf(ScopeLadderError);
        message = (e as Error).message;
      }
      expect(message).toContain('may not be accepted');
      expect(message).toContain('nested scoping may be off');
      expect(message).toContain('get_scope_ladder({ refresh: true })');
    });
  it('more values than levels raises', () => {
    expect(() => deep().address(['a', 'b', 'c', 'd'])).toThrow(/4 values for 3 levels/);
    expect(() => dflt().address(['a', 'b', 'c'])).toThrow(ScopeLadderError);
  });
  it.each([[[]], ['acme'], [['acme', '']], [['acme', null]], [[1]]])('empty or malformed values raise: %j', (bad) => {
    expect(() => deep().address(bad as never)).toThrow(ScopeLadderError);
  });
  it('the error is an InvalidInputError', () => {
    expect(new ScopeLadderError('x')).toBeInstanceOf(InvalidInputError);
  });
  it('keys', () => { expect(deep().keys).toEqual(['customer', 'team', 'user']); });
});

describe('address(): B2C deeper ladder (tenant level takes the identity value)', () => {
  it('pins the rule', () => {
    const a = deep({ b2c: true }).address(['payments', 'dana']);
    expect(a.scope_path).toEqual({ customer: 'dana', team: 'payments', user: 'dana' });
    expect([a.user_id, a.customer_id]).toEqual(['dana', null]);
    expect('customer_id' in a.read_kwargs()).toBe(false);
    expect('customer_id' in a.write_kwargs()).toBe(false);
  });
  it('a list that stops before the identity raises', () => {
    expect(() => deep({ b2c: true }).address(['payments'])).toThrow(/identity/);
  });
  it('too many values raise', () => {
    expect(() => deep({ b2c: true }).address(['acme', 'payments', 'dana'])).toThrow(ScopeLadderError);
  });
  it('still needs ratification', () => {
    expect(() => deep({ b2c: true, ratified: false }).address(['payments', 'dana'])).toThrow(ScopeLadderError);
  });
});

describe('kwargs', () => {
  it('read uses scope_path and write uses scope', () => {
    const a = new ScopeAddress({ user_id: 'dana', customer_id: 'acme', scope_path: { customer: 'acme', user: 'dana' } });
    expect(a.read_kwargs()).toEqual({ user_id: 'dana', customer_id: 'acme', scope_path: { customer: 'acme', user: 'dana' } });
    expect(a.write_kwargs()).toEqual({ user_id: 'dana', customer_id: 'acme', scope: { customer: 'acme', user: 'dana' } });
  });
  it('empty entries are omitted and a null path is never emitted', () => {
    const a = new ScopeAddress({ customer_id: 'acme' });
    expect(a.read_kwargs()).toEqual({ customer_id: 'acme' });
    expect(Object.keys(a.write_kwargs())).toEqual(['customer_id']);
    expect(new ScopeAddress().read_kwargs()).toEqual({});
    expect(new ScopeAddress({ scope_path: {} }).read_kwargs()).toEqual({});
  });
  it('kwargs are copies', () => {
    const a = new ScopeAddress({ scope_path: { customer: 'acme' } });
    a.read_kwargs().scope_path!['x'] = 'y';
    expect(a.scope_path).toEqual({ customer: 'acme' });
  });
});

describe('ladderFromWhoami: key missing, null, malformed, valid', () => {
  it('valid block', () => {
    const l = ladderFromWhoami({ user_context_isolation: 'strict', scope_ladder: DEEP_BLOCK })!;
    expect(l.source).toBe('server');
    expect(l.keys).toEqual(['customer', 'team', 'user']);
    expect(l.ladder_ratified && l.nested_scoping_enabled && !l.is_default).toBe(true);
    expect(l.levels[1]!.role).toBeNull();
  });
  it('orders by position', () => {
    const l = ladderFromWhoami({ scope_ladder: { ...DEEP_BLOCK, levels: [...DEEP_BLOCK.levels].reverse() } })!;
    expect(l.keys).toEqual(['customer', 'team', 'user']);
  });
  it.each([[{ user_context_isolation: 'strict' }], [{ user_context_isolation: 'strict', scope_ladder: null }]])(
    'missing key and explicit null infer the default: %j', (w) => {
      const l = ladderFromWhoami(w)!;
      expect([l.source, l.is_default, l.keys]).toEqual(['inferred', true, ['customer', 'user']]);
    });
  it('B2C is inferred as user only', () => {
    const l = ladderFromWhoami({ user_context_isolation: 'equals_customer', scope_ladder: null })!;
    expect([l.source, l.keys, l.b2c]).toEqual(['inferred', ['user'], true]);
  });
  it.each([[{}], [{ scope_ladder: null }], [{ user_context_isolation: null }]])('nothing known is null: %j', (w) => {
    expect(ladderFromWhoami(w)).toBeNull();
  });
  it.each([
    ['nope'], [5], [[]], [{ levels: 'x' }], [{ levels: [] }], [{ levels: [{ label: 'x', position: 1 }] }],
    [{ levels: ['x'] }], [{ levels: [{ key: 'a', position: 'z' }] }],
  ])('malformed block is absent with a warning, never an exception: %j', (block) => {
    const warnings: string[] = [];
    const l = ladderFromWhoami({ user_context_isolation: 'strict', scope_ladder: block }, undefined, (m) => warnings.push(m))!;
    expect(l.source).toBe('inferred');
    expect(warnings.join()).toContain('malformed scope_ladder');
  });
  it('malformed block and unknown isolation is null', () => {
    expect(ladderFromWhoami({ scope_ladder: 'nope' })).toBeNull();
  });
});

// ---------------------------------------------------------------------------

type Reply = { status: number; body: unknown };
function rig(first: Reply['body'] | Error = { user_context_isolation: 'strict', scope_ladder: DEEP_BLOCK }) {
  const clock = { t: 1000 };
  const state: {
    whoami: unknown | Error; whoamiCalls: number; sent: { path: string; body: Record<string, unknown> }[];
    refuse: string | null; refuseStatus: number; gate: Promise<void> | null; sentPaths: string[];
  } = { whoami: first, whoamiCalls: 0, sent: [], refuse: null, refuseStatus: 400, gate: null, sentPaths: [] };
  const warnings: string[] = [];
  const fetchImpl = (async (url: string, init?: { body?: string }) => {
    const path = new URL(url).pathname;
    if (path.endsWith('/whoami')) {
      state.whoamiCalls += 1;
      // What this request reads is decided when it is sent. `gate` holds the
      // answer back until the test resolves it, so a test can land another
      // request while this one is still in flight.
      const reply = state.whoami;
      const gate = state.gate;
      if (gate !== null) await gate;
      if (reply instanceof Error) throw reply;
      return new Response(JSON.stringify(reply), { status: 200 });
    }
    state.sentPaths.push(path);
    // A multipart body (FormData) is not JSON; record it as empty.
    const raw = init?.body;
    state.sent.push({ path, body: typeof raw === 'string' ? (JSON.parse(raw) as Record<string, unknown>) : {} });
    if (state.refuse !== null) return new Response(JSON.stringify({ detail: state.refuse }), { status: state.refuseStatus });
    return new Response(JSON.stringify({ context: {} }), { status: 200 });
  }) as unknown as typeof fetch;
  const c = new SynapClient({
    _force_new: true, apiKey: 'synap_k', clientId: 'test-client', instanceId: 'inst_0123456789abcdef',
    fetchImpl, retryPolicy: null, _clock: () => clock.t,
    logger: (_level: string, m: string) => { warnings.push(m); },
  } as never);
  return { c, clock, state, warnings };
}

describe('get_scope_ladder', () => {
  it('loads through whoami even when the ids are already known', async () => {
    const { c, state } = rig();
    const l = await c.get_scope_ladder();
    expect(l!.keys).toEqual(['customer', 'team', 'user']);
    expect(state.whoamiCalls).toBe(1);
  });
  it('a fresh cache hit makes no whoami call', async () => {
    const { c, clock, state } = rig();
    const first = await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS - 1;
    expect(await c.get_scope_ladder()).toBe(first);
    expect(state.whoamiCalls).toBe(1);
  });
  it('expiry re-reads', async () => {
    const { c, clock, state } = rig();
    await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1;
    state.whoami = { user_context_isolation: 'strict', scope_ladder: DEFAULT_BLOCK };
    expect((await c.get_scope_ladder())!.is_default).toBe(true);
    expect(state.whoamiCalls).toBe(2);
  });
  it('a failed re-read keeps the old ladder, warns, and backs off', async () => {
    const { c, clock, state, warnings } = rig();
    const first = await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1;
    state.whoami = new Error('down');
    expect(await c.get_scope_ladder()).toBe(first);
    expect(warnings.join()).toContain('keeping the last known');
    expect(state.whoamiCalls).toBe(2);
    clock.t += LADDER_RETRY_BACKOFF_MS - 1;
    expect(await c.get_scope_ladder()).toBe(first);
    expect(state.whoamiCalls).toBe(2);
    clock.t += 2;
    state.whoami = { user_context_isolation: 'strict', scope_ladder: DEFAULT_BLOCK };
    expect((await c.get_scope_ladder())!.is_default).toBe(true);
    expect(state.whoamiCalls).toBe(3);
  });
  it('refresh forces a read', async () => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    await c.get_scope_ladder({ refresh: true });
    expect(state.whoamiCalls).toBe(2);
  });
  it('a refresh failure is thrown, not hidden', async () => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    state.whoami = new Error('down');
    await expect(c.get_scope_ladder({ refresh: true })).rejects.toThrow();
  });
  it('a first-read failure is thrown', async () => {
    const { c } = rig(new Error('down'));
    await expect(c.get_scope_ladder()).rejects.toThrow();
  });
  it('no block infers the default ladder; nothing known is null', async () => {
    const a = rig({ user_context_isolation: 'equals_customer', scope_ladder: null });
    const l = await a.c.get_scope_ladder();
    expect([l!.source, l!.keys]).toEqual(['inferred', ['user']]);
    const b = rig({ client_id: 'c', scope_ladder: null });
    expect(await b.c.get_scope_ladder()).toBeNull();
  });
  it('invalidation keeps the old ladder if the re-read fails', async () => {
    const { c, state } = rig();
    const first = await c.get_scope_ladder();
    state.refuse = 'x';
    await expect(c.user.context.fetch({ user_id: 'dana', search_query: ['q'], scope_path: { customer: 'acme' } } as never)).rejects.toBeInstanceOf(InvalidInputError);
    state.whoami = new Error('down');
    expect(await c.get_scope_ladder()).toBe(first);
  });
});

// Fix 1: an unknown ladder (null) is cached like any other answer.
describe('an unknown ladder is cached', () => {
  const UNKNOWN = { client_id: 'c', scope_ladder: null };
  it('five sequential calls make one whoami request', async () => {
    const { c, state } = rig(UNKNOWN);
    for (let i = 0; i < 5; i += 1) expect(await c.get_scope_ladder()).toBeNull();
    expect(state.whoamiCalls).toBe(1);
  });
  it('after the TTL it is read again', async () => {
    const { c, clock, state } = rig(UNKNOWN);
    await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(2);
  });
  it('a failed re-read backs off for 60 s', async () => {
    const { c, clock, state } = rig(UNKNOWN);
    await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1;
    state.whoami = new Error('down');
    expect(await c.get_scope_ladder()).toBeNull();
    expect(await c.get_scope_ladder()).toBeNull();
    expect(state.whoamiCalls).toBe(2);
    clock.t += LADDER_RETRY_BACKOFF_MS + 1;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(3);
  });
});

// Fix 2: concurrent callers share one read.
describe('single flight', () => {
  it('five concurrent cold calls make one whoami request', async () => {
    const { c, state } = rig();
    const results = await Promise.all([1, 2, 3, 4, 5].map(() => c.get_scope_ladder()));
    expect(state.whoamiCalls).toBe(1);
    for (const r of results) expect(r).toBe(results[0]);
    expect(results[0]!.keys).toEqual(['customer', 'team', 'user']);
  });
  it('a shared failure reaches every waiter', async () => {
    const { c, state } = rig(new Error('down'));
    const results = await Promise.allSettled([1, 2, 3, 4, 5].map(() => c.get_scope_ladder()));
    expect(state.whoamiCalls).toBe(1);
    expect(results.every((r) => r.status === 'rejected')).toBe(true);
  });
  it('a shared failed re-read keeps the old ladder for all', async () => {
    const { c, clock, state } = rig();
    const first = await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1;
    state.whoami = new Error('down');
    const results = await Promise.all([1, 2, 3, 4, 5].map(() => c.get_scope_ladder()));
    expect(state.whoamiCalls).toBe(2);
    for (const r of results) expect(r).toBe(first);
  });
});

// Fix 3: invalidation storm and the in-flight race.
describe('invalidation storm and race', () => {
  const refuse = async (c: SynapClient, state: ReturnType<typeof rig>['state']) => {
    state.refuse = 'whatever the server says';
    await c.user.context.fetch({ user_id: 'dana', search_query: ['q'], scope_path: { customer: 'acme' } } as never)
      .catch(() => undefined);
    state.refuse = null;
  };
  it('five refusals, each followed by a read, make one extra request', async () => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    for (let i = 0; i < 5; i += 1) {
      await refuse(c, state);
      await c.get_scope_ladder();
    }
    expect(state.whoamiCalls).toBe(2);
  });
  it('after the minimum interval it reads again', async () => {
    const { c, clock, state } = rig();
    await c.get_scope_ladder();
    await refuse(c, state);
    await c.get_scope_ladder();
    await refuse(c, state);
    clock.t += LADDER_INVALIDATION_MIN_INTERVAL_MS + 100;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(3);
  });
  it('a read in flight during an invalidation is returned but not stored', async () => {
    const { c, state } = rig({ user_context_isolation: 'strict', scope_ladder: DEFAULT_BLOCK });
    let release!: () => void;
    state.gate = new Promise<void>((resolve) => { release = resolve; });
    const inflight = c.get_scope_ladder(); // whoami is sent and held at the gate
    await Promise.resolve();
    expect(state.whoamiCalls).toBe(1);
    await refuse(c, state); // the 400 lands while that read is still in flight
    state.gate = null;
    state.whoami = { user_context_isolation: 'strict', scope_ladder: DEEP_BLOCK };
    release();
    const old = await inflight;
    expect(old!.is_default).toBe(true); // its own caller gets what it read
    const fresh = await c.get_scope_ladder(); // but it was not stored: this re-reads
    expect(state.whoamiCalls).toBe(2);
    expect(fresh!.keys).toEqual(['customer', 'team', 'user']);
  });
  it('the invalidated flag clears after a stored read', async () => {
    const { c, clock, state } = rig();
    await c.get_scope_ladder();
    await refuse(c, state);
    await c.get_scope_ladder(); // the re-read, stored
    expect(state.whoamiCalls).toBe(2);
    // Past the 5 s minimum and well inside the 300 s TTL. A flag that stayed
    // set would read again here.
    clock.t += LADDER_INVALIDATION_MIN_INTERVAL_MS + 1000;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(2);
  });
});

// A refresh caller is never answered with the stale ladder.
describe('refresh never inherits stale-on-failure', () => {
  it('refresh does not join a keep-on-failure read; the other waiters keep the stale ladder', async () => {
    const { c, clock, state } = rig();
    const first = await c.get_scope_ladder();
    clock.t += LADDER_TTL_MS + 1; // expired: the next read keeps the old ladder on failure
    state.whoami = new Error('down');
    let release!: () => void;
    state.gate = new Promise<void>((resolve) => { release = resolve; });
    const waiters = [1, 2, 3, 4].map(() => c.get_scope_ladder());
    await Promise.resolve();
    const joiner = c.get_scope_ladder({ refresh: true });
    await Promise.resolve();
    release();
    const results = await Promise.allSettled([...waiters, joiner]);
    expect(results[4]!.status).toBe('rejected'); // the refresh caller was not handed the stale ladder
    for (const r of results.slice(0, 4)) {
      expect(r.status).toBe('fulfilled');
      expect((r as PromiseFulfilledResult<ScopeLadder | null>).value).toBe(first);
    }
  });
  it('refresh callers share a refresh read', async () => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    let release!: () => void;
    state.gate = new Promise<void>((resolve) => { release = resolve; });
    const calls = [1, 2, 3].map(() => c.get_scope_ladder({ refresh: true }));
    await Promise.resolve();
    release();
    await Promise.all(calls);
    expect(state.whoamiCalls).toBe(2);
  });
});

// Fix 4: any 400 on a call that carried a path invalidates; the wording is not read.
// Verbatim from synap/cloud/shared/scope.py and scope_ladder.py.
const SERVER_REFUSALS = [
  'scope path is empty',
  'nested scoping is not enabled for this instance, so a scope path cannot be honoured. ' +
    'Sending one and having it ignored would widen the scope rather than narrow it, so it is refused instead.',
  "this client's scope ladder has not been accepted yet, so a scope path cannot be honoured. " +
    'Accept the ladder in the dashboard first.',
  'scope path resolved to an empty chain',
  'client 3f2b0c1e-0000-0000-0000-000000000000 has no scope ladder',
  "'squad' is not a level on this ladder; levels are ['customer', 'team', 'user']",
  "'client' is the root level and is implied by the client, it is not part of the path",
  'path skips levels: supply team',
  'path skips team, which the ladder marks optional. Skipping a level is not supported yet: there is no ' +
    'representation for a node whose parent is two rungs above it.',
  'path levels are not contiguous from the root',
  "no 'team' node 'payments' under customer 'acme' (closed_tree is on for this client)",
  "no 'team' node 'payments' for this client",
  'client 3f2b0c1e-0000-0000-0000-000000000000 has no root node',
  "'payments' exists at level 'team' under this parent but is deactivated. Reactivate it deliberately; " +
    'resolving through it would make reads depend on which worker is warm.',
  "'payments' is already in use at level 'team' for this client, and that level requires ids to be " +
    'unique across the whole client',
  "'payments' already exists under this parent at level 'region', cannot also be a 'team'",
];

describe('a 400 on a call that carried a path invalidates the ladder', () => {
  it.each(SERVER_REFUSALS)('any server refusal wording: %s', async (detail) => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    state.refuse = detail;
    await expect(c.user.context.fetch({ user_id: 'dana', search_query: ['q'], scope_path: PATH } as never))
      .rejects.toBeInstanceOf(InvalidInputError);
    state.refuse = null;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(2);
  });
  it.each([
    ['no scope_path', {}],
    ['null scope_path', { scope_path: null }],
    ['empty scope_path', { scope_path: {} }],
  ])('a 400 on a call with %s does not, even if the words look like a ladder error', async (_n, extra) => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    state.refuse = SERVER_REFUSALS[5]!;
    await expect(c.user.context.fetch({ user_id: 'dana', search_query: ['q'], ...extra } as never))
      .rejects.toBeInstanceOf(InvalidInputError);
    state.refuse = null;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(1);
  });
  it('a write that carried `scope` counts', async () => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    state.refuse = 'x';
    await expect(c.memories.create({ document: 'hello', user_id: 'dana', customer_id: 'acme', scope: PATH } as never))
      .rejects.toBeInstanceOf(InvalidInputError);
    state.refuse = null;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(2);
  });
  it.each([401, 404, 405, 409, 413, 415, 422])('a %i answer does not, only a 400 does', async (status) => {
    const { c, state } = rig();
    await c.get_scope_ladder();
    state.refuse = 'whatever';
    state.refuseStatus = status;
    await expect(c.user.context.fetch({ user_id: 'dana', search_query: ['q'], scope_path: PATH } as never)).rejects.toBeInstanceOf(Error);
    state.refuse = null;
    await c.get_scope_ladder();
    expect(state.whoamiCalls).toBe(1);
  });
  describe('memories.create with file parts (the path sits inside the manifest)', () => {
    const create = (c: SynapClient, scope?: Record<string, string> | null) =>
      c.memories.create({
        document: [textPart('notes'), filePart({ data: new Uint8Array([1, 2, 3]), filename: 'a.txt', media_type: 'text/plain' })],
        user_id: 'dana', customer_id: 'acme', ...(scope === undefined ? {} : { scope }),
      } as never);
    it('a refusal invalidates the ladder', async () => {
      const { c, state } = rig();
      await c.get_scope_ladder();
      state.refuse = 'whatever';
      await expect(create(c, PATH)).rejects.toBeInstanceOf(InvalidInputError);
      expect(state.sentPaths).toContain('/api/v1/memories/create-multipart');
      state.refuse = null;
      await c.get_scope_ladder();
      expect(state.whoamiCalls).toBe(2);
    });
    it.each([['no scope', undefined], ['scope: {}', {}], ['scope: null', null]])('a manifest with %s does not', async (_n, scope) => {
      const { c, state } = rig();
      await c.get_scope_ladder();
      state.refuse = 'whatever';
      await expect(create(c, scope as never)).rejects.toBeInstanceOf(InvalidInputError);
      state.refuse = null;
      await c.get_scope_ladder();
      expect(state.whoamiCalls).toBe(1);
    });
    it('an unreadable manifest is no path and never throws', () => {
      const form = new FormData();
      form.set('manifest', 'not json {');
      expect(carriesScopePath(form)).toBe(false);
      const arr = new FormData();
      arr.set('manifest', '[1,2]');
      expect(carriesScopePath(arr)).toBe(false);
      const ok = new FormData();
      ok.set('manifest', JSON.stringify({ document: [], scope: { customer: 'acme' } }));
      expect(carriesScopePath(ok)).toBe(true);
    });
  });
  it('carriesScopePath: bodies, batches, multipart, and data.delete', () => {
    expect(carriesScopePath({ user_id: 'd', scope: PATH })).toBe(true);
    expect(carriesScopePath({ user_id: 'd' })).toBe(false);
    expect(carriesScopePath({ scope: null })).toBe(false);
    expect(carriesScopePath({ scope: {} })).toBe(false);
    expect(carriesScopePath({ scope: 'user' })).toBe(false); // data.delete: a plain string
    expect(carriesScopePath({ messages: [{ role: 'user' }, { role: 'user', scope: { customer: 'acme' } }] })).toBe(true);
    expect(carriesScopePath({ messages: [{ role: 'user' }] })).toBe(false);
    const form = new FormData();
    form.set('scope', JSON.stringify({ customer: 'acme' }));
    expect(carriesScopePath(form)).toBe(true);
    expect(carriesScopePath(new FormData())).toBe(false);
    expect(carriesScopePath(undefined)).toBe(false);
  });
});

const PATH = { customer: 'acme', team: 'payments', user: 'dana' };
const CONV = '11111111-2222-3333-4444-555555555555';
const ALL = ['conversation', 'user', 'customer', 'client'];

describe('unified fetch carries the path', () => {
  const call = (c: SynapClient, extra: Record<string, unknown> = {}) =>
    c.fetch({
      conversation_id: CONV, user_id: 'dana', customer_id: 'acme', search_query: ['q'],
      scopes: ALL, include_conversation_context: false, ...extra,
    } as never);

  it('every sub-fetch that runs gets it', async () => {
    const { c, state } = rig();
    await call(c, { scope_path: PATH });
    expect(state.sent.map((s) => s.path).sort()).toEqual([
      '/v1/context/client/fetch', '/v1/context/conversation/fetch',
      '/v1/context/customer/fetch', '/v1/context/user/fetch',
    ]);
    for (const s of state.sent) expect(s.body['scope'], s.path).toEqual(PATH);
  });
  it('only the scopes that run get it', async () => {
    const { c, state } = rig();
    await c.fetch({ user_id: 'dana', customer_id: 'acme', search_query: ['q'], scopes: ['customer'], scope_path: PATH } as never);
    expect(state.sent.map((s) => s.path)).toEqual(['/v1/context/customer/fetch']);
  });
  it('without it no request carries a path (bodies unchanged)', async () => {
    const { c, state } = rig();
    await call(c);
    expect(state.sent).toHaveLength(4);
    for (const s of state.sent) {
      expect('scope' in s.body || 'scope_path' in s.body, s.path).toBe(false);
    }
    const bare = rig();
    await call(bare.c, { scope_path: null });
    const empty = rig();
    await call(empty.c, { scope_path: {} });
    const key = (r: typeof state) => JSON.stringify(r.sent.map((s) => [s.path, s.body]).sort());
    expect(key(bare.state)).toBe(key(state));
    expect(key(empty.state)).toBe(key(state));
  });
});

describe('unified fetch with an empty path', () => {
  // Python: `if scope_path:`. null and {} are "no path": the key is not sent.
  it.each([['null', null], ['an empty object', {}], ['undefined', undefined]])(
    'scope_path: %s sends no `scope` and no `scope_path` to any scope', async (_n, value) => {
      const { c, state } = rig();
      await c.fetch({
        conversation_id: CONV, user_id: 'dana', customer_id: 'acme', search_query: ['q'],
        scopes: ALL, include_conversation_context: false, scope_path: value,
      } as never);
      expect(state.sent).toHaveLength(4);
      for (const s of state.sent) {
        expect(Object.keys(s.body), s.path).not.toContain('scope');
        expect(Object.keys(s.body), s.path).not.toContain('scope_path');
      }
    });
});

describe('as_tool carries the path', () => {
  const target = () => {
    const calls: { where: string; opts: Record<string, unknown> }[] = [];
    const sub = (where: string) => ({ context: { fetch: async (o: Record<string, unknown>) => { calls.push({ where, opts: o }); return {}; } } });
    return {
      calls,
      t: {
        fetch: async (o: Record<string, unknown>) => { calls.push({ where: 'unified', opts: o }); return { formatted_context: '', scopes_queried: [], total_items: 0 }; },
        conversation: sub('conversation'), user: sub('user'), customer: sub('customer'), client: sub('client'),
      },
    };
  };
  const schemaOf = (tool: unknown) => JSON.stringify(tool, (k, v) => (k === 'handler' ? undefined : v));

  it.each(['unified', 'conversation', 'user', 'customer', 'client'])(
    'never in the LLM-visible schema (%s)', (scope) => {
      const { t } = target();
      const base = { scope, user_id: 'dana', customer_id: 'acme', conversation_id: CONV };
      const plain = buildTool(t as never, base);
      const withPath = buildTool(t as never, { ...base, scope_path: PATH });
      expect(schemaOf(withPath)).toBe(schemaOf(plain));
      expect(schemaOf(withPath)).not.toContain('scope_path');
    });
  it.each(['unified', 'conversation', 'user', 'customer', 'client'])(
    'the handler closes it over and the model cannot override it (%s)', async (scope) => {
      const { t, calls } = target();
      const tool = buildTool(t as never, { scope, user_id: 'dana', customer_id: 'acme', conversation_id: CONV, scope_path: PATH });
      await tool.handler({ search_query: ['q'], scope_path: { customer: 'evil' }, scope: { customer: 'evil' } });
      expect(calls).toHaveLength(1);
      expect(calls[0]!.opts['scope_path']).toEqual(PATH);
    });
  it('without it the handler passes no path key', async () => {
    const { t, calls } = target();
    await buildTool(t as never, { scope: 'user', user_id: 'dana' }).handler({ search_query: ['q'] });
    expect('scope_path' in calls[0]!.opts).toBe(false);
  });
  it('the client method forwards the option', async () => {
    const { c, state } = rig();
    await c.as_tool({ scope: 'customer', customer_id: 'acme', scope_path: PATH }).handler({ search_query: ['q'] });
    expect(state.sent[0]!.body['scope']).toEqual(PATH);
  });
});

it('is exported from the package entry', async () => {
  const mod = await import('../index.js');
  expect(mod.ScopeLadder).toBe(ScopeLadder);
  expect(mod.ScopeAddress).toBe(ScopeAddress);
  expect(mod.ScopeLadderError).toBe(ScopeLadderError);
});
