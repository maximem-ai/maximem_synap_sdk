/**
 * The scope guard: per-call ids must agree with a fixed scope path, and a path
 * needs an SDK that carries it. Mirrors `test_ensure_ids_match_path.py` in
 * `synap_integrations_common`; vendored byte-identical like `scope-guard.ts`.
 */
import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import {
  SynapScopeError,
  assertSdkSupportsScopePath,
  ensureIdsMatchPath,
  fingerprint,
} from '../scope-guard.js';

const PATH = { customer: 'acme', user: 'alice' };

describe('ensureIdsMatchPath', () => {
  it('matching ids pass', () => {
    ensureIdsMatchPath(PATH, { userId: 'alice', customerId: 'acme', where: 't' });
  });

  it('a user mismatch throws, names the key and where, and is a SynapScopeError', () => {
    let caught: unknown;
    try {
      ensureIdsMatchPath(PATH, { userId: 'bob', customerId: 'acme', where: 'Store.search' });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(SynapScopeError);
    const err = caught as SynapScopeError;
    expect(err.message).toMatch(/^ensure_ids_match_path: user_id does not match the scope_path's 'user' level \(Store\.search\): id fingerprint [0-9a-f]{8}, path fingerprint [0-9a-f]{8}\. The server would use the path, so this call would read or write another identity's memory\.$/);
    expect(err.operation).toBe('ensure_ids_match_path');
    expect(err.context.key).toBe('user');
    expect(err.context.where).toBe('Store.search');
  });

  it('a customer mismatch throws', () => {
    expect(() => ensureIdsMatchPath(PATH, { userId: 'alice', customerId: 'rival' }))
      .toThrow(/customer_id does not match/);
  });

  it('the exposure case, bob and rival, is refused', () => {
    expect(() => ensureIdsMatchPath(PATH, { userId: 'bob', customerId: 'rival', where: 'x' }))
      .toThrow(SynapScopeError);
  });

  it('only the keys present in the path are checked', () => {
    ensureIdsMatchPath({ user: 'alice' }, { userId: 'alice', customerId: 'anything' });
    expect(() => ensureIdsMatchPath({ user: 'alice' }, { userId: 'bob' })).toThrow(SynapScopeError);
    ensureIdsMatchPath({ team: 'payments' }, { userId: 'bob', customerId: 'rival' });
  });

  it.each([undefined, null, ''])('an empty id (%j) is fine: the path supplies it', (empty) => {
    ensureIdsMatchPath(PATH, { userId: empty, customerId: empty });
    ensureIdsMatchPath(PATH, { userId: 'alice', customerId: empty });
  });

  it('B2C: customer == user passes with no customer id or an equal one', () => {
    const b2c = { customer: 'alice', user: 'alice' };
    ensureIdsMatchPath(b2c, { userId: 'alice', customerId: undefined });
    ensureIdsMatchPath(b2c, { userId: 'alice', customerId: 'alice' });
    expect(() => ensureIdsMatchPath(b2c, { userId: 'alice', customerId: 'rival' }))
      .toThrow(SynapScopeError);
  });

  it.each([undefined, null, {}, []])('a falsy or empty path (%j) is a no-op', (falsy) => {
    ensureIdsMatchPath(falsy as never, { userId: 'anyone', customerId: 'anyone' });
  });

  it.each([
    { user: 'a/b' }, { user: '' }, { user: 3 }, { '': 'x' }, { user: null }, { team: 'x/' },
  ])('an odd path %j is refused even when no ids are given', (bad) => {
    expect(() => ensureIdsMatchPath(bad as never)).toThrow(/scope_path/);
  });

  it('a non-mapping path is refused', () => {
    expect(() => ensureIdsMatchPath(['acme', 'alice'] as never)).toThrow(/mapping/);
    expect(() => ensureIdsMatchPath('acme/alice' as never)).toThrow(/mapping/);
  });

  it('messages and context never contain the raw ids', () => {
    let err: SynapScopeError | undefined;
    try {
      ensureIdsMatchPath(
        { customer: 'acme-secret', user: 'alice-secret' },
        { userId: 'bob-secret', customerId: 'rival-secret', where: 'w' },
      );
    } catch (e) {
      err = e as SynapScopeError;
    }
    const blob = String(err?.message) + JSON.stringify(err?.context);
    for (const raw of ['acme-secret', 'alice-secret', 'bob-secret', 'rival-secret']) {
      expect(blob).not.toContain(raw);
    }
    expect(err?.context.id_fp).not.toBe(err?.context.path_fp);
    expect(String(err?.context.id_fp)).toHaveLength(8);
  });

  it('a bad path value is not echoed', () => {
    expect(() => ensureIdsMatchPath({ user: 'secret/value' })).toThrow(
      expect.objectContaining({ message: expect.not.stringContaining('secret') }),
    );
  });
});

describe('fingerprint', () => {
  it('is the first 8 hex of sha256, as in Python', () => {
    // Python: hashlib.sha256(v.encode()).hexdigest()[:8]
    expect(fingerprint('alice')).toBe('2bd806c9');
    expect(fingerprint('bob')).toBe('81b637d8');
    expect(fingerprint('')).toBe('e3b0c442');
  });

  it('agrees with node:crypto on odd and long inputs, including multi-block ones', () => {
    for (const v of ['é☃ x', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(64), 'a'.repeat(200), '😀'.repeat(40)]) {
      expect(fingerprint(v)).toBe(createHash('sha256').update(v, 'utf8').digest('hex').slice(0, 8));
    }
  });
});

describe('assertSdkSupportsScopePath', () => {
  const NEW_SDK = { get_scope_ladder: () => null };

  it('passes when the SDK has get_scope_ladder', () => {
    assertSdkSupportsScopePath(NEW_SDK, PATH, 'pkg');
  });

  it('throws at the minimum version when the SDK lacks it', () => {
    expect(() => assertSdkSupportsScopePath({}, PATH, 'createThing')).toThrow(
      /createThing: scopePath needs @maximem\/synap-js-sdk >= 0\.7\.1/,
    );
    expect(() => assertSdkSupportsScopePath({ get_scope_ladder: 'x' }, PATH, 'p')).toThrow(/0\.7\.1/);
    expect(() => assertSdkSupportsScopePath(null, PATH, 'p')).toThrow(/0\.7\.1/);
  });

  it.each([undefined, null, {}])('is a no-op without a path (%j), whatever the SDK', (none) => {
    assertSdkSupportsScopePath({}, none, 'pkg');
    assertSdkSupportsScopePath(undefined, none, 'pkg');
  });
});
