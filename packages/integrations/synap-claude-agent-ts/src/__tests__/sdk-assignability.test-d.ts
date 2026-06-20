/**
 * Type-level regression guard (runs in CI via vitest's typecheck mode).
 *
 * Why this exists: this integration accepts `sdk: SynapSdkLike` (a duck-typed
 * surface in ../types.ts) and consumers pass the real `@maximem/synap-js-sdk`
 * `SynapClient`. Nothing at runtime proves the real client actually satisfies
 * that surface — the runtime tests use hand-written mocks. Only a type check
 * catches drift, and the sync process has reverted the enabling fix before
 * (e.g. dropping `| null` from the namespaced input id fields in the SDK's
 * types/index.d.ts, which silently breaks assignability). This test fails the
 * build the moment that happens.
 */

import { expectTypeOf, test } from 'vitest';
import type { SynapClient } from '@maximem/synap-js-sdk';
import type { SynapSdkLike } from '../types.js';

test('the real SynapClient satisfies the SynapSdkLike surface this integration consumes', () => {
  // Faithful reproduction of how consumers wire it: `const sdk: SynapSdkLike = client`.
  const _assignable: SynapSdkLike = undefined as unknown as SynapClient;
  void _assignable;

  // Vitest-native form of the same assertion.
  expectTypeOf<SynapClient>().toMatchTypeOf<SynapSdkLike>();
});
