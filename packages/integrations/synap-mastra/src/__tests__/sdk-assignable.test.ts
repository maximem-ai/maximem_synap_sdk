// The real SynapClient from @maximem/synap-js-sdk must be assignable to
// SynapSdkLike under `strict`, with no cast. This file is compiled by `tsc`
// (the build and `npm run typecheck`), so a regression fails there; the
// runtime assertions below only keep vitest from reporting an empty suite.

import { describe, expect, it, vi } from 'vitest';
import type { SynapClient } from '@maximem/synap-js-sdk';
import { SynapMemory } from '../memory.js';
import type { SynapSdkLike } from '../types.js';
import { makeSdk, RICH } from './helpers.js';

// Never called: it only has to type-check.
export function realClientIsAssignable(client: SynapClient): SynapSdkLike {
  return client;
}

describe('SynapSdkLike accepts the real SynapClient', () => {
  it('is checked at compile time', () => {
    expect(typeof realClientIsAssignable).toBe('function');
  });

  it('never sends search_query: null (the real SDK throws on a non-array)', async () => {
    const sdk = makeSdk({ fetchResponse: RICH() });
    const mem = new SynapMemory({ sdk, userId: 'alice' });
    await mem.getSystemMessage({ threadId: 't-1' });
    const callArg = (sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArg.search_query).toBeUndefined();
  });
});
