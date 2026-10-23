// The real SynapClient from @maximem/synap-js-sdk must be assignable to the
// types this package asks for, under `strict`, with no cast. This file is
// compiled by `tsc` (the build and `npm run typecheck`), so a regression fails
// there; the runtime assertion only keeps vitest from reporting an empty suite.

import { describe, expect, it } from 'vitest';
import type { SynapClient } from '@maximem/synap-js-sdk';
import { createSynapSearchTool, createSynapStoreTool } from '../tools.js';
import { createSynapStreamHooks, type SynapTurnSdkLike } from '../stream.js';
import type { SynapSdkLike } from '../types.js';

// Never called: it only has to type-check.
export function realClientIsAssignable(client: SynapClient) {
  const sdk: SynapSdkLike = client;
  const turn: SynapTurnSdkLike = client;
  createSynapStreamHooks({ sdk: client });
  createSynapSearchTool({ sdk: client });
  createSynapStoreTool({ sdk: client });
  return [sdk, turn];
}

describe('the real SynapClient is assignable to the SDK-like types', () => {
  it('is checked at compile time', () => {
    expect(typeof realClientIsAssignable).toBe('function');
  });
});
