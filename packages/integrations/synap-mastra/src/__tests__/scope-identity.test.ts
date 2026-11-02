// A path fixed at construction must agree with the ids fixed beside it. The
// server lets the path's `user` and `customer` win over the flat ids, so a
// disagreement would address another identity while the caller believes it
// addressed its own. Refused once, at construction, before any SDK call.

import { describe, expect, it } from 'vitest';

import { SynapScopeError } from '../index.js';
import { SynapMemory } from '../memory.js';
import { synapSearchTool, synapStoreTool } from '../tools.js';
import { makeSdk } from './helpers.js';

const PATH = { customer: 'acme', user: 'alice' };

const oldSdk = () => {
  const sdk = makeSdk();
  delete (sdk as { get_scope_ladder?: unknown }).get_scope_ladder;
  return sdk;
};

describe('ids vs path, at construction', () => {
  it('SynapMemory refuses another userId, and another customerId', () => {
    expect(() => new SynapMemory({ sdk: makeSdk(), userId: 'bob', customerId: 'acme', scopePath: PATH }))
      .toThrow(SynapScopeError);
    expect(() => new SynapMemory({ sdk: makeSdk(), userId: 'alice', customerId: 'globex', scopePath: PATH }))
      .toThrow(SynapScopeError);
  });

  it('the tools refuse the same', () => {
    for (const make of [synapSearchTool, synapStoreTool]) {
      expect(() => make({ sdk: makeSdk(), userId: 'bob', customerId: 'acme', scopePath: PATH }))
        .toThrow(SynapScopeError);
      expect(() => make({ sdk: makeSdk(), userId: 'alice', customerId: 'globex', scopePath: PATH }))
        .toThrow(SynapScopeError);
    }
  });

  it('the refusal names no raw id and no SDK call was made', () => {
    const sdk = makeSdk();
    let err: Error | undefined;
    try {
      synapSearchTool({ sdk, userId: 'bob-secret', customerId: 'acme', scopePath: PATH });
    } catch (e) {
      err = e as Error;
    }
    expect(err).toBeInstanceOf(SynapScopeError);
    for (const raw of ['bob-secret', 'alice', 'acme']) expect(err!.message).not.toContain(raw);
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it('matching ids pass; an empty customerId passes (the path supplies it)', () => {
    expect(() => new SynapMemory({ sdk: makeSdk(), userId: 'alice', customerId: 'acme', scopePath: PATH })).not.toThrow();
    expect(() => new SynapMemory({ sdk: makeSdk(), userId: 'alice', scopePath: PATH })).not.toThrow();
  });

  it('B2C: customer == user passes with no customerId or an equal one', () => {
    const b2c = { customer: 'alice', user: 'alice' };
    expect(() => synapSearchTool({ sdk: makeSdk(), userId: 'alice', scopePath: b2c })).not.toThrow();
    expect(() => synapSearchTool({ sdk: makeSdk(), userId: 'alice', customerId: 'alice', scopePath: b2c })).not.toThrow();
  });

  it('a path with no user / customer key accepts any ids', () => {
    expect(() => synapSearchTool({ sdk: makeSdk(), userId: 'bob', customerId: 'globex', scopePath: { team: 'eng' } }))
      .not.toThrow();
  });

  it('without a path nothing is checked', () => {
    expect(() => new SynapMemory({ sdk: makeSdk(), userId: 'bob', customerId: 'globex' })).not.toThrow();
    expect(() => synapStoreTool({ sdk: makeSdk(), userId: 'bob', customerId: 'globex', scopePath: {} })).not.toThrow();
  });
});

describe('an SDK without get_scope_ladder is refused when a path is set', () => {
  it('SynapMemory and both tools throw at construction, naming 0.7.1', () => {
    expect(() => new SynapMemory({ sdk: oldSdk(), userId: 'alice', scopePath: PATH })).toThrow(/>= 0\.7\.1/);
    expect(() => synapSearchTool({ sdk: oldSdk(), userId: 'alice', scopePath: PATH })).toThrow(/>= 0\.7\.1/);
    expect(() => synapStoreTool({ sdk: oldSdk(), userId: 'alice', scopePath: PATH })).toThrow(/>= 0\.7\.1/);
  });

  it('without a path (or an empty one) an old SDK still works', () => {
    expect(() => new SynapMemory({ sdk: oldSdk(), userId: 'alice' })).not.toThrow();
    expect(() => synapSearchTool({ sdk: oldSdk(), userId: 'alice', scopePath: {} })).not.toThrow();
  });
});
