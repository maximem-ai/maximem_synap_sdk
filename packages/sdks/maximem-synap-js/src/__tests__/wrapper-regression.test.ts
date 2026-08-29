import { describe, it, expect } from 'vitest';
import { SynapClient } from '../client.js';

/**
 * No regression against the 0.3.x Python-subprocess wrapper.
 *
 * Surface parity with the pip SDK is one axis; this is the other. Anything the
 * wrapper exposed still has to exist, or upgrading breaks working code with a
 * TypeError. The list below is taken from the removed `src/synap-client.js`.
 */
const WRAPPER_TOP_LEVEL = [
  'addMemory', 'deleteMemory',
  'fetchClientContext', 'fetchCustomerContext', 'fetchUserContext',
  'getContextForPrompt', 'getMemories', 'searchMemory',
  'init', 'shutdown',
];

const WRAPPER_NAMESPACED: [string, string][] = [
  ['user.context', 'fetch'],
  ['customer.context', 'fetch'],
  ['client.context', 'fetch'],
  ['conversation.context', 'fetch'],
  ['conversation.context', 'get_context_for_prompt'],
  ['conversation', 'record_message'],
  ['memories', 'create'],
];

function resolve(client: SynapClient, dotted: string): Record<string, unknown> | undefined {
  let cur: unknown = client;
  for (const part of dotted.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur as Record<string, unknown> | undefined;
}

const client = () =>
  new SynapClient({ _force_new: true, apiKey: 'k', fetchImpl: (async () => new Response('{}')) as unknown as typeof fetch });

describe('no regression against the 0.3.x wrapper', () => {
  it.each(WRAPPER_TOP_LEVEL)('client.%s still exists', (name) => {
    expect(typeof (client() as unknown as Record<string, unknown>)[name]).toBe('function');
  });

  it.each(WRAPPER_NAMESPACED)('client.%s.%s still exists', (ns, method) => {
    expect(typeof resolve(client(), ns)?.[method]).toBe('function');
  });

  it('accepts both id spellings, as the wrapper did', async () => {
    // The wrapper read `pickDefined(args.user_id, args.userId)` everywhere.
    const calls: string[] = [];
    const c = new SynapClient({ _force_new: true, apiKey: 'k',
      fetchImpl: (async (_u: string, i: RequestInit) => {
        calls.push(i.body as string);
        return new Response('{"context":{}}');
      }) as unknown as typeof fetch,
    });
    await c.user.context.fetch({ user_id: 'u1' });
    await c.user.context.fetch({ userId: 'u1' });
    expect(calls[0]).toBe(calls[1]);
  });

  it('keeps the wrapper defaults that differ between methods', async () => {
    const bodies: Record<string, unknown>[] = [];
    const c = new SynapClient({ _force_new: true, apiKey: 'k',
      fetchImpl: (async (_u: string, i: RequestInit) => {
        bodies.push(JSON.parse(i.body as string));
        return new Response('{"context":{}}');
      }) as unknown as typeof fetch,
    });
    await c.searchMemory({ userId: 'u1', query: 'x' });
    await c.getMemories({ userId: 'u1' });
    // searchMemory defaulted to 10, getMemories to 100. Easy to "harmonise"
    // by accident, and it changes how much a caller gets back.
    expect(bodies[0]!['max_results']).toBe(10);
    expect(bodies[1]!['max_results']).toBe(100);
    expect(bodies[0]!['search_query']).toEqual(['x']);
    expect(bodies[1]!['search_query']).toEqual([]);
  });

  it('flattens context the way the bridge did', async () => {
    const c = new SynapClient({ _force_new: true, apiKey: 'k',
      fetchImpl: (async () => new Response(JSON.stringify({
        context: {
          facts: [{ id: 'f', content: 'fact text', confidence: 0.9, source: 'chat' }],
          preferences: [{ id: 'p', content: 'pref text', strength: 0.8, source: 'chat' }],
          episodes: [{ id: 'e', summary: 'episode text', significance: 0.7 }],
          emotions: [{ id: 'm', context: 'emotion text', intensity: 0.6 }],
          temporal_events: [{ id: 't', content: 'event text', temporal_confidence: 0.5, source: 'chat' }],
        },
      }))) as unknown as typeof fetch,
    });
    const { results, count } = await c.searchMemory({ userId: 'u1', query: 'x' });
    expect(count).toBe(5);
    expect(results.map((r) => r.context_type))
      .toEqual(['fact', 'preference', 'episode', 'emotion', 'temporal_event']);
    // `memory` comes from a different source field per type.
    expect(results.map((r) => r.memory))
      .toEqual(['fact text', 'pref text', 'episode text', 'emotion text', 'event text']);
    // `score` likewise.
    expect(results.map((r) => r.score)).toEqual([0.9, 0.8, 0.7, 0.6, 0.5]);
    // Episodes and emotions carry NO source key. The bridge never set one, and
    // adding it would silently change the shape consumers destructure.
    expect('source' in results[2]!).toBe(false);
    expect('source' in results[3]!).toBe(false);
    expect(results[0]!.source).toBe('chat');
  });
});
