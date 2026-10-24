import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SynapClient } from '../client.js';
import { conversationStateContext, hasQuery, profileContext } from '../context/no-query.js';
import type { Json } from '../context/types.js';

/**
 * A fetch with no query, against the corpus captured from Python
 * (`synap/sdk/scripts/gen_no_query_corpus.py`). The Python half is
 * `synap/tests/sdk/test_no_query_fetch.py`.
 *
 * Without a query the server embedded an empty string and ranked memories
 * against it. A conversation fetch now answers with the conversation's state,
 * a user fetch with the profile when there is one; everything else, and a user
 * with no profile, runs the fetch it always ran.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(
  readFileSync(path.resolve(here, '..', '..', '..', 'CONTRACT', 'conformance', 'no_query_fetch.json'), 'utf8'),
) as {
  query_cases: Array<{ search_query: unknown; expected: boolean }>;
  conversation_cases: Array<{ name: string; prompt: Json; conversation_id: string; expected: unknown }>;
  profile_cases: Array<{ name: string; profile: unknown; expected: unknown }>;
};

describe('the rules, against the Python corpus', () => {
  it.each(corpus.query_cases)('has a query: $search_query', (c) => {
    expect(hasQuery(c.search_query)).toBe(c.expected);
  });

  it.each(corpus.conversation_cases)('conversation state: $name', (c) => {
    expect(conversationStateContext(c.prompt, c.conversation_id)).toEqual(c.expected);
  });

  it.each(corpus.profile_cases)('profile: $name', (c) => {
    expect(profileContext(c.profile)).toEqual(c.expected);
  });
});

const INSTANCE_ID = 'inst_0123456789abcdef';
const CONV = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

const PROMPT = corpus.conversation_cases.find((c) => c.name === 'compacted_with_tail')!.prompt;
const PROFILE = { profile: { attributes: { city: { value: 'Pune' } }, overview: 'Frequent rider.' } };

/** A server that answers each route, recording the paths and bodies it saw. */
function server(routes: { profile?: () => Response } = {}) {
  const seen: Array<{ path: string; body: Json | null }> = [];
  const fetchImpl = (async (url: string, init?: { body?: string }) => {
    const p = new URL(url).pathname;
    seen.push({ path: p, body: init?.body ? (JSON.parse(init.body) as Json) : null });
    if (p.endsWith('/context-for-prompt')) return json({ context_for_prompt: PROMPT });
    if (p.endsWith('/profile')) return routes.profile ? routes.profile() : json(PROFILE);
    return json({ context: { facts: [{ id: 'f1', content: 'searched', confidence: 0.9, source: 'vector' }] } });
  }) as unknown as typeof fetch;
  const client = new SynapClient({
    _force_new: true, apiKey: 'k', clientId: 'cli_test', instanceId: INSTANCE_ID, fetchImpl,
  });
  const searches = () => seen.filter((s) => s.path.startsWith('/v1/context/'));
  return { client, seen, searches };
}

describe('conversation.context.fetch', () => {
  it('with no query answers with the conversation state, and searches nothing', async () => {
    const { client, searches } = server();
    const ctx = await client.conversation.context.fetch({ conversation_id: CONV });
    expect(searches()).toHaveLength(0);
    expect(ctx['conversation_context']).toEqual(conversationStateContext(PROMPT, CONV)['conversation_context']);
    expect(ctx['facts']).toEqual([]);
  });

  it('with blank queries is the same', async () => {
    const { client, searches } = server();
    await client.conversation.context.fetch({ conversation_id: CONV, search_query: ['', '  '] });
    expect(searches()).toHaveLength(0);
  });

  it.each(['blank', [123], ['ok', null]])('refuses malformed query input before the no-query path', async (search_query) => {
    const { client, seen } = server();
    await expect(client.conversation.context.fetch({
      conversation_id: CONV, search_query: search_query as string[],
    })).rejects.toThrow('search_query must be an array of strings');
    expect(seen).toEqual([]);
  });

  it('with a query searches, as before', async () => {
    const { client, searches } = server();
    await client.conversation.context.fetch({ conversation_id: CONV, search_query: ['refund'] });
    expect(searches().map((s) => s.path)).toEqual(['/v1/context/conversation/fetch']);
  });

  it('with no query still carries a turn recorded here and not yet stored', async () => {
    const { client } = server();
    await client.conversation.record_message({
      conversation_id: CONV, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'I fly from Berlin',
    });
    const ctx = await client.conversation.context.fetch({ conversation_id: CONV });
    const turns = ((ctx['conversation_context'] as Json)['recent_turns'] as Array<{ content: string }>);
    expect(turns.map((t) => t.content)).toContain('I fly from Berlin');
    await client.shutdown();
  });

  it('ranked, with no query, has an empty items list', async () => {
    const { client } = server();
    const ctx = await client.conversation.context.fetch({ conversation_id: CONV, response_shape: 'ranked' });
    expect(ctx['items']).toEqual([]);
  });
});

describe('user.context.fetch', () => {
  it('with no query answers with the profile, and searches nothing', async () => {
    const { client, searches } = server();
    const ctx = await client.user.context.fetch({ user_id: 'u1' });
    expect(searches()).toHaveLength(0);
    expect(ctx['profile']).toEqual(PROFILE.profile);
  });

  it('with no profile (404) runs the fetch it always ran', async () => {
    const { client, searches } = server({ profile: () => json({ detail: 'not found' }, 404) });
    const ctx = await client.user.context.fetch({ user_id: 'u1' });
    expect(searches().map((s) => s.path)).toEqual(['/v1/context/user/fetch']);
    expect((ctx['facts'] as unknown[]).length).toBe(1);
  });

  it('with an empty profile runs the fetch it always ran', async () => {
    const { client, searches } = server({ profile: () => json({ profile: { attributes: {}, overview: null } }) });
    await client.user.context.fetch({ user_id: 'u1' });
    expect(searches()).toHaveLength(1);
  });

  it('when the profile read fails runs the fetch it always ran', async () => {
    const { client, searches } = server({ profile: () => json({ detail: 'boom' }, 422) });
    await client.user.context.fetch({ user_id: 'u1' });
    expect(searches()).toHaveLength(1);
  });

  it('in summary mode is unchanged', async () => {
    const { client, seen, searches } = server();
    await client.user.context.fetch({ user_id: 'u1', context_mode: 'conversation-summary' });
    expect(seen.some((s) => s.path.endsWith('/profile'))).toBe(false);
    expect(searches()).toHaveLength(1);
  });

  it('with a query searches and never reads the profile', async () => {
    const { client, seen, searches } = server();
    await client.user.context.fetch({ user_id: 'u1', search_query: ['home address'] });
    expect(seen.some((s) => s.path.endsWith('/profile'))).toBe(false);
    expect(searches()).toHaveLength(1);
  });
});

describe('customer and client fetches', () => {
  it('with no query run the fetch they always ran', async () => {
    const { client, searches } = server();
    await client.customer.context.fetch({ customer_id: 'c1' });
    await client.client.context.fetch({});
    expect(searches().map((s) => s.path)).toEqual(['/v1/context/customer/fetch', '/v1/context/client/fetch']);
  });
});

describe('the unified fetch', () => {
  it('with no query: conversation state plus the profile, no searches', async () => {
    const { client, searches } = server();
    const merged = await client.fetch({ conversation_id: CONV, user_id: 'u1' });
    expect(searches()).toHaveLength(0);
    expect(merged.scopes_queried).toEqual(['conversation', 'user']);
    expect(merged.profile).toEqual(PROFILE.profile);
    expect(merged.conversation_context).toEqual(PROMPT);
    expect(merged.formatted_context).toContain('Priya reported order ORD-1 broken.');
  });

  it('with no query and only a conversation id answers with its state', async () => {
    const { client, searches } = server();
    const merged = await client.fetch({ conversation_id: CONV });
    expect(searches()).toHaveLength(0);
    expect(merged.scopes_queried).toEqual(['conversation']);
    expect(merged.conversation_context).toEqual(PROMPT);
  });

  it('with no query and the conversation context turned off adds no turns, and searches nothing', async () => {
    // pydantic-ai and Mastra build the system prompt this way: the framework
    // already holds the message history.
    const { client, seen, searches } = server();
    const merged = await client.fetch({ conversation_id: CONV, user_id: 'u1', include_conversation_context: false });
    expect(searches()).toHaveLength(0);
    expect(seen.some((s) => s.path.endsWith('/context-for-prompt'))).toBe(false);
    expect(merged.conversation_context).toBeNull();
    expect(merged.profile).toEqual(PROFILE.profile);
    expect(merged.formatted_context).not.toContain('Priya');
  });

  it('with a query searches every scope, as before', async () => {
    const { client, searches } = server();
    await client.fetch({ conversation_id: CONV, user_id: 'u1', search_query: ['refund'] });
    expect(searches().map((s) => s.path).sort()).toEqual(
      ['/v1/context/conversation/fetch', '/v1/context/user/fetch']);
  });
});
