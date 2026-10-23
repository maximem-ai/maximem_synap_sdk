import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFetchRequest } from '../context/fetch.js';
import {
  normalizeContextResponse,
  normalizeEmotion,
  normalizeEpisode,
  normalizeFact,
  normalizePreference,
  normalizeTemporalEvent,
} from '../context/normalize.js';
import type { FetchOptions, RawContextItem } from '../context/types.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = (p: string) => readFileSync(path.resolve(here, '..', p), 'utf8');

/**
 * Two additions, both optional, and both of which must leave a caller who does
 * not use them exactly where they were.
 *
 * `max_results_per_query` gives each search query a share of the response, so a
 * query with many strong matches cannot take every place. `max_results` stays
 * the ceiling. `matched_queries` on each item says which query found it.
 *
 * The Python half is `synap/tests/sdk/test_max_results_per_query_parity.py`.
 * The golden request bodies both must build are in
 * `CONTRACT/conformance/request_payloads.json`, checked by
 * `payload-parity.test.ts`.
 */
describe('max_results_per_query reaches every fetch surface', () => {
  const SCOPES = [
    { scope: 'user' as const, required: { user_id: 'u1' } },
    { scope: 'customer' as const, required: { customer_id: 'c1' } },
    { scope: 'client' as const, required: {} },
    { scope: 'conversation' as const, required: { conversation_id: 'conv-1' } },
  ];
  const QUERIES = { search_query: ['diet', 'restaurants'] };

  it.each(SCOPES)('$scope scope sends it when provided', ({ scope, required }) => {
    const opts = { ...required, ...QUERIES, max_results_per_query: 3 } as FetchOptions;
    expect(buildFetchRequest(scope, opts).body['max_results_per_query']).toBe(3);
  });

  it.each(SCOPES)('$scope scope leaves max_results alone', ({ scope, required }) => {
    const opts = { ...required, ...QUERIES, max_results_per_query: 3 } as FetchOptions;
    expect(buildFetchRequest(scope, opts).body['max_results']).toBe(10);
  });

  it.each(SCOPES)('$scope scope omits the key when unset', ({ scope, required }) => {
    const opts = { ...required, ...QUERIES } as FetchOptions;
    expect('max_results_per_query' in buildFetchRequest(scope, opts).body).toBe(false);
  });

  it.each(SCOPES)('$scope scope treats 0 as unset, as Python does', ({ scope, required }) => {
    const opts = { ...required, ...QUERIES, max_results_per_query: 0 } as FetchOptions;
    expect('max_results_per_query' in buildFetchRequest(scope, opts).body).toBe(false);
  });

  it('accepts the camelCase spelling too, as every other option does', () => {
    const opts = { user_id: 'u1', ...QUERIES, maxResultsPerQuery: 4 } as FetchOptions;
    expect(buildFetchRequest('user', opts).body['max_results_per_query']).toBe(4);
  });

  it('a request that does not use it is byte-identical to before', () => {
    const body = buildFetchRequest('user', { user_id: 'u1', ...QUERIES } as FetchOptions).body;
    expect(Object.keys(body).sort()).toEqual(
      ['conversation_id', 'max_results', 'mode', 'search_query', 'types', 'user_id'],
    );
  });
});

describe('the unified fetch threads it to every branch', () => {
  it('carries it in `shared`, which all four branches spread', () => {
    const client = src('client.ts');
    const sharedBlock = client.slice(
      client.indexOf('const shared = {'),
      client.indexOf('const labels: string[] = []'),
    );
    expect(sharedBlock).toContain('max_results_per_query: maxResultsPerQuery,');
    expect(client).toMatch(/max_results_per_query\?: number;/);
  });

  it('never serves a request that names a share from the prefetch cache', () => {
    // A prefetched bundle was not assembled under a share, and its items do
    // not say which query found them.
    expect(src('client.ts')).toContain(
      "if (o['max_results_per_query'] ?? o['maxResultsPerQuery']) return null;",
    );
  });
});

describe('matched_queries on returned items', () => {
  const item: RawContextItem = {
    id: 'm-1', content: 'c', confidence: 0.8, source: 'vector',
    matched_queries: ['diet', 'restaurants'],
  };
  const NORMALISERS = [
    { name: 'fact', fn: normalizeFact },
    { name: 'preference', fn: normalizePreference },
    { name: 'episode', fn: normalizeEpisode },
    { name: 'emotion', fn: normalizeEmotion },
    { name: 'temporal event', fn: normalizeTemporalEvent },
  ];

  it.each(NORMALISERS)('a normalised $name carries it as matchedQueries', ({ fn }) => {
    expect(fn(item).matchedQueries).toEqual(['diet', 'restaurants']);
  });

  it.each(NORMALISERS)('a $name from a server without the field gets an empty array', ({ fn }) => {
    const { matched_queries: _dropped, ...older } = item;
    expect(fn(older).matchedQueries).toEqual([]);
  });

  it.each(NORMALISERS)('a null from the server is an empty array for a $name', ({ fn }) => {
    expect(fn({ ...item, matched_queries: null as unknown as string[] }).matchedQueries)
      .toEqual([]);
  });

  it('is declared on the raw item, so the namespaced fetches expose it typed', () => {
    // The raw surface returns the server's items untouched, so the field was
    // already there at runtime the moment the server sent it. This is the type.
    const typed: string[] | undefined = item.matched_queries;
    expect(typed).toEqual(['diet', 'restaurants']);
  });

  it('survives normalizeContextResponse for every group', () => {
    const out = normalizeContextResponse({
      context: {
        facts: [item], preferences: [item], episodes: [item],
        emotions: [item], temporal_events: [item],
      },
    }) as unknown as Record<string, Array<{ matchedQueries: string[] }>>;
    for (const group of ['facts', 'preferences', 'episodes', 'emotions', 'temporalEvents']) {
      expect(out[group]?.[0]?.matchedQueries, group).toEqual(['diet', 'restaurants']);
    }
  });
});

/**
 * `retrieval` is a SIBLING of `context` in the response body. The Python SDK
 * dropped it for exactly that reason, so a retrieval that lost its embedding
 * call (16 of 267 fetches, a user holding 122 memories, zero items served)
 * looked identical to "no relevant memories". The normaliser must read it off
 * the envelope, and must report null on a healthy body.
 */
describe('retrieval health from the envelope', () => {
  it('surfaces a failed retrieval', () => {
    const out = normalizeContextResponse({
      context: {},
      retrieval: { degraded: true, level: 'failed', code: 'timeout', warnings: ['Retrieval failed (timeout)'] },
    });
    expect(out.retrieval).toEqual({
      degraded: true, level: 'failed', code: 'timeout', warnings: ['Retrieval failed (timeout)'],
    });
  });

  it('is null on a healthy body', () => {
    expect(normalizeContextResponse({ context: {} }).retrieval).toBeNull();
  });
});
