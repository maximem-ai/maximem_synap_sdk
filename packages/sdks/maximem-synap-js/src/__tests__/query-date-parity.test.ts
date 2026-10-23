import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFetchRequest } from '../context/fetch.js';
import type { FetchOptions } from '../context/types.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = (p: string) => readFileSync(path.resolve(here, '..', p), 'utf8');

/**
 * `query_date` tells the server the moment the question is being asked, so it
 * can resolve a period the question names ("in March", "two months ago")
 * instead of counting back from now. The server has accepted it on all four
 * fetch routes for a while and no SDK exposed it, so the filter never ran.
 *
 * These tests exist because threading one argument through four scopes is
 * exactly the change that half-lands. The Python edit missed the client branch
 * of its unified fetch, because that branch sits one indent deeper than the
 * other three, and the first version of its test asserted "three" and pinned
 * the miss as correct.
 */
describe('query_date reaches every fetch surface', () => {
  // Each scope's own required identifier. The conversation route rejects a
  // request without conversation_id, so one shared options bag cannot serve all
  // four.
  const SCOPES = [
    { scope: 'user' as const, required: { user_id: 'u1' } },
    { scope: 'customer' as const, required: { customer_id: 'c1' } },
    { scope: 'client' as const, required: {} },
    { scope: 'conversation' as const, required: { conversation_id: 'conv-1' } },
  ];

  it.each(SCOPES)('$scope scope sends it when provided', ({ scope, required }) => {
    const opts = { ...required, query_date: '2023-03-11' } as FetchOptions;
    expect(buildFetchRequest(scope, opts).body['query_date']).toBe('2023-03-11');
  });

  it.each(SCOPES)('$scope scope omits the key at its default', ({ scope, required }) => {
    expect('query_date' in buildFetchRequest(scope, required as FetchOptions).body)
      .toBe(false);
  });

  it.each(SCOPES)('$scope scope treats an empty string as absent', ({ scope, required }) => {
    const opts = { ...required, query_date: '' } as FetchOptions;
    expect('query_date' in buildFetchRequest(scope, opts).body).toBe(false);
  });

  it('accepts the camelCase spelling too, as every other option does', () => {
    const opts = { user_id: 'u1', queryDate: '2023-03-11' } as FetchOptions;
    expect(buildFetchRequest('user', opts).body['query_date']).toBe('2023-03-11');
  });

  it('is passed through verbatim, including the harness date shape', () => {
    // The server strips the parenthesised weekday itself. An SDK that
    // normalised it would be friction for no gain and a second place for the
    // format to drift.
    const raw = '2023/03/11 (Sat) 21:26';
    const opts = { user_id: 'u1', query_date: raw } as FetchOptions;
    expect(buildFetchRequest('user', opts).body['query_date']).toBe(raw);
  });
});

describe('the unified fetch threads it to every branch', () => {
  it('carries it in `shared`, which all four branches spread', () => {
    // A source check on purpose. The bug class here is a branch that does not
    // receive the option, and `shared` is what makes that impossible: anything
    // in it reaches all four. Threading four call sites by hand is what went
    // wrong on the Python side.
    const client = src('client.ts');
    const sharedBlock = client.slice(
      client.indexOf('const shared = {'),
      client.indexOf('const labels: string[] = []'),
    );
    expect(sharedBlock).toContain('query_date: queryDate,');
    expect(client).toContain('query_date: queryDate,');
    expect(client).toMatch(/query_date\?: string;/);
  });

  it('never serves a dated request from the prefetch cache', () => {
    // A prefetched bundle was assembled with no period restriction, so it
    // answers a different question. Same shape as the scope_path defect found
    // on production in August.
    expect(src('client.ts')).toContain(
      "if (o['query_date'] ?? o['queryDate']) return null;",
    );
  });
});
