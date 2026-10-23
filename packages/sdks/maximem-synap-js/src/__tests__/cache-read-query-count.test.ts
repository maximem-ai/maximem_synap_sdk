import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { distinctQueryCount } from '../context/anticipated.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = (p: string) => readFileSync(path.resolve(here, '..', p), 'utf8');

/**
 * A fetch served from the prefetch cache never reaches a fetch route, so the
 * `context_used` stream event is the only thing that tells the server it
 * happened. The server charges it per distinct search query, as it does a fetch
 * it answers itself, so the event carries the count.
 *
 * The count only: the event's own rule is that it never carries the caller's
 * words. The Python half is `synap/tests/sdk/test_cache_read_reports_query_count.py`.
 */
describe('the count a cache read reports', () => {
  it.each([
    { queries: ['diet'], expected: 1 },
    { queries: ['diet', 'restaurants'], expected: 2 },
    { queries: ['diet', 'diet'], expected: 1 },
    { queries: ['diet', '', 'restaurants'], expected: 2 },
    { queries: [], expected: 0 },
  ])('$queries is $expected, as in Python', ({ queries, expected }) => {
    expect(distinctQueryCount(queries)).toBe(expected);
  });

  it.each([undefined, null, 'diet', 3])('%s is not a query list, so zero', (notAList) => {
    expect(distinctQueryCount(notAList)).toBe(0);
  });
});

describe('the context_used event', () => {
  it('sends the count, read from either spelling of the option', () => {
    expect(src('client.ts')).toContain(
      "query_count: distinctQueryCount(options['search_query'] ?? options['searchQuery']),",
    );
  });

  it('has the field on the wire and no field for the caller\'s words', () => {
    const json = JSON.stringify(SYNAP_PROTO_DESCRIPTOR);
    // The definition, not the reference to it inside StreamEvent's oneof.
    const start = json.indexOf('"ContextUsedEvent":{');
    expect(start).toBeGreaterThan(-1);
    const event = json.slice(start, json.indexOf('"ContextAssembledEvent":{', start));
    expect(event).toContain('"query_count"');
    expect(event).not.toContain('"search_query"');
    expect(event).not.toContain('"search_queries"');
  });
});
