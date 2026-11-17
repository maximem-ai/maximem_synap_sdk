/**
 * A 504 is the gateway giving up, not the origin refusing: the origin may have finished. A
 * non-idempotent create is therefore sent once; a 503 (the origin's own answer) is retried.
 * Python twin: python/tests/test_http_504_no_retry_on_create.py.
 */
import { describe, it, expect } from 'vitest';
import { HttpTransport } from '../transport/http.js';

function transport(status: number, counter: { n: number }) {
  const fetchImpl = (async () => {
    counter.n += 1;
    return new Response(JSON.stringify({ detail: 'x' }), {
      status, headers: { 'content-type': 'application/json' },
    });
  }) as unknown as typeof fetch;
  return new HttpTransport({
    credentials: { apiKey: 'k', clientId: 'c', instanceId: 'i' },
    fetchImpl,
    retryPolicy: { maxAttempts: 3, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
}

describe('504 on a non-idempotent create', () => {
  it('is sent once', async () => {
    const c = { n: 0 };
    await transport(504, c).request('memories_create', { body: { document: 'x' } }).catch(() => null);
    expect(c.n).toBe(1);
  });

  it('a 503 is still retried', async () => {
    const c = { n: 0 };
    await transport(503, c).request('memories_create', { body: { document: 'x' } }).catch(() => null);
    expect(c.n).toBe(3);
  });
});
