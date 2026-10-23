/**
 * memories.forget: request shape, response parsing, error mapping.
 *
 * Mirrors synap/tests/sdk/memories/test_forget.py. The body must match what the
 * Python SDK sends, key set included.
 */
import { describe, it, expect, vi } from 'vitest';
import { HttpTransport, DEFAULT_BASE_URL } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { buildForgetBody, createMemoriesInterface } from '../memories/interface.js';
import {
  InvalidInputError, RateLimitError, ServiceUnavailableError,
} from '../errors.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'i' };

function memories(fetchImpl: typeof fetch, retry = 3) {
  const t = new HttpTransport({
    credentials: CREDS,
    fetchImpl,
    retryPolicy: { maxAttempts: retry, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
  return createMemoriesInterface(t);
}
const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const err = (status: number, body: unknown = { detail: 'boom' }) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

describe('contract entry', () => {
  it('is a non-idempotent POST under /api/v1', () => {
    expect(ENDPOINTS.memories_forget).toEqual({
      method: 'POST', path: '/api/v1/memories/forget', idempotent: false,
    });
  });
});

describe('forget request body', () => {
  it('sends every key, nulls included, and defaults mode to preview', () => {
    const body = buildForgetBody({ description: 'my old address', user_id: 'u1', customer_id: 'c1' });
    expect(body).toEqual({
      description: 'my old address', user_id: 'u1', customer_id: 'c1', scope: null, mode: 'preview',
    });
    expect(Object.keys(body)).toEqual(['description', 'user_id', 'customer_id', 'scope', 'mode']);
  });

  it('accepts camelCase ids, scope and direct mode', () => {
    const scope = { customer: 'c1', team: 't1', user: 'u1' };
    const body = buildForgetBody({
      description: 'x y z', userId: 'u1', customerId: 'c1', scope, mode: 'direct',
    });
    expect(body).toEqual({
      description: 'x y z', user_id: 'u1', customer_id: 'c1', scope, mode: 'direct',
    });
  });

  it.each(['', 'ab', 'x'.repeat(501)])('rejects description %j before any request', (d) => {
    expect(() => buildForgetBody({ description: d, user_id: 'u1' })).toThrow(InvalidInputError);
  });

  it('counts code points, as Python and the server do', () => {
    // 500 emoji: 500 code points, 1000 UTF-16 units. Was refused here only.
    expect(() => buildForgetBody({ description: '\u{1F600}'.repeat(500), user_id: 'u1' })).not.toThrow();
    expect(() => buildForgetBody({ description: '\u{1F600}'.repeat(501), user_id: 'u1' }))
      .toThrow(InvalidInputError);
  });

  it('rejects an unknown mode before any request', () => {
    expect(() => buildForgetBody({ description: 'my address', mode: 'purge' }))
      .toThrow(InvalidInputError);
  });
});

describe('memories.forget over the transport', () => {
  it('POSTs to the forget route with the body', async () => {
    const spy = vi.fn(async () => ok({ status: 'nothing_matches' }));
    await memories(spy as unknown as typeof fetch)
      .forget({ description: 'my old address', user_id: 'u1', customer_id: 'c1' });
    const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${DEFAULT_BASE_URL}/api/v1/memories/forget`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      description: 'my old address', user_id: 'u1', customer_id: 'c1', scope: null, mode: 'preview',
    });
  });

  it('returns the parsed response untouched', async () => {
    const payload = {
      status: 'partial', request_id: null,
      candidates: [{ memory_id: 'm1', content: 'lives at 1 Main St', memory_type: 'fact' }],
      deleted_ids: ['m1'], failed_ids: ['m2'], still_present_in: ['graph'],
      may_be_incomplete: true,
    };
    const r = await memories((async () => ok(payload)) as unknown as typeof fetch)
      .forget({ description: 'my address', user_id: 'u1', mode: 'direct' });
    expect(r).toEqual(payload);
    expect(r.status).not.toBe('deleted');
  });

  it('maps 400 and 422 to InvalidInputError', async () => {
    for (const status of [400, 422]) {
      const m = memories((async () => err(status)) as unknown as typeof fetch);
      await expect(m.forget({ description: 'my address' })).rejects.toBeInstanceOf(InvalidInputError);
    }
  });

  it('maps 429 to RateLimitError', async () => {
    const m = memories((async () => err(429)) as unknown as typeof fetch, 1);
    await expect(m.forget({ description: 'my address', user_id: 'u1' }))
      .rejects.toBeInstanceOf(RateLimitError);
  });

  it('maps 503 to ServiceUnavailableError once retries are spent', async () => {
    const spy = vi.fn(async () => err(503, { detail: 'matcher unavailable' }));
    const m = memories(spy as unknown as typeof fetch, 1);
    await expect(m.forget({ description: 'my address', user_id: 'u1', mode: 'direct' }))
      .rejects.toBeInstanceOf(ServiceUnavailableError);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('applies the normal retry policy to a 503 rejection', async () => {
    // Same rule as every other POST: a 503 is a rejection issued before the
    // request was processed, so a retry cannot double-act.
    let n = 0;
    const spy = vi.fn(async () => (++n === 1 ? err(503) : ok({ status: 'nothing_matches' })));
    const r = await memories(spy as unknown as typeof fetch)
      .forget({ description: 'my address', user_id: 'u1' });
    expect(r.status).toBe('nothing_matches');
    expect(spy).toHaveBeenCalledTimes(2);
  });
});

describe('forget evicts the local cache', () => {
  const build = (result: unknown, onForgotten: (ids: unknown) => void) => {
    const t = new HttpTransport({
      credentials: CREDS,
      fetchImpl: (async () => ok(result)) as unknown as typeof fetch,
      retryPolicy: { maxAttempts: 1, backoffBase: 0, backoffMax: 0, backoffJitter: false },
    });
    return createMemoriesInterface(t, onForgotten as never);
  };

  it('evicts that user after a forget that deleted something', async () => {
    const spy = vi.fn();
    await build({ status: 'deleted', deleted_ids: ['m1'] }, spy)
      .forget({ description: 'my phone number', user_id: 'u1', customer_id: 'c1', mode: 'direct' });
    expect(spy).toHaveBeenCalledWith({ userId: 'u1', customerId: 'c1' });
  });

  it('evicts after a forget that removed only copies', async () => {
    const spy = vi.fn();
    await build({ status: 'deleted', deleted_ids: [] }, spy)
      .forget({ description: 'my phone number', user_id: 'u1', mode: 'direct' });
    expect(spy).toHaveBeenCalledWith({ userId: 'u1', customerId: null });
  });

  it('still evicts after a partial forget', async () => {
    const spy = vi.fn();
    await build({ status: 'partial', deleted_ids: ['m1'], still_present_in: ['graph'] }, spy)
      .forget({ description: 'my phone number', user_id: 'u1', mode: 'direct' });
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('evicts nothing for a preview or no match', async () => {
    for (const result of [{ status: 'preview', candidates: [] }, { status: 'nothing_matches' }]) {
      const spy = vi.fn();
      await build(result, spy).forget({ description: 'my phone number', user_id: 'u1' });
      expect(spy).not.toHaveBeenCalled();
    }
  });

  it('never fails a successful delete because eviction threw', async () => {
    const r = await build({ status: 'deleted', deleted_ids: ['m1'] }, () => { throw new Error('cache'); })
      .forget({ description: 'my phone number', user_id: 'u1', mode: 'direct' });
    expect(r.status).toBe('deleted');
  });
});
