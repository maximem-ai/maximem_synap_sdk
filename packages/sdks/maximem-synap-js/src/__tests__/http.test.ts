import { describe, it, expect, vi } from 'vitest';
import { HttpTransport, isOutcomeUnknown, DEFAULT_BASE_URL } from '../transport/http.js';
import { ENDPOINTS, resolvePath } from '../transport/endpoints.js';
import {
  AuthenticationError, ContextNotFoundError, InsufficientCreditsError,
  InvalidInputError, NetworkTimeoutError, RateLimitError,
  ServiceUnavailableError, ConflictError, SynapError,
} from '../errors.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'i' };

function transport(fetchImpl: typeof fetch, opts = {}) {
  return new HttpTransport({
    credentials: CREDS, fetchImpl,
    retryPolicy: { maxAttempts: 3, backoffBase: 0, backoffMax: 0, backoffJitter: false },
    ...opts,
  });
}
const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const err = (status: number, body: unknown = {}, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers });

describe('HttpTransport', () => {
  it('builds the URL, method and headers from the contract', async () => {
    const spy = vi.fn(async () => ok({ ok: true }));
    await transport(spy as unknown as typeof fetch).request('whoami');
    const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${DEFAULT_BASE_URL}/api/v1/auth/whoami`);
    expect(init.method).toBe('GET');
    const h = init.headers as Record<string, string>;
    expect(h['Authorization']).toBe('Bearer k');
    expect(h['X-Client-ID']).toBe('c');
    expect(h['X-Instance-ID']).toBe('i');
    expect(h['X-Correlation-ID']).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('preserves the /api/v1 vs /v1 prefix split', async () => {
    // Normalising these to one prefix 404s in production (gotcha G-E).
    expect(ENDPOINTS.whoami.path.startsWith('/api/v1/')).toBe(true);
    expect(ENDPOINTS.memories_create.path.startsWith('/api/v1/')).toBe(true);
    expect(ENDPOINTS.context_fetch_user.path.startsWith('/v1/')).toBe(true);
    expect(ENDPOINTS.context_fetch_user.path.startsWith('/api/')).toBe(false);
    expect(ENDPOINTS.telemetry_batch.path.startsWith('/v1/')).toBe(true);
  });

  it('never produces a double slash from a trailing-slash base URL', async () => {
    const spy = vi.fn(async () => ok({}));
    await transport(spy as unknown as typeof fetch, { baseUrl: 'https://x.test/' }).request('whoami');
    expect(spy.mock.calls[0]![0]).toBe('https://x.test/api/v1/auth/whoami');
  });

  it('substitutes and URL-encodes path parameters', () => {
    expect(resolvePath('memories_get', { memory_id: 'abc' })).toBe('/api/v1/memories/abc');
    // An unencoded slash would silently retarget the request at another route.
    expect(resolvePath('memories_get', { memory_id: 'a/b?c' })).toBe('/api/v1/memories/a%2Fb%3Fc');
    expect(() => resolvePath('memories_get', {})).toThrow(/requires path parameter/);
    expect(() => resolvePath('memories_get', { memory_id: '' })).toThrow(/requires path parameter/);
  });

  describe('status mapping', () => {
    it.each([
      [400, InvalidInputError], [422, InvalidInputError],
      [401, AuthenticationError], [403, AuthenticationError],
      [404, ContextNotFoundError], [402, InsufficientCreditsError],
      [409, ConflictError], [429, RateLimitError],
      [503, ServiceUnavailableError], [500, ServiceUnavailableError],
    ])('maps %i to the right error class', async (status, Klass) => {
      const t = transport((async () => err(status)) as unknown as typeof fetch);
      await expect(t.request('whoami')).rejects.toBeInstanceOf(Klass);
    });

    // The fixture is the server's real 402 body, copied from
    // synap/cloud/application/credits/enforcement.py. The previous version of
    // this test invented `required_credits`, the same key the parser was
    // wrongly reading, so both agreed with each other and disagreed with
    // production: requiredCredits was always null against a real server.
    it('extracts credit details from a 402', async () => {
      const t = transport((async () =>
        err(402, {
          detail: 'no credits',
          error: 'insufficient_credits',
          balance_credits: 3,
          minimum_required_credits: 10,
          recovery_url: '/v1/credits/balance',
          redeem_url: '/credits/redeem',
          support_contact: 'support@maximem.ai',
        })
      ) as unknown as typeof fetch);
      await t.request('whoami').catch((e: InsufficientCreditsError) => {
        expect(e.balanceCredits).toBe(3);
        expect(e.requiredCredits).toBe(10);
        expect(e.recoveryUrl).toBe('/v1/credits/balance');
        expect(e.redeemUrl).toBe('/credits/redeem');
        expect(e.message).toContain('no credits');
      });
      expect.assertions(5);
    });

    it('still reads the legacy required_credits key', async () => {
      const t = transport((async () =>
        err(402, { detail: 'no credits', balance_credits: 3, required_credits: 10 })
      ) as unknown as typeof fetch);
      await t.request('whoami').catch((e: InsufficientCreditsError) => {
        expect(e.requiredCredits).toBe(10);
        expect(e.recoveryUrl).toBeNull();
      });
      expect.assertions(2);
    });

    it('reads Retry-After from a 429', async () => {
      const t = transport((async () => err(429, {}, { 'retry-after': '12' })) as unknown as typeof fetch);
      await t.request('whoami').catch((e: RateLimitError) => {
        expect(e.retryAfterSeconds).toBe(12);
      });
      expect.assertions(1);
    });

    it('tolerates a missing or junk Retry-After', async () => {
      for (const headers of [{}, { 'retry-after': 'soon' }, { 'retry-after': '' }]) {
        const t = transport((async () => err(429, {}, headers)) as unknown as typeof fetch);
        await t.request('whoami').catch((e: RateLimitError) => {
          expect(e.retryAfterSeconds).toBeNull();
        });
      }
    });

    it('does not choke on a non-JSON error body', async () => {
      const t = transport((async () => new Response('<html>502</html>', { status: 502 })) as unknown as typeof fetch);
      await expect(t.request('whoami')).rejects.toBeInstanceOf(ServiceUnavailableError);
    });
  });

  describe('retry behaviour end to end', () => {
    it('retries a transient failure on a GET and succeeds', async () => {
      let n = 0;
      const spy = vi.fn(async () => (++n < 3 ? err(503) : ok({ done: true })));
      const out = await transport(spy as unknown as typeof fetch).request<{ done: boolean }>('whoami');
      expect(out.done).toBe(true);
      expect(spy).toHaveBeenCalledTimes(3);
    });

    it('gives up after maxAttempts', async () => {
      const spy = vi.fn(async () => err(503));
      await expect(transport(spy as unknown as typeof fetch).request('whoami')).rejects.toBeInstanceOf(ServiceUnavailableError);
      expect(spy).toHaveBeenCalledTimes(3);
    });

    it('does not retry a 401', async () => {
      const spy = vi.fn(async () => err(401));
      await expect(transport(spy as unknown as typeof fetch).request('whoami')).rejects.toBeInstanceOf(AuthenticationError);
      expect(spy).toHaveBeenCalledTimes(1);
    });

    // The billing-critical path.
    it('does NOT retry memories_create after an ambiguous timeout', async () => {
      const spy = vi.fn(async () => { throw Object.assign(new Error('aborted'), { name: 'AbortError' }); });
      const t = transport(spy as unknown as typeof fetch);
      await expect(t.request('memories_create', { body: { content: 'x' } }))
        .rejects.toBeInstanceOf(NetworkTimeoutError);
      // One attempt only: a retry here could ingest and bill twice.
      expect(spy).toHaveBeenCalledTimes(1);
    });

    it('DOES retry memories_create when the connection was refused', async () => {
      let n = 0;
      const spy = vi.fn(async () => {
        if (++n < 2) throw Object.assign(new Error('refused'), { cause: { code: 'ECONNREFUSED' } });
        return ok({ id: 'm1' });
      });
      const t = transport(spy as unknown as typeof fetch);
      await expect(t.request('memories_create', { body: {} })).resolves.toEqual({ id: 'm1' });
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it('retries a context fetch POST because it is declared idempotent', async () => {
      let n = 0;
      const spy = vi.fn(async () => (++n < 2 ? err(503) : ok({ items_by_type: {} })));
      await transport(spy as unknown as typeof fetch).request('context_fetch_user', { body: {} });
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it('reuses one correlation id across retries', async () => {
      let n = 0;
      const spy = vi.fn(async () => (++n < 3 ? err(503) : ok({})));
      await transport(spy as unknown as typeof fetch).request('whoami');
      const ids = spy.mock.calls.map((c) => ((c as unknown as [string, RequestInit])[1].headers as Record<string, string>)['X-Correlation-ID']);
      expect(new Set(ids).size).toBe(1);
    });
  });

  describe('response handling', () => {
    it('returns undefined for 204 and for an empty body', async () => {
      await expect(transport((async () => new Response(null, { status: 204 })) as unknown as typeof fetch).request('whoami')).resolves.toBeUndefined();
      await expect(transport((async () => new Response('', { status: 200 })) as unknown as typeof fetch).request('whoami')).resolves.toBeUndefined();
    });

    it('raises a clear error on malformed JSON in a 200', async () => {
      const t = transport((async () => new Response('{oops', { status: 200 })) as unknown as typeof fetch);
      await expect(t.request('whoami')).rejects.toThrow(/Malformed JSON/);
    });

    it('sends a JSON body on POST but not on GET', async () => {
      const spy = vi.fn(async () => ok({}));
      const t = transport(spy as unknown as typeof fetch);
      await t.request('memories_create', { body: { a: 1 } });
      expect((spy.mock.calls[0] as unknown as [string, RequestInit])[1].body).toBe('{"a":1}');
      spy.mockClear();
      await t.request('whoami', { body: { a: 1 } });
      expect((spy.mock.calls[0] as unknown as [string, RequestInit])[1].body).toBeUndefined();
    });

    it('appends query parameters and drops undefined ones', async () => {
      const spy = vi.fn(async () => ok({}));
      await transport(spy as unknown as typeof fetch)
        .request('credits_ledger', { query: { limit: 10, cursor: undefined, active: true } });
      const url = spy.mock.calls[0]![0] as string;
      expect(url).toContain('limit=10');
      expect(url).toContain('active=true');
      expect(url).not.toContain('cursor');
    });
  });

  describe('outcome classification', () => {
    it('treats definitive HTTP answers as known', () => {
      for (const e of [
        new RateLimitError('x'), new ServiceUnavailableError('x'), new AuthenticationError('x'),
        new InvalidInputError('x'), new ConflictError('x'), new InsufficientCreditsError('x'),
        new ContextNotFoundError('x'),
      ]) expect(isOutcomeUnknown(e)).toBe(false);
    });

    it('treats timeouts as ambiguous', () => {
      expect(isOutcomeUnknown(new NetworkTimeoutError('x'))).toBe(true);
    });

    it('treats connection-level failures as known-not-processed', () => {
      for (const code of ['ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'ECONNRESET', 'EHOSTUNREACH']) {
        expect(isOutcomeUnknown(Object.assign(new Error('x'), { cause: { code } }))).toBe(false);
      }
    });

    it('defaults an unrecognised error to ambiguous', () => {
      // Biased deliberately: an unknown failure must not license a retry that
      // could double-bill.
      expect(isOutcomeUnknown(new Error('mystery'))).toBe(true);
      expect(isOutcomeUnknown(new SynapError('mystery'))).toBe(true);
    });
  });

  it('close() is idempotent and stops the heartbeat', () => {
    const t = transport((async () => ok({})) as unknown as typeof fetch);
    t.startHeartbeat();
    t.close();
    t.close();
    t.startHeartbeat(); // must not restart after close
  });
});
