import { describe, it, expect } from 'vitest';
import { shouldRetry, backoffDelay, DEFAULT_RETRY_POLICY } from '../transport/retry.js';
import {
  NetworkTimeoutError, RateLimitError, ServiceUnavailableError,
  AuthenticationError, InsufficientCreditsError, InvalidInputError,
} from '../errors.js';

const P = DEFAULT_RETRY_POLICY;
const decide = (o: Partial<Parameters<typeof shouldRetry>[0]>) =>
  shouldRetry({
    error: new ServiceUnavailableError('x'),
    attempt: 1, policy: P, method: 'GET', outcomeUnknown: false, ...o,
  });

describe('retry policy', () => {
  it('retries transient errors on idempotent methods', () => {
    for (const method of ['GET', 'HEAD', 'OPTIONS', 'PUT', 'DELETE']) {
      expect(decide({ method, outcomeUnknown: true }).retry).toBe(true);
    }
  });

  it('never retries permanent errors', () => {
    for (const error of [
      new AuthenticationError('x'), new InvalidInputError('x'), new InsufficientCreditsError('x'),
    ]) {
      expect(decide({ error }).retry).toBe(false);
      expect(decide({ error }).reason).toBe('not_transient');
    }
  });

  it('never retries a non-Synap error', () => {
    expect(decide({ error: new Error('boom') }).retry).toBe(false);
    expect(decide({ error: 'a string' }).retry).toBe(false);
    expect(decide({ error: null }).retry).toBe(false);
  });

  it('stops at maxAttempts', () => {
    expect(decide({ attempt: 3 }).retry).toBe(false);
    expect(decide({ attempt: 3 }).reason).toBe('max_attempts_exhausted');
    expect(decide({ attempt: 2 }).retry).toBe(true);
  });

  // ── The double-billing gate (defect D / gotcha G-L) ─────────────────────────
  describe('non-idempotent methods', () => {
    it('retries POST when the request provably never landed', () => {
      // A connection-level failure: the server never saw it, so nothing was
      // ingested and nothing was billed.
      const d = decide({ method: 'POST', outcomeUnknown: false });
      expect(d.retry).toBe(true);
    });

    it('refuses to retry POST when the outcome is unknown', () => {
      // A read timeout. The ingest may have completed and only the response
      // been lost; retrying would store and bill twice.
      const d = decide({
        method: 'POST', outcomeUnknown: true, error: new NetworkTimeoutError('x'),
      });
      expect(d.retry).toBe(false);
      expect(d.reason).toBe('non_idempotent_outcome_unknown');
    });

    it('honours an explicit idempotent override for read-only POSTs', () => {
      // /v1/context/*/fetch is POST for the body but is a read.
      const d = decide({ method: 'POST', outcomeUnknown: true, idempotent: true });
      expect(d.retry).toBe(true);
    });

    it('can mark an idempotent-by-verb route as unsafe', () => {
      expect(decide({ method: 'PUT', outcomeUnknown: true, idempotent: false }).retry).toBe(false);
    });

    it('is case-insensitive about the method', () => {
      expect(decide({ method: 'post', outcomeUnknown: true }).retry).toBe(false);
      expect(decide({ method: 'get', outcomeUnknown: true }).retry).toBe(true);
    });
  });

  describe('backoff', () => {
    const fixed = { ...P, backoffJitter: false };

    it('grows exponentially and caps', () => {
      const e = new ServiceUnavailableError('x');
      expect(backoffDelay(e, 1, fixed)).toBe(1);
      expect(backoffDelay(e, 2, fixed)).toBe(2);
      expect(backoffDelay(e, 3, fixed)).toBe(4);
      expect(backoffDelay(e, 9, fixed)).toBe(fixed.backoffMax);
    });

    it('honours Retry-After over its own guess', () => {
      // Coming back sooner than told just gets rejected again and burns quota.
      const e = new RateLimitError('x', { retryAfterSeconds: 7 });
      expect(backoffDelay(e, 1, fixed)).toBe(7);
    });

    it('still caps a hostile Retry-After', () => {
      const e = new RateLimitError('x', { retryAfterSeconds: 9999 });
      expect(backoffDelay(e, 1, fixed)).toBe(fixed.backoffMax);
    });

    it('keeps jitter within [0, cap]', () => {
      const e = new ServiceUnavailableError('x');
      for (let i = 0; i < 200; i++) {
        const d = backoffDelay(e, 3, P);
        expect(d).toBeGreaterThanOrEqual(0);
        expect(d).toBeLessThanOrEqual(4);
      }
    });
  });
});

describe('retry backoff holds the event loop', () => {
  it('settles a retried request when the backoff is the only pending work', async () => {
    // Regression: the backoff sleep was `unref`'d, copying the heartbeat
    // pattern. A heartbeat must not hold the process open; a backoff must,
    // because a caller is awaiting it. With it unref'd a process with no other
    // work exited mid-backoff and the request never settled.
    const { HttpTransport } = await import('../transport/http.js');
    let attempts = 0;
    const transport = new HttpTransport({
      credentials: { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' },
      retryPolicy: { maxAttempts: 2, backoffBase: 0.05, backoffJitter: false },
      fetchImpl: (async () => {
        attempts += 1;
        return attempts === 1
          ? new Response(JSON.stringify({ detail: 'transient' }), { status: 503 })
          : new Response(JSON.stringify({ ok: true }), {
              status: 200, headers: { 'content-type': 'application/json' },
            });
      }) as unknown as typeof fetch,
    });

    const result = await transport.request<{ ok: boolean }>('whoami', {});
    expect(attempts).toBe(2);
    expect(result.ok).toBe(true);
  });
});
