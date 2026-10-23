/**
 * Configuration page findings.
 *
 * - retryPolicy.maxAttempts 0 sent no request and threw "Request failed with
 *   no recorded error".
 * - apiKey: "" did not fall back to SYNAP_API_KEY (`??`), while Python's
 *   `api_key or os.environ.get(...)` does.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { HttpTransport } from '../transport/http.js';
import { ServiceUnavailableError } from '../errors.js';
import { SynapClient } from '../client.js';

afterEach(() => {
  delete process.env['SYNAP_API_KEY'];
});

describe('retryPolicy.maxAttempts below 1', () => {
  it('still makes one attempt and throws the real error', async () => {
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      return new Response(JSON.stringify({ detail: 'down' }), { status: 503 });
    }) as typeof fetch;
    const t = new HttpTransport({
      credentials: { apiKey: 'k', clientId: 'c', instanceId: 'i' }, fetchImpl,
      retryPolicy: { maxAttempts: 0, backoffBase: 0, backoffMax: 0, backoffJitter: false },
    });
    const err = await t.request('whoami').then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(ServiceUnavailableError);
    expect(calls).toBe(1);
  });
});

describe('apiKey: ""', () => {
  it('falls back to SYNAP_API_KEY, as Python does', () => {
    process.env['SYNAP_API_KEY'] = 'synap_from_env';
    expect(() => new SynapClient({ _force_new: true, apiKey: '', instanceId: 'inst_0123456789abcdef' } as never))
      .not.toThrow();
  });
});
