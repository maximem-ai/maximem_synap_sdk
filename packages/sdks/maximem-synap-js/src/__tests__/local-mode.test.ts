/**
 * Local mode: the SDK against the open-source server, which has no auth.
 *
 * On when there is no API key (option or SYNAP_API_KEY) and a base URL is set.
 * Mirrors `python/tests/test_local_mode.py`. JS has no telemetry sender (it
 * reports over the gRPC stream), so there is no telemetry case here.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SynapClient } from '../index.js';
import {
  AuthenticationError, ContextNotFoundError, ServiceUnavailableError, SynapPermanentError,
} from '../errors.js';
import { GrpcStreamClient } from '../grpc/stream-client.js';

const ENV = [
  'SYNAP_API_KEY', 'SYNAP_API_TOKEN', 'SYNAP_BASE_URL', 'SYNAP_CLIENT_ID', 'SYNAP_INSTANCE_ID',
];
beforeEach(() => { for (const k of ENV) delete process.env[k]; });
afterEach(() => { for (const k of ENV) delete process.env[k]; vi.restoreAllMocks(); });

const CLOUD_ONLY = 'Not available in the open-source server';
const BASE = 'http://core.test';

type Call = { url: string; headers: Record<string, string> };

function fakeCore(opts: { whoami?: () => Response } = {}) {
  const calls: Call[] = [];
  const fetchImpl = (async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), headers: { ...(init?.headers as Record<string, string>) } });
    const path = new URL(String(url)).pathname;
    if (path === '/api/v1/auth/whoami') {
      if (opts.whoami) return opts.whoami();
      return Response.json({
        client_id: 'cli_core', instance_id: 'inst_0123456789abcdef',
        user_context_isolation: 'equals_customer',
      });
    }
    if (path.startsWith('/v1/credits/')) {
      return Response.json({ detail: CLOUD_ONLY }, { status: 404 });
    }
    if (path === '/v1/data/delete') {
      return Response.json({ detail: CLOUD_ONLY }, { status: 501 });
    }
    return Response.json({});
  }) as typeof fetch;
  return { calls, fetchImpl };
}

const local = (fetchImpl: typeof fetch, extra: Record<string, unknown> = {}) =>
  new SynapClient({ _force_new: true, baseUrl: BASE, fetchImpl, retryPolicy: null, ...extra });

describe('local mode on', () => {
  it('initialises with no key and learns identity from whoami', async () => {
    const { calls, fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    expect(c.client_id).toBe('cli_core');
    expect(c.instance_id).toBe('inst_0123456789abcdef');
    expect(calls.map((x) => new URL(x.url).pathname)).toEqual(['/api/v1/auth/whoami']);
  });

  it('turns on from SYNAP_BASE_URL too', async () => {
    process.env['SYNAP_BASE_URL'] = BASE;
    const { fetchImpl } = fakeCore();
    const c = new SynapClient({ _force_new: true, fetchImpl });
    await c.initialize();
    expect(c.instance_id).toBe('inst_0123456789abcdef');
  });

  it('fails as before with no key and no base URL', () => {
    expect(() => new SynapClient({ _force_new: true })).toThrow(AuthenticationError);
    expect(() => new SynapClient({ _force_new: true })).toThrow(/No Synap API key found/);
  });

  it('treats an empty SYNAP_API_KEY as no key', async () => {
    process.env['SYNAP_API_KEY'] = '';
    const { fetchImpl } = fakeCore();
    await expect(local(fetchImpl).initialize()).resolves.toBeUndefined();
  });

  it('sends SYNAP_API_TOKEN as a Bearer token on every request', async () => {
    process.env['SYNAP_API_TOKEN'] = 'tok-1';
    const { calls, fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    await expect(c.credits.get_balance()).rejects.toBeInstanceOf(ContextNotFoundError);
    expect(calls).toHaveLength(2);
    expect(calls.map((x) => x.headers['Authorization'])).toEqual(['Bearer tok-1', 'Bearer tok-1']);
  });

  it('sends no Authorization header at all without a token', async () => {
    const { calls, fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    await expect(c.credits.get_balance()).rejects.toBeInstanceOf(ContextNotFoundError);
    expect(calls).toHaveLength(2);
    for (const call of calls) expect(Object.keys(call.headers)).not.toContain('Authorization');
  });

  it('throws when whoami cannot be reached', async () => {
    const fetchImpl = (async () => { throw new TypeError('fetch failed'); }) as typeof fetch;
    const c = local(fetchImpl);
    await expect(c.initialize()).rejects.toBeInstanceOf(AuthenticationError);
    await expect(c.initialize()).rejects.toThrow(/Could not reach the Synap server at http:\/\/core\.test/);
  });

  it('throws when whoami answers with an error status', async () => {
    const { fetchImpl } = fakeCore({ whoami: () => Response.json({ detail: 'no' }, { status: 500 }) });
    await expect(local(fetchImpl).initialize()).rejects.toThrow(/whoami failed/);
  });

  it('throws when whoami returns no identity', async () => {
    const { fetchImpl } = fakeCore({ whoami: () => Response.json({}) });
    await expect(local(fetchImpl).initialize()).rejects.toThrow(/did not identify itself/);
  });

  it('does not stay silent on the first call either, and asks again next time', async () => {
    let up = false;
    const { fetchImpl: ok } = fakeCore();
    const fetchImpl = (async (u: string, i?: RequestInit) => {
      if (!up) throw new TypeError('fetch failed');
      return ok(u, i);
    }) as typeof fetch;
    const c = local(fetchImpl);
    await expect(c.credits.get_balance()).rejects.toBeInstanceOf(AuthenticationError);
    up = true;
    await expect(c.credits.get_balance()).rejects.toBeInstanceOf(ContextNotFoundError);
  });

  it('listen() throws the open-source message and opens no channel', async () => {
    const connect = vi.spyOn(GrpcStreamClient.prototype, 'connect');
    const { fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    const error = await c.instance.listen().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(SynapPermanentError);
    expect((error as Error).message).toContain(CLOUD_ONLY);
    expect(connect).not.toHaveBeenCalled();
    expect(c.instance.is_listening).toBe(false);
  });

  it('surfaces a Cloud-only 404 body in the exception message', async () => {
    const { fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    const error = await c.credits.get_balance().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ContextNotFoundError);
    expect((error as Error).message).toContain(CLOUD_ONLY);
  });

  it('surfaces a Cloud-only 501 body in the exception message', async () => {
    const { fetchImpl } = fakeCore();
    const c = local(fetchImpl);
    await c.initialize();
    const error = await c.data
      .delete({ scope: 'user', user_id: 'u1', dry_run: true })
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ServiceUnavailableError);
    expect((error as Error).message).toContain(CLOUD_ONLY);
  });
});

describe('with a key nothing changes', () => {
  it('sends the key as the Bearer token, even when a token is set', async () => {
    process.env['SYNAP_API_TOKEN'] = 'tok-should-not-be-used';
    const { calls, fetchImpl } = fakeCore();
    const c = local(fetchImpl, { apiKey: 'sk-cloud' });
    await c.initialize();
    expect(calls.map((x) => x.headers['Authorization'])).toEqual(['Bearer sk-cloud']);
  });

  it('keeps a failed whoami non-fatal', async () => {
    const fetchImpl = (async () => { throw new TypeError('fetch failed'); }) as typeof fetch;
    const c = local(fetchImpl, { apiKey: 'sk-cloud' });
    await expect(c.initialize()).resolves.toBeUndefined();
  });

  it('does not block listen() before it tries to open a stream', async () => {
    const connect = vi.spyOn(GrpcStreamClient.prototype, 'connect').mockRejectedValue(new Error('stub'));
    const { fetchImpl } = fakeCore();
    const c = local(fetchImpl, { apiKey: 'sk-cloud' });
    await c.initialize();
    await c.instance.listen().catch(() => undefined);
    expect(connect).toHaveBeenCalled();
  });
});
