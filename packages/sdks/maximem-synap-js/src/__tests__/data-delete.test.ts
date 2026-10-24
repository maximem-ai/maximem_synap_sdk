/**
 * data.delete: validation, request shape, wait loop, local clear, errors.
 *
 * Mirrors synap/tests/sdk/data/test_delete.py. Messages and body key order
 * must match the Python SDK.
 */
import { describe, it, expect, vi } from 'vitest';
import { HttpTransport, DEFAULT_BASE_URL } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { buildDeleteBody, createDataInterface, MAX_WAIT_CALLS } from '../data/interface.js';
import { AuthenticationError, InvalidInputError, RateLimitError } from '../errors.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_test' };
const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const err = (status: number) =>
  new Response(JSON.stringify({ detail: 'boom' }), { status, headers: { 'content-type': 'application/json' } });

function data(fetchImpl: typeof fetch, onDeleted?: Parameters<typeof createDataInterface>[1]) {
  const t = new HttpTransport({
    credentials: CREDS,
    fetchImpl,
    retryPolicy: { maxAttempts: 1, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
  return createDataInterface(t, onDeleted);
}

describe('contract entries', () => {
  it('data_delete is an idempotent POST under /v1', () => {
    expect(ENDPOINTS.data_delete).toEqual({ method: 'POST', path: '/v1/data/delete', idempotent: true });
    expect(ENDPOINTS.data_delete_status.path).toBe('/v1/data/delete/{deletion_id}');
  });
});

describe('validation (same messages as Python)', () => {
  const cases: Array<[Record<string, unknown>, string]> = [
    [{ scope: 'client', user_id: 'u' }, "Invalid scope 'client'"],
    [{ scope: { customer: 'a' }, user_id: 'u' }, 'A dict is not accepted'],
    [{ scope: 'user', confirm: 'u' }, "user_id is required for scope='user'"],
    [{ scope: 'customer', user_id: 'u', confirm: 'u' }, 'customer_id is required'],
    [{ scope: 'conversation', user_id: 'u', confirm: 'c' }, 'conversation_id is required'],
    [{ scope: 'user', user_id: '  ', confirm: 'x' }, 'user_id must not be empty'],
    [{ scope: 'user', user_id: 'u' }, 'confirm is required unless dry_run=True'],
    [{ scope: 'user', user_id: 'u', confirm: 'other' }, 'confirm must equal the user_id'],
    [{ scope: 'conversation', user_id: 'u', conversation_id: 'c', confirm: 'u' },
      'confirm must equal the conversation_id'],
    [{ scope: 'user', user_id: 'u', confirm: 'u', dry_run: 'yes' }, 'dry_run must be a boolean'],
  ];
  it.each(cases)('refuses %j before any request', async (opts, msg) => {
    const spy = vi.fn(async () => ok({}));
    const d = data(spy as unknown as typeof fetch);
    await expect(d.delete(opts as never)).rejects.toThrow(msg);
    await expect(d.delete(opts as never)).rejects.toBeInstanceOf(InvalidInputError);
    expect(spy).not.toHaveBeenCalled();
  });

  it('dry run needs no confirm', () => {
    expect(() => buildDeleteBody({ scope: 'user', userId: 'u', dryRun: true }, 'i')).not.toThrow();
  });
});

describe('request', () => {
  it('sends every key, in Python order, with the instance', async () => {
    const spy = vi.fn(async () => ok({ status: 'complete', deletion_id: 'd1' }));
    await data(spy as unknown as typeof fetch)
      .delete({ scope: 'user', userId: 'u1', customerId: 'c1', confirm: 'u1' });
    const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${DEFAULT_BASE_URL}/v1/data/delete`);
    expect(init.method).toBe('POST');
    const body = JSON.parse(init.body as string);
    expect(body).toEqual({
      scope: 'user', instance_id: 'inst_test', user_id: 'u1', customer_id: 'c1',
      conversation_id: null, dry_run: false, confirm: 'u1',
    });
    expect(Object.keys(body)).toEqual([
      'scope', 'instance_id', 'user_id', 'customer_id', 'conversation_id', 'dry_run', 'confirm']);
  });

  it('status GETs the audit record', async () => {
    const spy = vi.fn(async () => ok({ status: 'partial', still_present_in: ['graph'] }));
    const d = data(spy as unknown as typeof fetch);
    const r = await d.status('d-1');
    const url = (spy.mock.calls[0] as unknown as [string])[0];
    expect(url.startsWith(`${DEFAULT_BASE_URL}/v1/data/delete/d-1?instance_id=`)).toBe(true);
    expect(r.still_present_in).toEqual(['graph']);
    await expect(d.status('')).rejects.toBeInstanceOf(InvalidInputError);
  });

  it('a partial result is returned, not thrown, with every field filled', async () => {
    const r = await data((async () => ok({
      status: 'partial', fully_removed: false, still_present_in: ['graph'],
    })) as unknown as typeof fetch).delete({ scope: 'user', user_id: 'u', confirm: 'u' });
    expect(r.fully_removed).toBe(false);
    expect(r.continue).toBe(false);
    expect(r.counts).toEqual({});
    expect(r.local_cleared).toEqual([]);
  });
});

describe('wait loop', () => {
  const opts = { scope: 'customer', customer_id: 'c', confirm: 'c' } as const;

  it('repeats while the server says continue', async () => {
    const answers = [
      { status: 'in_progress', continue: true },
      { status: 'in_progress', continue: true },
      { status: 'complete', continue: false },
    ];
    const spy = vi.fn(async () => ok(answers.shift()));
    const r = await data(spy as unknown as typeof fetch).delete(opts);
    expect(spy).toHaveBeenCalledTimes(3);
    expect(r.status).toBe('complete');
  });

  it('wait:false makes one call', async () => {
    const spy = vi.fn(async () => ok({ status: 'in_progress', continue: true }));
    const r = await data(spy as unknown as typeof fetch).delete({ ...opts, wait: false });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(r.continue).toBe(true);
  });

  it('gives up after 50 calls', async () => {
    const spy = vi.fn(async () => ok({ status: 'in_progress', continue: true }));
    const r = await data(spy as unknown as typeof fetch).delete(opts);
    expect(spy).toHaveBeenCalledTimes(MAX_WAIT_CALLS);
    expect(MAX_WAIT_CALLS).toBe(50);
    expect(r.continue).toBe(true);
  });
});

describe('local clear', () => {
  it('runs after a real delete and fills local_cleared', async () => {
    const hook = vi.fn(() => ['context_cache']);
    const r = await data((async () => ok({ status: 'complete' })) as unknown as typeof fetch, hook)
      .delete({ scope: 'conversation', user_id: 'u', conversation_id: 'cv', confirm: 'cv' });
    expect(hook).toHaveBeenCalledWith({
      scope: 'conversation', userId: 'u', customerId: null, conversationId: 'cv',
    });
    expect(r.local_cleared).toEqual(['context_cache']);
  });

  it('does not run for a dry run', async () => {
    const hook = vi.fn(() => ['x']);
    await data((async () => ok({ status: 'dry_run' })) as unknown as typeof fetch, hook)
      .delete({ scope: 'user', user_id: 'u', dry_run: true });
    expect(hook).not.toHaveBeenCalled();
  });
});

describe('errors', () => {
  it('maps 403 and 429', async () => {
    const o = { scope: 'user', user_id: 'u', confirm: 'u' } as const;
    await expect(data((async () => err(403)) as unknown as typeof fetch).delete(o))
      .rejects.toBeInstanceOf(AuthenticationError);
    await expect(data((async () => err(429)) as unknown as typeof fetch).delete(o))
      .rejects.toBeInstanceOf(RateLimitError);
  });
});


describe('pessimistic defaults', () => {
  it('fully_removed is false unless the server said true', async () => {
    const r = await data((async () => ok({})) as unknown as typeof fetch)
      .delete({ scope: 'user', user_id: 'u', customer_id: 'c', dry_run: true });
    expect(r.fully_removed).toBe(false);
    expect(r.status).toBe('unknown');
  });
});
