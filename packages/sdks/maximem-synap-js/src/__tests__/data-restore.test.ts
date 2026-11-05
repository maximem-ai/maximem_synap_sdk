/**
 * data.restore and the pending_purge status: validation, request shape, local clear.
 *
 * Mirrors test_python_sdk_restore_validates_and_posts in
 * synap/tests/application/soft_delete/test_stage2_restore_pg.py. Messages and
 * body key order must match the Python SDK.
 */
import { describe, it, expect, vi } from 'vitest';
import { HttpTransport } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { buildDeleteBody, buildRestoreBody, createDataInterface, MAX_RESTORE_IDS } from '../data/interface.js';
import { InvalidInputError } from '../errors.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_test' };
const ok = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

function data(fetchImpl: typeof fetch, onDeleted?: Parameters<typeof createDataInterface>[1]) {
  const t = new HttpTransport({
    credentials: CREDS,
    fetchImpl,
    retryPolicy: { maxAttempts: 1, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
  return createDataInterface(t, onDeleted);
}

describe('data.restore', () => {
  it('is an idempotent POST under /v1', () => {
    expect(ENDPOINTS.data_restore).toEqual({ method: 'POST', path: '/v1/data/restore', idempotent: true });
  });

  const bad: Array<[unknown, string]> = [
    ['one-id', 'memory_ids must be a list of memory ids'],
    [[], 'memory_ids must not be empty'],
    [[''], 'memory_ids must not contain empty ids'],
    [Array(MAX_RESTORE_IDS + 1).fill('a'), 'at most 200 memory_ids per call'],
  ];
  it.each(bad)('refuses %j before any request', async (ids, msg) => {
    const spy = vi.fn(async () => ok({}));
    await expect(data(spy as unknown as typeof fetch).restore(ids as string[]))
      .rejects.toThrow(InvalidInputError);
    expect(() => buildRestoreBody(ids, 'i')).toThrow(msg);
    expect(spy).not.toHaveBeenCalled();
  });

  it('posts the ids with the instance and fills the result', async () => {
    const spy = vi.fn(async (_url: unknown, _init?: RequestInit) =>
      ok({ restored_ids: ['m1'], not_restorable_ids: ['m2'] }));
    const res = await data(spy as unknown as typeof fetch).restore(['m1', 'm2']);
    const [url, init] = spy.mock.calls[0] as [string, RequestInit];
    expect(String(url)).toContain('/v1/data/restore');
    expect(JSON.parse(String(init.body))).toEqual({ memory_ids: ['m1', 'm2'], instance_id: 'inst_test' });
    expect(res.restored_ids).toEqual(['m1']);
    expect(res.not_restorable_ids).toEqual(['m2']);
    expect(res.still_missing_in).toEqual([]);
    expect(res.fully_restored).toBe(false);
  });
});

describe('B1 stage 4 options', () => {
  it('restores by deletion_id, and refuses both or neither', async () => {
    const spy = vi.fn(async (_url: unknown, _init?: RequestInit) => ok({ restored_ids: ['a'] }));
    await data(spy as unknown as typeof fetch).restore({ deletionId: 'del-1' });
    const [, init] = spy.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({ deletion_id: 'del-1', instance_id: 'inst_test' });
    expect(() => buildRestoreBody(['a'], 'i', 'del-1')).toThrow('send exactly one of memory_ids or deletion_id');
    expect(() => buildRestoreBody(null, 'i', null)).toThrow('send exactly one of memory_ids or deletion_id');
    expect(() => buildRestoreBody(null, 'i', ' ')).toThrow('deletion_id must not be empty');
  });

  it('sends permanent only when true, so older bodies are unchanged', () => {
    const base = { scope: 'user', user_id: 'u', confirm: 'u' };
    expect(buildDeleteBody(base, 'i')).not.toHaveProperty('permanent');
    expect(buildDeleteBody({ ...base, permanent: true }, 'i')).toMatchObject({ permanent: true });
    expect(() => buildDeleteBody({ ...base, permanent: 'yes' as unknown as boolean }, 'i'))
      .toThrow('permanent must be a boolean');
  });
});

describe('pending_purge', () => {
  it('clears local copies and carries the purge date', async () => {
    const spy = vi.fn(async () => ok({ status: 'pending_purge', purge_after: '2026-12-03T00:00:00Z' }));
    const onDeleted = vi.fn(() => ['cache']);
    const res = await data(spy as unknown as typeof fetch, onDeleted)
      .delete({ scope: 'user', user_id: 'u', confirm: 'u' });
    expect(res.status).toBe('pending_purge');
    expect(res.purge_after).toBe('2026-12-03T00:00:00Z');
    expect(onDeleted).toHaveBeenCalledOnce();
    expect(res.local_cleared).toEqual(['cache']);
  });
});
