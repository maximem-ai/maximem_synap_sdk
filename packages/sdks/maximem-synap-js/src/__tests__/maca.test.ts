/**
 * maca: the MACA plugin calls of a Synap Core server.
 * Mirrors synap/tests/sdk/test_maca_interface.py; messages must match the Python SDK.
 */
import { describe, it, expect, vi } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { HttpTransport } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { createMacaNamespace } from '../maca/interface.js';
import { InvalidInputError } from '../errors.js';

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_test' };
const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

function maca(fetchImpl: typeof fetch) {
  const t = new HttpTransport({
    credentials: CREDS,
    fetchImpl,
    retryPolicy: { maxAttempts: 1, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
  return createMacaNamespace(t);
}

describe('contract entries', () => {
  it('generate is a non-idempotent POST and the job read an idempotent GET', () => {
    expect(ENDPOINTS.maca_generate).toEqual({ method: 'POST', path: '/v1/maca/generate', idempotent: false });
    expect(ENDPOINTS.maca_job).toEqual({ method: 'GET', path: '/v1/maca/jobs/{job_id}', idempotent: true });
  });
});

function usecaseFile(text = '# Agent\nA support assistant.\n') {
  const dir = mkdtempSync(join(tmpdir(), 'maca-'));
  const file = join(dir, 'use-case.md');
  writeFileSync(file, text);
  return { dir, file };
}

describe('generate', () => {
  it('uploads the file as multipart with the relationship', async () => {
    const { dir, file } = usecaseFile();
    try {
      const spy = vi.fn(async () => ok({ job_id: 'maca_1', status: 'queued', message: 'Sending.' }, 202));
      const job = await maca(spy as unknown as typeof fetch).generate({ file_path: file, user_relationship: 'b2c' });
      const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toContain('/v1/maca/generate');
      const form = init.body as FormData;
      expect(form).toBeInstanceOf(FormData);
      expect(form.get('user_relationship')).toBe('b2c');
      const upload = form.get('file') as File;
      expect(upload.name).toBe('use-case.md');
      expect(await upload.text()).toBe('# Agent\nA support assistant.\n');
      expect(job.job_id).toBe('maca_1');
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it('refuses a bad relationship, a missing, empty or large file before any request', async () => {
    const empty = usecaseFile('');
    const big = usecaseFile('x'.repeat(256 * 1024 + 1));
    const good = usecaseFile();
    try {
      const spy = vi.fn(async () => ok({}));
      const m = maca(spy as unknown as typeof fetch);
      await expect(m.generate({ file_path: good.file } as never)).rejects.toThrow("user_relationship must be 'b2c' or 'b2b'");
      await expect(m.generate({ file_path: good.file, user_relationship: 'internal' as never })).rejects.toThrow(InvalidInputError);
      await expect(m.generate({ file_path: join(good.dir, 'nope.md'), user_relationship: 'b2c' })).rejects.toThrow('readable file');
      await expect(m.generate({ file_path: empty.file, user_relationship: 'b2c' })).rejects.toThrow('empty');
      await expect(m.generate({ file_path: big.file, user_relationship: 'b2c' })).rejects.toThrow('larger than 256 KB');
      expect(spy).not.toHaveBeenCalled();
    } finally {
      for (const x of [empty, big, good]) rmSync(x.dir, { recursive: true });
    }
  });
});

describe('get_job and wait_for_completion', () => {
  it('reads the config when it is there', async () => {
    const spy = vi.fn(async () => ok({ job_id: 'j', status: 'succeeded', config_file: 'memory.generated.yml', yaml: 'a: 1\n', removed_fields: ['x'] }));
    const job = await maca(spy as unknown as typeof fetch).get_job('j');
    expect((spy.mock.calls[0] as unknown as [string])[0]).toContain('/v1/maca/jobs/j');
    expect(job.yaml).toBe('a: 1\n');
    expect(job.removed_fields).toEqual(['x']);
  });

  it('returns when the job ends', async () => {
    const replies = [{ job_id: 'j', status: 'running' }, { job_id: 'j', status: 'invalid', problems: ['p'] }];
    const spy = vi.fn(async () => ok(replies.shift()));
    const job = await maca(spy as unknown as typeof fetch).wait_for_completion('j', { timeout_seconds: 5, poll_interval_seconds: 0 });
    expect(job.status).toBe('invalid');
    expect(job.problems).toEqual(['p']);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('gives up', async () => {
    const spy = vi.fn(async () => ok({ job_id: 'j', status: 'running' }));
    await expect(maca(spy as unknown as typeof fetch).wait_for_completion('j', { timeout_seconds: 0 })).rejects.toThrow(/did not finish/);
  });
});
