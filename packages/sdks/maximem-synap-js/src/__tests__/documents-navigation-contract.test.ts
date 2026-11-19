/**
 * P22: documents.index and documents.read in the JS SDK (D166, D167). Mirrors
 * synap/tests/sdk/test_documents_navigation_contract.py; the requests and the
 * server responses are CONTRACT/conformance/documents_navigation.json.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SynapClient } from '../client.js';
import { ContextNotFoundError, InvalidInputError } from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const FIXTURE = JSON.parse(readFileSync(
  resolve(__dirname, '../../../CONTRACT/conformance/documents_navigation.json'), 'utf8')) as {
  requests: Array<{
    name: string; method: 'index' | 'read'; js: Record<string, unknown>; path: string;
    params: Record<string, string>;
  }>;
  bad: Array<{ method: 'index' | 'read'; args: Record<string, unknown> }>;
  responses: Record<string, unknown>;
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

interface Call { url: string; init: RequestInit }

function capture(respond: (call: Call) => Response = () => json({})) {
  const calls: Call[] = [];
  const fetchImpl = (async (url: string, init: RequestInit = {}) => {
    if (url.endsWith('/api/v1/auth/whoami')) return json({ client_id: 'c', instance_id: INSTANCE_ID });
    calls.push({ url, init });
    return respond({ url, init });
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

function client(fetchImpl: typeof fetch) {
  return new SynapClient({
    _force_new: true, apiKey: 'k', instanceId: INSTANCE_ID, fetchImpl, retryPolicy: null,
  });
}

afterEach(() => vi.restoreAllMocks());

describe('documents.index and documents.read', () => {
  for (const c of FIXTURE.requests) {
    it(`sends the contract request: ${c.name}`, async () => {
      const { calls, fetchImpl } = capture();
      await client(fetchImpl).documents[c.method](c.js as never);
      const url = new URL(calls[0]!.url);
      expect(url.pathname).toBe(c.path);
      expect(calls[0]!.init.method).toBe('GET');
      expect(Object.fromEntries(url.searchParams)).toEqual(c.params);
    });
  }

  for (const [name, response] of Object.entries(FIXTURE.responses)) {
    it(`returns the ${name} response untouched`, async () => {
      const { fetchImpl } = capture(() => json(response));
      const docs = client(fetchImpl).documents;
      const got = name === 'read'
        ? await docs.read({ document_key: 'k', section_key: 's1' })
        : await docs.index('k');
      expect(got).toEqual(response);
    });
  }

  it.each(FIXTURE.bad.map((b) => [b.method, b.args] as const))(
    'refuses %s %j before any request', async (method, args) => {
      const { calls, fetchImpl } = capture();
      const docs = client(fetchImpl).documents;
      await expect((docs[method] as (a: never) => Promise<unknown>)(args as never))
        .rejects.toThrow(InvalidInputError);
      expect(calls).toHaveLength(0);
    });

  it('looks a document_id up first and sends the scope on', async () => {
    const { calls, fetchImpl } = capture((c) => json(
      c.url.includes('/lookup') ? { document_key: 'k9' } : { sections: [] }));
    await client(fetchImpl).documents.index({ document_id: 'msa-1', customer_id: 'acme' });
    expect(new URL(calls[0]!.url).pathname).toBe('/api/v1/documents/lookup');
    expect(Object.fromEntries(new URL(calls[0]!.url).searchParams))
      .toEqual({ document_id: 'msa-1', customer_id: 'acme' });
    expect(new URL(calls[1]!.url).pathname).toBe('/api/v1/documents/k9/index');
    expect(Object.fromEntries(new URL(calls[1]!.url).searchParams)).toEqual({ customer_id: 'acme' });
  });

  it('keeps the server code on a not-ready 404 and a too-large 422', async () => {
    const body = (error: string) => ({ error, message: `${error} happened`,
      detail: { code: error, error, message: `${error} happened` } });
    const notReady = capture(() => json(body('document_not_ready'), 404));
    const err = await client(notReady.fetchImpl).documents.read({ document_key: 'k', pages: '1-1' })
      .then(() => null, (e: unknown) => e) as ContextNotFoundError & { serverCode?: string };
    expect(err).toBeInstanceOf(ContextNotFoundError);
    expect(err.serverCode).toBe('document_not_ready');
    const big = capture(() => json(body('read_too_large'), 422));
    const err2 = await client(big.fetchImpl).documents.read({ document_key: 'k', pages: '1-30' })
      .then(() => null, (e: unknown) => e) as InvalidInputError & { serverCode?: string };
    expect(err2).toBeInstanceOf(InvalidInputError);
    expect(err2.serverCode).toBe('read_too_large');
  });
});
