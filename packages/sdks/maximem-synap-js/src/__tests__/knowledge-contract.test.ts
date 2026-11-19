/**
 * P21: the knowledge namespace in the JS SDK (D165, D167). Mirrors
 * synap/tests/sdk/test_knowledge_contract.py; the wire bodies and the server
 * responses are CONTRACT/conformance/knowledge_query.json.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SynapClient } from '../client.js';
import { InvalidInputError } from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const FIXTURE = JSON.parse(readFileSync(
  resolve(__dirname, '../../../CONTRACT/conformance/knowledge_query.json'), 'utf8')) as {
  requests: Array<{ name: string; js: Record<string, unknown>; body: Record<string, unknown> }>;
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

describe('knowledge.query', () => {
  for (const c of FIXTURE.requests) {
    it(`sends the contract body: ${c.name}`, async () => {
      const { calls, fetchImpl } = capture();
      await client(fetchImpl).knowledge.query(c.js as never);
      expect(calls[0]!.url).toMatch(/\/api\/v1\/knowledge\/query$/);
      expect(calls[0]!.init.method).toBe('POST');
      const body = JSON.parse(String(calls[0]!.init.body)) as Record<string, unknown>;
      expect(body).toEqual(c.body);
      expect(Object.keys(body)).toEqual(Object.keys(c.body));
    });
  }

  for (const [name, response] of Object.entries(FIXTURE.responses)) {
    it(`returns the ${name} response untouched`, async () => {
      const { fetchImpl } = capture(() => json(response));
      expect(await client(fetchImpl).knowledge.query({ target: 'Obligation' })).toEqual(response);
    });
  }

  it.each([
    [{ target: '' }], [{ target: 'O', limit: 0 }], [{ target: 'O', limit: 501 }],
    [{ target: 'O', scope: { a: 1 } }],
  ])('refuses %j before any request', async (bad) => {
    const { calls, fetchImpl } = capture();
    await expect(client(fetchImpl).knowledge.query(bad as never)).rejects.toThrow(InvalidInputError);
    expect(calls).toHaveLength(0);
  });
});

describe('knowledge.describe', () => {
  it('is a GET with the scope as query parameters', async () => {
    const { calls, fetchImpl } = capture();
    const c = client(fetchImpl);
    await c.knowledge.describe({ customerId: 'acme', scope: { customer: 'acme', team: 't' } });
    const url = new URL(calls[0]!.url);
    expect(url.pathname).toBe('/api/v1/knowledge/describe');
    expect(calls[0]!.init.method).toBe('GET');
    expect(Object.fromEntries(url.searchParams)).toEqual({
      customer_id: 'acme', scope: '{"customer":"acme","team":"t"}',
    });
    await c.knowledge.describe();
    expect(new URL(calls[1]!.url).search).toBe('');
  });
});
