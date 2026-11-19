/**
 * P20: memory records in the JS SDK. Mirrors
 * synap/tests/sdk/test_memory_records_sdk.py; the golden bodies are in
 * CONTRACT/conformance/memory_payloads.json (payload-parity.test.ts).
 *
 * A list of dicts is a records load only with document_type 'memory-record';
 * a list with any other type is parts exactly as before.
 */
import { describe, expect, it } from 'vitest';

import { SynapClient } from '../client.js';
import { buildCreateBody } from '../memories/interface.js';
import { InvalidInputError } from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const QUEUED = {
  ingestion_id: 'ing_1', document_id: 'records-1', status: 'queued', queued_at: '2026-11-04T00:00:00Z',
};
const RECORDS = [
  { id: 'crm-1', content: 'Prefers EUR', memory_type: 'preferences' },
  { content: 'Renewal on 30 Nov 2026', memory_type: 'temporal_events' },
];

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

function capture(respond: () => Response = () => json(QUEUED)) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fetchImpl = (async (url: string, init: RequestInit = {}) => {
    if (url.endsWith('/api/v1/auth/whoami')) return json({ client_id: 'c', instance_id: INSTANCE_ID });
    calls.push({ url, init });
    return respond();
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

const client = (fetchImpl: typeof fetch) => new SynapClient({
  _force_new: true, apiKey: 'k', instanceId: INSTANCE_ID, fetchImpl, retryPolicy: null,
});

describe('memory records', () => {
  it('sends a list of records as it is, with on_conflict last', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: RECORDS, document_type: 'memory-record', user_id: 'dana', customer_id: 'acme',
      on_conflict: 'reject',
    });
    expect(calls[0]!.url.endsWith('/api/v1/memories/create')).toBe(true);
    const body = JSON.parse(String(calls[0]!.init.body)) as Record<string, unknown>;
    expect(body['document']).toEqual(RECORDS);
    expect(body['on_conflict']).toBe('reject');
    expect(Object.keys(body).at(-1)).toBe('on_conflict');
  });

  it('leaves on_conflict out when not set', () => {
    expect(buildCreateBody({ document: RECORDS[0]!, document_type: 'memory-record' }))
      .not.toHaveProperty('on_conflict');
  });

  it('keeps a list with any other type as parts', () => {
    expect(buildCreateBody({ document: ['one', 'two'], document_type: 'document' })['document'])
      .toBe('one\ntwo');
    expect(() => buildCreateBody({ document: RECORDS, document_type: 'document' }))
      .toThrow(InvalidInputError);
  });

  it('refuses an unknown on_conflict before sending', () => {
    expect(() => buildCreateBody({
      document: RECORDS, document_type: 'memory-record',
      on_conflict: 'overwrite' as unknown as 'reject',
    })).toThrow(InvalidInputError);
  });

  it('sends records in a batch item as they are', async () => {
    const { calls, fetchImpl } = capture(() => json({
      batch_id: 'b', total: 1, succeeded: 1, failed: 0, results: [QUEUED],
    }));
    await client(fetchImpl).memories.batch_create({
      documents: [{ document: RECORDS, document_type: 'memory-record', onConflict: 'keep_both' }],
    });
    const body = JSON.parse(String(calls[0]!.init.body)) as { documents: Record<string, unknown>[] };
    expect(body.documents[0]!['document']).toEqual(RECORDS);
    expect(body.documents[0]!['on_conflict']).toBe('keep_both');
  });
});
