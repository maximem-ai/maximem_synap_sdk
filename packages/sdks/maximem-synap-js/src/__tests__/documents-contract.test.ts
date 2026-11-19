/**
 * P3: the Documents contract in the JS SDK. Mirrors
 * synap/tests/sdk/test_documents_contract.py; the golden create bodies are
 * CONTRACT/conformance/memory_payloads.json (payload-parity.test.ts) and the
 * error codes are http_status_errors.json (status-errors-parity.test.ts).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SynapClient } from '../client.js';
import { HttpTransport } from '../transport/http.js';
import { createConversationNamespace } from '../conversation/interface.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { buildFetchRequest, documentFiltersOf } from '../context/fetch.js';
import { buildCreateBody, resetCreateFromFileWarning } from '../memories/interface.js';
import { filePart } from '../memories/parts.js';
import { AmbiguousDocumentError, FiltersNeedQueryError, InvalidInputError } from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const CREDS = { apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID };
const CONV = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
const PDF = new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 1, 2, 3]);
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
const LEGACY_KEYS = [
  'document', 'document_type', 'document_id', 'document_created_at',
  'user_id', 'customer_id', 'scope', 'mode', 'metadata',
];
const QUEUED = {
  ingestion_id: 'ing_1', document_id: 'doc_1', status: 'queued', queued_at: '2026-10-19T00:00:00Z',
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

interface Call { url: string; init: RequestInit }

function capture(respond: (call: Call) => Response = () => json(QUEUED)) {
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

describe('create: the Documents fields', () => {
  it('sends none of them unless set', () => {
    expect(Object.keys(buildCreateBody({ document: 'hello' }))).toEqual(LEGACY_KEYS);
  });

  it('sends them after objective, in Python\'s order', () => {
    const body = buildCreateBody({
      document: 'hello',
      participants: { Dana: { user_id: 'u1' } },
      source: { system: 'slack' },
      conversation_id: 'conv-1',
      plan: { extraction: 'detailed' },
      utility: { read_frequency: 'rare' },
      document_format: 'csv',
      objective: 'record',
    });
    expect(Object.keys(body).slice(LEGACY_KEYS.length)).toEqual([
      'objective', 'document_format', 'utility', 'plan', 'conversation_id', 'source', 'participants',
    ]);
  });

  it('takes the camelCase spellings', () => {
    const body = buildCreateBody({ document: 'x', documentFormat: 'html', conversationId: 'c1' });
    expect(body['document_format']).toBe('html');
    expect(body['conversation_id']).toBe('c1');
  });

  it('refuses an unknown document_format and a bad conversation_id', () => {
    expect(() => buildCreateBody({ document: 'x', document_format: 'pdf' as never }))
      .toThrow(/Invalid document_format/);
    expect(() => buildCreateBody({ document: 'x', conversation_id: '' })).toThrow(/conversation_id/);
    expect(() => buildCreateBody({ document: 'x', conversation_id: 'x'.repeat(201) }))
      .toThrow(/conversation_id/);
    expect(() => buildCreateBody({ document: 'x', source: 'slack' as never })).toThrow(/source must be a dict/);
  });

  it('sends a free-text document_type as given', () => {
    expect(buildCreateBody({ document: 'x', document_type: 'vendor-sow' })['document_type']).toBe('vendor-sow');
  });
});

describe('FilePart: per-file type and identity (D10, D171)', () => {
  it('puts them in the manifest part only when set', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: [
        filePart({ data: PNG, filename: 'a.png' }),
        filePart({ data: PDF, filename: 'msa.pdf', documentType: 'contract', document_id: 'msa-1' }),
      ],
    });
    const form = calls[0]!.init.body as FormData;
    expect(JSON.parse(form.get('manifest') as string).document).toEqual([
      { type: 'file', field: 'file_0', filename: 'a.png', media_type: 'image/png' },
      {
        type: 'file', field: 'file_1', filename: 'msa.pdf', media_type: 'application/pdf',
        document_type: 'contract', document_id: 'msa-1',
      },
    ]);
  });

  it('refuses an empty one', () => {
    expect(() => filePart({ data: PNG, document_id: '' })).toThrow(/FilePart.document_id/);
  });

  it('record_message refuses a part carrying them, before anything is sent', async () => {
    const { calls, fetchImpl } = capture();
    const transport = new HttpTransport({ credentials: CREDS, fetchImpl, retryPolicy: null });
    const ns = createConversationNamespace(transport, undefined, () => {}, () => {});
    await expect(ns.record_message({
      conversation_id: CONV, role: 'user', user_id: 'u1',
      content: ['hi', filePart({ data: PNG, document_type: 'contract' })],
    })).rejects.toThrow(/memories\.create/);
    expect(calls).toHaveLength(0);
  });
});

describe('batch_create', () => {
  const BATCH = { batch_id: 'b', total: 1, succeeded: 1, failed: 0, results: [QUEUED] };

  it('keeps today\'s JSON body for a text batch, new fields only when set', async () => {
    const { calls, fetchImpl } = capture(() => json(BATCH));
    await client(fetchImpl).memories.batch_create({
      documents: [{ document: 'a', user_id: 'u1' }, { document: 'b', objective: 'record', source: { system: 'jira' } }],
    });
    const body = JSON.parse(calls[0]!.init.body as string);
    expect(Object.keys(body)).toEqual(['documents', 'fail_fast']);
    expect(Object.keys(body.documents[0])).toEqual(LEGACY_KEYS);
    expect(Object.keys(body.documents[1])).toEqual([...LEGACY_KEYS, 'objective', 'source']);
  });

  it('sends a batch with files as multipart with an items manifest', async () => {
    const { calls, fetchImpl } = capture(() => json(BATCH));
    await client(fetchImpl).memories.batch_create({
      documents: [
        { document: 'plain text' },
        {
          document: ['cover note', filePart({ data: PDF, filename: 'msa.pdf', document_type: 'contract', document_id: 'msa-1' })],
          user_id: 'u1',
        },
      ],
      fail_fast: true,
    });
    expect(calls[0]!.url).toMatch(/\/api\/v1\/memories\/batch$/);
    const form = calls[0]!.init.body as FormData;
    const m = JSON.parse(form.get('manifest') as string);
    expect(Object.keys(m)).toEqual(['items', 'fail_fast']);
    expect(m.fail_fast).toBe(true);
    expect(m.items[0].document).toBe('plain text');
    expect(m.items[0].document_type).toBe('ai-chat-conversation');
    expect(m.items[1].document_type).toBeNull();
    expect(m.items[1].document).toEqual([
      { type: 'text', text: 'cover note' },
      {
        type: 'file', field: 'file_1_0', filename: 'msa.pdf', media_type: 'application/pdf',
        document_type: 'contract', document_id: 'msa-1',
      },
    ]);
    expect((form.get('file_1_0') as File).name).toBe('msa.pdf');
  });

  it('refuses more than 100 items before sending', async () => {
    const { calls, fetchImpl } = capture(() => json(BATCH));
    await expect(client(fetchImpl).memories.batch_create({
      documents: Array.from({ length: 101 }, () => ({ document: 'a' })),
    })).rejects.toThrow(/at most 100 items/);
    expect(calls).toHaveLength(0);
  });
});

describe('create_from_file is deprecated (D164)', () => {
  it('warns once per process, naming the replacement, and sends the same request', async () => {
    resetCreateFromFileWarning();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { calls, fetchImpl } = capture();
    const c = client(fetchImpl);
    await c.memories.create_from_file({ user_id: 'u1', text: 'hello' });
    await c.memories.create_from_file({ user_id: 'u1', text: 'hello' });
    const deprecations = warn.mock.calls.filter((a) => String(a[0]).includes('create_from_file'));
    expect(deprecations).toHaveLength(1);
    expect(String(deprecations[0]![0])).toContain('memories.create(document=FilePart(');
    expect(calls[0]!.url).toMatch(/\/api\/v1\/memories\/upload$/);
  });
});

describe('documents namespace (T12, D178)', () => {
  it('gets by key', async () => {
    const { calls, fetchImpl } = capture(() => json({ document_key: 'dk_1' }));
    expect(await client(fetchImpl).documents.get('dk_1')).toEqual({ document_key: 'dk_1' });
    expect(calls[0]!.url).toMatch(/\/api\/v1\/documents\/dk_1$/);
    expect(calls[0]!.init.method).toBe('GET');
  });

  it('looks a caller id up first', async () => {
    const answers = [{ document_key: 'dk_7' }, { document_key: 'dk_7', version: 3 }];
    const { calls, fetchImpl } = capture(() => json(answers.shift()));
    const out = await client(fetchImpl).documents.get({ document_id: 'msa-1', user_id: 'u1', customer_id: 'c1' });
    expect(out).toEqual({ document_key: 'dk_7', version: 3 });
    const lookup = new URL(calls[0]!.url);
    expect(lookup.pathname).toBe('/api/v1/documents/lookup');
    expect(Object.fromEntries(lookup.searchParams)).toEqual({ document_id: 'msa-1', user_id: 'u1', customer_id: 'c1' });
    expect(new URL(calls[1]!.url).pathname).toBe('/api/v1/documents/dk_7');
  });

  it('sends a scope to lookup as JSON', async () => {
    const answers = [{ document_key: 'dk_7' }, {}];
    const { calls, fetchImpl } = capture(() => json(answers.shift()));
    await client(fetchImpl).documents.versions({ document_id: 'd', scope: { customer: 'acme', team: 't' } });
    expect(new URL(calls[0]!.url).searchParams.get('scope')).toBe('{"customer":"acme","team":"t"}');
    expect(new URL(calls[1]!.url).pathname).toBe('/api/v1/documents/dk_7/versions');
  });

  it('takes exactly one of key or id', async () => {
    const { fetchImpl } = capture();
    const c = client(fetchImpl);
    await expect(c.documents.get({})).rejects.toBeInstanceOf(InvalidInputError);
    await expect(c.documents.get({ document_key: 'dk', document_id: 'd' })).rejects.toBeInstanceOf(InvalidInputError);
  });

  it('deletes softly unless permanent', async () => {
    const { calls, fetchImpl } = capture(() => json({}));
    const c = client(fetchImpl);
    await c.documents.delete('dk_1');
    expect(calls[0]!.init.method).toBe('DELETE');
    expect(new URL(calls[0]!.url).search).toBe('');
    await c.documents.delete({ document_key: 'dk_1', permanent: true });
    expect(new URL(calls[1]!.url).searchParams.get('permanent')).toBe('true');
  });

  it('restores', async () => {
    const { calls, fetchImpl } = capture(() => json({}));
    await client(fetchImpl).documents.restore('dk_1');
    expect(calls[0]!.init.method).toBe('POST');
    expect(new URL(calls[0]!.url).pathname).toBe('/api/v1/documents/dk_1/restore');
  });

  it('lists with only the filters given', async () => {
    const { calls, fetchImpl } = capture(() => json({ items: [] }));
    await client(fetchImpl).documents.list({ documentType: 'contract', cursor: 'c2' });
    const url = new URL(calls[0]!.url);
    expect(url.pathname).toBe('/api/v1/documents');
    expect(Object.fromEntries(url.searchParams)).toEqual({ document_type: 'contract', cursor: 'c2' });
  });

  it('returns the original as bytes', async () => {
    const { calls, fetchImpl } = capture(() => new Response(PDF, { status: 200 }));
    const out = await client(fetchImpl).documents.original({ document_key: 'dk_1', version: 2 });
    expect(Array.from(out)).toEqual(Array.from(PDF));
    expect(new URL(calls[0]!.url).pathname).toBe('/api/v1/documents/dk_1/versions/2/original');
  });

  it('raises AmbiguousDocumentError on a 409 lookup and deletes nothing', async () => {
    const body = { error: 'ambiguous_document', message: 'two', matches: [{ scope: 'user' }, { scope: 'client' }] };
    const { calls, fetchImpl } = capture(() => json({ ...body, detail: { code: 'ambiguous_document', ...body } }, 409));
    const err = await client(fetchImpl).documents.delete({ document_id: 'd' }).then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(AmbiguousDocumentError);
    expect((err as AmbiguousDocumentError).matches).toEqual([{ scope: 'user' }, { scope: 'client' }]);
    expect(calls).toHaveLength(1);
  });
});

describe('fetch filters (D123, D185)', () => {
  const FILTERS = { kinds: ['passage'], document_types: ['contract'], document_ids: ['msa-1'], as_of: '2025-01-01' };

  it.each(['user', 'customer', 'client', 'conversation'] as const)('absent unless set, last when set: %s', (scope) => {
    const ids = { user_id: 'u1', customer_id: 'c1', conversation_id: CONV };
    const plain = buildFetchRequest(scope, { ...ids, search_query: ['q'] }).body;
    for (const k of Object.keys(FILTERS)) expect(plain).not.toHaveProperty(k);
    const body = buildFetchRequest(scope, { ...ids, search_query: ['q'], ...FILTERS }).body;
    expect(Object.keys(body).slice(-4)).toEqual(Object.keys(FILTERS));
    for (const [k, v] of Object.entries(FILTERS)) expect(body[k]).toEqual(v);
  });

  it('takes camelCase and a Date', () => {
    const out = documentFiltersOf({
      search_query: ['q'], documentTypes: ['policy'], asOf: new Date(Date.UTC(2025, 0, 1)),
    });
    expect(out).toEqual({ document_types: ['policy'], as_of: '2025-01-01T00:00:00.000Z' });
  });

  it('refuses filters without a query, before any request', async () => {
    const { calls, fetchImpl } = capture(() => json({ context: {} }));
    await expect(client(fetchImpl).user.context.fetch({ user_id: 'u1', kinds: ['passage'] }))
      .rejects.toBeInstanceOf(FiltersNeedQueryError);
    await expect(client(fetchImpl).fetch({ user_id: 'u1', document_types: ['policy'] }))
      .rejects.toThrow(/documents\.get/);
    expect(calls).toHaveLength(0);
  });

  it('takes source_kind_hint (T54): one of the kinds, sent last, no query needed', () => {
    const body = buildFetchRequest('user', {
      user_id: 'u1', search_query: ['q'], kinds: ['passage'], sourceKindHint: 'passage',
    }).body;
    expect(Object.keys(body).slice(-2)).toEqual(['kinds', 'source_kind_hint']);
    expect(documentFiltersOf({ source_kind_hint: 'record' })).toEqual({ source_kind_hint: 'record' });
    expect(() => documentFiltersOf({ search_query: ['q'], source_kind_hint: 'chunk' }))
      .toThrow(/source_kind_hint/);
  });

  it('refuses an unknown kind and a non-list', () => {
    expect(() => documentFiltersOf({ search_query: ['q'], kinds: ['chunk'] })).toThrow(/Invalid kinds/);
    expect(() => documentFiltersOf({ search_query: ['q'], document_ids: 'x' as never })).toThrow(/document_ids/);
  });

  it('the unified fetch passes them to each scope', async () => {
    const { calls, fetchImpl } = capture(() => json({ context: {} }));
    await client(fetchImpl).fetch({ user_id: 'u1', search_query: ['q'], document_types: ['policy'] });
    const sent = calls.filter((c) => c.url.includes('/v1/context/'));
    expect(sent.length).toBeGreaterThan(0);
    for (const c of sent) expect(JSON.parse(c.init.body as string).document_types).toEqual(['policy']);
  });
});

describe('anticipation cache: kind, type and id filters, citations (D120, T22)', () => {
  function cache(items: Record<string, unknown>[]) {
    const c = new AnticipationCache();
    c.store({
      bundleId: 'b1', entityId: 'u1', conversationId: null, scopeRung: null,
      itemsByType: { facts: items as never }, ttlHintSeconds: null, bundleType: 'anticipation',
      searchQueries: ['termination notice'],
    } as never);
    return c;
  }
  const item = (i: number, content: string, extra: Record<string, unknown> = {}) =>
    ({ item_id: `i${i}`, content, confidence: 0.9, ...extra });
  const ids = (hit: ReturnType<AnticipationCache['lookup']>) =>
    Object.values(hit?.itemsByType ?? {}).flat().map((x) => (x as { item_id: string }).item_id).sort();

  it('narrows by kind; an item without one is a memory', () => {
    const c = cache([
      item(1, 'termination notice is ninety days'),
      item(2, 'termination notice period ninety days in the MSA', { kind: 'passage', document_type: 'contract', document_id: 'msa-1' }),
    ]);
    const q = { searchQuery: ['termination notice'], entityId: 'u1' };
    expect(ids(c.lookup(q))).toEqual(['i1', 'i2']);
    expect(ids(c.lookup({ ...q, filters: { kinds: ['memory'] } }))).toEqual(['i1']);
    expect(ids(c.lookup({ ...q, filters: { kinds: ['passage'] } }))).toEqual(['i2']);
    expect(c.lookup({ ...q, filters: { document_types: ['policy'] } })).toBeNull();
    expect(ids(c.lookup({ ...q, filters: { document_ids: ['msa-1'] } }))).toEqual(['i2']);
  });

  it('filters before the per-query share', () => {
    const c = cache([
      item(1, 'termination notice: the termination notice is ninety days'),
      item(2, 'termination notice: the contract termination notice clause', { kind: 'passage', document_type: 'contract', document_id: 'msa-1' }),
      item(3, 'renewal date renewal date of the contract', { kind: 'passage', document_type: 'contract', document_id: 'msa-1' }),
    ]);
    const q = { searchQuery: ['termination notice', 'renewal date'], entityId: 'u1', maxItems: 2, maxResultsPerQuery: 1 };
    expect(ids(c.lookup(q))).toEqual(['i1', 'i3']);
    expect(ids(c.lookup({ ...q, filters: { kinds: ['passage'] } }))).toEqual(['i2', 'i3']);
  });

  it('never answers as_of', () => {
    const c = cache([item(1, 'termination notice is ninety days')]);
    expect(c.lookup({ searchQuery: ['termination notice'], entityId: 'u1', filters: { as_of: '2025-01-01' } })).toBeNull();
  });

  it('serves the citation on a hit', () => {
    const cite = { document_key: 'dk_1', document_id: 'msa-1', version: 2, section_path: ['9', '9.2'], page: 14 };
    const c = cache([item(2, 'termination notice ninety days', {
      kind: 'passage', document_type: 'contract', document_id: 'msa-1', citation_json: JSON.stringify(cite),
    })]);
    const served = Object.values(c.lookup({ searchQuery: ['termination notice'], entityId: 'u1' })!.itemsByType).flat()[0] as Record<string, unknown>;
    expect(served['citation']).toEqual(cite);
    expect(served).not.toHaveProperty('citation_json');
  });
});
