/**
 * Images in conversations: content parts on `memories.create`,
 * `conversation.record_message` and `instance.send_message`.
 *
 * Mirrors the Python SDK's tests for the same change. Two things matter most:
 *
 * 1. Every call that worked before sends exactly the bytes it sent before. A
 *    string, a dict, or a list with no file in it never reaches the new code
 *    path on the wire.
 * 2. Python and JS send the same shapes: the manifest, the form field names,
 *    the inline wire parts, the placeholder text and the gRPC metadata key.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { SynapClient } from '../client.js';
import { HttpTransport } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { createConversationNamespace, buildRecordMessageBody } from '../conversation/interface.js';
import { createInstanceNamespace, CONTENT_PARTS_METADATA_KEY } from '../instance/interface.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { buildCreateBody, buildCreateManifest } from '../memories/interface.js';
import {
  OBJECTIVES, MAX_FILE_BYTES, MAX_STREAM_IMAGE_BYTES, filePart, textPart, guessMediaType,
  fetchUrlBytes, placeholderText, toBase64,
} from '../memories/parts.js';
import * as root from '../index.js';
import { ContextNotFoundError, InvalidInputError, ListeningNotActiveError } from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const CREDS = { apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID };
const CONV = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';

/** The PNG signature plus a few bytes. The server sniffs; the SDK does not. */
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
const PNG_B64 = Buffer.from(PNG).toString('base64');

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

const QUEUED = {
  ingestion_id: 'ing_1', document_id: 'doc_1', status: 'queued', queued_at: '2026-10-19T00:00:00Z',
};

interface Call { url: string; init: RequestInit }

/**
 * A fetch that answers Synap routes with `respond` and anything else from
 * `external`, recording every call.
 */
function capture(
  respond: (call: Call) => Response = () => json(QUEUED),
  external: Record<string, () => Response | Promise<Response>> = {},
) {
  const calls: Call[] = [];
  const fetchImpl = (async (url: string, init: RequestInit = {}) => {
    // The current SDK resolves the instance identity on the first write. That
    // bootstrap request is unrelated to the content route under test.
    if (!url.endsWith('/api/v1/auth/whoami')) calls.push({ url, init });
    const ext = external[url];
    if (ext !== undefined) return ext();
    return respond({ url, init });
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

function client(fetchImpl: typeof fetch) {
  return new SynapClient({
    _force_new: true, apiKey: 'k', instanceId: INSTANCE_ID, fetchImpl, retryPolicy: null,
  });
}

const synapCalls = (calls: Call[]) => calls.filter((c) => c.url.includes('synap-cloud'));

/** The exact JSON body a string document produced before this change. */
const LEGACY_HELLO =
  '{"document":"hello","document_type":"ai-chat-conversation","document_id":null,' +
  '"document_created_at":null,"user_id":null,"customer_id":null,"scope":null,' +
  '"mode":"long-range","metadata":{}}';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('exports', () => {
  it('exports the part helpers, OBJECTIVES and the types from the package root', () => {
    expect(root.textPart).toBe(textPart);
    expect(root.filePart).toBe(filePart);
    expect(root.OBJECTIVES).toBe(OBJECTIVES);
  });

  it('defines the same objectives as Python', () => {
    expect([...OBJECTIVES]).toEqual([
      'conversation', 'bootstrap', 'knowledge-base', 'record', 'profile',
    ]);
  });
});

describe('textPart and filePart', () => {
  it('textPart builds a text part', () => {
    expect(textPart('hi')).toEqual({ type: 'text', text: 'hi' });
    expect(() => textPart(3 as never)).toThrow(InvalidInputError);
  });

  it('requires exactly one of data, path or url', () => {
    // Python's messages, word for word.
    expect(() => filePart({})).toThrow(/^FilePart takes exactly one of data, path or url$/);
    expect(() => filePart({ data: PNG, url: 'https://x/a.png' }))
      .toThrow('FilePart takes exactly one of data, path or url (got data, url)');
    expect(() => filePart({ path: '/a.png', url: 'https://x/a.png' })).toThrow(InvalidInputError);
    // null counts as absent, as None does in Python.
    expect(filePart({ data: PNG, path: null, url: null }).data).toBe(PNG);
  });

  it('defaults filename to the basename of path or url, else "file"', () => {
    expect(filePart({ path: '/tmp/x/scan.png' }).filename).toBe('scan.png');
    expect(filePart({ url: 'https://cdn.example.com/a/b/photo.jpeg?sig=1#f' }).filename)
      .toBe('photo.jpeg');
    expect(filePart({ url: 'https://cdn.example.com/' }).filename).toBe('file');
    expect(filePart({ data: PNG }).filename).toBe('file');
    expect(filePart({ data: PNG, filename: 'mine.webp' }).filename).toBe('mine.webp');
  });

  it('defaults media_type to a guess from filename, else application/octet-stream', () => {
    expect(filePart({ path: '/a/scan.png' }).media_type).toBe('image/png');
    expect(filePart({ path: '/a/scan.JPG' }).media_type).toBe('image/jpeg');
    expect(filePart({ path: '/a/scan.webp' }).media_type).toBe('image/webp');
    expect(filePart({ data: PNG }).media_type).toBe('application/octet-stream');
    expect(filePart({ data: PNG, media_type: 'image/png' }).media_type).toBe('image/png');
    expect(filePart({ data: PNG, mediaType: 'image/jpeg' }).media_type).toBe('image/jpeg');
  });

  it('guesses like Python mimetypes for the extensions that matter', () => {
    expect(guessMediaType('a.png')).toBe('image/png');
    expect(guessMediaType('a.jpg')).toBe('image/jpeg');
    expect(guessMediaType('a.PDF')).toBe('application/pdf');
    expect(guessMediaType('.png')).toBeNull();
    expect(guessMediaType('noext')).toBeNull();
    expect(guessMediaType('a.unknownext')).toBeNull();
  });

  it('takes a File\'s own name and type when none is given', () => {
    const file = new File([PNG], 'from-browser.bin', { type: 'image/png' });
    const part = filePart({ data: file });
    expect(part.filename).toBe('from-browser.bin');
    // .bin guesses nothing, so the Blob's own type is the next best answer.
    expect(part.media_type).toBe('image/png');
  });
});

describe('memories.create: the JSON route is byte-identical', () => {
  it('a string document sends exactly the bytes it always did', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: 'hello' });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toMatch(/\/api\/v1\/memories\/create$/);
    expect(calls[0]!.init.body).toBe(LEGACY_HELLO);
  });

  it('a list with no file is the "\\n"-joined string, byte for byte', async () => {
    const { calls, fetchImpl } = capture();
    const c = client(fetchImpl);
    await c.memories.create({ document: 'a\nb\nc' });
    await c.memories.create({ document: ['a', textPart('b'), 'c'] });
    expect(calls[1]!.url).toBe(calls[0]!.url);
    expect(calls[1]!.init.body).toBe(calls[0]!.init.body);
  });

  it('a single TextPart is its text', () => {
    expect(JSON.stringify(buildCreateBody({ document: textPart('hello') }))).toBe(LEGACY_HELLO);
  });

  it('matches the three new corpus cases', () => {
    // CONTRACT/conformance/memory_payloads.json carries these once the Python
    // side captures them; asserted here too so JS does not wait on that.
    const base = JSON.parse(LEGACY_HELLO) as Record<string, unknown>;
    expect(buildCreateBody({ document: 'hello', objective: 'bootstrap' }))
      .toEqual({ ...base, objective: 'bootstrap' });
    expect(JSON.stringify(buildCreateBody({ document: 'hello', scope: { client: 'self' } })))
      .toBe(LEGACY_HELLO);
    expect(buildCreateBody({ document: ['a', 'b'] })).toEqual({ ...base, document: 'a\nb' });
  });

  it('a dict document passes through untouched', () => {
    const doc = { messages: [{ role: 'user', content: 'hi' }] };
    expect(buildCreateBody({ document: doc }).document).toBe(doc);
  });

  it('a dict that happens to look like a part is still a dict, as on main', async () => {
    // Python always passes a dict through. Only textPart()/filePart() make a
    // top-level part; a plain object is a dict document, exactly as before.
    const doc = { type: 'text', text: 'hi', source: 'crm' };
    const body = buildCreateBody({ document: doc });
    expect(body['document']).toBe(doc);
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: doc });
    expect(calls[0]!.url).toMatch(/\/api\/v1\/memories\/create$/);
    expect(calls[0]!.init.body).toBe(LEGACY_HELLO.replace(
      '"document":"hello"', '"document":{"type":"text","text":"hi","source":"crm"}',
    ));
  });

  it('a dict with type "file" is a dict too, never read as a file', async () => {
    const doc = { type: 'file', path: '/etc/hosts' };
    expect(buildCreateBody({ document: doc })['document']).toBe(doc);
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: doc });
    expect(calls).toHaveLength(1);
    expect(JSON.parse(calls[0]!.init.body as string).document).toEqual(doc);
  });

  it('the brand never reaches the wire or deep equality', () => {
    expect(textPart('x')).toEqual({ type: 'text', text: 'x' });
    expect(JSON.stringify(filePart({ path: '/a.png' })))
      .toBe('{"type":"file","path":"/a.png","filename":"a.png","media_type":"image/png"}');
  });

  it('batch_create: dicts stay dicts, branded parts are unpacked', async () => {
    const { calls, fetchImpl } = capture(() => json({ batch_id: 'b', results: [] }));
    const doc = { type: 'text', text: 'hi', source: 'crm' };
    await client(fetchImpl).memories.batch_create({
      documents: [{ document: doc }, { document: textPart('plain') }],
    });
    const body = JSON.parse(calls[0]!.init.body as string);
    expect(body.documents[0].document).toEqual(doc);
    expect(body.documents[1].document).toBe('plain');
  });

  it('sends objective only when set, and last', () => {
    expect('objective' in buildCreateBody({ document: 'x' })).toBe(false);
    expect('objective' in buildCreateBody({ document: 'x', objective: null })).toBe(false);
    const body = buildCreateBody({ document: 'x', objective: 'free text works' });
    expect(body['objective']).toBe('free text works');
    expect(Object.keys(body).at(-1)).toBe('objective');
  });

  it('refuses an empty list explicitly (an empty array is truthy)', async () => {
    const { calls, fetchImpl } = capture();
    await expect(client(fetchImpl).memories.create({ document: [] }))
      .rejects.toThrow(InvalidInputError);
    await expect(client(fetchImpl).memories.create({ document: [] }))
      .rejects.toThrow('document is an empty list; pass at least one part');
    expect(calls).toHaveLength(0);
  });

  it('still refuses an empty string the way it did', () => {
    expect(() => buildCreateBody({ document: '' })).toThrow('document is required');
    expect(() => buildCreateBody({ document: textPart('') })).toThrow('document is required');
  });

  it('refuses a list item that is not a string or a part', () => {
    expect(() => buildCreateBody({ document: ['a', 3 as never] }))
      .toThrow('document[1] is a number; expected a string, TextPart or FilePart');
    expect(() => buildCreateBody({ document: ['a', { type: 'audio' } as never] }))
      .toThrow(InvalidInputError);
  });

  it('buildCreateBody refuses a FilePart: those go multipart', () => {
    expect(() => buildCreateBody({ document: [filePart({ data: PNG })] }))
      .toThrow(/cannot carry a FilePart/);
  });
});

describe('scope { client: "self" }', () => {
  it('on its own is sent as scope: null', () => {
    expect(buildCreateBody({ document: 'x', scope: { client: 'self' } })['scope']).toBeNull();
  });

  it('with user_id or customer_id is refused', () => {
    expect(() => buildCreateBody({ document: 'x', scope: { client: 'self' }, user_id: 'u' }))
      .toThrow(InvalidInputError);
    expect(() => buildCreateBody({ document: 'x', scope: { client: 'self' }, customerId: 'c' }))
      .toThrow(/cannot be combined with user_id or customer_id/);
  });

  it('"self" alongside other entries is refused, under any key', () => {
    expect(() => buildCreateBody({ document: 'x', scope: { client: 'self', customer: 'acme' } }))
      .toThrow('"self" can only be used alone, as scope={"client": "self"}; got {"client":"self","customer":"acme"}');
    expect(() => buildCreateBody({ document: 'x', scope: { customer: 'self', user: 'dana' } }))
      .toThrow(InvalidInputError);
  });

  it('every other use of the client key passes through untouched', () => {
    expect(buildCreateBody({ document: 'x', scope: { client: 'acme' } })['scope'])
      .toEqual({ client: 'acme' });
    expect(buildCreateBody({ document: 'x', scope: { client: 'acme', customer: 'c' } })['scope'])
      .toEqual({ client: 'acme', customer: 'c' });
    expect(buildCreateBody({ document: 'x', scope: { customer: 'acme', user: 'dana' } })['scope'])
      .toEqual({ customer: 'acme', user: 'dana' });
  });

  it('applies to the multipart manifest too', () => {
    const parts = [textPart('a')];
    expect(buildCreateManifest({ document: parts, scope: { client: 'self' } }, parts, [])['scope'])
      .toBeNull();
    expect(() => buildCreateManifest(
      { document: parts, scope: { client: 'self' }, user_id: 'u' }, parts, [],
    )).toThrow(InvalidInputError);
  });
});

describe('memories.create: file parts go multipart', () => {
  it('posts manifest + file_0 to /create-multipart, with Python\'s manifest shape', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: ['Here is my receipt', filePart({ data: PNG, filename: 'scan.png' })],
      user_id: 'u1',
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toMatch(/\/api\/v1\/memories\/create-multipart$/);
    const form = calls[0]!.init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect([...form.keys()]).toEqual(['manifest', 'file_0']);

    expect(form.get('manifest')).toBe(JSON.stringify({
      document: [
        { type: 'text', text: 'Here is my receipt' },
        { type: 'file', field: 'file_0', filename: 'scan.png', media_type: 'image/png' },
      ],
      document_type: null,
      document_id: null,
      document_created_at: null,
      user_id: 'u1',
      customer_id: null,
      scope: null,
      mode: 'long-range',
      metadata: {},
    }));

    const file = form.get('file_0') as File;
    expect(file.name).toBe('scan.png');
    expect(file.type).toBe('image/png');
    expect(new Uint8Array(await file.arrayBuffer())).toEqual(PNG);
  });

  it('numbers files in order of appearance, counting files only', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: [
        filePart({ data: PNG, filename: 'one.png' }),
        'between',
        filePart({ data: new Uint8Array([9, 9]).buffer, filename: 'two.jpg' }),
        filePart({ data: new Blob([new Uint8Array([7])]), filename: 'three.webp' }),
      ],
    });
    const form = calls[0]!.init.body as FormData;
    expect([...form.keys()]).toEqual(['manifest', 'file_0', 'file_1', 'file_2']);
    const manifest = JSON.parse(form.get('manifest') as string) as { document: unknown[] };
    expect(manifest.document).toEqual([
      { type: 'file', field: 'file_0', filename: 'one.png', media_type: 'image/png' },
      { type: 'text', text: 'between' },
      { type: 'file', field: 'file_1', filename: 'two.jpg', media_type: 'image/jpeg' },
      { type: 'file', field: 'file_2', filename: 'three.webp', media_type: 'image/webp' },
    ]);
    expect([...new Uint8Array(await (form.get('file_1') as File).arrayBuffer())]).toEqual([9, 9]);
    expect([...new Uint8Array(await (form.get('file_2') as File).arrayBuffer())]).toEqual([7]);
  });

  it('a single FilePart as the document goes multipart', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: filePart({ data: PNG }) });
    const form = calls[0]!.init.body as FormData;
    const manifest = JSON.parse(form.get('manifest') as string) as { document: unknown[] };
    expect(manifest.document).toEqual([
      { type: 'file', field: 'file_0', filename: 'file', media_type: 'application/octet-stream' },
    ]);
  });

  it('a literal { type: "file" } gets the same defaults as filePart()', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: [{ type: 'file', data: PNG, filename: 'lit.png' }],
    });
    const manifest = JSON.parse((calls[0]!.init.body as FormData).get('manifest') as string) as {
      document: Array<Record<string, unknown>>;
    };
    expect(manifest.document[0]).toEqual({
      type: 'file', field: 'file_0', filename: 'lit.png', media_type: 'image/png',
    });
  });

  it('carries every other field into the manifest', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({
      document: [filePart({ data: PNG, filename: 'a.png' })],
      document_type: 'document',
      documentId: 'd1',
      document_created_at: new Date('2026-10-19T10:00:00Z'),
      user_id: 'u1',
      customer_id: 'c1',
      scope: { customer: 'c1', team: 't', user: 'u1' },
      mode: 'fast',
      metadata: { k: 'v' },
      objective: 'record',
    });
    const manifest = JSON.parse((calls[0]!.init.body as FormData).get('manifest') as string);
    expect(Object.keys(manifest)).toEqual([
      'document', 'document_type', 'document_id', 'document_created_at', 'user_id',
      'customer_id', 'scope', 'mode', 'metadata', 'objective',
    ]);
    expect(manifest).toMatchObject({
      document_type: 'document',
      document_id: 'd1',
      document_created_at: '2026-10-19T10:00:00.000Z',
      user_id: 'u1',
      customer_id: 'c1',
      scope: { customer: 'c1', team: 't', user: 'u1' },
      mode: 'fast',
      metadata: { k: 'v' },
      objective: 'record',
    });
  });

  it('never sets a JSON content-type, and keeps the credentials on the Synap request', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: [filePart({ data: PNG })] });
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBeUndefined();
    expect(headers['Authorization']).toBe('Bearer k');
  });

  it('validates the rest before reading any file', async () => {
    const { calls, fetchImpl } = capture();
    await expect(client(fetchImpl).memories.create({
      document: [filePart({ url: 'https://files.example.com/a.png' })],
      document_type: 'nope',
    })).rejects.toThrow(/Invalid document_type 'nope'/);
    expect(calls, 'the url must not be fetched for a call that cannot be sent').toHaveLength(0);
  });

  it('returns the attachments the server reports', async () => {
    const attachments = [
      { filename: 'a.png', media_type: 'image/png', status: 'read', error: null },
      { filename: 'b.png', media_type: 'image/png', status: 'failed', error: 'unreadable' },
    ];
    const { fetchImpl } = capture(() => json({ ...QUEUED, attachments }));
    const result = await client(fetchImpl).memories.create({
      document: [filePart({ data: PNG, filename: 'a.png' }), filePart({ data: PNG, filename: 'b.png' })],
    });
    expect(result.attachments).toEqual(attachments);
    expect(result.ingestion_id).toBe('ing_1');
  });

  it('turns a 404 from an older server into a clear upgrade message', async () => {
    const { fetchImpl } = capture(() => json({ detail: 'Not Found' }, 404));
    const err = await client(fetchImpl).memories.create({ document: [filePart({ data: PNG })] })
      .catch((e: unknown) => e);
    // InvalidInputError, as Python raises, not the generic "context not found".
    expect(err).toBeInstanceOf(InvalidInputError);
    expect(err).not.toBeInstanceOf(ContextNotFoundError);
    expect((err as Error).message).toBe(
      'This Synap server does not support file parts in memories.create. ' +
        'Upgrade the Synap server, or send text only.',
    );
  });

  it('maps 405 from an older server to the same upgrade message', async () => {
    const { fetchImpl } = capture(() => json({ detail: 'Method Not Allowed' }, 405));
    const err = await client(fetchImpl).memories.create({ document: [filePart({ data: PNG })] })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InvalidInputError);
    expect((err as Error).message).toMatch(/does not support file parts in memories\.create/);
  });

  it.each([
    [413, 'File too large: scan.png is over 10 MB'],
    [415, 'Only image attachments (png, jpeg, webp) are supported in memories.create today; use memories.create_from_file for documents.'],
  ])('maps %i to InvalidInputError carrying the server detail', async (status, detail) => {
    const { fetchImpl } = capture(() => json({ detail }, status));
    const err = await client(fetchImpl).memories.create({ document: [filePart({ data: PNG })] })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InvalidInputError);
    expect((err as Error).message).toBe(`HTTP ${status}: ${detail}`);
  });

  it('maps 405 on any other route to InvalidInputError, as Python does', async () => {
    const { fetchImpl } = capture(() => json({ detail: 'Method Not Allowed' }, 405));
    const err = await client(fetchImpl).memories.create({ document: 'hello' })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InvalidInputError);
    expect((err as Error).message).toBe('HTTP 405: Method Not Allowed');
  });

  it('leaves a 404 on plain /create exactly as it was', async () => {
    const { fetchImpl } = capture(() => json({ detail: 'Instance not found' }, 404));
    const err = await client(fetchImpl).memories.create({ document: 'hello' })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ContextNotFoundError);
    expect((err as Error).message).toBe('HTTP 404: Instance not found');
  });

  it('is declared non-idempotent and so is never retried on an unknown outcome', async () => {
    expect(ENDPOINTS.memories_create_multipart).toEqual({
      method: 'POST', path: '/api/v1/memories/create-multipart', idempotent: false,
    });
    let n = 0;
    const fetchImpl = (async (url: string) => {
      // Identity bootstrap is idempotent and may retry independently. The
      // assertion is about the multipart write itself.
      if (url.endsWith('/api/v1/auth/whoami')) return json(QUEUED);
      n += 1;
      // A timeout: the server may have ingested it, so a retry could double it.
      throw Object.assign(new Error('timed out'), { name: 'AbortError' });
    }) as unknown as typeof fetch;
    const c = new SynapClient({ _force_new: true, apiKey: 'k', instanceId: INSTANCE_ID, fetchImpl });
    await expect(c.memories.create({ document: [filePart({ data: PNG })] })).rejects.toThrow();
    expect(n).toBe(1);
  });

  it('refuses a file over 10 MB before sending anything', async () => {
    const { calls, fetchImpl } = capture();
    await expect(client(fetchImpl).memories.create({
      document: [filePart({ data: new Uint8Array(MAX_FILE_BYTES + 1), filename: 'big.png' })],
    })).rejects.toThrow('big.png is 10485761 bytes; the limit is 10485760 bytes per file');
    expect(calls).toHaveLength(0);
    // Exactly at the limit is fine.
    await client(fetchImpl).memories.create({
      document: [filePart({ data: new Uint8Array(MAX_FILE_BYTES) })],
    });
    expect(calls).toHaveLength(1);
  });

  it('batch_create refuses a FilePart and joins a text-only list', async () => {
    const { calls, fetchImpl } = capture(() => json({ batch_id: 'b', results: [] }));
    const c = client(fetchImpl);
    await expect(c.memories.batch_create({
      documents: [{ document: [filePart({ data: PNG })] }],
    })).rejects.toThrow(/memories\.batch_create cannot carry a FilePart/);
    expect(calls).toHaveLength(0);
    await c.memories.batch_create({ documents: [{ document: ['a', 'b'] }] });
    const body = JSON.parse(calls[0]!.init.body as string);
    expect(body.documents[0].document).toBe('a\nb');
  });
});

describe('file sources: path and url', () => {
  it('reads a path from disk, naming and typing it from the basename', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'synap-parts-'));
    const file = path.join(dir, 'receipt.jpeg');
    writeFileSync(file, PNG);
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create({ document: [filePart({ path: file })] });
    const form = calls[0]!.init.body as FormData;
    const sent = form.get('file_0') as File;
    expect(sent.name).toBe('receipt.jpeg');
    expect(sent.type).toBe('image/jpeg');
    expect(new Uint8Array(await sent.arrayBuffer())).toEqual(PNG);
  });

  it('fetches a url client-side, with no Synap credentials', async () => {
    const url = 'https://files.example.com/img/cat.webp';
    const { calls, fetchImpl } = capture(undefined, {
      [url]: () => new Response(PNG, { status: 200 }),
    });
    await client(fetchImpl).memories.create({ document: ['look', filePart({ url })] });
    expect(calls.map((c) => c.url)).toEqual([url, expect.stringMatching(/create-multipart$/)]);
    // The API key must never go to somebody else's server.
    expect(calls[0]!.init.headers).toBeUndefined();
    expect(calls[0]!.init.method).toBe('GET');
    const form = calls[1]!.init.body as FormData;
    const manifest = JSON.parse(form.get('manifest') as string) as { document: unknown[] };
    expect(manifest.document[1]).toEqual({
      type: 'file', field: 'file_0', filename: 'cat.webp', media_type: 'image/webp',
    });
    expect(new Uint8Array(await (form.get('file_0') as File).arrayBuffer())).toEqual(PNG);
  });

  it('takes the response content-type when the filename says nothing', async () => {
    const url = 'https://files.example.com/render?id=7';
    const { calls, fetchImpl } = capture(undefined, {
      [url]: () => new Response(PNG, {
        status: 200, headers: { 'content-type': 'Image/PNG; charset=binary' },
      }),
    });
    await client(fetchImpl).memories.create({ document: [filePart({ url })] });
    const manifest = JSON.parse((calls[1]!.init.body as FormData).get('manifest') as string);
    expect(manifest.document[0]).toEqual({
      type: 'file', field: 'file_0', filename: 'render', media_type: 'image/png',
    });
  });

  it('keeps a media type the filename gave over the response content-type', async () => {
    const url = 'https://files.example.com/a.jpg';
    const { calls, fetchImpl } = capture(undefined, {
      [url]: () => new Response(PNG, { status: 200, headers: { 'content-type': 'image/png' } }),
    });
    await client(fetchImpl).memories.create({ document: [filePart({ url })] });
    const manifest = JSON.parse((calls[1]!.init.body as FormData).get('manifest') as string);
    expect(manifest.document[0].media_type).toBe('image/jpeg');
  });

  it('an unreadable path is an InvalidInputError', async () => {
    const { calls, fetchImpl } = capture();
    await expect(client(fetchImpl).memories.create({
      document: [filePart({ path: '/definitely/not/here.png' })],
    })).rejects.toThrow(/Could not read FilePart path '\/definitely\/not\/here.png'/);
    expect(calls).toHaveLength(0);
  });

  it('a non-2xx url is an error and nothing is posted', async () => {
    const url = 'https://files.example.com/gone.png';
    const { calls, fetchImpl } = capture(undefined, {
      [url]: () => new Response('nope', { status: 404 }),
    });
    await expect(client(fetchImpl).memories.create({ document: [filePart({ url })] }))
      .rejects.toThrow(`Fetching FilePart url '${url}' returned HTTP 404`);
    expect(synapCalls(calls)).toHaveLength(0);
  });

  it('refuses a url whose declared length is over 10 MB without reading it', async () => {
    const url = 'https://files.example.com/huge.png';
    const fetchImpl = (async () => new Response('x', {
      status: 200, headers: { 'content-length': String(MAX_FILE_BYTES + 1) },
    })) as unknown as typeof fetch;
    await expect(fetchUrlBytes(url, fetchImpl))
      .rejects.toThrow(`FilePart url '${url}' is over the limit of 10485760 bytes per file`);
  });

  it('refuses a streamed url body once it passes 10 MB', async () => {
    const chunk = new Uint8Array(4 * 1024 * 1024);
    let pulled = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled += 1;
        if (pulled > 10) controller.close();
        else controller.enqueue(chunk);
      },
    });
    const fetchImpl = (async () => new Response(stream, { status: 200 })) as unknown as typeof fetch;
    await expect(fetchUrlBytes('https://files.example.com/s.png', fetchImpl))
      .rejects.toThrow(/over the limit of 10485760 bytes per file/);
    // It stopped reading at the third chunk rather than draining all ten.
    expect(pulled).toBeLessThan(6);
  });

  it('times out a url after 30 s', async () => {
    vi.useFakeTimers();
    const fetchImpl = ((_url: string, init: RequestInit) => new Promise((_, reject) => {
      init.signal?.addEventListener('abort', () => reject(new Error('aborted')));
    })) as unknown as typeof fetch;
    const pending = fetchUrlBytes('https://slow.example.com/a.png', fetchImpl);
    const settled = expect(pending).rejects.toThrow(/timed out after 30 s/);
    await vi.advanceTimersByTimeAsync(30_000);
    await settled;
  });

  it('refuses a url that is not http or https', async () => {
    const fetchImpl = vi.fn() as unknown as typeof fetch;
    await expect(fetchUrlBytes('file:///etc/passwd', fetchImpl))
      .rejects.toThrow(/only http and https are supported/);
    await expect(fetchUrlBytes('not a url', fetchImpl)).rejects.toThrow(/not a valid URL/);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});

describe('base64 without Buffer', () => {
  it('encodes the same with btoa as with Buffer', () => {
    const big = new Uint8Array(70_000).map((_, i) => i % 256);
    const withBuffer = toBase64(big);
    vi.stubGlobal('Buffer', undefined);
    expect(toBase64(big)).toBe(withBuffer);
    expect(toBase64(PNG)).toBe(PNG_B64);
  });
});

describe('conversation.record_message', () => {
  function namespace(fetchImpl: typeof fetch) {
    const turns: Array<{ content: string; role: string }> = [];
    const transport = new HttpTransport({ credentials: CREDS, fetchImpl, retryPolicy: null });
    const ns = createConversationNamespace(transport, undefined, () => {}, (t) => {
      turns.push({ content: t.content, role: t.role });
    });
    return { ns, turns };
  }
  const base = { conversation_id: CONV, role: 'user', user_id: 'u1' };

  it('a string content sends the same bytes as before', async () => {
    const { calls, fetchImpl } = capture(() => json({ message_id: 'm1' }));
    await namespace(fetchImpl).ns.record_message({ ...base, content: 'hi' });
    expect(calls[0]!.init.body).toBe(
      `{"conversation_id":"${CONV}","role":"user","content":"hi","user_id":"u1","metadata":{}}`,
    );
  });

  it('a list with no file is the joined string, byte for byte', async () => {
    const { calls, fetchImpl } = capture(() => json({ message_id: 'm1' }));
    const { ns, turns } = namespace(fetchImpl);
    await ns.record_message({ ...base, content: 'a\nb' });
    await ns.record_message({ ...base, content: ['a', textPart('b')] });
    expect(calls[1]!.init.body).toBe(calls[0]!.init.body);
    expect(turns.map((t) => t.content)).toEqual(['a\nb', 'a\nb']);
  });

  it('sends text and inline-image wire parts, and mirrors placeholder text locally', async () => {
    const { calls, fetchImpl } = capture(() => json({ message_id: 'm1' }));
    const { ns, turns } = namespace(fetchImpl);
    await ns.record_message({
      ...base,
      content: ['What is this?', filePart({ data: PNG, filename: 'scan.png' })],
      metadata: { k: 'v' },
    });
    const body = JSON.parse(calls[0]!.init.body as string);
    expect(body.content).toEqual([
      { type: 'text', text: 'What is this?' },
      { type: 'image', filename: 'scan.png', media_type: 'image/png', data: PNG_B64 },
    ]);
    // Key order of an image part, matching Python's.
    expect(Object.keys(body.content[1])).toEqual(['type', 'filename', 'media_type', 'data']);
    // Base64 goes in content only, never in metadata.
    expect(body.metadata).toEqual({ k: 'v' });
    // The short-term mirror was `String(array)` => "[object Object],..." before.
    expect(turns).toEqual([{ role: 'user', content: 'What is this?\n[image: scan.png]' }]);
  });

  it('refuses an empty list before sending', async () => {
    const { calls, fetchImpl } = capture();
    await expect(namespace(fetchImpl).ns.record_message({ ...base, content: [] }))
      .rejects.toThrow('content is an empty list; pass at least one part');
    expect(calls).toHaveLength(0);
  });

  it('validates the rest of the message before reading any file', async () => {
    const { calls, fetchImpl } = capture();
    await expect(namespace(fetchImpl).ns.record_message({
      conversation_id: CONV, role: 'user', content: [filePart({ url: 'https://x.example.com/a.png' })],
    } as never)).rejects.toThrow('user_id is required');
    expect(calls).toHaveLength(0);
  });

  it('buildRecordMessageBody refuses an unread FilePart', () => {
    expect(() => buildRecordMessageBody({ ...base, content: [filePart({ data: PNG })] }))
      .toThrow(/FilePart, whose bytes have to be read first/);
  });

  it('record_messages_batch accepts parts per message', async () => {
    const { calls, fetchImpl } = capture(() => json({
      results: [{ message_id: 'm1' }, { message_id: 'm2' }], errors: [],
    }));
    const { ns, turns } = namespace(fetchImpl);
    await ns.record_messages_batch([
      { ...base, content: 'plain' },
      { ...base, content: [filePart({ data: PNG, filename: 'x.png' })] },
    ]);
    const body = JSON.parse(calls[0]!.init.body as string);
    expect(body.messages[0].content).toBe('plain');
    expect(body.messages[1].content).toEqual([
      { type: 'image', filename: 'x.png', media_type: 'image/png', data: PNG_B64 },
    ]);
    expect(turns.map((t) => t.content)).toEqual(['plain', '[image: x.png]']);
  });

  it('record_messages_batch names the failing message, as Python does', async () => {
    const { calls, fetchImpl } = capture();
    await expect(namespace(fetchImpl).ns.record_messages_batch([
      { ...base, content: 'ok' },
      { ...base, content: [] },
    ])).rejects.toThrow('messages[1].content is an empty list; pass at least one part');
    expect(calls).toHaveLength(0);
  });
});

describe('instance.send_message', () => {
  async function listening(fetchUrl?: typeof fetch) {
    const events: Array<Record<string, unknown>> = [];
    const turns: Array<{ content: string }> = [];
    const fakeClient = {
      isConnected: true,
      lastError: null,
      sendConversationEvent(e: Record<string, unknown>) { events.push(e); },
      sendSessionControl() {},
      sendContextUsed() {},
      sendContextAssembled() {},
      async connect() {},
      async disconnect() {},
    };
    vi.doMock('../grpc/stream-client.js', () => ({
      GrpcStreamClient: class {
        constructor() { return fakeClient as never; }
      },
    }));
    const ns = createInstanceNamespace(
      () => CREDS,
      new AnticipationCache(),
      (t) => { turns.push({ content: t.content }); },
      fetchUrl,
    );
    await ns.listen({ host: 'localhost', port: 1, useTls: false });
    return { ns, events, turns };
  }
  const base = { conversation_id: CONV, user_id: 'u1' };

  it('a string content is sent exactly as before, with no parts key', async () => {
    const h = await listening();
    await h.ns.send_message({ ...base, content: 'hi', metadata: { a: 'b' } });
    expect(h.events[0]!['content']).toBe('hi');
    expect(h.events[0]!['metadata']).toEqual({ a: 'b' });
  });

  it('a list with no file is the joined string, with no parts key', async () => {
    const h = await listening();
    await h.ns.send_message({ ...base, content: ['a', textPart('b')] });
    expect(h.events[0]!['content']).toBe('a\nb');
    expect(h.events[0]!['metadata']).toEqual({});
  });

  it('puts the parts in metadata.synap_content_parts and placeholder text in content', async () => {
    const h = await listening();
    await h.ns.send_message({
      ...base,
      content: ['Look:', filePart({ data: PNG, filename: 'a.png' }), 'and', filePart({ data: PNG })],
      metadata: { keep: 'me' },
    });
    const event = h.events[0]!;
    expect(CONTENT_PARTS_METADATA_KEY).toBe('synap_content_parts');
    expect(event['content']).toBe('Look:\n[image: a.png]\nand\n[image: file]');
    const metadata = event['metadata'] as Record<string, string>;
    expect(metadata['keep']).toBe('me');
    expect(metadata['synap_content_parts']).toBe(JSON.stringify([
      { type: 'text', text: 'Look:' },
      { type: 'image', filename: 'a.png', media_type: 'image/png', data: PNG_B64 },
      { type: 'text', text: 'and' },
      { type: 'image', filename: 'file', media_type: 'application/octet-stream', data: PNG_B64 },
    ]));
    // The local mirror gets the placeholder, never base64.
    expect(h.turns).toEqual([{ content: 'Look:\n[image: a.png]\nand\n[image: file]' }]);
  });

  it('refuses more than 2.5 MB of raw image bytes in one event', async () => {
    const h = await listening();
    expect(MAX_STREAM_IMAGE_BYTES).toBe(2_621_440);
    const half = new Uint8Array(MAX_STREAM_IMAGE_BYTES / 2);
    // Exactly at the cap is fine.
    await h.ns.send_message({ ...base, content: [filePart({ data: half }), filePart({ data: half })] });
    expect(h.events).toHaveLength(1);
    const err = await h.ns.send_message({
      ...base, content: [filePart({ data: half }), filePart({ data: new Uint8Array(half.length + 1) })],
    }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InvalidInputError);
    expect((err as Error).message).toMatch(/come to more than 2621440 bytes/);
    expect((err as Error).message).toMatch(/memories\.create or conversation\.record_message/);
    expect(h.events).toHaveLength(1);
  });

  it('reads a url part with the fetch it was given', async () => {
    const url = 'https://files.example.com/p.png';
    const fetchUrl = vi.fn(async () => new Response(PNG, { status: 200 })) as unknown as typeof fetch;
    const h = await listening(fetchUrl);
    await h.ns.send_message({ ...base, content: [filePart({ url })] });
    expect(fetchUrl).toHaveBeenCalledWith(url, expect.objectContaining({ method: 'GET' }));
    expect(h.events[0]!['content']).toBe('[image: p.png]');
  });

  it('a string send still goes out synchronously, in the call', async () => {
    const h = await listening();
    void h.ns.send_message({ ...base, content: 'now' });
    expect(h.events.map((e) => e['content'])).toEqual(['now']);
  });

  it('keeps call order for un-awaited sends: image, then text, then a thought', async () => {
    const url = 'https://files.example.com/slow.png';
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const fetchUrl = (async () => {
      await gate;
      return new Response(PNG, { status: 200 });
    }) as unknown as typeof fetch;
    const h = await listening(fetchUrl);
    const first = h.ns.send_message({ ...base, content: ['see', filePart({ url })] });
    const second = h.ns.send_message({ ...base, content: 'after the image' });
    const third = h.ns.record_thinking({ ...base, content: 'thinking' });
    // Nothing may overtake the image while it is still being read.
    await Promise.resolve();
    expect(h.events).toHaveLength(0);
    release();
    await Promise.all([first, second, third]);
    expect(h.events.map((e) => e['content'])).toEqual([
      'see\n[image: slow.png]', 'after the image', 'thinking',
    ]);
    // Once the queue drains, a string send is synchronous again.
    void h.ns.send_message({ ...base, content: 'later' });
    expect(h.events.at(-1)!['content']).toBe('later');
  });

  it('a failed image send does not block the sends behind it', async () => {
    const h = await listening();
    const big = new Uint8Array(MAX_STREAM_IMAGE_BYTES + 1);
    const failed = h.ns.send_message({ ...base, content: [filePart({ data: big })] });
    const next = h.ns.send_message({ ...base, content: 'still sent' });
    await expect(failed).rejects.toThrow(InvalidInputError);
    await next;
    expect(h.events.map((e) => e['content'])).toEqual(['still sent']);
  });

  it('still requires an active stream', async () => {
    const ns = createInstanceNamespace(() => CREDS, new AnticipationCache());
    await expect(ns.send_message({ ...base, content: [filePart({ data: PNG })] }))
      .rejects.toThrow(ListeningNotActiveError);
  });
});

describe('placeholder rendering', () => {
  it('keeps text, renders each image as [image: <filename>], joins with "\\n"', () => {
    expect(placeholderText([
      { type: 'text', text: 'a' },
      { type: 'image', filename: 'x.png' },
      { type: 'image' },
    ])).toBe('a\n[image: x.png]\n[image: image]');
  });
});
