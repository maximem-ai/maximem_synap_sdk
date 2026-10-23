/**
 * memories.create_from_file against the Python SDK: the multipart form it
 * uploads, and what it refuses before uploading.
 *
 * The corpus (CONTRACT/conformance/fetch_scopes.json, `create_from_file`) is
 * the form Python hands its transport, from scripts/gen_fetch_corpus.py. A
 * `{"__bytes__": "..."}` arg stands for file content. Declared: the JSON-valued
 * fields (`metadata`, `scope`) are compared parsed, since Python's json.dumps
 * spaces them and JSON.stringify does not.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SynapClient } from '../client.js';
import { InvalidInputError } from '../errors.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const inPublicMirror = !existsSync(path.resolve(here, '../../../CONTRACT'));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/fetch_scopes.json');

interface UploadCorpus {
  base_args: Record<string, unknown>;
  forms: { args: Record<string, unknown>; path: string; form: Record<string, string>; files: Record<string, [string, string]> }[];
  invalid: { args: Record<string, unknown>; raises: string; is_invalid_input: boolean }[];
}

const JSON_FIELDS = new Set(['metadata', 'scope']);

function toArgs(base: Record<string, unknown>, extra: Record<string, unknown>) {
  const args: Record<string, unknown> = { ...base, ...extra };
  const f = args['file'] as { __bytes__?: string } | undefined;
  if (f && typeof f === 'object' && '__bytes__' in f) args['file'] = Buffer.from(f.__bytes__ ?? '');
  return args;
}

function client() {
  const sent: { path: string; body: FormData }[] = [];
  const fetchImpl = (async (url: string, init?: { body?: unknown }) => {
    sent.push({ path: new URL(url).pathname, body: init?.body as FormData });
    return new Response(JSON.stringify({
      ingestion_id: '6c4a1f9e-2b7d-4e0a-9c1f-1a2b3c4d5e6f', document_id: 'd1',
      status: 'queued', queued_at: '2026-10-23T00:00:00+00:00',
    }), { status: 200 });
  }) as unknown as typeof fetch;
  const c = new SynapClient({ _force_new: true, apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef', fetchImpl });
  return { sent, c };
}

async function readForm(body: FormData) {
  const form: Record<string, unknown> = {};
  const files: Record<string, [string, string]> = {};
  for (const [k, v] of body.entries()) {
    if (typeof v === 'string') form[k] = JSON_FIELDS.has(k) ? JSON.parse(v) : v;
    else files[k] = [(v as File).name, await (v as Blob).text()];
  }
  return { form, files };
}

describe.skipIf(inPublicMirror)('memories.create_from_file parity with the Python SDK', () => {
  if (inPublicMirror) return;
  const corpus = (JSON.parse(readFileSync(corpusPath, 'utf8')) as { create_from_file: UploadCorpus }).create_from_file;

  it.each(corpus.forms.map((f, i) => ({ ...f, i, label: JSON.stringify(f.args) })))(
    'uploads the form Python uploads for $label',
    async ({ args, path: p, form, files }) => {
      const { sent, c } = client();
      await c.memories.create_from_file(toArgs(corpus.base_args, args) as never);
      expect(sent).toHaveLength(1);
      expect(sent[0]!.path).toBe(p);
      const got = await readForm(sent[0]!.body);
      const want = Object.fromEntries(Object.entries(form).map(([k, v]) => [k, JSON_FIELDS.has(k) ? JSON.parse(v) : v]));
      expect(got.form).toEqual(want);
      expect(got.files).toEqual(files);
    },
  );

  it.each(corpus.invalid.map((c, i) => ({ ...c, i, label: JSON.stringify(c.args) })))(
    'refuses $label before uploading, as Python',
    async ({ args, raises, is_invalid_input }) => {
      const { sent, c } = client();
      const err = await c.memories.create_from_file(toArgs(corpus.base_args, args) as never)
        .then(() => null, (e: unknown) => e);
      expect((err as Error | null)?.name).toBe(raises);
      expect(err instanceof InvalidInputError).toBe(is_invalid_input);
      expect(sent).toEqual([]);
    },
  );
});
