/**
 * instance.send_message against the Python SDK: the event written to the
 * stream, and what is refused before anything is written.
 *
 * The corpus (CONTRACT/conformance/fetch_scopes.json, `send_message`) is the
 * payload Python hands its stream transport, from scripts/gen_fetch_corpus.py.
 *
 * Declared, not hidden:
 *  - JS writes every proto field, empty ones included (`tool_name: ''`,
 *    `search_queries: []`); Python leaves them out. proto3 reads an empty value
 *    and an absent one the same, so empty values are dropped before comparing.
 *  - `timestamp_ms`, `event_id` and `sent_at_ms` are stamped by JS here and by
 *    Python's transport later, so they are not compared.
 *  - `*_json` fields: Python's json.dumps puts a space after `:` and `,`, JS's
 *    JSON.stringify does not. Compared as parsed JSON.
 *  - An auto-opened session id is random on both sides: compared as "<auto>".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const inPublicMirror = !existsSync(path.resolve(here, '../../../CONTRACT'));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/fetch_scopes.json');

type Event = Record<string, unknown>;
interface SendCorpus {
  base_args: Record<string, unknown>;
  events: { args: Record<string, unknown>; event: Event }[];
  invalid: { args: Record<string, unknown>; raises: string; is_invalid_input: boolean }[];
  b2c_customer_id: { raises: string; is_invalid_input: boolean };
}

const STAMPS = new Set(['timestamp_ms', 'event_id', 'sent_at_ms']);

function normalise(event: Event, args: Record<string, unknown>): Event {
  const out: Event = {};
  for (const [k, v] of Object.entries(event)) {
    if (STAMPS.has(k)) continue;
    if (v === '' || v === null || v === undefined) continue;
    if (Array.isArray(v) && v.length === 0) continue;
    if (k !== 'metadata' && typeof v === 'object' && !Array.isArray(v) && Object.keys(v as object).length === 0) continue;
    out[k] = k.endsWith('_json') && typeof v === 'string' ? JSON.parse(v) : v;
  }
  if (!('session_id' in args) && !('sessionId' in args)) out['session_id'] = '<auto>';
  return out;
}

let written: Event[] = [];

beforeEach(() => {
  vi.resetModules();
  written = [];
  vi.doMock('../grpc/stream-client.js', () => ({
    GrpcStreamClient: class {
      async connect() {}
      async disconnect() {}
      get isConnected() { return true; }
      sendConversationEvent(e: Event) { written.push(e); }
      sendSessionControl() {}
      sendContextUsed() {}
      sendContextAssembled() {}
    },
  }));
});

async function client(isolation = 'strict') {
  const { SynapClient } = await import('../client.js');
  const fetchImpl = (async (url: string) => {
    const p = new URL(url).pathname;
    const body = p.endsWith('/whoami')
      ? { client_id: 'c', instance_id: 'inst_0123456789abcdef', user_context_isolation: isolation }
      : {};
    return new Response(JSON.stringify(body), { status: 200 });
  }) as unknown as typeof fetch;
  const c = new SynapClient({ _force_new: true, apiKey: 'synap_k', fetchImpl });
  await c.initialize();
  await c.instance.listen();
  return c;
}

describe.skipIf(inPublicMirror)('instance.send_message parity with the Python SDK', () => {
  if (inPublicMirror) return;
  const corpus = (JSON.parse(readFileSync(corpusPath, 'utf8')) as { send_message: SendCorpus }).send_message;

  it.each(corpus.events.map((e, i) => ({ ...e, i, label: JSON.stringify(e.args) })))(
    'writes the event Python writes for $label',
    async ({ args, event }) => {
      const c = await client();
      await c.instance.send_message({ ...corpus.base_args, ...args } as never);
      expect(written).toHaveLength(1);
      expect(normalise(written[0]!, args)).toEqual(normalise(event, args));
      await c.shutdown();
    },
  );

  it.each(corpus.invalid.map((e, i) => ({ ...e, i, label: JSON.stringify(e.args) })))(
    'refuses $label before writing, as Python',
    async ({ args, raises, is_invalid_input }) => {
      const c = await client();
      const { InvalidInputError } = await import('../errors.js');
      const err = await c.instance.send_message({ ...corpus.base_args, ...args } as never)
        .then(() => null, (e: unknown) => e);
      expect((err as Error | null)?.name).toBe(raises);
      expect(err instanceof InvalidInputError).toBe(is_invalid_input);
      expect(written).toEqual([]);
      await c.shutdown();
    },
  );

  it('refuses a customer_id on a B2C instance before writing, as Python', async () => {
    const c = await client('equals_customer');
    const err = await c.instance.send_message(corpus.base_args as never).then(() => null, (e: unknown) => e);
    expect((err as Error | null)?.name).toBe(corpus.b2c_customer_id.raises);
    expect(written).toEqual([]);
    await c.shutdown();
  });
});
