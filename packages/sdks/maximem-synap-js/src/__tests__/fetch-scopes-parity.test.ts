/**
 * The four scoped fetches against the Python SDK: what each posts, and what
 * each refuses before posting.
 *
 * The corpus (CONTRACT/conformance/fetch_scopes.json) is generated from the
 * public Python fetch path by scripts/gen_fetch_corpus.py. payload-parity
 * covers user scope only, so client, customer and conversation could drop an
 * option without a failing test: they dropped `scope_path` from 0.4.4 to 0.5.9.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFetchRequest, fetchContext, type Scope } from '../context/fetch.js';
import type { FetchOptions, Json } from '../context/types.js';
import type { HttpTransport } from '../transport/http.js';
import { ENDPOINTS } from '../transport/endpoints.js';
import { SynapClient } from '../client.js';
import { InvalidInputError } from '../errors.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const inPublicMirror = !existsSync(path.resolve(here, '../../../CONTRACT'));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/fetch_scopes.json');

interface Corpus {
  base_args: Record<Scope, Record<string, unknown>>;
  payloads: Record<Scope, { args: Record<string, unknown>; path: string; payload: Json }[]>;
  invalid: Record<Scope, { args: Record<string, unknown>; raises: string | null }[]>;
  uninitialized: Record<Scope, { raises: string | null; sent: number }>;
  b2c_customer_id: {
    call: Scope | 'unified';
    args: Record<string, unknown>;
    raises: string;
    is_invalid_input: boolean;
  }[];
}

/**
 * Wire differences that are declared, not hidden. Each one names why the
 * server cannot tell the two apart.
 */
const KNOWN_WIRE_DIFFS: Record<string, string> = {
  search_query:
    'Python sends null when no query is given, JS sends []. The server reads ' +
    '`request.search_query or []`, so both mean "no query".',
};

function normalise(body: Json): Json {
  const out = { ...body };
  if (out['search_query'] === null) out['search_query'] = [];
  return out;
}

function recordingTransport() {
  const sent: { name: string; body: unknown }[] = [];
  const transport = {
    checkCustomerId: () => undefined,
    request: async (name: string, opts?: { body?: unknown }) => {
      sent.push({ name, body: opts?.body });
      return { context: {} };
    },
  } as unknown as HttpTransport;
  return { transport, sent };
}

describe.skipIf(inPublicMirror)('scoped fetch parity with the Python SDK', () => {
  if (inPublicMirror) return;
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as Corpus;
  const scopes = Object.keys(corpus.payloads) as Scope[];

  it('declares every wire difference it normalises', () => {
    expect(Object.keys(KNOWN_WIRE_DIFFS)).toEqual(['search_query']);
  });

  const payloadCases = scopes.flatMap((scope) =>
    corpus.payloads[scope].map((c, i) => ({ scope, i, label: JSON.stringify(c.args) })),
  );

  it.each(payloadCases)('$scope payload[$i] $label', ({ scope, i }) => {
    const c = corpus.payloads[scope][i]!;
    const args = { ...corpus.base_args[scope], ...c.args } as FetchOptions;
    const { endpoint, body } = buildFetchRequest(scope, args);
    expect(ENDPOINTS[endpoint].path).toBe(c.path);
    expect(normalise(body)).toEqual(normalise(c.payload));
  });

  const invalidCases = scopes.flatMap((scope) =>
    corpus.invalid[scope].map((c, i) => ({ scope, i, label: JSON.stringify(c.args) })),
  );

  it.each(invalidCases)('$scope refuses $label before any request', async ({ scope, i }) => {
    const c = corpus.invalid[scope][i]!;
    const args = { ...corpus.base_args[scope], ...c.args } as FetchOptions;
    const { transport, sent } = recordingTransport();
    const err = await fetchContext(transport, scope, args).then(() => null, (e: unknown) => e);
    expect(c.raises).not.toBeNull();
    expect((err as Error | null)?.name).toBe(c.raises);
    expect(sent).toEqual([]);
  });

  /**
   * Declared difference: Python raises AuthenticationError("SDK not
   * initialized") on a fetch before initialize(). JS resolves identity on the
   * first request instead, so no existing JS caller breaks. What must match is
   * the outcome Python's rule protects: identity is known before the fetch.
   */
  it.each(scopes)('%s: Python refuses before initialize(), JS resolves identity first', async (scope) => {
    expect(corpus.uninitialized[scope]).toEqual({ raises: 'AuthenticationError', sent: 0 });

    const calls: string[] = [];
    const fetchImpl = (async (url: string) => {
      calls.push(new URL(url).pathname);
      const body = url.includes('whoami')
        ? { client_id: 'cli_1', instance_id: 'inst_0123456789abcdef' }
        : { context: {} };
      return new Response(JSON.stringify(body), { status: 200 });
    }) as unknown as typeof fetch;
    const client = new SynapClient({ _force_new: true, apiKey: 'synap_k', fetchImpl });
    const args = corpus.base_args[scope] as FetchOptions;
    const ns = scope === 'conversation'
      ? client.conversation.context
      : (client as unknown as Record<string, { context: { fetch: (o: FetchOptions) => Promise<unknown> } }>)[scope]!.context;
    await Promise.all([ns.fetch(args), ns.fetch(args)]);

    expect(calls[0]).toBe('/api/v1/auth/whoami');
    expect(calls.filter((p) => p.endsWith('/whoami'))).toHaveLength(1);
    expect(client.client_id).toBe('cli_1');
  });

  /**
   * A customer_id on a B2C instance: same class, same parent, nothing sent.
   * Until 0.5.11 / 0.5.12 Python raised a plain ValueError and JS an
   * InvalidInputError, so one `catch` could not handle both SDKs.
   */
  it.each(corpus.b2c_customer_id.map((c, i) => ({ ...c, i, label: JSON.stringify(c.args) })))(
    'B2C $call $label raises $raises',
    async ({ call, args, raises, is_invalid_input }) => {
      const calls: string[] = [];
      const fetchImpl = (async (url: string) => {
        calls.push(new URL(url).pathname);
        const body = url.includes('whoami')
          ? { client_id: 'cli_1', instance_id: 'inst_0123456789abcdef', user_context_isolation: 'equals_customer' }
          : { context: {} };
        return new Response(JSON.stringify(body), { status: 200 });
      }) as unknown as typeof fetch;
      const client = new SynapClient({ _force_new: true, apiKey: 'synap_k', fetchImpl });
      const fetcher = call === 'unified'
        ? (o: FetchOptions) => client.fetch(o as never)
        : call === 'conversation'
          ? (o: FetchOptions) => client.conversation.context.fetch(o)
          : (client as unknown as Record<string, { context: { fetch: (o: FetchOptions) => Promise<unknown> } }>)[call]!.context.fetch;

      const err = await fetcher(args as FetchOptions).then(() => null, (e: unknown) => e);

      expect((err as Error | null)?.name).toBe(raises);
      expect(err instanceof InvalidInputError).toBe(is_invalid_input);
      expect(calls.filter((p) => !p.endsWith('/whoami'))).toEqual([]);
    },
  );
});
