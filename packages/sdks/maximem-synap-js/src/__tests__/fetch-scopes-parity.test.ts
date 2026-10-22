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

const SERVER_FACT = { id: 'srv', content: 'from the server', confidence: 0.9 };
const SUMMARY_FACT = 'Summary: prefers email over calls';

/**
 * A real gRPC server speaking the committed descriptor. Copied from
 * `anticipation-rung.test.ts` rather than shared, as that suite explains.
 */
async function startServer() {
  const grpc = await import('@grpc/grpc-js');
  const protoLoader = await import('@grpc/proto-loader');
  const { SYNAP_PROTO_DESCRIPTOR } = await import('../grpc/descriptor.js');
  const pkgDef = protoLoader.fromJSON(
    SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoLoader.fromJSON>[0],
    { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true },
  );
  const pkg = grpc.loadPackageDefinition(pkgDef) as unknown as {
    synap: { v1: { SynapService: { service: never } } };
  };
  let push: ((m: Record<string, unknown>) => void) | null = null;
  const queued: Array<Record<string, unknown>> = [];
  let ready: () => void = () => {};
  const connected = new Promise<void>((r) => { ready = r; });

  const server = new grpc.Server();
  server.addService(pkg.synap.v1.SynapService.service, {
    Listen: (call: { on: (e: string, h: (a?: unknown) => void) => void; write: (m: unknown) => void; end: () => void }) => {
      push = (m) => call.write(m);
      for (const m of queued.splice(0)) call.write(m);
      ready();
      call.on('data', () => {});
      call.on('end', () => call.end());
      call.on('error', () => {});
    },
    IngestTelemetry: () => {},
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (e, p) =>
      e ? reject(e) : resolve(p));
  });
  return {
    port, connected,
    push: (m: Record<string, unknown>) => { if (push) push(m); else queued.push(m); },
    stop: () => new Promise<void>((r) => server.tryShutdown(() => r())),
  };
}

async function until(predicate: () => boolean, ms = 3000) {
  const deadline = Date.now() + ms;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('timed out');
    await new Promise((r) => setTimeout(r, 10));
  }
}
const inPublicMirror = !existsSync(path.resolve(here, '../../../CONTRACT'));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/fetch_scopes.json');

interface Corpus {
  base_args: Record<Scope, Record<string, unknown>>;
  payloads: Record<Scope, { args: Record<string, unknown>; path: string; payload: Json }[]>;
  invalid: Record<Scope, { args: Record<string, unknown>; raises: string | null; cache_consulted: boolean }[]>;
  uninitialized: Record<Scope, { raises: string | null; sent: number }>;
  b2c_customer_id: {
    call: Scope | 'unified';
    args: Record<string, unknown>;
    raises: string;
    is_invalid_input: boolean;
  }[];
  summary_mode_cache: { args: Record<string, unknown>; cache_consulted: boolean; sent: number }[];
  summary_injection: {
    call: Scope;
    owner: { user_id?: string; customer_id?: string };
    args: Record<string, unknown>;
    turns: string[][];
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
    let consulted = false;
    const hooks = { beforeFetch: () => { consulted = true; return null; } };
    const err = await fetchContext(transport, scope, args, hooks).then(() => null, (e: unknown) => e);
    expect(c.raises).not.toBeNull();
    expect((err as Error | null)?.name).toBe(c.raises);
    expect(sent).toEqual([]);
    // Refused before the anticipation cache, as Python refuses it, so a value
    // a prefetched bundle would have answered is still refused.
    expect(consulted).toBe(c.cache_consulted);
  });

  it.each(corpus.summary_mode_cache.map((c, i) => ({ ...c, i, label: JSON.stringify(c.args) })))(
    'user $label: cache consulted = $cache_consulted',
    async ({ args, cache_consulted, sent: expectedSent }) => {
      const { transport, sent } = recordingTransport();
      let consulted = false;
      const hooks = { beforeFetch: () => { consulted = true; return null; } };
      await fetchContext(transport, 'user', { ...corpus.base_args.user, ...args } as FetchOptions, hooks);
      expect(consulted).toBe(cache_consulted);
      expect(sent).toHaveLength(expectedSent);
    },
  );

  /**
   * The periodic user-summary splice, turn by turn, on every scope. The bundle
   * arrives over a real local gRPC stream, the only writer of the client's
   * cache, so this is the whole path: stream, cache, fetch, splice.
   */
  it.each(corpus.summary_injection.map((c) => ({ ...c, label: JSON.stringify(c.owner) })))(
    '$call splices the user summary on the same turns as Python (owner $label)',
    async ({ call, owner, args, turns }) => {
      const server = await startServer();
      const fetchImpl = (async () =>
        new Response(JSON.stringify({ context: { facts: [SERVER_FACT] } }), { status: 200 })
      ) as unknown as typeof fetch;
      const client = new SynapClient({
        _force_new: true, apiKey: 'synap_k', clientId: 'cli_corpus',
        instanceId: 'inst_0123456789abcdef', fetchImpl,
      });
      await client.instance.listen({ host: '127.0.0.1', port: server.port, use_tls: false });
      await server.connected;
      server.push({
        context_bundle: {
          bundle_id: 'b-summary',
          bundle_type: 'user_summary',
          anticipation_user_id: owner.user_id ?? '',
          anticipation_customer_id: owner.customer_id ?? '',
          items_by_type: { facts: { items: [{ item_id: 's1', content: SUMMARY_FACT, scope: 'user' }] } },
        },
      });
      await until(() => client.anticipation_cache_snapshot().total_entries > 0);

      const fetcher = call === 'conversation'
        ? (o: FetchOptions) => client.conversation.context.fetch(o)
        : (client as unknown as Record<string, { context: { fetch: (o: FetchOptions) => Promise<unknown> } }>)[call]!.context.fetch;
      const seen: string[][] = [];
      for (let t = 0; t < turns.length; t += 1) {
        const r = (await fetcher(args as FetchOptions)) as { facts?: { content?: string }[] };
        seen.push((r.facts ?? []).map((f) => String(f.content)));
      }
      expect(seen).toEqual(turns);

      await client.instance.stop_listening();
      await client.shutdown();
      await server.stop();
    },
  );

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
