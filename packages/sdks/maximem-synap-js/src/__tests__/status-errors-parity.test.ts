/**
 * HTTP status -> exception, against the Python SDK.
 *
 * The corpus (CONTRACT/conformance/http_status_errors.json) is generated from
 * the Python transport by scripts/gen_status_corpus.py. Python had no branch
 * for 403, so an access refusal came out transient there and as
 * AuthenticationError here; every other unhandled 4xx was transient in Python
 * and a bare SynapError here. Nothing compared the two.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { HttpTransport } from '../transport/http.js';
import { SynapError, SynapPermanentError } from '../errors.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const inPublicMirror = !existsSync(path.resolve(here, '../../../CONTRACT'));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/http_status_errors.json');

/** Declared, not hidden: each names why it is not fixed here. */
const KNOWN_STATUS_DIFFS: Record<number, { js: string; reason: string }> = {
  504: {
    js: 'AgentUnavailableError',
    reason:
      'Python raises ServiceUnavailableError. The class name differs and is an open product call. ' +
      'Retry behaviour is the same: both treat a 504 as outcome-unknown, so a GET retries and a ' +
      'non-idempotent POST is sent once (python/tests/test_http_504_no_retry_on_create.py).',
  },
};

describe.skipIf(inPublicMirror)('HTTP status -> exception parity with the Python SDK', () => {
  if (inPublicMirror) return;
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as {
    cases: { status: number; raises: string; transient: boolean; permanent: boolean }[];
  };

  it.each(corpus.cases)('HTTP $status raises $raises', async ({ status, raises, transient, permanent }) => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ detail: 'corpus' }), {
        status, headers: { 'content-type': 'application/json' },
      })) as unknown as typeof fetch;
    const t = new HttpTransport({
      credentials: { apiKey: 'k', clientId: 'c', instanceId: 'i' }, fetchImpl, retryPolicy: null,
    });
    const err = await t.request('whoami').then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(SynapError);
    const known = KNOWN_STATUS_DIFFS[status];
    expect((err as Error).name).toBe(known ? known.js : raises);
    expect((err as SynapError).transient).toBe(transient);
    expect(err instanceof SynapPermanentError).toBe(permanent);
  });

  it('declares only the differences it still has', () => {
    for (const status of Object.keys(KNOWN_STATUS_DIFFS).map(Number)) {
      const c = corpus.cases.find((x) => x.status === status);
      expect(c, `HTTP ${status} is declared but not in the corpus`).toBeDefined();
      expect(c!.raises, `HTTP ${status} now matches Python: remove it from KNOWN_STATUS_DIFFS`)
        .not.toBe(KNOWN_STATUS_DIFFS[status]!.js);
    }
  });
});

/**
 * The Documents error codes: a body naming one raises its own class, with the
 * same message and the same fields read from it (snake_case in Python,
 * camelCase here). Python's `code` is `serverCode` here.
 */
describe.skipIf(inPublicMirror)('Documents error codes, against the Python SDK', () => {
  if (inPublicMirror) return;
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as {
    coded_cases: {
      status: number; body: unknown; raises: string; message: string; code: string | null;
      fields: Record<string, unknown>; transient: boolean; permanent: boolean;
    }[];
  };
  const camel = (k: string) => k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

  it.each(corpus.coded_cases.map((c, i) => ({ i, label: `${c.status} ${c.raises}` })))(
    'coded[$i] $label',
    async ({ i }) => {
      const c = corpus.coded_cases[i]!;
      const fetchImpl = (async () =>
        new Response(JSON.stringify(c.body), {
          status: c.status, headers: { 'content-type': 'application/json' },
        })) as unknown as typeof fetch;
      const t = new HttpTransport({
        credentials: { apiKey: 'k', clientId: 'c', instanceId: 'i' }, fetchImpl, retryPolicy: null,
      });
      const err = (await t.request('whoami').then(() => null, (e: unknown) => e)) as SynapError;
      // A coded 504 (knowledge query's query_timeout) differs in the same, already
      // declared way as a plain 504 (KNOWN_STATUS_DIFFS above): an open product call.
      const known = KNOWN_STATUS_DIFFS[c.status];
      expect(err.name).toBe(known ? known.js : c.raises);
      expect(err.serverCode).toBe(c.code);
      expect(err.transient).toBe(c.transient);
      expect(err instanceof SynapPermanentError).toBe(c.permanent);
      if (c.code !== null) expect(err.message).toBe(c.message);
      const rec = err as unknown as Record<string, unknown>;
      for (const [k, v] of Object.entries(c.fields)) {
        expect(rec[camel(k)], k).toEqual(v);
      }
    },
  );
});
