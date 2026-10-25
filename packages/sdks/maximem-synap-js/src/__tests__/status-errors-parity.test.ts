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
  405: { js: 'InvalidInputError', reason: 'An unsupported image route is caller input, not retryable.' },
  413: { js: 'InvalidInputError', reason: 'An over-limit image payload is caller input, not retryable.' },
  415: { js: 'InvalidInputError', reason: 'An unsupported image media type is caller input, not retryable.' },
  504: {
    js: 'AgentUnavailableError',
    reason:
      'Python raises ServiceUnavailableError. Both are transient, so retrying is ' +
      'the same; which class a 504 should be is an open product call.',
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
