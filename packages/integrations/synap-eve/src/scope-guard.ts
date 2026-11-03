/**
 * Two refusals that keep a fixed `scopePath` honest.
 *
 * 1. `ensureIdsMatchPath`: per-call ids must agree with the path.
 *    The server lets the path win: it takes `user_id` and `customer_id` from the
 *    path keys `user` and `customer` over the flat ids it is also sent. A package
 *    that fixes a path when it is built but takes the user per call (an eve
 *    session principal) would let one caller read and write another's memory
 *    while believing it addressed its own. Call this before every SDK call that
 *    sends both, and once at construction where the ids are fixed too.
 *
 * 2. `assertSdkSupportsScopePath`: an SDK older than 0.7.1 silently ignores the
 *    path (`sdk.fetch` sends ids only; `send_message` drops it), so a call that
 *    was meant for one level would be filed and read at another. Refuse at
 *    construction instead. The marker is `get_scope_ladder`, which exists on
 *    every SDK that carries the path and on none that does not.
 *
 * `ensureIdsMatchPath` mirrors `ensure_ids_match_path` in
 * `synap_integrations_common/scope.py`: same rules, same message text. Keep
 * them in step.
 *
 * ⚠ This file is vendored, not imported: byte-identical in `synap-eve`,
 * `synap-mastra`, `synap-claude-agent-ts` and `synap-vercel-adk`, for the
 * reason `stream-events.ts` is (no shared runtime package). The vendoring tests
 * fail the build if a copy drifts. Change one, copy it to the other three.
 */

/** The lowest `@maximem/synap-js-sdk` that sends `scope_path`. */
export const MIN_SDK_VERSION_FOR_SCOPE_PATH = '0.7.1';

/**
 * Raised when per-call ids disagree with a fixed scope path, or the path is
 * malformed. Same shape as Python's `SynapIntegrationError`: `operation`,
 * `context`, message `"<operation>: <message>"`. Never carries a raw id.
 */
export class SynapScopeError extends Error {
  readonly operation: string;
  readonly context: Record<string, unknown>;

  constructor(operation: string, message: string, context?: Record<string, unknown>) {
    super(`${operation}: ${message}`);
    this.name = 'SynapScopeError';
    this.operation = operation;
    this.context = context ?? {};
  }
}

// SHA-256 in plain JS. `node:crypto` is not importable on the Edge and Workers
// builds (see util/correlation.ts in synap-vercel-adk), and a fingerprint has to
// be synchronous because it is built inside an error message. The first 8 hex
// characters equal Python's `sha256(...).hexdigest()[:8]`, so a log line from the
// Python and the TypeScript packages can be matched; `scope-guard.test.ts` checks
// this against `node:crypto`.
const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function sha256Hex(message: string): string {
  const bytes = new TextEncoder().encode(message);
  const bitLen = bytes.length * 8;
  const padded = new Uint8Array(((bytes.length + 9 + 63) >> 6) << 6);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(bitLen / 0x100000000));
  view.setUint32(padded.length - 4, bitLen >>> 0);
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const w = new Uint32Array(64);
  const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const t1 = (hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  return Array.from(h, (x) => x.toString(16).padStart(8, '0')).join('');
}

/** A short hash of an id, safe to put in an error message or a log. */
export function fingerprint(value: unknown): string {
  return sha256Hex(String(value)).slice(0, 8);
}

/**
 * Refuse per-call ids that disagree with a scope path fixed elsewhere.
 *
 * - `scopePath` falsy or empty: no-op.
 * - For each reserved key (`user`, `customer`) present in the path: an id that
 *   is set and differs throws; an empty / absent id is fine (the path supplies it).
 * - B2C: a path whose `customer` equals its `user` with `customerId` left out
 *   passes, as does a `customerId` equal to that value.
 * - The path itself must map non-empty string keys to non-empty string values
 *   without `/`; otherwise it throws.
 *
 * Messages carry short fingerprints of the ids, never the ids themselves.
 */
export function ensureIdsMatchPath(
  scopePath: Readonly<Record<string, string>> | null | undefined,
  ids: { userId?: string | null; customerId?: string | null; where?: string } = {},
): void {
  const op = 'ensure_ids_match_path';
  const where = ids.where ?? '';
  const whereTxt = where ? ` (${where})` : '';
  // Falsy (Python's `not scope_path`): nothing to check.
  if (!scopePath || (Array.isArray(scopePath) && scopePath.length === 0)) return;
  if (typeof scopePath !== 'object' || Array.isArray(scopePath)) {
    throw new SynapScopeError(op, `scope_path must be a mapping${whereTxt}.`);
  }
  const entries = Object.entries(scopePath);
  if (entries.length === 0) return;
  for (const [key, value] of entries) {
    if (typeof key !== 'string' || key === '') {
      throw new SynapScopeError(op, `scope_path keys must be non-empty strings${whereTxt}.`);
    }
    if (typeof value !== 'string' || value === '' || value.includes('/')) {
      throw new SynapScopeError(
        op,
        `scope_path value for level '${key}' must be a non-empty string without '/'${whereTxt}.`,
      );
    }
  }
  const pairs: Array<['user' | 'customer', string | null | undefined]> = [
    ['user', ids.userId],
    ['customer', ids.customerId],
  ];
  for (const [key, given] of pairs) {
    if (
      Object.prototype.hasOwnProperty.call(scopePath, key) &&
      given !== undefined && given !== null && given !== '' &&
      given !== scopePath[key]
    ) {
      throw new SynapScopeError(
        op,
        `${key}_id does not match the scope_path's '${key}' level${whereTxt}: ` +
          `id fingerprint ${fingerprint(given)}, path fingerprint ` +
          `${fingerprint(scopePath[key])}. The server would use the path, ` +
          'so this call would read or write another identity\'s memory.',
        { key, id_fp: fingerprint(given), path_fp: fingerprint(scopePath[key]), where },
      );
    }
  }
}

/**
 * Throw at construction when a `scopePath` is set on an SDK that cannot carry
 * it. No-op without a path, so a package used without one never changes.
 */
export function assertSdkSupportsScopePath(
  sdk: unknown,
  scopePath: Readonly<Record<string, string>> | null | undefined,
  where: string,
): void {
  if (!scopePath || Object.keys(scopePath).length === 0) return;
  const marker = (sdk as { get_scope_ladder?: unknown } | null | undefined)?.get_scope_ladder;
  if (typeof marker !== 'function') {
    throw new TypeError(
      `${where}: scopePath needs @maximem/synap-js-sdk >= ${MIN_SDK_VERSION_FOR_SCOPE_PATH}. ` +
        'The SDK passed in has no get_scope_ladder, so it is older and would silently ignore the ' +
        'path and read and write at the wrong level. Upgrade the SDK, or remove scopePath.',
    );
  }
}
