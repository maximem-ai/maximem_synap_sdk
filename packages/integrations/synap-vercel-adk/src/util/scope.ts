/**
 * The scope-ladder path: how it is written on the wire and how it is compared.
 *
 * `scopePath` is the level of an account's scope ladder a call reads or writes
 * at, for an account whose ladder has a level between customer and user, for
 * example `{ customer: 'acme', team: 'eng', user: 'u1' }`. Unset on a default
 * ladder, where every request is exactly what it was.
 *
 * On the wire the key is `scope` in every REST body (context fetch, conversation
 * record) and the map field `scope` (14) on a stream conversation event. Omitted
 * when absent or empty, as the JS SDK does (`fetch.ts`, `conversation/interface.ts`).
 */

/** `{ scope: {...} }` when a path was given and has entries, else `{}`, so an unchanged request stays byte-identical. */
export function scopeField(
  path: Readonly<Record<string, string>> | undefined | null,
): { scope?: Record<string, string> } {
  return path && Object.keys(path).length > 0 ? { scope: { ...path } } : {};
}

// The rung a path names, written so a request and a pushed bundle compare as
// strings. The recipe is the shared contract's `rung_key`
// (synap/sdk/CONTRACT/behavior/anticipation.json) and is written once per
// implementation by necessity; its golden vectors are checked in
// `__tests__/scope-path.test.ts`. Do not re-tune it here.
const RUNG_SEPARATOR = '|';
const RUNG_ASSIGN = '=';

/** The rung a path names, canonically, or null for no path. */
export function rungKey(path: Readonly<Record<string, string>> | null | undefined): string | null {
  if (!path) return null;
  const pairs: [string, string][] = [];
  for (const [k, v] of Object.entries(path)) {
    if (k === '' || v === null || v === undefined || String(v) === '') continue;
    pairs.push([String(k), String(v)]);
  }
  if (pairs.length === 0) return null;
  pairs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return pairs
    .map(([k, v]) => {
      const escaped = v
        .split('\\').join('\\\\')
        .split(RUNG_SEPARATOR).join('\\p')
        .split(RUNG_ASSIGN).join('\\e');
      return `${k}${RUNG_ASSIGN}${escaped}`;
    })
    .join(RUNG_SEPARATOR);
}

/**
 * May a bundle stamped `bundleRung` answer a request at `requestRung`? Equality,
 * on purpose: a subset rule would hand one person's context to another. Absent
 * on both sides (the proto's empty string counts as absent) serves, so a caller
 * with no path sees no change.
 */
export function rungServes(
  bundleRung: string | null | undefined,
  requestRung: string | null | undefined,
): boolean {
  return (bundleRung || null) === (requestRung || null);
}
