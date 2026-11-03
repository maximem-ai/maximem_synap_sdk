/**
 * `scope-guard.ts` (and its test) is copied, not imported: this package carries
 * the same file as `synap-eve`, `synap-mastra` and `synap-claude-agent-ts`, for
 * the same reason they carry `stream-events.ts` (no shared runtime package). Their
 * own vendoring test covers those three; this one holds this copy to them.
 *
 * It reads siblings through relative paths, so it only runs from a checkout of
 * the monorepo. Published tarballs ship `dist` and `README.md` only.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const HERE = dirname(fileURLToPath(import.meta.url));
const MINE = join(HERE, '..', '..');
const INTEGRATIONS = join(MINE, '..');
const REFERENCE = 'synap-eve';
const FILES = [
  join('src', 'scope-guard.ts'),
  join('src', '__tests__', 'scope-guard.test.ts'),
] as const;

describe('scope-guard is byte-identical to the other TypeScript integrations', () => {
  for (const relative of FILES) {
    it(`${relative} matches ${REFERENCE}`, () => {
      const reference = join(INTEGRATIONS, REFERENCE, relative);
      expect(existsSync(reference), `${REFERENCE}/${relative} is missing`).toBe(true);
      const expected = readFileSync(reference);
      expect(expected.length, 'suspiciously short').toBeGreaterThan(1000);
      const actual = readFileSync(join(MINE, relative));
      expect(actual.toString('utf8'), `${relative} has drifted from ${REFERENCE}`)
        .toBe(expected.toString('utf8'));
      expect(actual.equals(expected)).toBe(true);
    });
  }
});
