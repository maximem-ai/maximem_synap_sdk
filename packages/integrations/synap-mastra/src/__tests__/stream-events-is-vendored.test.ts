/**
 * The guard that makes vendoring safe instead of a drift factory.
 *
 * `stream-events.ts` is copied, not imported. The three TypeScript
 * integrations (`claude-agent-ts`, `mastra`, `eve`) declare zero runtime
 * dependencies on purpose: they duck-type the SDK so a user brings their own
 * version of it. A shared package would be a new public npm name that has to
 * be published before any of the three could be, which blocks shipping.
 *
 * The cost of that choice is drift. This test is what pays it: it reads all
 * three copies off disk and fails the build the moment one of them differs by
 * a single byte. Change one, copy it to the other two.
 *
 * It reads the siblings through relative paths, so it only runs from a checkout
 * of the monorepo. Published tarballs ship `dist` and `README.md` only, so no
 * user ever sees this file.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

/** `<repo>/integrations`, from `<repo>/integrations/<pkg>/src/__tests__`. */
const INTEGRATIONS = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

/** Every package that carries a copy. The first is the one the others match. */
const PACKAGES = ['synap-claude-agent-ts', 'synap-mastra', 'synap-eve'] as const;

/** Both files are vendored: the helper and the tests that hold it honest. */
const VENDORED = [
  ['stream-events.ts', join('src', 'stream-events.ts')],
  ['stream-events.test.ts', join('src', '__tests__', 'stream-events.test.ts')],
  // This file too. Otherwise somebody can weaken the guard in one package and
  // the other two never notice, which is the same drift one level up.
  ['stream-events-is-vendored.test.ts',
    join('src', '__tests__', 'stream-events-is-vendored.test.ts')],
] as const;

function copyPath(pkg: string, relative: string): string {
  return join(INTEGRATIONS, pkg, relative);
}

describe('the vendored stream helper is byte-identical in all three packages', () => {
  it('finds a copy in every package', () => {
    // A missing copy is the other way this drifts: a package quietly stops
    // carrying the helper and this file would otherwise compare one file
    // against itself and pass.
    for (const pkg of PACKAGES) {
      for (const [, relative] of VENDORED) {
        expect(existsSync(copyPath(pkg, relative)), `${pkg}/${relative} is missing`).toBe(true);
      }
    }
  });

  for (const [name, relative] of VENDORED) {
    it(`${name} is the same file in ${PACKAGES.join(', ')}`, () => {
      const [reference, ...rest] = PACKAGES;
      const expected = readFileSync(copyPath(reference, relative));

      // A truncated or emptied helper would make every copy "identical".
      expect(expected.length, `${reference}/${relative} is suspiciously short`)
        .toBeGreaterThan(1000);

      for (const pkg of rest) {
        const actual = readFileSync(copyPath(pkg, relative));
        // Compared as text first, because that is the assertion that prints a
        // usable diff when somebody edits one copy.
        expect(
          actual.toString('utf8'),
          `${pkg}/${relative} has drifted from ${reference}/${relative}`,
        ).toBe(expected.toString('utf8'));
        // And as bytes, which also catches an encoding or BOM difference that
        // decodes to the same text.
        expect(
          actual.equals(expected),
          `${pkg}/${relative} decodes the same but is not byte-identical`,
        ).toBe(true);
      }
    });
  }
});
