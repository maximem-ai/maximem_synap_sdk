/**
 * `SDK_VERSION` must equal the version in package.json.
 *
 * The release workflow has a step that syncs them, and that step is the only
 * thing that ever did. A publish run by hand skips it, and 0.4.8 shipped that
 * way: the package was 0.4.8 and every request it made announced 0.4.7 in its
 * User-Agent, with every telemetry event reporting the same wrong number.
 *
 * Nothing catches that at review time, because both files look correct on
 * their own. This test is what makes them wrong together.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { SDK_VERSION } from '../version.js';

describe('the version the SDK reports', () => {
  it('is the version the package publishes', () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(
      readFileSync(join(here, '..', '..', 'package.json'), 'utf8'),
    ) as { version: string };

    expect(SDK_VERSION).toBe(pkg.version);
  });
});
