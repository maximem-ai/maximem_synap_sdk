import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, it, expect } from 'vitest';

// The vendored proto must stay byte-identical to the canonical source of truth.
// Drift here means the gRPC client is missing fields/events the server sends —
// e.g. context_used / context_assembled telemetry or bundle composition fields.
const here = path.dirname(fileURLToPath(import.meta.url));
const vendored = path.resolve(here, '../../proto/synap_service.proto');
const canonical = path.resolve(here, '../../../../synap/proto/synap_service.proto');

describe('proto parity with canonical source of truth', () => {
  it('vendored proto is byte-identical to synap/proto/synap_service.proto', () => {
    if (!existsSync(canonical)) {
      // Canonical only exists inside the monorepo. When the package is consumed
      // standalone (published tarball) there is nothing to compare against.
      return;
    }
    expect(readFileSync(vendored, 'utf8')).toBe(readFileSync(canonical, 'utf8'));
  });
});
