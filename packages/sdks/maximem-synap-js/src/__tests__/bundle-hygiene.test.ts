import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Guards what must NOT be in the published main entry.
 *
 * This exists because the invariant broke silently. Adding
 * `@grpc/proto-loader` as a runtime dependency without adding it to tsup's
 * `external` bundled proto-loader AND protobufjs into `dist/index.js`, taking
 * it from 72KB to 463KB. Separately, `splitting: false` inlined the lazily
 * imported stream client into the main entry and brought the
 * `import('@grpc/grpc-js')` call with it, which is what makes an Edge build
 * fail on `node:http2`.
 *
 * Neither showed up in any test: the SDK worked fine in Node. It would have
 * been found by a customer at deploy time.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(here, '../../dist');

const ENTRIES = ['index.js', 'index.cjs'];
/**
 * Only reachable through the `/grpc` subpath or a lazy chunk, never the main
 * entry: gRPC, and the local embedding model's native runtime and tokenizer
 * (cache/onnx-encoder.ts).
 */
const FORBIDDEN = [
  '@grpc/grpc-js', '@grpc/proto-loader', 'protobufjs', 'ContextBundleProto',
  'onnxruntime-node', '@huggingface/tokenizers',
];

describe('published bundle hygiene', () => {
  const built = existsSync(path.join(dist, 'index.js'));

  it.runIf(built).each(ENTRIES)('%s does not reference gRPC, the proto or the local model runtime', (entry) => {
    const source = readFileSync(path.join(dist, entry), 'utf8');
    for (const needle of FORBIDDEN) {
      expect(
        source.includes(needle),
        `${entry} references ${needle}; it must stay behind the /grpc subpath. ` +
          'Check tsup external + splitting.',
      ).toBe(false);
    }
  });

  it.runIf(built).each(ENTRIES)('%s imports no node builtins eagerly', (entry) => {
    const source = readFileSync(path.join(dist, entry), 'utf8');
    // A static `node:` import fails the build on Edge and Workers. The only
    // permitted form is a lazy import() inside the code path that needs it.
    const statics = [
      ...source.matchAll(/^\s*import\s+[^;]*?from\s*['"](node:[^'"]+)['"]/gm),
      ...source.matchAll(/^\s*require\(\s*['"](node:[^'"]+)['"]\s*\)/gm),
    ].map((m) => m[1]);
    expect(statics).toEqual([]);
  });

  it.runIf(built)('keeps the main entry well under the pre-regression ceiling', () => {
    // 463KB was the accident. 200KB leaves headroom for real features while
    // still failing loudly if a heavy transitive dep gets bundled again.
    // (Raised from 120KB: 0.5.8 shipped at 124KB and history search, ranked
    // results and semantic matching took the entry to 170KB.)
    const bytes = readFileSync(path.join(dist, 'index.js'), 'utf8').length;
    expect(bytes / 1024).toBeLessThan(200);
  });

  it.runIf(built)('still exposes gRPC through the subpath', () => {
    const source = readFileSync(path.join(dist, 'grpc', 'index.js'), 'utf8');
    expect(source).toContain('@grpc/grpc-js');
  });
});
