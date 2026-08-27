import { defineConfig } from 'tsup';

// Dual ESM + CJS. Both are required: the ecosystem is mid-migration, and a
// CJS-only or ESM-only build strands one half of it. The cost is the dual-package
// hazard -- a consumer can end up with two copies of every error class, making
// `err instanceof SynapError` silently false. That is why every error also
// carries a `.code` string (see src/errors.ts and gotcha G-G).
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'grpc/index': 'src/grpc/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  target: 'node20',
  // Splitting is REQUIRED, not a size optimisation. `instance.listen()` reaches
  // the stream client through a lazy import(); without splitting, esbuild
  // inlines that local module into the main entry and drags the
  // `import('@grpc/grpc-js')` call in with it. A bundler building for Edge then
  // follows it and fails on node:http2 -- which is the whole reason gRPC lives
  // behind a subpath. With splitting on, the stream client is its own chunk and
  // the main entry never mentions grpc. Guarded by bundle-hygiene.test.ts.
  splitting: true,
  // The gRPC packages are optionalDependencies loaded through a lazy import().
  // Bundling them drags ~5MB of transitive deps into every build, including
  // Edge builds that can never execute them.
  //
  // proto-loader MUST be listed alongside grpc-js. Leaving it out bundled
  // proto-loader AND protobufjs into the main entry and took dist/index.js
  // from 72KB to 463KB, which is exactly the cost the subpath split exists to
  // avoid. `npm run size` guards the number now.
  external: ['@grpc/grpc-js', '@grpc/proto-loader', 'protobufjs', 'undici'],
});
