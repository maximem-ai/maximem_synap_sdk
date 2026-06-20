import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // Run type-level assertions (*.test-d.ts) as part of `vitest run`, so CI's
    // `npm test` enforces them. See src/__tests__/sdk-assignability.test-d.ts.
    typecheck: {
      enabled: true,
      include: ['src/**/*.test-d.ts'],
      tsconfig: './tsconfig.json',
    },
  },
});
