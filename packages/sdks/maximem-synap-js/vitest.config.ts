import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/__tests__/**/*.test.ts'],
    environment: 'node',
    // No test downloads or runs the real embedding model: a test that wants
    // one passes its own encoder.
    env: { SYNAP_SDK_LOCAL_MODEL: 'off' },
  },
});
