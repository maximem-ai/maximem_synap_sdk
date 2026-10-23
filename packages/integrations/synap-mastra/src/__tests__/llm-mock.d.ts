// @mastra/core ships this test helper without a declaration file.
declare module '@mastra/core/test-utils/llm-mock' {
  export function createMockModel(options: Record<string, unknown>): unknown;
}
