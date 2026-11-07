import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    // @langchain/langgraph-checkpoint-validation registers its suite with the
    // global describe/it/expect.
    globals: true,
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // Integration files share engines; run them one after another.
    fileParallelism: false
  }
})
