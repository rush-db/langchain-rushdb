import { describe, expect, it } from 'vitest'

import { RushDBVectorStore, clientFromEnv, isSelfHosted } from '../src/index.js'

describe('clientFromEnv', () => {
  it('builds a Cloud client', () => {
    expect(isSelfHosted(clientFromEnv({ RUSHDB_API_KEY: 'key', RUSHDB_URL: 'https://example.test' }))).toBe(false)
  })

  it('builds a self-hosted client', () => {
    const client = clientFromEnv({
      RUSHDB_ENGINE_URL: 'http://127.0.0.1:3000',
      RUSHDB_ENGINE_TOKEN: 't',
      RUSHDB_SCOPE: 'tenant-a'
    })
    expect(isSelfHosted(client)).toBe(true)
  })

  it.each([
    [{}],
    [{ RUSHDB_API_KEY: 'k', RUSHDB_ENGINE_URL: 'http://x' }],
    [{ RUSHDB_ENGINE_URL: 'http://x' }]
  ])('refuses %j', (env) => {
    expect(() => clientFromEnv(env)).toThrow()
  })
})

describe('managed embeddings', () => {
  it('are refused on a self-hosted engine', () => {
    const client = clientFromEnv({ RUSHDB_ENGINE_URL: 'http://x', RUSHDB_ENGINE_TOKEN: 't' })
    expect(() => new RushDBVectorStore('managed', { client })).toThrow(/needs RushDB Cloud/)
  })

  it('are allowed on RushDB Cloud', () => {
    const client = clientFromEnv({ RUSHDB_API_KEY: 'k' })
    expect(new RushDBVectorStore('managed', { client }).managed).toBe(true)
  })
})
