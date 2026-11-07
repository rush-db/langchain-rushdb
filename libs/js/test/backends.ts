/**
 * Backends for the integration tests; the same variables as the Python suite.
 *
 * - `self_hosted`: `RUSHDB_TEST_ENGINE_URL` + `RUSHDB_TEST_ENGINE_TOKEN`
 * - `self_hosted_multitenant`: `RUSHDB_TEST_MT_ENGINE_URL` + `RUSHDB_TEST_MT_ENGINE_TOKEN`
 *   (each client gets its own tenant)
 * - `cloud`: `RUSHDB_TEST_API_KEY` (+ optional `RUSHDB_TEST_URL`)
 *
 * A backend that is not configured is skipped, never silently passed; CI sets
 * `RUSHDB_TEST_REQUIRE` to the backends that must run.
 */
import { randomUUID } from 'node:crypto'

import RushDB from '@rushdb/javascript-sdk'
import { describe, it } from 'vitest'

import type { Client } from '../src/connection.js'

export const BACKENDS = ['self_hosted', 'self_hosted_multitenant', 'cloud'] as const
export type Backend = (typeof BACKENDS)[number]

export function makeClient(backend: Backend): Client | undefined {
  const env = process.env
  if (backend === 'self_hosted' && env.RUSHDB_TEST_ENGINE_URL) {
    return new RushDB(env.RUSHDB_TEST_ENGINE_TOKEN, { direct: true, engineUrl: env.RUSHDB_TEST_ENGINE_URL })
  }
  if (backend === 'self_hosted_multitenant' && env.RUSHDB_TEST_MT_ENGINE_URL) {
    return new RushDB(env.RUSHDB_TEST_MT_ENGINE_TOKEN, {
      direct: true,
      engineUrl: env.RUSHDB_TEST_MT_ENGINE_URL,
      scope: `t${unique().toLowerCase()}`
    })
  }
  if (backend === 'cloud' && env.RUSHDB_TEST_API_KEY) {
    return env.RUSHDB_TEST_URL
      ? new RushDB(env.RUSHDB_TEST_API_KEY, { url: env.RUSHDB_TEST_URL })
      : new RushDB(env.RUSHDB_TEST_API_KEY)
  }
  return undefined
}

/** A suffix that keeps one test's labels apart from every other run. */
export function unique(): string {
  return randomUUID().replace(/-/g, '').slice(0, 10).toUpperCase()
}

/**
 * `describe` once per backend. `body` gets a factory for fresh clients (a new
 * tenant per call on the multitenant engine).
 */
export function describeBackends(name: string, body: (client: () => Client, backend: Backend) => void): void {
  const required = (process.env.RUSHDB_TEST_REQUIRE ?? '').split(',')
  for (const backend of BACKENDS) {
    const configured = makeClient(backend) !== undefined
    if (!configured && required.includes(backend)) {
      describe(`${name} [${backend}]`, () => {
        it('backend is configured', () => {
          throw new Error(`backend ${backend} is required but not configured`)
        })
      })
      continue
    }
    describe.skipIf(!configured)(`${name} [${backend}]`, () => {
      body(() => makeClient(backend) as Client, backend)
    })
  }
}

/**
 * Delete a test's records and the vector and text indexes on its labels.
 * Every test uses fresh labels; leaving their indexes behind would exhaust the
 * engine's 256 vector spaces per tenant.
 */
export async function dropLabels(client: Client, ...labels: string[]): Promise<void> {
  const wanted = new Set(labels)
  for (const label of labels) await client.records.delete({ labels: [label] } as any)
  for (const index of ((await client.ai.indexes.find()).data ?? []) as any[]) {
    if (wanted.has(index.label)) await client.ai.indexes.delete(index.id)
  }
  for (const index of ((await client.textIndexes.find()).data ?? []) as any[]) {
    if (wanted.has(index.label)) await client.textIndexes.delete(index.name)
  }
}
