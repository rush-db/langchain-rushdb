/** Client construction and the record helpers every component shares. */
import RushDB from '@rushdb/javascript-sdk'

export type Client = InstanceType<typeof RushDB>
export type Row = Record<string, any>

export const PAGE_SIZE = 1000
export const FAR_FUTURE = '9999-12-31T23:59:59.000Z'
export const CONFLICT_RETRIES = 8

/**
 * Build a RushDB client from the environment.
 *
 * RushDB Cloud: `RUSHDB_API_KEY` (and optionally `RUSHDB_URL`).
 * Self-hosted engine: `RUSHDB_ENGINE_URL` and `RUSHDB_ENGINE_TOKEN` (and
 * `RUSHDB_SCOPE` on a multitenant engine). Setting both is an error.
 */
export function clientFromEnv(env: Record<string, string | undefined> = process.env): Client {
  const apiKey = env.RUSHDB_API_KEY
  const engineUrl = env.RUSHDB_ENGINE_URL
  if (apiKey && engineUrl) {
    throw new Error(
      'Both RUSHDB_API_KEY (RushDB Cloud) and RUSHDB_ENGINE_URL (self-hosted) are set; ' +
        'set the variables of one mode only.'
    )
  }
  if (engineUrl) {
    const token = env.RUSHDB_ENGINE_TOKEN
    if (!token) throw new Error('RUSHDB_ENGINE_URL is set but RUSHDB_ENGINE_TOKEN is not.')
    return new RushDB(token, { direct: true, engineUrl, scope: env.RUSHDB_SCOPE || undefined })
  }
  if (apiKey) {
    return env.RUSHDB_URL ? new RushDB(apiKey, { url: env.RUSHDB_URL }) : new RushDB(apiKey)
  }
  throw new Error(
    'No RushDB connection configured: set RUSHDB_API_KEY for RushDB Cloud, or ' +
      'RUSHDB_ENGINE_URL and RUSHDB_ENGINE_TOKEN for a self-hosted engine.'
  )
}

export function isSelfHosted(client: Client): boolean {
  return client.direct !== undefined
}

/** The data of every record `search` matches, page by page. */
export async function findAll(client: Client, search: Row): Promise<Row[]> {
  const out: Row[] = []
  for (let skip = 0; ; skip += PAGE_SIZE) {
    const page = await client.records.find({ ...search, skip, limit: PAGE_SIZE } as any)
    const rows = page.data.map((record: any) => record.data as Row)
    out.push(...rows)
    if (rows.length < PAGE_SIZE) return out
  }
}

/**
 * Create `rows` in one engine commit, each with its inline `vectors`.
 *
 * `records.importJson` rather than `createMany`: rows may hold flat arrays,
 * which the engine stores as array properties.
 */
export async function createRows(
  client: Client,
  label: string,
  rows: Row[],
  vectors?: Array<Row[] | undefined>,
  transaction?: any
): Promise<void> {
  if (!rows.length) return
  const data = rows.map((row, i) => (vectors?.[i]?.length ? { ...row, $vectors: vectors[i] } : row))
  await client.records.importJson(
    { label, data, options: { returnResult: false, suggestTypes: true } as any },
    transaction
  )
}

export async function deleteIds(client: Client, ids: string[]): Promise<void> {
  if (ids.length) await client.records.deleteById(ids)
}

/**
 * Delete records by id and create rows in one transaction.
 *
 * Concurrent transactions can conflict; the engine reports that as retryable,
 * so the transaction is replayed with backoff.
 */
export async function transact(
  client: Client,
  remove: string[],
  create: Array<[string, Row[]]>
): Promise<void> {
  if (!remove.length && !create.some(([, rows]) => rows.length)) return
  for (let attempt = 0; attempt < CONFLICT_RETRIES; attempt += 1) {
    // Commit and roll back through `client`: the SDK's `Transaction` methods
    // use the most recently created client, not the one that began it.
    const tx = await client.tx.begin()
    let committing = false
    try {
      for (const id of remove) await client.records.deleteById(id, tx)
      for (const [label, rows] of create) await createRows(client, label, rows, undefined, tx)
      committing = true
      await client.tx.commit(tx)
      return
    } catch (error) {
      // A failed commit has already ended the transaction.
      if (!committing) await rollback(client, tx, error)
      const conflict = String((error as Error)?.message ?? error).includes('transaction_conflict')
      if (!conflict || attempt === CONFLICT_RETRIES - 1) throw error
      await sleep(20 * 2 ** attempt * (1 + Math.random()))
    }
  }
}

async function rollback(client: Client, tx: unknown, cause: unknown): Promise<void> {
  try {
    await client.tx.rollback(tx as never)
  } catch (error) {
    console.warn(`[@rushdb/langchain] rollback after ${String(cause)} also failed: ${String(error)}`)
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export function isScalar(value: unknown): value is string | number | boolean {
  return (
    typeof value === 'string' ||
    typeof value === 'boolean' ||
    (typeof value === 'number' && Number.isFinite(value))
  )
}

/** The filterable properties of `obj`: top-level scalars and flat arrays of one kind. */
export function mirror(prefix: string, obj: Record<string, unknown>): Row {
  const out: Row = {}
  for (const [key, value] of Object.entries(obj)) {
    if (isScalar(value)) {
      out[`${prefix}.${key}`] = value
    } else if (
      Array.isArray(value) &&
      value.length &&
      value.every(isScalar) &&
      new Set(value.map((v) => typeof v)).size === 1
    ) {
      out[`${prefix}.${key}`] = [...value]
    }
  }
  return out
}

export function toJson(value: unknown): string {
  return JSON.stringify(value)
}

export function b64(data: Uint8Array): string {
  return Buffer.from(data).toString('base64')
}

export function unb64(data: string): Uint8Array {
  return new Uint8Array(Buffer.from(data, 'base64'))
}

export function iso(date: Date): string {
  return date.toISOString()
}

export function nowIso(): string {
  return new Date().toISOString()
}
