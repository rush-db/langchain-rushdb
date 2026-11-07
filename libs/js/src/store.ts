/** RushDB-backed LangGraph long-term memory store. */
import {
  BaseStore,
  type GetOperation,
  type IndexConfig,
  type Item,
  type ListNamespacesOperation,
  type MatchCondition,
  type Operation,
  type OperationResults,
  type PutOperation,
  type SearchItem,
  type SearchOperation,
  getTextAtPath
} from '@langchain/langgraph-checkpoint'

import {
  type Client,
  type Row,
  FAR_FUTURE,
  clientFromEnv,
  createRows,
  deleteIds,
  findAll,
  isScalar,
  iso,
  mirror,
  nowIso,
  toJson,
  transact
} from './connection.js'

export const VALUE_PREFIX = 'value'
const SEARCH_POOL_MAX = 10_000

/** LangGraph's TTL settings, in minutes. */
export interface TTLConfig {
  /** TTL of every item put; none when omitted. */
  defaultTtl?: number
  /** Reads extend an item's expiry by its TTL (default true). */
  refreshOnRead?: boolean
  /** Interval of the sweeper `start()` runs (default 5). */
  sweepIntervalMinutes?: number
}

export interface RushDBStoreConfig {
  /** A RushDB client; built from the environment when omitted. */
  client?: Client
  /** `dims`, `embeddings` and optional `fields`, to enable semantic `search({ query })`. */
  index?: IndexConfig
  /**
   * RushDB has no record TTL, so expiry is an `expires_at` property every read
   * filters on; `sweep()` (or the sweeper `start()` runs) deletes expired items.
   */
  ttl?: TTLConfig
  itemLabel?: string
  vectorLabel?: string
}

export function namespaceString(namespace: string[]): string {
  return namespace.join('.')
}

/** Records whose namespace equals `prefix` or starts with it. */
export function prefixWhere(prefix: string[]): Record<string, unknown> {
  if (!prefix.length) return {}
  const ns = namespaceString(prefix)
  return { $or: [{ ns }, { ns: { $startsWith: `${ns}.` } }] }
}

/**
 * LangGraph `BaseStore` on RushDB, Cloud or self-hosted alike.
 *
 * Items are records of `itemLabel`; with `index` configured, every embedded
 * text of an item is a record of `vectorLabel` carrying an external vector
 * (see the repository's `spec/README.md`).
 */
export class RushDBStore extends BaseStore {
  readonly client: Client
  readonly itemLabel: string
  readonly vectorLabel: string
  readonly ttl?: TTLConfig
  readonly indexConfig?: IndexConfig
  private readonly indexFields: string[]
  private vectorIndexReady = false
  private sweeper?: ReturnType<typeof setInterval>

  constructor(config: RushDBStoreConfig = {}) {
    super()
    this.client = config.client ?? clientFromEnv()
    this.itemLabel = config.itemLabel ?? 'LG_STORE_ITEM'
    this.vectorLabel = config.vectorLabel ?? 'LG_STORE_VECTOR'
    this.ttl = config.ttl
    this.indexConfig = config.index
    this.indexFields = config.index ? (config.index.fields ?? ['$']) : []
  }

  // ── BaseStore ──────────────────────────────────────────────────────────

  async batch<Op extends Operation[]>(operations: Op): Promise<OperationResults<Op>> {
    const results: unknown[] = []
    for (const op of operations) {
      if ('namespacePrefix' in op) {
        results.push(await this.searchOp(op as SearchOperation))
      } else if ('value' in op) {
        await this.putOp(op as PutOperation)
        results.push(undefined)
      } else if ('key' in op && 'namespace' in op) {
        results.push(await this.getOp(op as GetOperation))
      } else {
        results.push(await this.listNamespacesOp(op as ListNamespacesOperation))
      }
    }
    return results as OperationResults<Op>
  }

  /** Starts the TTL sweeper when the store has a TTL. */
  start(): void {
    if (this.ttl) this.startTtlSweeper()
  }

  stop(): void {
    this.stopTtlSweeper()
  }

  // ── get ────────────────────────────────────────────────────────────────

  private live(): Record<string, unknown> {
    return { expires_at: { $gt: nowIso() } }
  }

  private get refreshOnRead(): boolean {
    return this.ttl?.refreshOnRead ?? true
  }

  private async itemRows(namespace: string[], key: string): Promise<Row[]> {
    return findAll(this.client, {
      labels: [this.itemLabel],
      where: { ns: namespaceString(namespace), key }
    })
  }

  private async getOp(op: GetOperation): Promise<Item | null> {
    const rows = (await this.itemRows(op.namespace, op.key)).filter((r) => !expired(r))
    if (!rows.length) return null
    if (this.refreshOnRead) await this.refresh([rows[0]])
    return toItem(rows[0])
  }

  private async refresh(rows: Row[]): Promise<void> {
    for (const row of rows) {
      const minutes = row.ttl_minutes
      if (typeof minutes !== 'number') continue
      const expires = iso(new Date(Date.now() + minutes * 60_000))
      await this.client.records.update({
        target: row.__id,
        label: this.itemLabel,
        data: { expires_at: expires }
      } as any)
      row.expires_at = expires
    }
  }

  // ── put / delete ───────────────────────────────────────────────────────

  private async putOp(op: PutOperation): Promise<void> {
    const existing = await this.itemRows(op.namespace, op.key)
    const oldVectors = await this.vectorIds(op.namespace, op.key)
    if (op.value === null) {
      await deleteIds(this.client, [...existing.map((r) => r.__id as string), ...oldVectors])
      return
    }
    const now = nowIso()
    const createdAt = (existing[0]?.created_at as string | undefined) ?? now
    const ttl = this.ttl?.defaultTtl
    const expiresAt = ttl !== undefined ? iso(new Date(Date.now() + ttl * 60_000)) : FAR_FUTURE
    const value = { ...op.value }
    const mirrored = mirror(VALUE_PREFIX, value)
    const item: Row = {
      ns: namespaceString(op.namespace),
      ns_depth: op.namespace.length,
      key: op.key,
      lg_value: toJson(value),
      created_at: createdAt,
      updated_at: now,
      expires_at: expiresAt,
      ...mirrored
    }
    if (ttl !== undefined) item.ttl_minutes = ttl
    await transact(
      this.client,
      existing.map((r) => r.__id as string),
      [[this.itemLabel, [item]]]
    )
    await deleteIds(this.client, oldVectors)
    const texts = this.texts(value, op.index)
    if (texts.length && this.indexConfig) {
      const vectors = await this.indexConfig.embeddings.embedDocuments(texts.map(([, text]) => text))
      await this.ensureVectorIndex()
      const rows = texts.map(([field, text]) => ({
        ns: item.ns,
        ns_depth: item.ns_depth,
        key: op.key,
        field,
        text,
        ...mirrored
      }))
      await createRows(
        this.client,
        this.vectorLabel,
        rows,
        vectors.map((vector) => [{ propertyName: 'text', vector, similarityFunction: 'cosine' }])
      )
    }
  }

  private texts(value: Record<string, unknown>, index: PutOperation['index']): Array<[string, string]> {
    if (!this.indexConfig || index === false) return []
    const fields = Array.isArray(index) ? index : this.indexFields
    const out: Array<[string, string]> = []
    for (const path of fields) {
      for (const text of getTextAtPath(value, path)) {
        if (text) out.push([path, text])
      }
    }
    return out
  }

  private async vectorIds(namespace: string[], key: string): Promise<string[]> {
    if (!this.indexConfig) return []
    const rows = await findAll(this.client, {
      labels: [this.vectorLabel],
      where: { ns: namespaceString(namespace), key }
    })
    return rows.map((r) => r.__id as string)
  }

  private async ensureVectorIndex(): Promise<void> {
    if (this.vectorIndexReady || !this.indexConfig) return
    const dims = this.indexConfig.dims
    const existing: Row[] = ((await this.client.ai.indexes.find()).data ?? []) as Row[]
    const found = existing.some(
      (index) =>
        index.label === this.vectorLabel &&
        index.propertyName === 'text' &&
        index.sourceType === 'external' &&
        index.dimensions === dims &&
        index.similarityFunction === 'cosine'
    )
    if (!found) {
      await this.client.ai.indexes.create({
        label: this.vectorLabel,
        propertyName: 'text',
        external: true,
        dimensions: dims,
        similarityFunction: 'cosine'
      })
    }
    this.vectorIndexReady = true
  }

  // ── search ─────────────────────────────────────────────────────────────

  private async searchOp(op: SearchOperation): Promise<SearchItem[]> {
    const limit = op.limit ?? 10
    const offset = op.offset ?? 0
    const [pushed0, exact] = pushdown(op.filter)
    let pushed = pushed0
    if (op.namespacePrefix.length) {
      const prefix = prefixWhere(op.namespacePrefix)
      pushed = Object.keys(pushed).length ? { $and: [pushed, prefix] } : prefix
    }
    let items: SearchItem[]
    if (op.query && this.indexConfig) {
      // Liveness is checked on the item: vectors carry no expiry.
      items = await this.vectorSearch(op, pushed, offset, limit)
    } else {
      const where = Object.keys(pushed).length ? { $and: [this.live(), pushed] } : this.live()
      const search = { labels: [this.itemLabel], where, orderBy: { updated_at: 'desc' } }
      let rows: Row[]
      if (exact) {
        const page = await this.client.records.find({ ...search, skip: offset, limit } as any)
        rows = page.data.map((r: any) => r.data as Row)
      } else {
        const matched = (await findAll(this.client, search)).filter((r) => matches(value(r), op.filter))
        rows = matched.slice(offset, offset + limit)
      }
      items = rows.map((r) => toSearchItem(r, undefined))
    }
    if (this.refreshOnRead && items.length) await this.refreshItems(items)
    return items
  }

  private async vectorSearch(
    op: SearchOperation,
    where: Record<string, unknown>,
    offset: number,
    limit: number
  ): Promise<SearchItem[]> {
    const index = this.indexConfig as IndexConfig
    const vector = await index.embeddings.embedQuery(op.query ?? '')
    await this.ensureVectorIndex()
    const wanted = offset + limit
    let pool = Math.max(wanted * Math.max(1, this.indexFields.length) * 2, 10)
    for (;;) {
      const params: Row = {
        labels: [this.vectorLabel],
        propertyName: 'text',
        queryVector: vector,
        similarityFunction: 'cosine',
        limit: pool
      }
      if (Object.keys(where).length) params.where = where
      const result = await this.client.records.vectorSearch(params as any)
      const hits = result.data.map((r: any) => r.data as Row)
      const best = new Map<string, [string, string, number]>()
      for (const hit of hits) {
        const ident = JSON.stringify([hit.ns, hit.key])
        const score = Number(hit.__score ?? 0)
        const seen = best.get(ident)
        if (!seen || score > seen[2]) best.set(ident, [hit.ns, hit.key, score])
      }
      const ranked = [...best.values()].sort((a, b) => b[2] - a[2])
      const items: SearchItem[] = []
      for (const [ns, key, score] of ranked) {
        const rows = (
          await findAll(this.client, { labels: [this.itemLabel], where: { ns, key } })
        ).filter((r) => !expired(r))
        if (rows.length && matches(value(rows[0]), op.filter)) items.push(toSearchItem(rows[0], score))
        if (items.length >= wanted) break
      }
      if (items.length >= wanted || hits.length < pool || pool >= SEARCH_POOL_MAX) {
        return items.slice(offset, offset + limit)
      }
      pool = Math.min(pool * 4, SEARCH_POOL_MAX)
    }
  }

  private async refreshItems(items: SearchItem[]): Promise<void> {
    for (const item of items) {
      const rows = (await this.itemRows(item.namespace, item.key)).filter(
        (r) => typeof r.ttl_minutes === 'number'
      )
      await this.refresh(rows)
    }
  }

  // ── namespaces ─────────────────────────────────────────────────────────

  private async listNamespacesOp(op: ListNamespacesOperation): Promise<string[][]> {
    const rows = await findAll(this.client, {
      labels: [this.itemLabel],
      where: this.live(),
      select: { items: { $count: '*' } },
      groupBy: ['ns']
    })
    const seen = new Map<string, string[]>()
    for (const row of rows) {
      const ns = row.ns
      if (typeof ns !== 'string' || !ns) continue
      let path = ns.split('.')
      if (op.matchConditions?.length && !op.matchConditions.every((c) => doesMatch(c, path))) continue
      if (op.maxDepth !== undefined) path = path.slice(0, op.maxDepth)
      seen.set(JSON.stringify(path), path)
    }
    const ordered = [...seen.values()].sort(compareNamespaces)
    const offset = op.offset ?? 0
    return ordered.slice(offset, offset + (op.limit ?? ordered.length))
  }

  // ── TTL sweeping ───────────────────────────────────────────────────────

  /** Delete expired items and their vectors; returns how many items expired. */
  async sweep(): Promise<number> {
    const items = await findAll(this.client, {
      labels: [this.itemLabel],
      where: { expires_at: { $lte: nowIso() } }
    })
    for (const row of items) {
      const vectors = await this.vectorIds(row.ns.split('.'), row.key)
      await deleteIds(this.client, [row.__id as string, ...vectors])
    }
    return items.length
  }

  /** Run `sweep()` every `sweepIntervalMinutes` (default: the TTL config's, else 5). */
  startTtlSweeper(sweepIntervalMinutes?: number): void {
    if (this.sweeper) return
    const minutes = sweepIntervalMinutes ?? this.ttl?.sweepIntervalMinutes ?? 5
    this.sweeper = setInterval(() => {
      this.sweep().catch((error) => console.warn(`[@rushdb/langchain] TTL sweep failed: ${String(error)}`))
    }, minutes * 60_000)
    this.sweeper.unref?.()
  }

  stopTtlSweeper(): void {
    if (this.sweeper) clearInterval(this.sweeper)
    this.sweeper = undefined
  }
}

// ── helpers ──────────────────────────────────────────────────────────────

function expired(row: Row): boolean {
  return typeof row.expires_at === 'string' && new Date(row.expires_at).getTime() <= Date.now()
}

function value(row: Row): Record<string, any> {
  const raw = row.lg_value
  const loaded = typeof raw === 'string' ? JSON.parse(raw) : {}
  return loaded && typeof loaded === 'object' && !Array.isArray(loaded) ? loaded : {}
}

function toItem(row: Row): Item {
  return {
    value: value(row),
    key: row.key,
    namespace: (row.ns as string).split('.'),
    createdAt: new Date(row.created_at),
    updatedAt: new Date(row.updated_at)
  }
}

function toSearchItem(row: Row, score: number | undefined): SearchItem {
  return { ...toItem(row), score }
}

function compareNamespaces(a: string[], b: string[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1
  }
  return a.length - b.length
}

/**
 * The part of `filter` the engine answers exactly, and whether that is all of it.
 *
 * Only equality on top-level scalars is pushed down: comparisons and `$ne`
 * keep LangGraph's semantics (number coercion, missing fields) in the client.
 */
export function pushdown(filter?: Record<string, any>): [Record<string, unknown>, boolean] {
  if (!filter || !Object.keys(filter).length) return [{}, true]
  const pushed: Record<string, unknown> = {}
  let exact = true
  for (const [key, val] of Object.entries(filter)) {
    if (isScalar(val)) {
      pushed[`${VALUE_PREFIX}.${key}`] = val
    } else if (
      val !== null &&
      typeof val === 'object' &&
      !Array.isArray(val) &&
      Object.keys(val).length === 1 &&
      '$eq' in val &&
      isScalar(val.$eq)
    ) {
      pushed[`${VALUE_PREFIX}.${key}`] = val.$eq
    } else {
      exact = false
    }
  }
  return [pushed, exact]
}

export function matches(item: Record<string, any>, filter?: Record<string, any>): boolean {
  if (!filter) return true
  return Object.entries(filter).every(([k, v]) => compare(item[k], v))
}

function isObject(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function compare(itemValue: unknown, filterValue: unknown): boolean {
  if (isObject(filterValue)) {
    if (Object.keys(filterValue).some((k) => k.startsWith('$'))) {
      return Object.entries(filterValue).every(([op, operand]) => operator(itemValue, op, operand))
    }
    if (!isObject(itemValue)) return false
    return Object.entries(filterValue).every(([k, v]) => compare(itemValue[k], v))
  }
  if (Array.isArray(filterValue)) {
    return (
      Array.isArray(itemValue) &&
      itemValue.length === filterValue.length &&
      filterValue.every((f, i) => compare(itemValue[i], f))
    )
  }
  return itemValue === filterValue
}

function operator(value: unknown, op: string, operand: unknown): boolean {
  switch (op) {
    case '$eq':
      return value === operand
    case '$ne':
      return value !== operand
    case '$gt':
    case '$gte':
    case '$lt':
    case '$lte': {
      const left = Number(value)
      const right = Number(operand)
      if (value === null || value === undefined || Number.isNaN(left) || Number.isNaN(right)) return false
      if (op === '$gt') return left > right
      if (op === '$gte') return left >= right
      if (op === '$lt') return left < right
      return left <= right
    }
    case '$in':
      return Array.isArray(operand) && operand.includes(value)
    case '$nin':
      return Array.isArray(operand) ? !operand.includes(value) : true
    default:
      throw new Error(`Unsupported operator: ${op}`)
  }
}

function doesMatch(condition: MatchCondition, key: string[]): boolean {
  const { matchType, path } = condition
  if (path.length > key.length) return false
  if (matchType === 'prefix') return path.every((p, i) => p === '*' || key[i] === p)
  return path.every((p, i) => p === '*' || key[key.length - path.length + i] === p)
}
