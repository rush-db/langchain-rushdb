/** RushDB vector store for LangChain. */
import { randomUUID } from 'node:crypto'

import { Document, type DocumentInterface } from '@langchain/core/documents'
import type { EmbeddingsInterface } from '@langchain/core/embeddings'
import { VectorStore } from '@langchain/core/vectorstores'
import type { CreateEmbeddingIndexParams } from '@rushdb/javascript-sdk'

import {
  type Client,
  type Row,
  clientFromEnv,
  createRows,
  deleteIds,
  findAll,
  isSelfHosted,
  mirror,
  toJson
} from './connection.js'
import { type Filter, translateFilter } from './filters.js'

export type Similarity = 'cosine' | 'dot' | 'euclidean'

export const ID_PROPERTY = 'lc_id'
export const METADATA_PROPERTY = 'lc_metadata'
export const METADATA_PREFIX = 'metadata'

export interface RushDBVectorStoreConfig {
  /** A RushDB client; built from the environment when omitted. */
  client?: Client
  /** The record label documents are stored under. */
  label?: string
  /** The property holding the document text. */
  textProperty?: string
  /** The vector index similarity. */
  similarity?: Similarity
  /** Also create a text (BM25) index, needed by `RushDBHybridRetriever`. */
  hybrid?: boolean
  /** `eventual` (engine default) or `immediate`. */
  textIndexConsistency?: 'eventual' | 'immediate'
}

export interface SearchOptions {
  k?: number
  /** A LangChain-style filter on document metadata. */
  filter?: Filter
  /** A raw RushDB `where`; it can also match related records. */
  where?: Record<string, unknown>
}

/** Stands in for `embeddings` when RushDB Cloud computes them. */
const MANAGED: EmbeddingsInterface = {
  embedDocuments: async () => {
    throw new Error('Managed embeddings are computed by RushDB Cloud.')
  },
  embedQuery: async () => {
    throw new Error('Managed embeddings are computed by RushDB Cloud.')
  }
}

/**
 * LangChain vector store backed by RushDB, Cloud or self-hosted alike.
 *
 * Each document is one record of `label`: its text, its full metadata as JSON
 * and the metadata's top-level scalar fields as filterable `metadata.<key>`
 * properties (see the repository's `spec/README.md`). Vectors live in a RushDB
 * external vector index created on first use.
 *
 * `embeddings` turn texts into vectors on every backend. `"managed"` lets
 * RushDB Cloud compute them instead; a self-hosted engine refuses it, as it
 * never computes embeddings.
 */
export class RushDBVectorStore extends VectorStore {
  declare FilterType: Filter

  readonly client: Client
  readonly managed: boolean
  readonly label: string
  readonly textProperty: string
  readonly similarity: Similarity
  readonly hybrid: boolean
  readonly textIndexConsistency: 'eventual' | 'immediate'
  private indexReady = false
  private textIndexReady = false

  constructor(embeddings: EmbeddingsInterface | 'managed', config: RushDBVectorStoreConfig = {}) {
    super(embeddings === 'managed' ? MANAGED : embeddings, config)
    this.client = config.client ?? clientFromEnv()
    this.managed = embeddings === 'managed'
    if (this.managed && isSelfHosted(this.client)) {
      throw new Error(
        'embedding "managed" needs RushDB Cloud: a self-hosted engine never computes ' +
          'embeddings. Pass a LangChain Embeddings instance instead.'
      )
    }
    this.label = config.label ?? 'LC_DOCUMENT'
    this.textProperty = config.textProperty ?? 'page_content'
    if (
      this.textProperty === ID_PROPERTY ||
      this.textProperty === METADATA_PROPERTY ||
      this.textProperty.startsWith('__')
    ) {
      throw new Error(`textProperty '${this.textProperty}' is reserved.`)
    }
    this.similarity = config.similarity ?? 'cosine'
    this.hybrid = config.hybrid ?? false
    this.textIndexConsistency = config.textIndexConsistency ?? 'eventual'
  }

  _vectorstoreType(): string {
    return 'rushdb'
  }

  // ── indexes ────────────────────────────────────────────────────────────

  /** Create the vector index (and the text index when `hybrid`) if missing. */
  async ensureIndexes(dimensions?: number): Promise<void> {
    if (!this.indexReady) {
      await this.ensureVectorIndex(dimensions)
      this.indexReady = true
    }
    if (this.hybrid && !this.textIndexReady) {
      await this.ensureTextIndex()
      this.textIndexReady = true
    }
  }

  private async ensureVectorIndex(dimensions?: number): Promise<void> {
    const existing: Row[] = ((await this.client.ai.indexes.find()).data ?? []) as Row[]
    for (const index of existing) {
      if (index.label !== this.label || index.propertyName !== this.textProperty) continue
      if (this.managed && index.sourceType !== 'external') return
      if (
        !this.managed &&
        index.sourceType === 'external' &&
        index.similarityFunction === this.similarity &&
        (dimensions === undefined || index.dimensions === dimensions)
      ) {
        return
      }
    }
    const params: CreateEmbeddingIndexParams = { label: this.label, propertyName: this.textProperty }
    if (!this.managed) {
      params.external = true
      params.dimensions = dimensions ?? (await this.embeddings.embedQuery('dimension probe')).length
      params.similarityFunction = this.similarity
    }
    await this.client.ai.indexes.create(params)
  }

  private async ensureTextIndex(): Promise<void> {
    const existing: Row[] = ((await this.client.textIndexes.find()).data ?? []) as Row[]
    if (existing.some((i) => i.label === this.label && i.property === this.textProperty)) return
    await this.client.textIndexes.create({
      label: this.label,
      property: this.textProperty,
      consistency: this.textIndexConsistency
    })
  }

  // ── writes ─────────────────────────────────────────────────────────────

  async addDocuments(documents: DocumentInterface[], options?: { ids?: string[] }): Promise<string[]> {
    if (this.managed) return this.write(documents, undefined, options?.ids)
    const vectors = await this.embeddings.embedDocuments(documents.map((d) => d.pageContent))
    return this.addVectors(vectors, documents, options)
  }

  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[],
    options?: { ids?: string[] }
  ): Promise<string[]> {
    if (this.managed) throw new Error('addVectors needs caller-computed embeddings, not managed ones.')
    return this.write(documents, vectors, options?.ids)
  }

  /**
   * Store `documents`; a document whose id already exists replaces it.
   *
   * New documents are written in one atomic import; records of existing ids
   * are deleted first.
   */
  private async write(
    documents: DocumentInterface[],
    vectors: number[][] | undefined,
    idsArg: string[] | undefined
  ): Promise<string[]> {
    if (idsArg && idsArg.length !== documents.length) {
      throw new Error('ids must have one entry per document.')
    }
    const ids = documents.map((doc, i) => idsArg?.[i] || doc.id || randomUUID())
    if (!documents.length) return []
    await this.ensureIndexes(vectors?.[0]?.length)
    const rows = documents.map((doc, i) => this.row(doc, ids[i]))
    const entries = vectors?.map((vector) => [
      { propertyName: this.textProperty, vector, similarityFunction: this.similarity }
    ])
    await deleteIds(this.client, await this.recordIds(ids))
    await createRows(this.client, this.label, rows, entries)
    return ids
  }

  private row(doc: DocumentInterface, id: string): Row {
    const metadata = { ...(doc.metadata ?? {}) }
    return {
      [ID_PROPERTY]: id,
      [this.textProperty]: doc.pageContent,
      [METADATA_PROPERTY]: toJson(metadata),
      ...mirror(METADATA_PREFIX, metadata)
    }
  }

  private async recordIds(ids: string[]): Promise<string[]> {
    if (!ids.length) return []
    const rows = await findAll(this.client, {
      labels: [this.label],
      where: { [ID_PROPERTY]: { $in: ids } }
    })
    return rows.map((row) => row.__id as string)
  }

  /** Delete the documents with `ids`, or every document with `deleteAll: true`. */
  async delete(params: { ids?: string[]; deleteAll?: boolean } = {}): Promise<void> {
    if (params.deleteAll) {
      await this.client.records.delete({ labels: [this.label] } as any)
      return
    }
    if (!params.ids) throw new Error('delete needs `ids`, or `deleteAll: true`.')
    await deleteIds(this.client, await this.recordIds(params.ids))
  }

  // ── reads ──────────────────────────────────────────────────────────────

  async getByIds(ids: string[]): Promise<Document[]> {
    if (!ids.length) return []
    const rows = await findAll(this.client, {
      labels: [this.label],
      where: { [ID_PROPERTY]: { $in: ids } }
    })
    return rows.map((row) => this.document(row))
  }

  private document(row: Row): Document {
    const raw = row[METADATA_PROPERTY]
    const metadata = typeof raw === 'string' && raw ? JSON.parse(raw) : {}
    return new Document({
      id: row[ID_PROPERTY],
      pageContent: row[this.textProperty] ?? '',
      metadata
    })
  }

  /** The RushDB vector search request; `where` is a raw RushDB filter. */
  searchParams(
    query: string | undefined,
    vector: number[] | undefined,
    k: number,
    filter?: Filter,
    where?: Record<string, unknown>
  ): Row {
    const params: Row = { labels: [this.label], propertyName: this.textProperty, limit: k }
    if (this.managed) {
      params.query = query
    } else {
      params.queryVector = vector
      params.similarityFunction = this.similarity
    }
    let combined = translateFilter(filter, METADATA_PREFIX)
    if (where && Object.keys(where).length) {
      combined = Object.keys(combined).length ? { $and: [combined, where] } : { ...where }
    }
    if (Object.keys(combined).length) params.where = combined
    return params
  }

  async runSearch(params: Row): Promise<Array<[Document, number]>> {
    const vector = params.queryVector as number[] | undefined
    await this.ensureIndexes(vector?.length)
    const result = await this.client.records.vectorSearch(params)
    return result.data.map((record: any) => {
      const row = record.data as Row
      return [this.document(row), Number(row.__score ?? 0)]
    })
  }

  /** Search with a LangChain `filter`, a raw RushDB `where`, or both. */
  async searchWithScore(query: string, options: SearchOptions = {}): Promise<Array<[Document, number]>> {
    const vector = this.managed ? undefined : await this.embeddings.embedQuery(query)
    return this.runSearch(this.searchParams(query, vector, options.k ?? 4, options.filter, options.where))
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: Filter
  ): Promise<Array<[Document, number]>> {
    if (this.managed) throw new Error('Search by vector needs caller-computed embeddings, not managed ones.')
    return this.runSearch(this.searchParams(undefined, query, k, filter))
  }

  async similaritySearchWithScore(query: string, k = 4, filter?: Filter): Promise<Array<[Document, number]>> {
    return this.searchWithScore(query, { k, filter })
  }

  async similaritySearch(query: string, k = 4, filter?: Filter): Promise<Document[]> {
    return (await this.searchWithScore(query, { k, filter })).map(([doc]) => doc)
  }

  /** Results with relevance in [0, 1] (see `relevanceFn`). */
  async similaritySearchWithRelevanceScores(
    query: string,
    k = 4,
    filter?: Filter
  ): Promise<Array<[Document, number]>> {
    const relevance = relevanceFn(this.similarity)
    return (await this.searchWithScore(query, { k, filter })).map(([doc, score]) => [doc, relevance(score)])
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    config: RushDBVectorStoreConfig = {}
  ): Promise<RushDBVectorStore> {
    const docs = texts.map(
      (text, i) =>
        new Document({
          pageContent: text,
          metadata: (Array.isArray(metadatas) ? metadatas[i] : metadatas) ?? {}
        })
    )
    return RushDBVectorStore.fromDocuments(docs, embeddings, config)
  }

  static async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface,
    config: RushDBVectorStoreConfig = {}
  ): Promise<RushDBVectorStore> {
    const store = new RushDBVectorStore(embeddings, config)
    await store.addDocuments(docs)
    return store
  }
}

/** Map an engine score (higher is closer) to a relevance in [0, 1]. */
export function relevanceFn(similarity: Similarity): (score: number) => number {
  if (similarity === 'cosine') return (s) => (1 + s) / 2
  if (similarity === 'dot') return (s) => 1 / (1 + Math.exp(-s))
  return (s) => s
}
