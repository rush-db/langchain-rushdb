/** Hybrid (vector + BM25) retriever over a `RushDBVectorStore`. */
import type { Document } from '@langchain/core/documents'
import { BaseRetriever, type BaseRetrieverInput } from '@langchain/core/retrievers'

import type { Row } from './connection.js'
import type { Filter } from './filters.js'
import type { RushDBVectorStore } from './vectorstores.js'

export interface RushDBHybridRetrieverInput extends BaseRetrieverInput {
  vectorstore: RushDBVectorStore
  k?: number
  fusion?: 'rrf' | 'dbsf'
  rrfK?: number
  vectorWeight?: number
  textWeight?: number
  prefetchLimit?: number
  allTerms?: boolean
  phrase?: boolean
  /** A LangChain-style filter on document metadata. */
  filter?: Filter
  /** A raw RushDB `where`; it can also match related records. */
  where?: Record<string, unknown>
}

/**
 * Retrieve with RushDB hybrid search: a vector leg and a BM25 keyword leg,
 * filtered before ranking and fused in the engine (RRF or DBSF).
 *
 * The vector store must be created with `hybrid: true` so its text index
 * exists. Each returned document carries the fused score in
 * `metadata.score`.
 */
export class RushDBHybridRetriever extends BaseRetriever {
  static lc_name(): string {
    return 'RushDBHybridRetriever'
  }

  lc_namespace = ['rushdb', 'retrievers']

  vectorstore: RushDBVectorStore
  k: number
  fusion: 'rrf' | 'dbsf'
  rrfK?: number
  vectorWeight?: number
  textWeight?: number
  prefetchLimit?: number
  allTerms: boolean
  phrase: boolean
  filter?: Filter
  where?: Record<string, unknown>

  constructor(fields: RushDBHybridRetrieverInput) {
    super(fields)
    this.vectorstore = fields.vectorstore
    this.k = fields.k ?? 4
    this.fusion = fields.fusion ?? 'rrf'
    this.rrfK = fields.rrfK
    this.vectorWeight = fields.vectorWeight
    this.textWeight = fields.textWeight
    this.prefetchLimit = fields.prefetchLimit
    this.allTerms = fields.allTerms ?? false
    this.phrase = fields.phrase ?? false
    this.filter = fields.filter
    this.where = fields.where
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const store = this.vectorstore
    if (!store.hybrid) {
      throw new Error('RushDBHybridRetriever needs a RushDBVectorStore with hybrid: true.')
    }
    const vector = store.managed ? undefined : await store.embeddings.embedQuery(query)
    const params = store.searchParams(query, vector, this.k, this.filter, this.where)
    const text: Row = { query, property: store.textProperty }
    if (this.allTerms) text.allTerms = true
    if (this.phrase) text.phrase = true
    if (this.textWeight !== undefined) text.weight = this.textWeight
    if (this.prefetchLimit !== undefined) {
      text.prefetchLimit = this.prefetchLimit
      params.prefetchLimit = this.prefetchLimit
    }
    if (this.vectorWeight !== undefined) params.weight = this.vectorWeight
    params.text = text
    params.fusion = this.fusion
    if (this.rrfK !== undefined) params.rrfK = this.rrfK
    return (await store.runSearch(params)).map(([doc, score]) => {
      doc.metadata.score = score
      return doc
    })
  }
}
