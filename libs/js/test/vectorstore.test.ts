/** Vector store and hybrid retriever, once per backend. */
import { Document } from '@langchain/core/documents'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { FilterError, RushDBHybridRetriever, RushDBVectorStore } from '../src/index.js'
import { clearLabels, describeBackends, label } from './backends.js'
import { KeywordEmbeddings } from './keyword-embeddings.js'

const DOCS = [
  new Document({ id: 'd1', pageContent: 'apple pie recipe', metadata: { kind: 'food', year: 2021, tags: ['sweet'] } }),
  new Document({ id: 'd2', pageContent: 'ocean currents', metadata: { kind: 'science', year: 2024, nested: { a: 1 } } }),
  new Document({ id: 'd3', pageContent: 'graph databases for agent memory', metadata: { kind: 'tech', year: 2026 } }),
  new Document({ id: 'd4', pageContent: 'apple orchard graph', metadata: { kind: 'food', year: 2026 } })
]

describeBackends('RushDBVectorStore', (makeClient) => {
  let store: RushDBVectorStore

  beforeEach(async () => {
    store = new RushDBVectorStore(new KeywordEmbeddings(), {
      client: makeClient(),
      label: label('LC_VS'),
      hybrid: true,
      textIndexConsistency: 'immediate'
    })
    await clearLabels(store.client, store.label)
  })

  afterEach(async () => {
    await clearLabels(store.client, store.label)
  })

  // ── the behaviours LangChain's standard vector store suite checks ──────

  it('is empty before writes', async () => {
    expect(await store.similaritySearch('apple', 4)).toEqual([])
  })

  it('adds documents and returns their ids', async () => {
    const ids = await store.addDocuments([
      new Document({ pageContent: 'foo', metadata: { id: 1 } }),
      new Document({ pageContent: 'bar', metadata: { id: 2 } })
    ])
    expect(ids).toHaveLength(2)
    const found = await store.getByIds(ids)
    expect(found.map((d) => d.pageContent).sort()).toEqual(['bar', 'foo'])
  })

  it('deletes documents by id, ignoring missing ones', async () => {
    await store.addDocuments(DOCS)
    await store.delete({ ids: ['d1', 'missing'] })
    expect((await store.getByIds(['d1', 'd2'])).map((d) => d.id)).toEqual(['d2'])
  })

  it('is idempotent and replaces documents by id', async () => {
    await store.addDocuments(DOCS)
    await store.addDocuments(DOCS)
    await store.addDocuments([new Document({ id: 'd2', pageContent: 'ocean waves', metadata: { kind: 'x' } })])
    const docs = await store.getByIds(DOCS.map((d) => d.id as string))
    expect(docs).toHaveLength(4)
    const d2 = docs.find((d) => d.id === 'd2')
    expect(d2?.pageContent).toBe('ocean waves')
    expect(d2?.metadata).toEqual({ kind: 'x' })
  })

  it('takes ids from options', async () => {
    const ids = await store.addDocuments([new Document({ pageContent: 'foo' })], { ids: ['given'] })
    expect(ids).toEqual(['given'])
    expect((await store.getByIds(['given']))[0].pageContent).toBe('foo')
  })

  it('refuses delete without ids', async () => {
    await expect(store.delete()).rejects.toThrow(/deleteAll/)
  })

  // ── RushDB behaviour ──────────────────────────────────────────────────

  it('round-trips metadata', async () => {
    await store.addDocuments(DOCS)
    const [doc] = await store.getByIds(['d2'])
    expect(doc.metadata).toEqual({ kind: 'science', year: 2024, nested: { a: 1 } })
  })

  it('filters before ranking', async () => {
    await store.addDocuments(DOCS)
    const docs = await store.similaritySearch('apple', 4, { kind: 'food', year: { $gte: 2025 } })
    expect(docs.map((d) => d.id)).toEqual(['d4'])
  })

  it('applies logic and membership filters', async () => {
    await store.addDocuments(DOCS)
    const docs = await store.similaritySearch('graph', 4, {
      $or: [{ kind: { $in: ['science'] } }, { tags: 'sweet' }]
    })
    expect(new Set(docs.map((d) => d.id))).toEqual(new Set(['d1', 'd2']))
  })

  it('returns scores and relevance', async () => {
    await store.addDocuments(DOCS)
    const [[doc, score]] = await store.similaritySearchWithScore('ocean', 1)
    expect(doc.id).toBe('d2')
    expect(score).toBeCloseTo(1)
    const [[, relevance]] = await store.similaritySearchWithRelevanceScores('ocean', 1)
    expect(relevance).toBeCloseTo(1)
  })

  it('raises on an unsupported filter', async () => {
    await expect(store.similaritySearch('x', 4, { kind: { $regex: 'f.*' } })).rejects.toThrow(FilterError)
  })

  it('takes a raw where', async () => {
    await store.addDocuments(DOCS)
    const docs = await store.searchWithScore('apple', { k: 4, where: { 'metadata.year': 2021 } })
    expect(docs.map(([d]) => d.id)).toEqual(['d1'])
  })

  it('fuses keyword and vector legs', async () => {
    await store.addDocuments(DOCS)
    // "pie recipe" has no topic axis: the vector leg alone ranks d3/d4 first,
    // only the BM25 leg knows d1 is the match.
    expect((await store.similaritySearch('pie recipe', 1))[0].id).not.toBe('d1')
    const docs = await new RushDBHybridRetriever({ vectorstore: store, k: 2 }).invoke('pie recipe')
    expect(docs[0].id).toBe('d1')
    expect(docs.every((d) => 'score' in d.metadata)).toBe(true)
  })

  it('filters the hybrid retriever', async () => {
    await store.addDocuments(DOCS)
    const retriever = new RushDBHybridRetriever({ vectorstore: store, k: 4, filter: { kind: 'tech' } })
    expect((await retriever.invoke('graph')).map((d) => d.id)).toEqual(['d3'])
  })

  it('needs a text index for hybrid retrieval', async () => {
    const plain = new RushDBVectorStore(new KeywordEmbeddings(), {
      client: store.client,
      label: label('LC_NH')
    })
    await expect(new RushDBHybridRetriever({ vectorstore: plain }).invoke('x')).rejects.toThrow(/hybrid: true/)
  })

  it('works as a retriever', async () => {
    await store.addDocuments(DOCS)
    const docs = await store.asRetriever({ k: 1, filter: { kind: 'science' } }).invoke('x')
    expect(docs.map((d) => d.id)).toEqual(['d2'])
  })
})
