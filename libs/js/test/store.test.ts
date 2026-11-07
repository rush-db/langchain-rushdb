/** RushDBStore behaviour, once per backend. */
import { afterEach, beforeEach, expect, it } from 'vitest'

import { RushDBStore } from '../src/index.js'
import { describeBackends, dropLabels, unique } from './backends.js'
import { KeywordEmbeddings } from './keyword-embeddings.js'
import { sleep } from '../src/connection.js'

describeBackends('RushDBStore', (makeClient) => {
  let store: RushDBStore

  beforeEach(() => {
    const id = unique()
    store = new RushDBStore({
      client: makeClient(),
      itemLabel: `LG_ITEM_${id}`,
      vectorLabel: `LG_VEC_${id}`,
      index: { dims: 4, embeddings: new KeywordEmbeddings(), fields: ['text'] }
    })
  })

  afterEach(async () => {
    await dropLabels(store.client, store.itemLabel, store.vectorLabel)
  })

  it('round-trips put and get', async () => {
    const value = { text: 'likes apple', nested: { a: [1, { b: 2 }] }, n: 3 }
    await store.put(['users', 'u1'], 'k1', value)
    const item = await store.get(['users', 'u1'], 'k1')
    expect(item?.value).toEqual(value)
    expect(item?.namespace).toEqual(['users', 'u1'])
    expect(item?.createdAt).toBeInstanceOf(Date)
    expect(await store.get(['users', 'u2'], 'k1')).toBeNull()
  })

  it('keeps createdAt on update', async () => {
    await store.put(['a'], 'k', { text: 'v1' })
    const first = await store.get(['a'], 'k')
    await sleep(10)
    await store.put(['a'], 'k', { text: 'v2' })
    const second = await store.get(['a'], 'k')
    expect(second?.value).toEqual({ text: 'v2' })
    expect(second?.createdAt.getTime()).toBe(first?.createdAt.getTime())
    expect(second!.updatedAt.getTime()).toBeGreaterThan(first!.updatedAt.getTime())
    expect(await store.search(['a'])).toHaveLength(1)
  })

  it('deletes', async () => {
    await store.put(['a'], 'k', { text: 'apple' })
    await store.delete(['a'], 'k')
    expect(await store.get(['a'], 'k')).toBeNull()
    expect(await store.search(['a'], { query: 'apple' })).toEqual([])
  })

  it('searches by prefix and filters', async () => {
    await store.put(['users', 'u1', 'mem'], 'm1', { text: 'apple', kind: 'pref', score: 5 })
    await store.put(['users', 'u1', 'mem'], 'm2', { text: 'ocean', kind: 'fact', score: 9 })
    await store.put(['users', 'u10'], 'm3', { text: 'graph', kind: 'pref', score: 1 })
    const keys = async (prefix: string[], options = {}) =>
      new Set((await store.search(prefix, options)).map((i) => i.key))
    expect(await keys(['users', 'u1'])).toEqual(new Set(['m1', 'm2']))
    expect(await keys(['users'], { filter: { kind: 'pref' } })).toEqual(new Set(['m1', 'm3']))
    expect(await keys(['users'], { filter: { score: { $gt: 4 } } })).toEqual(new Set(['m1', 'm2']))
    expect(await keys(['users'], { filter: { kind: { $ne: 'pref' } } })).toEqual(new Set(['m2']))
    expect(await keys(['users'], { filter: { kind: { $in: ['fact'] } } })).toEqual(new Set(['m2']))
    expect(await store.search(['users'], { limit: 2, offset: 1 })).toHaveLength(2)
  })

  it('runs nested filters in the client', async () => {
    await store.put(['n'], 'a', { text: 'x', meta: { lang: 'en' } })
    await store.put(['n'], 'b', { text: 'y', meta: { lang: 'fr' } })
    expect((await store.search(['n'], { filter: { meta: { lang: 'fr' } } })).map((i) => i.key)).toEqual(['b'])
  })

  it('searches semantically', async () => {
    await store.put(['docs'], 'd1', { text: 'apple pie' })
    await store.put(['docs'], 'd2', { text: 'deep ocean' })
    await store.put(['other'], 'd3', { text: 'apple tree' })
    const results = await store.search(['docs'], { query: 'apple' })
    expect(results[0].key).toBe('d1')
    expect(results[0].score).toBeDefined()
    expect(results.map((r) => r.key)).not.toContain('d3')
    const filtered = await store.search(['docs'], { query: 'apple', filter: { text: 'deep ocean' } })
    expect(filtered.map((r) => r.key)).toEqual(['d2'])
  })

  it('skips vectors with index: false', async () => {
    await store.put(['docs'], 'plain', { text: 'apple' }, false)
    expect(await store.search(['docs'], { query: 'apple' })).toEqual([])
    expect(await store.get(['docs'], 'plain')).not.toBeNull()
  })

  it('lists namespaces', async () => {
    for (const ns of [['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'x'], ['z']]) await store.put(ns, 'k', { text: 'v' })
    expect(await store.listNamespaces()).toEqual([['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'x'], ['z']])
    expect(await store.listNamespaces({ prefix: ['a', 'b'] })).toEqual([['a', 'b', 'c'], ['a', 'b', 'd']])
    expect(await store.listNamespaces({ suffix: ['d'] })).toEqual([['a', 'b', 'd']])
    expect(await store.listNamespaces({ prefix: ['a', '*', 'c'] })).toEqual([['a', 'b', 'c']])
    expect(await store.listNamespaces({ maxDepth: 2 })).toEqual([['a', 'b'], ['a', 'x'], ['z']])
    expect(await store.listNamespaces({ limit: 1, offset: 1 })).toEqual([['a', 'b', 'd']])
  })

  it('hides and sweeps expired items', async () => {
    const ttlStore = new RushDBStore({
      client: store.client,
      itemLabel: store.itemLabel,
      ttl: { defaultTtl: 0.002, refreshOnRead: false }
    })
    await ttlStore.put(['t'], 'short', { v: 1 })
    await store.put(['t'], 'long', { v: 2 })
    expect(await ttlStore.get(['t'], 'short')).not.toBeNull()
    await sleep(300)
    expect(await ttlStore.get(['t'], 'short')).toBeNull()
    expect((await ttlStore.search(['t'])).map((i) => i.key)).toEqual(['long'])
    expect(await ttlStore.listNamespaces()).toEqual([['t']])
    expect(await ttlStore.sweep()).toBe(1)
    expect((await store.client.records.find({ labels: [store.itemLabel] } as any)).total).toBe(1)
  })

  it('refreshes TTL on read', async () => {
    const ttlStore = new RushDBStore({ client: store.client, itemLabel: store.itemLabel, ttl: { defaultTtl: 0.01 } })
    await ttlStore.put(['t'], 'k', { v: 1 })
    for (let i = 0; i < 3; i += 1) {
      await sleep(300)
      expect(await ttlStore.get(['t'], 'k')).not.toBeNull()
    }
  })
})
