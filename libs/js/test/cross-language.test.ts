/** Records written by the Python package are read by this one (spec/README.md). */
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

import RushDB from '@rushdb/javascript-sdk'
import { afterAll, describe, expect, it } from 'vitest'

import { RushDBStore, RushDBVectorStore } from '../src/index.js'
import { clearLabels, label } from './backends.js'
import { KeywordEmbeddings } from './keyword-embeddings.js'

const engineUrl = process.env.RUSHDB_TEST_ENGINE_URL
const token = process.env.RUSHDB_TEST_ENGINE_TOKEN ?? ''
const hasUv = spawnSync('uv', ['--version']).status === 0
const pythonProject = fileURLToPath(new URL('../../python', import.meta.url))
const script = fileURLToPath(new URL('./cross/write_from_python.py', import.meta.url))

describe.skipIf(!engineUrl || !hasUv)('Python → JS [self_hosted]', () => {
  const suffix = label('X')
  const client = new RushDB(token, { direct: true, engineUrl: engineUrl as string })

  afterAll(async () => {
    await clearLabels(client, `LC_X_${suffix}`, `LG_XI_${suffix}`, `LG_XV_${suffix}`)
  })

  it('reads what Python wrote', async () => {
    const run = spawnSync('uv', ['run', '--project', pythonProject, 'python', script, engineUrl as string, token, suffix], {
      encoding: 'utf8'
    })
    expect(run.status, run.stderr).toBe(0)

    const vectors = new RushDBVectorStore(new KeywordEmbeddings(), { client, label: `LC_X_${suffix}` })
    const [doc] = await vectors.getByIds(['py-doc'])
    expect(doc.pageContent).toBe('apple orchard')
    expect(doc.metadata).toEqual({ kind: 'food', tags: ['a', 'b'], nested: { x: 1 } })
    const hits = await vectors.similaritySearch('apple', 1, { tags: 'a' })
    expect(hits.map((d) => d.id)).toEqual(['py-doc'])

    const store = new RushDBStore({
      client,
      itemLabel: `LG_XI_${suffix}`,
      vectorLabel: `LG_XV_${suffix}`,
      index: { dims: 4, embeddings: new KeywordEmbeddings(), fields: ['text'] }
    })
    const item = await store.get(['users', 'u1'], 'py-item')
    expect(item?.value).toEqual({ text: 'apple memory', kind: 'pref', n: 2 })
    const found = await store.search(['users'], { query: 'apple', filter: { kind: 'pref' } })
    expect(found.map((i) => i.key)).toEqual(['py-item'])
    expect(await store.listNamespaces()).toEqual([['users', 'u1']])
  })
})
