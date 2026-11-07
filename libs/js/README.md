# @rushdb/langchain

LangChain.js and LangGraph.js integration for [RushDB](https://rushdb.com). It works
the same on RushDB Cloud and on a self-hosted RushDB engine.

| Component | LangChain / LangGraph interface | What it does |
|---|---|---|
| `RushDBVectorStore` | `VectorStore` | Documents with filterable metadata and vector search; the filter is applied before ranking |
| `RushDBHybridRetriever` | `BaseRetriever` | Vector + BM25 keyword search fused in the engine (RRF or DBSF) |
| `RushDBStore` | LangGraph `BaseStore` | Long-term agent memory: namespaces, filters, semantic search, TTL |
| `RushDBSaver` | LangGraph `BaseCheckpointSaver` | Thread checkpoints and pending writes (human-in-the-loop, time travel) |

```bash
npm install @rushdb/langchain @langchain/core @langchain/langgraph-checkpoint
```

## Connect

Every class takes a `@rushdb/javascript-sdk` client, or builds one from the
environment.

```ts
import RushDB from '@rushdb/javascript-sdk'

// RushDB Cloud
const client = new RushDB('RUSHDB_API_KEY')

// Self-hosted engine (docker run -e RUSHDB_TOKEN=... -p 127.0.0.1:3000:3000 rushdb/rushdb)
const client = new RushDB('ENGINE_TOKEN', { direct: true, engineUrl: 'http://127.0.0.1:3000' })
```

From the environment (`clientFromEnv()`, used when `client` is omitted):

| Mode | Variables |
|---|---|
| RushDB Cloud | `RUSHDB_API_KEY`, optional `RUSHDB_URL` |
| Self-hosted | `RUSHDB_ENGINE_URL`, `RUSHDB_ENGINE_TOKEN`, optional `RUSHDB_SCOPE` (multitenant engine) |

The rest of this page is identical for both modes.

## Vector store and hybrid retrieval

```ts
import { OpenAIEmbeddings } from '@langchain/openai'
import { RushDBHybridRetriever, RushDBVectorStore } from '@rushdb/langchain'

const store = new RushDBVectorStore(new OpenAIEmbeddings({ model: 'text-embedding-3-small' }), {
  client,
  hybrid: true
})
await store.addDocuments([
  { pageContent: 'RushDB stores JSON as a graph', metadata: { topic: 'db', year: 2026 } },
  { pageContent: 'LangGraph persists agent state', metadata: { topic: 'agents', year: 2025 } }
])

await store.similaritySearch('graph database', 2, { year: { $gte: 2026 } })

const retriever = new RushDBHybridRetriever({ vectorstore: store, k: 5, fusion: 'rrf' })
await retriever.invoke('how are agent checkpoints stored?')
```

Filters use the LangChain/MongoDB form on metadata:
- equality;
- `$eq $ne $gt $gte $lt $lte`;
- `$in $nin`;
- `$exists`;
- `$contains $startsWith $endsWith`;
- `$and $or $not`.

An unsupported filter throws `FilterError`; it is never ignored.
`store.searchWithScore(query, { k, filter, where })` also takes a raw RushDB `where`,
which can match related records.

Only top-level metadata values that are strings, numbers, booleans or flat arrays of
one of those are filterable. The full metadata, nested values included, is always
returned.

`store.delete({ ids })` deletes documents by id. `store.delete({ deleteAll: true })`
deletes every document of the store's label.

## Long-term memory store

```ts
import { RushDBStore } from '@rushdb/langchain'

const store = new RushDBStore({
  client,
  index: { dims: 1536, embeddings: new OpenAIEmbeddings({ model: 'text-embedding-3-small' }), fields: ['text'] },
  ttl: { defaultTtl: 60 * 24 * 30, refreshOnRead: true }
})
await store.put(['users', 'u1', 'memories'], 'm1', { text: 'Prefers dark mode', kind: 'pref' })
await store.search(['users', 'u1'], { query: 'ui preferences', filter: { kind: 'pref' } })
```

RushDB records have no native TTL. Expired items are hidden from every read.
`store.sweep()` deletes them, and `store.start()` runs it periodically.
LangGraph.js's `put` takes no TTL, so every item gets `ttl.defaultTtl`.

## Checkpointer

```ts
import { RushDBSaver } from '@rushdb/langchain'

const graph = builder.compile({ checkpointer: new RushDBSaver({ client }) })
await graph.invoke(inputs, { configurable: { thread_id: 't-1' } })
```

## Embeddings: the one mode difference

The default is the same everywhere: you pass a LangChain `Embeddings` object and the
connector stores and searches with the vectors it produces. On RushDB Cloud you can
instead let RushDB compute embeddings:

```ts
new RushDBVectorStore('managed', { client: cloudClient })
```

A self-hosted engine never computes embeddings, so `'managed'` throws there.

## Data layout

The records the connector writes (labels `LC_DOCUMENT`, `LG_STORE_ITEM`,
`LG_STORE_VECTOR`, `LG_CHECKPOINT*`) are specified in
[`spec/README.md`](https://github.com/rush-db/langchain-rushdb/blob/main/spec/README.md).
The Python package `langchain-rushdb` uses the same layout:
- **Vector-store documents and store items:** either package can read what the other
  wrote.
- **Checkpoints:** each language reads only its own, because the serializers differ.
