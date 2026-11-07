# langchain-rushdb

LangChain and LangGraph integration for [RushDB](https://rushdb.com). It works the same
on RushDB Cloud and on a self-hosted RushDB engine.

| Component | LangChain / LangGraph interface | What it does |
|---|---|---|
| `RushDBVectorStore` | `VectorStore` | Documents with filterable metadata and vector search; the filter is applied before ranking |
| `RushDBHybridRetriever` | `BaseRetriever` | Vector + BM25 keyword search fused in the engine (RRF or DBSF) |
| `RushDBStore` | LangGraph `BaseStore` | Long-term agent memory: namespaces, filters, semantic search, TTL |
| `RushDBSaver` | LangGraph `BaseCheckpointSaver` | Thread checkpoints and pending writes (human-in-the-loop, time travel) |

```bash
pip install langchain-rushdb
```

## Connect

Every class takes a `rushdb.RushDB` client, or builds one from the environment.

```python
from rushdb import RushDB

# RushDB Cloud
client = RushDB("RUSHDB_API_KEY")

# Self-hosted engine (docker run -e RUSHDB_TOKEN=... -p 127.0.0.1:3000:3000 rushdb/rushdb)
client = RushDB("ENGINE_TOKEN", direct=True, engine_url="http://127.0.0.1:3000")
```

From the environment (`client_from_env()`, used when `client` is omitted):

| Mode | Variables |
|---|---|
| RushDB Cloud | `RUSHDB_API_KEY`, optional `RUSHDB_URL` |
| Self-hosted | `RUSHDB_ENGINE_URL`, `RUSHDB_ENGINE_TOKEN`, optional `RUSHDB_SCOPE` (multitenant engine) |

The rest of this page is identical for both modes.

## Vector store and hybrid retrieval

```python
from langchain_openai import OpenAIEmbeddings
from langchain_rushdb import RushDBVectorStore, RushDBHybridRetriever

store = RushDBVectorStore(
    OpenAIEmbeddings(model="text-embedding-3-small"), client=client, hybrid=True
)
store.add_texts(
    ["RushDB stores JSON as a graph", "LangGraph persists agent state"],
    metadatas=[{"topic": "db", "year": 2026}, {"topic": "agents", "year": 2025}],
)

store.similarity_search("graph database", k=2, filter={"year": {"$gte": 2026}})

retriever = RushDBHybridRetriever(vectorstore=store, k=5, fusion="rrf")
retriever.invoke("how are agent checkpoints stored?")
```

Filters use the LangChain/MongoDB form on metadata: equality, `$eq $ne $gt $gte $lt $lte`,
`$in $nin`, `$exists`, `$contains $startsWith $endsWith`, `$and $or $not`. An
unsupported filter raises `FilterError`; it is never ignored. `where=` takes a raw
RushDB filter, which can also match related records.

Only top-level metadata values that are strings, numbers, booleans or flat arrays of
one of those are filterable. The full metadata, nested values included, is always
returned.

## Long-term memory store

```python
from langchain_rushdb import RushDBStore

store = RushDBStore(
    client=client,
    index={"dims": 1536, "embed": "openai:text-embedding-3-small", "fields": ["text"]},
    ttl={"default_ttl": 60 * 24 * 30, "refresh_on_read": True},
)
store.put(("users", "u1", "memories"), "m1", {"text": "Prefers dark mode", "kind": "pref"})
store.search(("users", "u1"), query="ui preferences", filter={"kind": "pref"})
```

RushDB records have no native TTL. Expired items are hidden from every read.
`store.sweep()` deletes them, and `store.start_ttl_sweeper()` runs `sweep()`
periodically.

## Checkpointer

```python
from langgraph.graph import StateGraph
from langchain_rushdb import RushDBSaver

graph = builder.compile(checkpointer=RushDBSaver(client=client))
graph.invoke(inputs, {"configurable": {"thread_id": "t-1"}})
```

## Embeddings: the one mode difference

The default is the same everywhere: you pass a LangChain `Embeddings` object and the
connector stores and searches with the vectors it produces. On RushDB Cloud you can
instead let RushDB compute embeddings:

```python
RushDBVectorStore("managed", client=cloud_client)
```

A self-hosted engine never computes embeddings, so `"managed"` raises a `ValueError`
there.

## Data layout

The records the connector writes (labels `LC_DOCUMENT`, `LG_STORE_ITEM`,
`LG_STORE_VECTOR`, `LG_CHECKPOINT*`) are specified in
[`spec/README.md`](https://github.com/rush-db/langchain-rushdb/blob/main/spec/README.md).
The JavaScript package `@rushdb/langchain` uses the same layout.
