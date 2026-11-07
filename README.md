# langchain-rushdb

LangChain and LangGraph integrations for [RushDB](https://rushdb.com). Both packages
work the same on RushDB Cloud and on a self-hosted RushDB engine.

| Package | Registry | Source |
|---|---|---|
| `langchain-rushdb` | PyPI | [`libs/python`](libs/python) |
| `@rushdb/langchain` | npm | [`libs/js`](libs/js) |

Each package provides four components:

- **Vector store**: documents with filterable metadata. Filters are applied before
  ranking.
- **Hybrid retriever**: vector and BM25 search, fused in the engine (RRF/DBSF).
- **LangGraph store**: long-term agent memory with namespaces, filters, semantic
  search and TTL.
- **LangGraph checkpointer**: thread checkpoints and pending writes.

Both packages follow one record layout, [`spec/README.md`](spec/README.md), and pass
the shared fixtures in [`spec/fixtures`](spec/fixtures).

## Development

Integration tests run against every configured backend. A backend that isn't
configured is skipped. CI requires the self-hosted backends, and Cloud whenever its
secret is set.

```bash
docker run -d -e RUSHDB_TOKEN=test-token -p 127.0.0.1:3917:3000 rushdb/rushdb:3.0.4
docker run -d -e RUSHDB_TOKEN=test-token -p 127.0.0.1:3918:3000 rushdb/rushdb:3.0.4 --multitenant --pgwire-port 0

export RUSHDB_TEST_ENGINE_URL=http://127.0.0.1:3917 RUSHDB_TEST_ENGINE_TOKEN=test-token
export RUSHDB_TEST_MT_ENGINE_URL=http://127.0.0.1:3918 RUSHDB_TEST_MT_ENGINE_TOKEN=test-token
export RUSHDB_TEST_API_KEY=...   # optional: RushDB Cloud test project

(cd libs/python && uv run pytest)
(cd libs/js && pnpm test)
```

Apache-2.0.
