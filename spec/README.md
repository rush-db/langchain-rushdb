# langchain-rushdb record layout (spec v1)

This is the single source of truth for how the Python (`langchain-rushdb`) and
JavaScript (`@rushdb/langchain`) packages store data in RushDB. Both packages
implement it, and both load the fixtures in [`fixtures/`](fixtures) in their unit
tests. A change to the layout is made here first, in the fixtures, and then in
both packages in the same change.

## Connection parity

Every class takes a RushDB SDK client, or builds one from the environment:

| Mode | Variables |
|---|---|
| RushDB Cloud | `RUSHDB_API_KEY`, optional `RUSHDB_URL` |
| Self-hosted engine | `RUSHDB_ENGINE_URL`, `RUSHDB_ENGINE_TOKEN`, optional `RUSHDB_SCOPE` (multitenant) |

Setting both groups is an error. No class, method or default differs by mode,
with one exception: `embedding="managed"` (Cloud computes the embeddings)
needs RushDB Cloud and is refused at construction against a self-hosted
engine. Embeddings computed by the caller (a LangChain `Embeddings`) are the
default everywhere.

The packages use only SDK calls that a self-hosted engine (SDK direct mode)
answers, under its rules:

- Inside a transaction: record creates and single-id deletes only. No
  upserts, no deletes by query, no inline vectors.
- Rows are created with `records.importJson` (`suggestTypes: true`), one call
  per batch, which the engine commits atomically. It accepts flat arrays
  (the SDK's `createMany` does not), and inline vectors go in each row's
  `$vectors`. Type inference changes only ISO-8601 strings, which become
  datetimes. Values are always read back from the JSON copy.
- A transaction refused with `transaction_conflict` (concurrent writers) is
  retried with exponential backoff, up to 8 attempts.
- The engine's `$gt`/`$gte`/`$lt`/`$lte` compare numbers and datetimes, not
  strings. Orderings on string ids are applied in the client.

## Mirrored fields

A JSON object stored by the connector is kept whole as a JSON string. Its
top-level entries are also mirrored as record properties under a prefix, so
they can be filtered by the engine:

- `<prefix>.<key>` holds the value when it is a string, a finite number, a boolean,
  or a non-empty array of only one of those kinds.
- Any other value (object, `null`, empty or mixed array) is not mirrored. The
  JSON string still holds it.

Prefixes: `metadata` (vector store documents and checkpoints), `value`
(store items).

## Filters

A filter is a JSON object in the MongoDB-style form LangChain uses:

| Form | Meaning |
|---|---|
| `{"k": v}` | equality (`v` a string, number or boolean) |
| `{"k": {"$eq"\|"$ne": v}}` | comparison |
| `{"k": {"$gt"\|"$gte"\|"$lt"\|"$lte": v}}` | range on numbers and ISO-8601 datetimes |
| `{"k": {"$in"\|"$nin": [v, ...]}}` | membership |
| `{"k": {"$exists": bool}}` | presence |
| `{"k": {"$contains"\|"$startsWith"\|"$endsWith": "s"}}` | string match |
| `{"$and"\|"$or": [filter, ...]}`, `{"$not": filter}` | logic |

Translation to a RushDB `where` renames every field `k` to `<prefix>.<k>` and
keeps the operators. Fields are combined with AND. An unknown operator, a key
that starts with `$` outside the logic operators, or an operand of the wrong
type is an error. It is never ignored. See [`fixtures/filters.json`](fixtures/filters.json).

## Vector store (`RushDBVectorStore`)

One record per document, label `LC_DOCUMENT` (configurable):

| Property | Type | Content |
|---|---|---|
| `lc_id` | string | The LangChain document id (generated UUID when not given) |
| `page_content` | string | Document text (property name configurable) |
| `lc_metadata` | string | The full metadata as JSON |
| `metadata.<key>` | mirrored | Filterable metadata |

The vector is held by an external vector index on (`label`, text property,
dimensions, similarity). The index is created on first use when missing.
Hybrid search adds a text index on the same property.

Scores come from the engine, and a higher score is always more similar. Relevance in
`[0, 1]`:

- cosine: `(1 + s) / 2`
- dot: `1 / (1 + e^-s)`
- euclidean: `s`, because the engine already returns `1 / (1 + d)`.

Writing documents whose ids already exist deletes the old records first, then
creates the new ones in one atomic `createMany`.

## Store (`RushDBStore`, LangGraph `BaseStore`)

Namespaces are tuples of non-empty strings without `.`, so they are stored
joined with `.`.

Store filters keep LangGraph's semantics (number coercion, nested objects,
`$ne` matching missing fields). The engine pre-filters equality on top-level
scalars; every other condition runs in the client after the namespace
narrowing.

Item: label `LG_STORE_ITEM` (configurable)

| Property | Type | Content |
|---|---|---|
| `ns` | string | Namespace joined with `.` |
| `ns_depth` | number | Namespace length |
| `key` | string | Item key |
| `lg_value` | string | The value as JSON |
| `created_at`, `updated_at` | datetime | ISO-8601 UTC |
| `expires_at` | datetime | Expiry. Items without a TTL carry `9999-12-31T23:59:59.000Z` |
| `ttl_minutes` | number | Only on items with a TTL. Read refreshes extend from now by it |
| `value.<key>` | mirrored | Filterable value fields |

Vector: label `LG_STORE_VECTOR` (configurable). There is one record per embedded text of an
item. One item can have several, one per indexed field path that yields text.

| Property | Type | Content |
|---|---|---|
| `ns`, `ns_depth`, `key`, `value.<key>` | | Copied from the item (no expiry: liveness is the item's) |
| `field` | string | The index field path (`$` = whole value) |
| `text` | string | The embedded text; carries the external vector index |

Search score is the maximum over an item's vectors. A namespace prefix
`(a, b)` matches `ns == "a.b"` or `ns` starting with `"a.b."`. Value filters run
in the engine on the mirrored fields when they can. Filters on nested
values run in the client with LangGraph's JSONB semantics, after the engine
narrows by namespace.

## Checkpointer (`RushDBSaver`, LangGraph `BaseCheckpointSaver`)

Serialized data is the saver's `serde.dumps_typed` output. The bytes are stored
base64-encoded (standard alphabet) next to their type. Checkpoints written by
one language are read by the same language. The record layout, not the
serialization, is shared: either package can list, count and delete any
thread.

Checkpoint: label `LG_CHECKPOINT`

| Property | Type | Content |
|---|---|---|
| `thread_id`, `checkpoint_ns`, `checkpoint_id` | string | Identity (`checkpoint_ns` may be `""`) |
| `parent_checkpoint_id` | string | Absent for the first checkpoint |
| `checkpoint_type`, `checkpoint` | string | Checkpoint without `channel_values` |
| `metadata_type`, `metadata` | string | Checkpoint metadata |
| `metadata.<key>` | mirrored | For `list(filter=...)` and `delete_for_runs` |

Channel value: label `LG_CHECKPOINT_BLOB`. There is one per (thread, ns, channel, version),
and it is written once.

| Property | Type | Content |
|---|---|---|
| `thread_id`, `checkpoint_ns`, `channel`, `version` | string | Identity (`version` stringified) |
| `blob_type` | string | `empty` when the channel had no value |
| `blob` | string | Absent when `blob_type` is `empty` |

Pending write: label `LG_CHECKPOINT_WRITE`

| Property | Type | Content |
|---|---|---|
| `thread_id`, `checkpoint_ns`, `checkpoint_id`, `task_id`, `task_path` | string | Identity |
| `idx` | number | Write index. Special channels use LangGraph's negative indexes |
| `channel`, `blob_type`, `blob` | string | The write |

A write with `idx >= 0` is kept as first written. A special write (`idx < 0`)
replaces the earlier one. Writes are returned in
`(task_path, task_id, idx)` order.

Each `put` stages its blobs and checkpoint in one transaction; each
`put_writes` does the same for its writes. `list` orders by `checkpoint_id`
descending in the engine. It applies `before` in the client, because ids are
strings. Equality on scalar metadata runs in the engine; other metadata filters
run in the client.
