"""RushDB-backed LangGraph long-term memory store."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from rushdb import RushDB

from ._connection import (
    FAR_FUTURE,
    client_from_env,
    create_rows,
    delete_ids,
    find_all,
    is_scalar,
    iso,
    mirror,
    now_iso,
    parse_iso,
    to_json,
    transact,
)

VALUE_PREFIX = "value"
SEARCH_POOL_MAX = 10_000


def namespace_string(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def prefix_where(prefix: tuple[str, ...]) -> dict[str, Any]:
    """Records whose namespace equals ``prefix`` or starts with it."""
    if not prefix:
        return {}
    ns = namespace_string(prefix)
    return {"$or": [{"ns": ns}, {"ns": {"$startsWith": ns + "."}}]}


class RushDBStore(BaseStore):
    """LangGraph ``BaseStore`` on RushDB, Cloud or self-hosted alike.

    Items are records of ``item_label``; with ``index`` configured, every
    embedded text of an item is a record of ``vector_label`` carrying an
    external vector (see the repository's ``spec/README.md``).

    Args:
        client: A ``rushdb.RushDB`` client; built from the environment when omitted.
        index: LangGraph ``IndexConfig`` (``dims``, ``embed``, optional ``fields``)
            to enable semantic ``search(query=...)``.
        ttl: LangGraph ``TTLConfig``. RushDB has no record TTL, so expiry is an
            ``expires_at`` property every read filters on; :meth:`sweep` (or the
            sweeper started by :meth:`start_ttl_sweeper`) deletes expired items.
        item_label: Label of item records.
        vector_label: Label of vector records.
    """

    supports_ttl = True

    def __init__(
        self,
        *,
        client: RushDB | None = None,
        index: IndexConfig | None = None,
        ttl: TTLConfig | None = None,
        item_label: str = "LG_STORE_ITEM",
        vector_label: str = "LG_STORE_VECTOR",
    ) -> None:
        self.client = client if client is not None else client_from_env()
        self.item_label = item_label
        self.vector_label = vector_label
        self.ttl_config = ttl
        self.index_config = index
        self.embeddings = ensure_embeddings(index["embed"]) if index else None
        self._index_fields = (
            [(path, tokenize_path(path)) for path in (index.get("fields") or ["$"])]
            if index
            else []
        )
        self._vector_index_ready = False
        self._sweeper: threading.Thread | None = None
        self._sweeper_stop = threading.Event()

    # ── BaseStore ──────────────────────────────────────────────────────────

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            else:
                raise ValueError(f"Unknown store operation {type(op).__name__}.")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    # ── get ────────────────────────────────────────────────────────────────

    def _live(self) -> dict[str, Any]:
        return {"expires_at": {"$gt": now_iso()}}

    def _item_rows(self, namespace: tuple[str, ...], key: str) -> list[dict[str, Any]]:
        return list(
            find_all(
                self.client,
                {
                    "labels": [self.item_label],
                    "where": {"ns": namespace_string(namespace), "key": key},
                },
            )
        )

    def _get(self, op: GetOp) -> Item | None:
        rows = [r for r in self._item_rows(op.namespace, op.key) if not _expired(r)]
        if not rows:
            return None
        row = rows[0]
        if op.refresh_ttl:
            self._refresh([row])
        return _item(row)

    def _refresh(self, rows: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc)
        for row in rows:
            minutes = row.get("ttl_minutes")
            if isinstance(minutes, (int, float)):
                expires = iso(now + timedelta(minutes=minutes))
                self.client.records.update(row["__id"], {"expires_at": expires})
                row["expires_at"] = expires

    # ── put / delete ───────────────────────────────────────────────────────

    def _put(self, op: PutOp) -> None:
        existing = self._item_rows(op.namespace, op.key)
        old_vectors = self._vector_ids(op.namespace, op.key)
        if op.value is None:
            delete_ids(self.client, [r["__id"] for r in existing] + old_vectors)
            return
        now = now_iso()
        created_at = existing[0].get("created_at", now) if existing else now
        ttl = op.ttl if isinstance(op.ttl, (int, float)) else None
        expires_at = (
            iso(datetime.now(timezone.utc) + timedelta(minutes=ttl))
            if ttl is not None
            else FAR_FUTURE
        )
        value = dict(op.value)
        mirrored = mirror(VALUE_PREFIX, value)
        item: dict[str, Any] = {
            "ns": namespace_string(op.namespace),
            "ns_depth": len(op.namespace),
            "key": op.key,
            "lg_value": to_json(value),
            "created_at": created_at,
            "updated_at": now,
            "expires_at": expires_at,
            **mirrored,
        }
        if ttl is not None:
            item["ttl_minutes"] = ttl
        transact(self.client, [r["__id"] for r in existing], [(self.item_label, [item])])
        delete_ids(self.client, old_vectors)
        texts = self._texts(value, op.index)
        if texts and self.embeddings is not None:
            vectors = self.embeddings.embed_documents([text for _, text in texts])
            self._ensure_vector_index()
            rows = [
                {
                    "ns": item["ns"],
                    "ns_depth": item["ns_depth"],
                    "key": op.key,
                    "field": field,
                    "text": text,
                    **mirrored,
                }
                for field, text in texts
            ]
            create_rows(
                self.client,
                self.vector_label,
                rows,
                [
                    [{"propertyName": "text", "vector": v, "similarityFunction": "cosine"}]
                    for v in vectors
                ],
            )

    def _texts(self, value: dict[str, Any], index: Any) -> list[tuple[str, str]]:
        if not self.index_config or index is False:
            return []
        fields = (
            [(path, tokenize_path(path)) for path in index]
            if isinstance(index, list)
            else self._index_fields
        )
        out: list[tuple[str, str]] = []
        for path, tokens in fields:
            for text in get_text_at_path(value, "$" if path == "$" else tokens):
                if text:
                    out.append((path, text))
        return out

    def _vector_ids(self, namespace: tuple[str, ...], key: str) -> list[str]:
        if self.index_config is None:
            return []
        rows = find_all(
            self.client,
            {
                "labels": [self.vector_label],
                "where": {"ns": namespace_string(namespace), "key": key},
            },
        )
        return [r["__id"] for r in rows]

    def _ensure_vector_index(self) -> None:
        if self._vector_index_ready:
            return
        assert self.index_config is not None
        dims = self.index_config["dims"]
        for index in self.client.ai.indexes.find().data or []:
            if (
                index.get("label") == self.vector_label
                and index.get("propertyName") == "text"
                and index.get("sourceType") == "external"
                and index.get("dimensions") == dims
                and index.get("similarityFunction") == "cosine"
            ):
                self._vector_index_ready = True
                return
        self.client.ai.indexes.create(
            {
                "label": self.vector_label,
                "propertyName": "text",
                "external": True,
                "dimensions": dims,
                "similarityFunction": "cosine",
            }
        )
        self._vector_index_ready = True

    # ── search ─────────────────────────────────────────────────────────────

    def _search(self, op: SearchOp) -> list[SearchItem]:
        pushed, exact = _pushdown(op.filter)
        if op.namespace_prefix:
            pushed = (
                {"$and": [pushed, prefix_where(op.namespace_prefix)]}
                if pushed
                else prefix_where(op.namespace_prefix)
            )
        if op.query and self.embeddings is not None:
            # Liveness is checked on the item: vectors carry no expiry.
            items = self._vector_search(op, pushed)
        else:
            where: dict[str, Any] = {"$and": [self._live(), pushed]} if pushed else self._live()
            search: dict[str, Any] = {
                "labels": [self.item_label],
                "where": where,
                "orderBy": {"updated_at": "desc"},
            }
            if exact:
                page = self.client.records.find({**search, "skip": op.offset, "limit": op.limit})
                rows = [r.data for r in page.data]
            else:
                matched = [
                    r for r in find_all(self.client, search) if _matches(_value(r), op.filter)
                ]
                rows = matched[op.offset : op.offset + op.limit]
            items = [_search_item(r, None) for r in rows]
        if op.refresh_ttl and items:
            self._refresh_items(items)
        return items

    def _vector_search(self, op: SearchOp, where: dict[str, Any]) -> list[SearchItem]:
        assert self.embeddings is not None
        vector = self.embeddings.embed_query(op.query or "")
        self._ensure_vector_index()
        wanted = op.offset + op.limit
        pool = max(wanted * max(1, len(self._index_fields)) * 2, 10)
        while True:
            params: dict[str, Any] = {
                "labels": [self.vector_label],
                "propertyName": "text",
                "queryVector": vector,
                "similarityFunction": "cosine",
                "limit": pool,
            }
            if where:
                params["where"] = where
            result = self.client.records.vector_search(params)
            hits = [r.data for r in result.data]
            best: dict[tuple[str, str], float] = {}
            for hit in hits:
                ident = (hit["ns"], hit["key"])
                score = float(hit.get("__score", 0.0))
                if ident not in best or score > best[ident]:
                    best[ident] = score
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
            items: list[SearchItem] = []
            for (ns, key), score in ranked:
                rows = [
                    r
                    for r in find_all(
                        self.client, {"labels": [self.item_label], "where": {"ns": ns, "key": key}}
                    )
                    if not _expired(r)
                ]
                if rows and _matches(_value(rows[0]), op.filter):
                    items.append(_search_item(rows[0], score))
                if len(items) >= wanted:
                    break
            if len(items) >= wanted or len(hits) < pool or pool >= SEARCH_POOL_MAX:
                return items[op.offset : op.offset + op.limit]
            pool = min(pool * 4, SEARCH_POOL_MAX)

    def _refresh_items(self, items: list[SearchItem]) -> None:
        for item in items:
            self._refresh(
                [
                    r
                    for r in self._item_rows(item.namespace, item.key)
                    if isinstance(r.get("ttl_minutes"), (int, float))
                ]
            )

    # ── namespaces ─────────────────────────────────────────────────────────

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = find_all(
            self.client,
            {
                "labels": [self.item_label],
                "where": self._live(),
                "select": {"items": {"$count": "*"}},
                "groupBy": ["ns"],
            },
        )
        namespaces: set[tuple[str, ...]] = set()
        for row in rows:
            ns = row.get("ns")
            if not isinstance(ns, str) or not ns:
                continue
            path = tuple(ns.split("."))
            if op.match_conditions and not all(_does_match(c, path) for c in op.match_conditions):
                continue
            if op.max_depth is not None:
                path = path[: op.max_depth]
            namespaces.add(path)
        ordered = sorted(namespaces)
        return ordered[op.offset : op.offset + op.limit]

    # ── TTL sweeping ───────────────────────────────────────────────────────

    def sweep(self) -> int:
        """Delete expired items and their vectors; returns how many items expired."""
        expired = {"expires_at": {"$lte": now_iso()}}
        items = list(find_all(self.client, {"labels": [self.item_label], "where": expired}))
        for row in items:
            namespace = tuple(row["ns"].split("."))
            delete_ids(self.client, [row["__id"], *self._vector_ids(namespace, row["key"])])
        return len(items)

    def start_ttl_sweeper(self, sweep_interval_minutes: float | None = None) -> threading.Thread:
        """Run :meth:`sweep` in a daemon thread every ``sweep_interval_minutes``."""
        interval = (
            sweep_interval_minutes or (self.ttl_config or {}).get("sweep_interval_minutes") or 5
        )
        if self._sweeper is not None and self._sweeper.is_alive():
            return self._sweeper
        self._sweeper_stop.clear()

        def run() -> None:
            while not self._sweeper_stop.wait(interval * 60):
                self.sweep()

        self._sweeper = threading.Thread(target=run, name="rushdb-store-ttl", daemon=True)
        self._sweeper.start()
        return self._sweeper

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        if self._sweeper is None:
            return True
        self._sweeper_stop.set()
        self._sweeper.join(timeout)
        return not self._sweeper.is_alive()


# ── helpers ────────────────────────────────────────────────────────────────


def _expired(row: Mapping[str, Any]) -> bool:
    expires = row.get("expires_at")
    return isinstance(expires, str) and parse_iso(expires) <= datetime.now(timezone.utc)


def _value(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("lg_value")
    loaded = json.loads(raw) if isinstance(raw, str) else {}
    return loaded if isinstance(loaded, dict) else {}


def _item(row: Mapping[str, Any]) -> Item:
    return Item(
        value=_value(row),
        key=row["key"],
        namespace=tuple(row["ns"].split(".")),
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
    )


def _search_item(row: Mapping[str, Any], score: float | None) -> SearchItem:
    return SearchItem(
        namespace=tuple(row["ns"].split(".")),
        key=row["key"],
        value=_value(row),
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
        score=score,
    )


def _pushdown(filter: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """The part of ``filter`` the engine answers exactly, and whether that is all of it.

    Only equality on top-level scalars is pushed down: comparisons and ``$ne``
    keep LangGraph's semantics (number coercion, missing fields) in the client.
    """
    if not filter:
        return {}, True
    pushed: dict[str, Any] = {}
    exact = True
    for key, value in filter.items():
        if is_scalar(value):
            pushed[f"{VALUE_PREFIX}.{key}"] = value
        elif isinstance(value, Mapping) and set(value) == {"$eq"} and is_scalar(value["$eq"]):
            pushed[f"{VALUE_PREFIX}.{key}"] = value["$eq"]
        else:
            exact = False
    return pushed, exact


def _matches(value: Mapping[str, Any], filter: Mapping[str, Any] | None) -> bool:
    if not filter:
        return True
    return all(_compare(value.get(k), v) for k, v in filter.items())


def _compare(item_value: Any, filter_value: Any) -> bool:
    if isinstance(filter_value, Mapping):
        if any(k.startswith("$") for k in filter_value):
            return all(_operator(item_value, op, operand) for op, operand in filter_value.items())
        if not isinstance(item_value, Mapping):
            return False
        return all(_compare(item_value.get(k), v) for k, v in filter_value.items())
    if isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(_compare(i, f) for i, f in zip(item_value, filter_value, strict=False))
        )
    return bool(item_value == filter_value)


def _operator(value: Any, op: str, operand: Any) -> bool:
    if op == "$eq":
        return bool(value == operand)
    if op == "$ne":
        return bool(value != operand)
    if op in ("$gt", "$gte", "$lt", "$lte"):
        try:
            left, right = float(value), float(operand)
        except (TypeError, ValueError):
            return False
        return {
            "$gt": left > right,
            "$gte": left >= right,
            "$lt": left < right,
            "$lte": left <= right,
        }[op]
    raise ValueError(f"Unsupported operator: {op}")


def _does_match(condition: MatchCondition, key: tuple[str, ...]) -> bool:
    path = condition.path
    if len(key) < len(path):
        return False
    pairs = (
        zip(key, path, strict=False)
        if condition.match_type == "prefix"
        else zip(reversed(key), reversed(path), strict=False)
    )
    return all(p == "*" or k == p for k, p in pairs)
