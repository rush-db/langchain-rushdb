"""RushDBStore behaviour, once per backend."""

import time
from datetime import datetime

import pytest

from langchain_rushdb import RushDBStore
from tests.conftest import clear_labels, label
from tests.integration.test_vectorstore import KeywordEmbeddings


@pytest.fixture
def store(client):
    s = RushDBStore(
        client=client,
        item_label=label("LG_ITEM"),
        vector_label=label("LG_VEC"),
        index={"dims": 4, "embed": KeywordEmbeddings(), "fields": ["text"]},
    )
    clear_labels(client, s.item_label, s.vector_label)
    yield s
    clear_labels(client, s.item_label, s.vector_label)


def test_put_get_round_trip(store):
    value = {"text": "likes apple", "nested": {"a": [1, {"b": 2}]}, "n": 3}
    store.put(("users", "u1"), "k1", value)
    item = store.get(("users", "u1"), "k1")
    assert item is not None
    assert item.value == value
    assert item.namespace == ("users", "u1") and item.key == "k1"
    assert isinstance(item.created_at, datetime)
    assert store.get(("users", "u2"), "k1") is None


def test_update_keeps_created_at(store):
    store.put(("a",), "k", {"text": "v1"})
    first = store.get(("a",), "k")
    time.sleep(0.01)
    store.put(("a",), "k", {"text": "v2"})
    second = store.get(("a",), "k")
    assert second.value == {"text": "v2"}
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at
    assert len(store.search(("a",))) == 1


def test_delete(store):
    store.put(("a",), "k", {"text": "apple"})
    store.delete(("a",), "k")
    assert store.get(("a",), "k") is None
    assert store.search(("a",), query="apple") == []


def test_search_prefix_and_filters(store):
    store.put(("users", "u1", "mem"), "m1", {"text": "apple", "kind": "pref", "score": 5})
    store.put(("users", "u1", "mem"), "m2", {"text": "ocean", "kind": "fact", "score": 9})
    store.put(("users", "u10"), "m3", {"text": "graph", "kind": "pref", "score": 1})
    assert {i.key for i in store.search(("users", "u1"))} == {"m1", "m2"}
    assert {i.key for i in store.search(("users",), filter={"kind": "pref"})} == {"m1", "m3"}
    assert {i.key for i in store.search(("users",), filter={"score": {"$gt": 4}})} == {"m1", "m2"}
    assert {i.key for i in store.search(("users",), filter={"kind": {"$ne": "pref"}})} == {"m2"}
    page = store.search(("users",), limit=2, offset=1)
    assert len(page) == 2


def test_nested_filter_runs_in_client(store):
    store.put(("n",), "a", {"text": "x", "meta": {"lang": "en"}})
    store.put(("n",), "b", {"text": "y", "meta": {"lang": "fr"}})
    assert [i.key for i in store.search(("n",), filter={"meta": {"lang": "fr"}})] == ["b"]


def test_semantic_search(store):
    store.put(("docs",), "d1", {"text": "apple pie"})
    store.put(("docs",), "d2", {"text": "deep ocean"})
    store.put(("other",), "d3", {"text": "apple tree"})
    results = store.search(("docs",), query="apple")
    assert results[0].key == "d1"
    assert results[0].score is not None
    assert "d3" not in {r.key for r in results}
    filtered = store.search(("docs",), query="apple", filter={"text": "deep ocean"})
    assert [r.key for r in filtered] == ["d2"]


def test_index_false_skips_vectors(store):
    store.put(("docs",), "plain", {"text": "apple"}, index=False)
    assert store.search(("docs",), query="apple") == []
    assert store.get(("docs",), "plain") is not None


def test_list_namespaces(store):
    for ns in [("a", "b", "c"), ("a", "b", "d"), ("a", "x"), ("z",)]:
        store.put(ns, "k", {"text": "v"})
    assert store.list_namespaces() == [("a", "b", "c"), ("a", "b", "d"), ("a", "x"), ("z",)]
    assert store.list_namespaces(prefix=("a", "b")) == [("a", "b", "c"), ("a", "b", "d")]
    assert store.list_namespaces(suffix=("d",)) == [("a", "b", "d")]
    assert store.list_namespaces(prefix=("a", "*", "c")) == [("a", "b", "c")]
    assert store.list_namespaces(max_depth=2) == [("a", "b"), ("a", "x"), ("z",)]
    assert store.list_namespaces(limit=1, offset=1) == [("a", "b", "d")]


# TTLs are long enough to outlast a slow network round trip.
TTL_MINUTES = 0.1  # 6 s


def test_ttl_hides_and_sweeps(client):
    store = RushDBStore(client=client, item_label=label("LG_TTL"), ttl={"default_ttl": TTL_MINUTES})
    clear_labels(client, store.item_label)
    store.put(("t",), "short", {"v": 1})
    store.put(("t",), "long", {"v": 2}, ttl=None)
    assert store.get(("t",), "short", refresh_ttl=False) is not None
    time.sleep(TTL_MINUTES * 60 + 1)
    assert store.get(("t",), "short") is None
    assert [i.key for i in store.search(("t",))] == ["long"]
    assert store.list_namespaces() == [("t",)]
    assert store.sweep() == 1
    assert client.records.find({"labels": [store.item_label]}).total == 1
    clear_labels(client, store.item_label)


def test_ttl_refresh_on_read(client):
    store = RushDBStore(
        client=client, item_label=label("LG_TTLR"), ttl={"default_ttl": TTL_MINUTES}
    )
    clear_labels(client, store.item_label)
    store.put(("t",), "k", {"v": 1})
    # Three reads TTL * 2/3 apart: 12 s in all, twice the TTL, so only the
    # refreshes keep the item alive.
    for _ in range(3):
        time.sleep(TTL_MINUTES * 60 * 2 / 3)
        assert store.get(("t",), "k") is not None
    clear_labels(client, store.item_label)


async def test_async_batch(store):
    await store.aput(("as",), "k", {"text": "apple"})
    item = await store.aget(("as",), "k")
    assert item.value == {"text": "apple"}
    assert [r.key for r in await store.asearch(("as",), query="apple")] == ["k"]
