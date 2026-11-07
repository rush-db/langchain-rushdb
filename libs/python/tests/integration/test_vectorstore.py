"""Vector store and hybrid retriever behaviour beyond the standard suite."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_rushdb import FilterError, RushDBHybridRetriever, RushDBVectorStore


class KeywordEmbeddings(Embeddings):
    """Deterministic 4-d embeddings: one axis per topic word."""

    AXES = ("apple", "ocean", "graph", "agent")

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [1.0 if word in lowered else 0.0 for word in self.AXES]
        return vector if any(vector) else [0.1, 0.1, 0.1, 0.1]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


DOCS = [
    Document(
        id="d1",
        page_content="apple pie recipe",
        metadata={"kind": "food", "year": 2021, "tags": ["sweet"]},
    ),
    Document(
        id="d2",
        page_content="ocean currents",
        metadata={"kind": "science", "year": 2024, "nested": {"a": 1}},
    ),
    Document(
        id="d3",
        page_content="graph databases for agent memory",
        metadata={"kind": "tech", "year": 2026},
    ),
    Document(id="d4", page_content="apple orchard graph", metadata={"kind": "food", "year": 2026}),
]


@pytest.fixture
def store(client, unique):
    s = RushDBVectorStore(
        KeywordEmbeddings(),
        client=client,
        label=f"LC_VS_{unique}",
        hybrid=True,
        text_index_consistency="immediate",
    )
    s.add_documents(DOCS)
    yield s
    s.delete()


def test_metadata_round_trip(store):
    [doc] = store.get_by_ids(["d2"])
    assert doc.metadata == {"kind": "science", "year": 2024, "nested": {"a": 1}}


def test_filter_before_ranking(store):
    docs = store.similarity_search("apple", k=4, filter={"kind": "food", "year": {"$gte": 2025}})
    assert [d.id for d in docs] == ["d4"]


def test_logic_and_membership_filters(store):
    docs = store.similarity_search(
        "graph", k=4, filter={"$or": [{"kind": {"$in": ["science"]}}, {"tags": "sweet"}]}
    )
    assert {d.id for d in docs} == {"d1", "d2"}


def test_scores_and_relevance(store):
    [(doc, score)] = store.similarity_search_with_score("ocean", k=1)
    assert doc.id == "d2" and score == pytest.approx(1.0)
    [(_, relevance)] = store.similarity_search_with_relevance_scores("ocean", k=1)
    assert relevance == pytest.approx(1.0)


def test_unsupported_filter_raises(store):
    with pytest.raises(FilterError):
        store.similarity_search("x", filter={"kind": {"$regex": "f.*"}})


def test_raw_where(store):
    docs = store.similarity_search("apple", k=4, where={"metadata.year": 2021})
    assert [d.id for d in docs] == ["d1"]


def test_hybrid_retriever_fuses_keyword_and_vector(store):
    # "pie recipe" has no topic axis: the vector leg alone ranks d3/d4 first,
    # only the BM25 leg knows d1 is the match.
    assert store.similarity_search("pie recipe", k=1)[0].id != "d1"
    docs = RushDBHybridRetriever(vectorstore=store, k=2).invoke("pie recipe")
    assert docs[0].id == "d1"
    assert all("score" in d.metadata for d in docs)


def test_hybrid_retriever_filter(store):
    retriever = RushDBHybridRetriever(vectorstore=store, k=4, filter={"kind": "tech"})
    assert [d.id for d in retriever.invoke("graph")] == ["d3"]


def test_hybrid_needs_text_index(client, unique):
    plain = RushDBVectorStore(KeywordEmbeddings(), client=client, label=f"LC_NH_{unique}")
    with pytest.raises(ValueError, match="hybrid=True"):
        RushDBHybridRetriever(vectorstore=plain).invoke("x")


def test_as_retriever(store):
    docs = store.as_retriever(search_kwargs={"k": 1, "filter": {"kind": "science"}}).invoke("x")
    assert [d.id for d in docs] == ["d2"]
