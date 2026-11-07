"""RushDB vector store for LangChain."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from rushdb import RushDB

from ._connection import (
    client_from_env,
    create_rows,
    delete_ids,
    find_all,
    is_self_hosted,
    mirror,
    to_json,
)
from .filters import translate_filter

Similarity = Literal["cosine", "dot", "euclidean"]

ID_PROPERTY = "lc_id"
METADATA_PROPERTY = "lc_metadata"
METADATA_PREFIX = "metadata"


class RushDBVectorStore(VectorStore):
    """LangChain vector store backed by RushDB, Cloud or self-hosted alike.

    Each document is one record of ``label``: its text, its full metadata as JSON
    and the metadata's top-level scalar fields as filterable ``metadata.<key>``
    properties (see the repository's ``spec/README.md``). Vectors live in a
    RushDB external vector index created on first use.

    Args:
        embedding: The embeddings that turn texts into vectors, on every backend.
            ``"managed"`` lets RushDB Cloud compute them instead; a self-hosted
            engine refuses it, as it never computes embeddings.
        client: A ``rushdb.RushDB`` client. When omitted it is built from the
            environment (``RUSHDB_API_KEY``, or ``RUSHDB_ENGINE_URL`` and
            ``RUSHDB_ENGINE_TOKEN``).
        label: The record label documents are stored under.
        text_property: The property holding the document text.
        similarity: The vector index similarity.
        hybrid: Also create a text (BM25) index, needed by
            :class:`~langchain_rushdb.RushDBHybridRetriever`.
        text_index_consistency: ``"eventual"`` (engine default) or ``"immediate"``.
    """

    def __init__(
        self,
        embedding: Embeddings | Literal["managed"],
        *,
        client: RushDB | None = None,
        label: str = "LC_DOCUMENT",
        text_property: str = "page_content",
        similarity: Similarity = "cosine",
        hybrid: bool = False,
        text_index_consistency: Literal["eventual", "immediate"] = "eventual",
    ) -> None:
        self.client = client if client is not None else client_from_env()
        self.managed = embedding == "managed"
        if self.managed and is_self_hosted(self.client):
            raise ValueError(
                'embedding="managed" needs RushDB Cloud: a self-hosted engine never computes '
                "embeddings. Pass a LangChain Embeddings instance instead."
            )
        self.embedding: Embeddings | None = None if self.managed else embedding  # type: ignore[assignment]
        if text_property in (ID_PROPERTY, METADATA_PROPERTY) or text_property.startswith("__"):
            raise ValueError(f"text_property {text_property!r} is reserved.")
        self.label = label
        self.text_property = text_property
        self.similarity: Similarity = similarity
        self.hybrid = hybrid
        self.text_index_consistency = text_index_consistency
        self._index_ready = False
        self._text_index_ready = False

    @property
    def embeddings(self) -> Embeddings | None:
        return self.embedding

    # ── indexes ────────────────────────────────────────────────────────────

    def ensure_indexes(self, dimensions: int | None = None) -> None:
        """Create the vector index (and the text index when ``hybrid``) if missing."""
        if not self._index_ready:
            self._ensure_vector_index(dimensions)
            self._index_ready = True
        if self.hybrid and not self._text_index_ready:
            self._ensure_text_index()
            self._text_index_ready = True

    def _ensure_vector_index(self, dimensions: int | None) -> None:
        existing = self.client.ai.indexes.find().data or []
        for index in existing:
            if index.get("label") != self.label or index.get("propertyName") != self.text_property:
                continue
            if self.managed and index.get("sourceType") != "external":
                return
            if (
                not self.managed
                and index.get("sourceType") == "external"
                and index.get("similarityFunction") == self.similarity
                and (dimensions is None or index.get("dimensions") == dimensions)
            ):
                return
        params: dict[str, Any] = {"label": self.label, "propertyName": self.text_property}
        if not self.managed:
            if dimensions is None:
                assert self.embedding is not None
                dimensions = len(self.embedding.embed_query("dimension probe"))
            params.update(external=True, dimensions=dimensions, similarityFunction=self.similarity)
        self.client.ai.indexes.create(params)

    def _ensure_text_index(self) -> None:
        for index in self.client.text_indexes.find().data or []:
            if index.get("label") == self.label and index.get("property") == self.text_property:
                return
        self.client.text_indexes.create(
            {
                "label": self.label,
                "property": self.text_property,
                "consistency": self.text_index_consistency,
            }
        )

    # ── writes ─────────────────────────────────────────────────────────────

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        docs = [
            Document(page_content=t, metadata=m or {}, id=(ids[i] if ids else None))
            for i, (t, m) in enumerate(zip(texts, metadatas, strict=True))
        ]
        return self.add_documents(docs, **kwargs)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Store ``documents``; a document whose id already exists replaces it.

        New documents are written in one atomic ``createMany``; records of
        existing ids are deleted first.
        """
        ids_arg: list[str] | None = kwargs.get("ids")
        if ids_arg is not None and len(ids_arg) != len(documents):
            raise ValueError("ids must have one entry per document.")
        ids = [
            (ids_arg[i] if ids_arg else None) or doc.id or str(uuid.uuid4())
            for i, doc in enumerate(documents)
        ]
        if not documents:
            return []
        rows = [self._row(doc, doc_id) for doc, doc_id in zip(documents, ids, strict=True)]
        vectors: list[list[dict[str, Any]]] | None = None
        if self.embedding is not None:
            embedded = self.embedding.embed_documents([doc.page_content for doc in documents])
            self.ensure_indexes(len(embedded[0]))
            vectors = [
                [
                    {
                        "propertyName": self.text_property,
                        "vector": vector,
                        "similarityFunction": self.similarity,
                    }
                ]
                for vector in embedded
            ]
        else:
            self.ensure_indexes()
        delete_ids(self.client, self._record_ids(ids))
        create_rows(self.client, self.label, rows, vectors)
        return ids

    def _row(self, doc: Document, doc_id: str) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        return {
            ID_PROPERTY: doc_id,
            self.text_property: doc.page_content,
            METADATA_PROPERTY: to_json(metadata),
            **mirror(METADATA_PREFIX, metadata),
        }

    def _record_ids(self, ids: Sequence[str]) -> list[str]:
        if not ids:
            return []
        rows = find_all(
            self.client,
            {"labels": [self.label], "where": {ID_PROPERTY: {"$in": list(ids)}}},
        )
        return [row["__id"] for row in rows]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete the documents with ``ids``; ``ids=None`` deletes every document."""
        if ids is None:
            self.client.records.delete({"labels": [self.label]})
            return True
        delete_ids(self.client, self._record_ids(ids))
        return True

    # ── reads ──────────────────────────────────────────────────────────────

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if not ids:
            return []
        rows = find_all(
            self.client,
            {"labels": [self.label], "where": {ID_PROPERTY: {"$in": list(ids)}}},
        )
        return [self._document(row) for row in rows]

    def _document(self, row: dict[str, Any]) -> Document:
        raw = row.get(METADATA_PROPERTY)
        metadata = json.loads(raw) if isinstance(raw, str) and raw else {}
        return Document(
            id=row.get(ID_PROPERTY),
            page_content=row.get(self.text_property, ""),
            metadata=metadata,
        )

    def search_params(
        self,
        query: str | None,
        vector: list[float] | None,
        k: int,
        filter: dict[str, Any] | None,
        where: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The RushDB vector search request; ``where`` is a raw RushDB filter."""
        params: dict[str, Any] = {
            "labels": [self.label],
            "propertyName": self.text_property,
            "limit": k,
        }
        if self.managed:
            params["query"] = query
        else:
            params["queryVector"] = vector
            params["similarityFunction"] = self.similarity
        combined = translate_filter(filter, METADATA_PREFIX)
        if where:
            combined = {"$and": [combined, where]} if combined else dict(where)
        if combined:
            params["where"] = combined
        return params

    def run_search(self, params: dict[str, Any]) -> list[tuple[Document, float]]:
        vector = params.get("queryVector")
        self.ensure_indexes(len(vector) if vector else None)
        result = self.client.records.vector_search(params)
        return [(self._document(r.data), float(r.data.get("__score", 0.0))) for r in result.data]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        where: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        vector = None if self.embedding is None else self.embedding.embed_query(query)
        return self.run_search(self.search_params(query, vector, k, filter, where))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        where: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        if self.managed:
            raise ValueError("Search by vector needs caller-computed embeddings, not managed ones.")
        params = self.search_params(None, embedding, k, filter, where)
        return [doc for doc, _ in self.run_search(params)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return relevance_fn(self.similarity)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> RushDBVectorStore:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store


def relevance_fn(similarity: Similarity) -> Callable[[float], float]:
    """Map an engine score (higher is closer) to a relevance in [0, 1]."""
    if similarity == "cosine":
        return lambda s: (1.0 + s) / 2.0
    if similarity == "dot":
        return lambda s: 1.0 / (1.0 + math.exp(-s))
    return lambda s: s
