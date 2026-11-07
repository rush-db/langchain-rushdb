"""Hybrid (vector + BM25) retriever over a :class:`RushDBVectorStore`."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .vectorstores import RushDBVectorStore


class RushDBHybridRetriever(BaseRetriever):
    """Retrieve with RushDB hybrid search: a vector leg and a BM25 keyword leg,
    filtered before ranking and fused in the engine (RRF or DBSF).

    The vector store must be created with ``hybrid=True`` so its text index
    exists. ``filter`` uses the LangChain filter form on document metadata;
    ``where`` is a raw RushDB filter, which can also match related records.
    Each returned document carries the fused score in ``metadata["score"]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: RushDBVectorStore
    k: int = 4
    fusion: Literal["rrf", "dbsf"] = "rrf"
    rrf_k: int | None = None
    vector_weight: float | None = None
    text_weight: float | None = None
    prefetch_limit: int | None = None
    all_terms: bool = False
    phrase: bool = False
    filter: dict[str, Any] | None = None
    where: dict[str, Any] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        store = self.vectorstore
        if not store.hybrid:
            raise ValueError("RushDBHybridRetriever needs a RushDBVectorStore with hybrid=True.")
        vector = None if store.embedding is None else store.embedding.embed_query(query)
        params = store.search_params(
            query,
            vector,
            kwargs.get("k", self.k),
            kwargs.get("filter", self.filter),
            kwargs.get("where", self.where),
        )
        text: dict[str, Any] = {"query": query, "property": store.text_property}
        if self.all_terms:
            text["allTerms"] = True
        if self.phrase:
            text["phrase"] = True
        if self.text_weight is not None:
            text["weight"] = self.text_weight
        if self.prefetch_limit is not None:
            text["prefetchLimit"] = self.prefetch_limit
            params["prefetchLimit"] = self.prefetch_limit
        if self.vector_weight is not None:
            params["weight"] = self.vector_weight
        params["text"] = text
        params["fusion"] = self.fusion
        if self.rrf_k is not None:
            params["rrfK"] = self.rrf_k
        docs = []
        for doc, score in store.run_search(params):
            doc.metadata["score"] = score
            docs.append(doc)
        return docs
