"""LangChain's standard vector store suite, once per backend."""

import pytest
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_rushdb import RushDBVectorStore
from tests.conftest import clear_labels, label


class TestRushDBVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self, client):
        store = RushDBVectorStore(self.get_embeddings(), client=client, label=label("LC_STD"))
        clear_labels(client, store.label)
        yield store
        clear_labels(client, store.label)
