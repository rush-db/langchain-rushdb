"""LangChain's standard vector store suite, once per backend."""

import pytest
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_rushdb import RushDBVectorStore
from tests.conftest import drop_labels


class TestRushDBVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self, client, unique):
        store = RushDBVectorStore(self.get_embeddings(), client=client, label=f"LC_STD_{unique}")
        yield store
        drop_labels(client, store.label)
