"""Write one vector-store document and one store item with the Python package.

Usage: write_from_python.py ENGINE_URL TOKEN SUFFIX
"""

import sys

from langchain_core.embeddings import Embeddings
from rushdb import RushDB

from langchain_rushdb import RushDBStore, RushDBVectorStore


class KeywordEmbeddings(Embeddings):
    AXES = ("apple", "ocean", "graph", "agent")

    def _embed(self, text):
        lowered = text.lower()
        vector = [1.0 if word in lowered else 0.0 for word in self.AXES]
        return vector if any(vector) else [0.1, 0.1, 0.1, 0.1]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


engine_url, token, suffix = sys.argv[1:4]
client = RushDB(token, direct=True, engine_url=engine_url)
RushDBVectorStore(KeywordEmbeddings(), client=client, label=f"LC_X_{suffix}").add_texts(
    ["apple orchard"],
    metadatas=[{"kind": "food", "tags": ["a", "b"], "nested": {"x": 1}}],
    ids=["py-doc"],
)
RushDBStore(
    client=client,
    item_label=f"LG_XI_{suffix}",
    vector_label=f"LG_XV_{suffix}",
    index={"dims": 4, "embed": KeywordEmbeddings(), "fields": ["text"]},
).put(("users", "u1"), "py-item", {"text": "apple memory", "kind": "pref", "n": 2})
