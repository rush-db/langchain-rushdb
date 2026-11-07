import pytest

from langchain_rushdb import RushDBVectorStore, client_from_env


def test_cloud_from_env():
    client = client_from_env({"RUSHDB_API_KEY": "key", "RUSHDB_URL": "https://example.test"})
    assert not client.direct
    assert client.base_url == "https://example.test/api/v1"


def test_self_hosted_from_env():
    client = client_from_env(
        {
            "RUSHDB_ENGINE_URL": "http://127.0.0.1:3000",
            "RUSHDB_ENGINE_TOKEN": "t",
            "RUSHDB_SCOPE": "tenant-a",
        }
    )
    assert client.direct
    assert client.engine.scope == "tenant-a"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"RUSHDB_API_KEY": "k", "RUSHDB_ENGINE_URL": "http://x"},
        {"RUSHDB_ENGINE_URL": "http://x"},
    ],
)
def test_env_errors(env):
    with pytest.raises(ValueError):
        client_from_env(env)


def test_managed_embeddings_refused_self_hosted():
    client = client_from_env({"RUSHDB_ENGINE_URL": "http://x", "RUSHDB_ENGINE_TOKEN": "t"})
    with pytest.raises(ValueError, match="needs RushDB Cloud"):
        RushDBVectorStore("managed", client=client)


def test_managed_embeddings_allowed_cloud():
    client = client_from_env({"RUSHDB_API_KEY": "k"})
    assert RushDBVectorStore("managed", client=client).managed
