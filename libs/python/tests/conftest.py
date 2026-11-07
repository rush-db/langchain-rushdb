"""Backends for the integration tests.

Every integration test runs once per configured backend:

- ``self_hosted``: ``RUSHDB_TEST_ENGINE_URL`` + ``RUSHDB_TEST_ENGINE_TOKEN``
- ``self_hosted_multitenant``: ``RUSHDB_TEST_MT_ENGINE_URL`` + ``RUSHDB_TEST_MT_ENGINE_TOKEN``
  (each test gets its own tenant)
- ``cloud``: ``RUSHDB_TEST_API_KEY`` (+ optional ``RUSHDB_TEST_URL``)

A backend that is not configured is skipped, never silently passed; CI sets
``RUSHDB_TEST_REQUIRE`` to the backends that must run.
"""

import os
import uuid

import pytest
from rushdb import RushDB

BACKENDS = ["self_hosted", "self_hosted_multitenant", "cloud"]


def _client(backend: str) -> RushDB | None:
    env = os.environ
    if backend == "self_hosted" and env.get("RUSHDB_TEST_ENGINE_URL"):
        return RushDB(
            env["RUSHDB_TEST_ENGINE_TOKEN"], direct=True, engine_url=env["RUSHDB_TEST_ENGINE_URL"]
        )
    if backend == "self_hosted_multitenant" and env.get("RUSHDB_TEST_MT_ENGINE_URL"):
        return RushDB(
            env["RUSHDB_TEST_MT_ENGINE_TOKEN"],
            direct=True,
            engine_url=env["RUSHDB_TEST_MT_ENGINE_URL"],
            scope=f"t{uuid.uuid4().hex[:12]}",
        )
    if backend == "cloud" and env.get("RUSHDB_TEST_API_KEY"):
        return RushDB(env["RUSHDB_TEST_API_KEY"], url=env.get("RUSHDB_TEST_URL") or None)
    return None


@pytest.fixture(params=BACKENDS)
def client(request):
    backend = request.param
    instance = _client(backend)
    if instance is None:
        required = os.environ.get("RUSHDB_TEST_REQUIRE", "").split(",")
        if backend in required:
            pytest.fail(f"backend {backend} is required but not configured")
        pytest.skip(f"backend {backend} not configured")
    return instance


def label(base: str) -> str:
    """A test label that is stable across runs of one CI job.

    RushDB never reclaims a dropped vector index's space (256 per tenant), so
    tests reuse their labels, and their indexes, instead of minting new ones.
    ``RUSHDB_TEST_LABEL_SUFFIX`` tells concurrent jobs apart; CI serializes
    runs, so one job's labels are never used twice at once.
    """
    return f"{base}_{os.environ.get('RUSHDB_TEST_LABEL_SUFFIX', 'LOCAL_PY')}"


def clear_labels(client: RushDB, *labels: str) -> None:
    """Delete every record of ``labels``; their indexes stay for the next run."""
    for name in labels:
        client.records.delete({"labels": [name]})
