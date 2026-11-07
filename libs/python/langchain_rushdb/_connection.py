"""Client construction and the record helpers every component shares."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

from rushdb import RushDB, RushDBError

logger = logging.getLogger("langchain_rushdb")

PAGE_SIZE = 1000
"""Records fetched per request when reading every match."""

FAR_FUTURE = "9999-12-31T23:59:59.000Z"


def client_from_env(env: Mapping[str, str] | None = None) -> RushDB:
    """Build a RushDB client from the environment.

    RushDB Cloud: ``RUSHDB_API_KEY`` (and optionally ``RUSHDB_URL``).
    Self-hosted engine: ``RUSHDB_ENGINE_URL`` and ``RUSHDB_ENGINE_TOKEN``
    (and ``RUSHDB_SCOPE`` on a multitenant engine). Setting both is an error.
    """
    env = os.environ if env is None else env
    api_key = env.get("RUSHDB_API_KEY")
    engine_url = env.get("RUSHDB_ENGINE_URL")
    if api_key and engine_url:
        raise ValueError(
            "Both RUSHDB_API_KEY (RushDB Cloud) and RUSHDB_ENGINE_URL (self-hosted) are set; "
            "set the variables of one mode only."
        )
    if engine_url:
        token = env.get("RUSHDB_ENGINE_TOKEN")
        if not token:
            raise ValueError("RUSHDB_ENGINE_URL is set but RUSHDB_ENGINE_TOKEN is not.")
        return RushDB(
            token,
            direct=True,
            engine_url=engine_url,
            scope=env.get("RUSHDB_SCOPE") or None,
        )
    if api_key:
        return RushDB(api_key, url=env.get("RUSHDB_URL") or None)
    raise ValueError(
        "No RushDB connection configured: set RUSHDB_API_KEY for RushDB Cloud, or "
        "RUSHDB_ENGINE_URL and RUSHDB_ENGINE_TOKEN for a self-hosted engine."
    )


def is_self_hosted(client: RushDB) -> bool:
    return bool(getattr(client, "direct", False))


def find_all(client: RushDB, search: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the data of every record ``search`` matches, page by page."""
    skip = 0
    while True:
        page = client.records.find({**search, "skip": skip, "limit": PAGE_SIZE})
        rows = [record.data for record in page.data]
        yield from rows
        if len(rows) < PAGE_SIZE:
            return
        skip += PAGE_SIZE


def create_rows(
    client: RushDB,
    label: str,
    rows: list[dict[str, Any]],
    vectors: list[list[dict[str, Any]]] | None = None,
    transaction: Any = None,
) -> None:
    """Create ``rows`` in one engine commit, each with its inline ``vectors``.

    ``records.import_json`` rather than ``create_many``: rows may hold flat
    arrays, which the engine stores as array properties.
    """
    if not rows:
        return
    data = [
        {**row, "$vectors": vectors[i]} if vectors and vectors[i] else row
        for i, row in enumerate(rows)
    ]
    client.records.import_json(
        data,
        label=label,
        options={"returnResult": False, "suggestTypes": True},
        transaction=transaction,
    )


CONFLICT_RETRIES = 8


def transact(
    client: RushDB,
    delete: list[str],
    create: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    """Delete records by id and create rows in one transaction.

    Concurrent transactions can conflict; the engine reports that as
    retryable, so the transaction is replayed with backoff.
    """
    if not delete and not any(rows for _, rows in create):
        return
    for attempt in range(CONFLICT_RETRIES):
        tx = client.tx.begin()
        committing = False
        try:
            for record in delete:
                client.records.delete_by_id(record, transaction=tx)
            for label, rows in create:
                create_rows(client, label, rows, transaction=tx)
            committing = True
            client.tx.commit(tx)
            return
        except BaseException as exc:
            # A failed commit has already ended the transaction.
            if not committing:
                _rollback(client, tx, exc)
            if (
                not isinstance(exc, RushDBError)
                or "transaction_conflict" not in str(exc)
                or attempt == CONFLICT_RETRIES - 1
            ):
                raise
            time.sleep(0.02 * 2**attempt * (1 + random.random()))


def _rollback(client: RushDB, tx: Any, cause: BaseException) -> None:
    try:
        client.tx.rollback(tx)
    except RushDBError as exc:
        logger.warning("rollback after %r also failed: %s", cause, exc)


def delete_ids(client: RushDB, ids: list[str]) -> None:
    if ids:
        client.records.delete_by_id(ids)


def is_scalar(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, str):
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)


def _kind(value: Any) -> type:
    return bool if isinstance(value, bool) else str if isinstance(value, str) else float


def mirror(prefix: str, obj: Mapping[str, Any]) -> dict[str, Any]:
    """The filterable properties of ``obj``: top-level scalars and flat arrays."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if is_scalar(value):
            out[f"{prefix}.{key}"] = value
        elif (
            isinstance(value, list)
            and value
            and all(is_scalar(v) for v in value)
            and len({_kind(v) for v in value}) == 1
        ):
            out[f"{prefix}.{key}"] = list(value)
    return out


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(data: str) -> bytes:
    return base64.b64decode(data)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    """Parse an engine datetime; the engine may echo midnight as a bare date."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def now_iso() -> str:
    return iso(datetime.now(timezone.utc))
