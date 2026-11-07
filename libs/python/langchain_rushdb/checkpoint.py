"""RushDB-backed LangGraph checkpointer."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from rushdb import RushDB

from ._connection import b64, client_from_env, delete_ids, find_all, mirror, transact, unb64

DEFAULT_LABEL = "LG_CHECKPOINT"
METADATA_PREFIX = "metadata"


class RushDBSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpointer on RushDB, Cloud or self-hosted alike.

    Checkpoints, channel values and pending writes are records of
    ``<label>``, ``<label>_BLOB`` and ``<label>_WRITE`` (see the
    repository's ``spec/README.md``); serialized bytes are stored base64-encoded.
    Each ``put`` and ``put_writes`` is one RushDB transaction.

    The async methods run the sync ones in a worker thread.

    Args:
        client: A ``rushdb.RushDB`` client; built from the environment when omitted.
        serde: The serializer; LangGraph's default when omitted.
        label: Label of checkpoint records; channel values and pending writes
            use ``<label>_BLOB`` and ``<label>_WRITE``. Separate labels keep
            separate applications in one RushDB project apart.
    """

    def __init__(
        self,
        *,
        client: RushDB | None = None,
        serde: SerializerProtocol | None = None,
        label: str = DEFAULT_LABEL,
    ) -> None:
        super().__init__(serde=serde)
        self.client = client if client is not None else client_from_env()
        self.label = label
        self.blob_label = f"{label}_BLOB"
        self.write_label = f"{label}_WRITE"

    # ── reads ──────────────────────────────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        where: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id := get_checkpoint_id(config):
            where["checkpoint_id"] = checkpoint_id
        page = self.client.records.find(
            {
                "labels": [self.label],
                "where": where,
                "orderBy": {"checkpoint_id": "desc"},
                "limit": 1,
            }
        )
        if not page.data:
            return None
        return self._tuple(page.data[0].data)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        where: dict[str, Any] = {}
        if config is not None:
            configurable = config["configurable"]
            if "thread_id" in configurable:
                where["thread_id"] = str(configurable["thread_id"])
            if configurable.get("checkpoint_ns") is not None:
                where["checkpoint_ns"] = configurable["checkpoint_ns"]
            if checkpoint_id := get_checkpoint_id(config):
                where["checkpoint_id"] = checkpoint_id
        # The engine compares numbers and datetimes only: `before` (a string
        # id) is applied here, on rows already ordered by id, newest first.
        before_id = get_checkpoint_id(before) if before is not None else None
        pushed, exact = _metadata_pushdown(filter)
        exact = exact and before_id is None
        where.update(pushed)
        search: dict[str, Any] = {
            "labels": [self.label],
            "orderBy": {"checkpoint_id": "desc"},
        }
        if where:
            search["where"] = where
        if exact and limit is not None:
            if limit <= 0:
                return
            rows: Iterator[dict[str, Any]] = (
                r.data for r in self.client.records.find({**search, "limit": limit}).data
            )
        else:
            rows = find_all(self.client, search)
        count = 0
        for row in rows:
            if limit is not None and count >= limit:
                return
            if before_id is not None and row["checkpoint_id"] >= before_id:
                continue
            checkpoint_tuple = self._tuple(row)
            if filter and not all(checkpoint_tuple.metadata.get(k) == v for k, v in filter.items()):
                continue
            count += 1
            yield checkpoint_tuple

    def _tuple(self, row: dict[str, Any]) -> CheckpointTuple:
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        checkpoint: Checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], unb64(row["checkpoint"]))
        )
        metadata: CheckpointMetadata = self.serde.loads_typed(
            (row["metadata_type"], unb64(row["metadata"]))
        )
        parent = row.get("parent_checkpoint_id")
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint["channel_versions"]
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent,
                    }
                }
                if parent
                else None
            ),
            pending_writes=self._load_writes(thread_id, checkpoint_ns, checkpoint_id),
        )

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        if not versions:
            return {}
        wanted = {(channel, str(version)) for channel, version in versions.items()}
        rows = find_all(
            self.client,
            {
                "labels": [self.blob_label],
                "where": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": {"$in": sorted(versions)},
                },
            },
        )
        values: dict[str, Any] = {}
        for row in rows:
            if (row["channel"], row["version"]) not in wanted or row["blob_type"] == "empty":
                continue
            values[row["channel"]] = self.serde.loads_typed(
                (row["blob_type"], unb64(row.get("blob", "")))
            )
        return values

    def _load_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> builtins.list[tuple[str, str, Any]]:
        rows = list(
            find_all(
                self.client,
                {
                    "labels": [self.write_label],
                    "where": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                },
            )
        )
        rows.sort(key=lambda r: writes_sort_key(r["task_path"], r["task_id"], int(r["idx"])))
        return [
            (
                r["task_id"],
                r["channel"],
                self.serde.loads_typed((r["blob_type"], unb64(r.get("blob", "")))),
            )
            for r in rows
        ]

    # ── writes ─────────────────────────────────────────────────────────────

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        parent_id = configurable.get("checkpoint_id")
        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]

        present = (
            {
                (r["channel"], r["version"])
                for r in find_all(
                    self.client,
                    {
                        "labels": [self.blob_label],
                        "where": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "channel": {"$in": sorted(new_versions)},
                        },
                    },
                )
            }
            if new_versions
            else set()
        )
        blobs = []
        for channel, version in new_versions.items():
            if (channel, str(version)) in present:
                continue
            row: dict[str, Any] = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "channel": channel,
                "version": str(version),
            }
            if channel in values:
                blob_type, data = self.serde.dumps_typed(values[channel])
                row.update(blob_type=blob_type, blob=b64(data))
            else:
                row["blob_type"] = "empty"
            blobs.append(row)

        full_metadata = get_checkpoint_metadata(config, metadata)
        checkpoint_type, checkpoint_data = self.serde.dumps_typed(stored)
        metadata_type, metadata_data = self.serde.dumps_typed(full_metadata)
        record: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "checkpoint_type": checkpoint_type,
            "checkpoint": b64(checkpoint_data),
            "metadata_type": metadata_type,
            "metadata": b64(metadata_data),
            **mirror(METADATA_PREFIX, full_metadata),
        }
        if parent_id:
            record["parent_checkpoint_id"] = parent_id
        replaced = [
            r["__id"]
            for r in find_all(
                self.client,
                {
                    "labels": [self.label],
                    "where": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint["id"],
                    },
                },
            )
        ]
        transact(self.client, replaced, [(self.blob_label, blobs), (self.label, [record])])
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]
        existing = {
            int(r["idx"]): r["__id"]
            for r in find_all(
                self.client,
                {
                    "labels": [self.write_label],
                    "where": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                    },
                },
            )
        }
        replaced: list[str] = []
        rows: list[dict[str, Any]] = []
        for position, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, position)
            if idx in existing:
                if idx >= 0:
                    continue
                replaced.append(existing[idx])
            blob_type, data = self.serde.dumps_typed(value)
            rows.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": idx,
                    "channel": channel,
                    "blob_type": blob_type,
                    "blob": b64(data),
                }
            )
        transact(self.client, replaced, [(self.write_label, rows)])

    def delete_thread(self, thread_id: str) -> None:
        for label in (self.label, self.blob_label, self.write_label):
            self.client.records.delete({"labels": [label], "where": {"thread_id": str(thread_id)}})

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete the checkpoints whose metadata ``run_id`` is one of ``run_ids``,
        with their pending writes."""
        if not run_ids:
            return
        rows = list(
            find_all(
                self.client,
                {
                    "labels": [self.label],
                    "where": {f"{METADATA_PREFIX}.run_id": {"$in": [str(r) for r in run_ids]}},
                },
            )
        )
        for row in rows:
            self.client.records.delete(
                {
                    "labels": [self.write_label],
                    "where": {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": row["checkpoint_ns"],
                        "checkpoint_id": row["checkpoint_id"],
                    },
                }
            )
        delete_ids(self.client, [r["__id"] for r in rows])

    # ── async (worker thread) ──────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await asyncio.to_thread(self.delete_for_runs, run_ids)


def _metadata_pushdown(filter: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """Equality on scalar metadata runs in the engine; the rest in the client."""
    if not filter:
        return {}, True
    pushed: dict[str, Any] = {}
    exact = True
    for key, value in filter.items():
        if isinstance(value, (str, bool, int, float)):
            pushed[f"{METADATA_PREFIX}.{key}"] = value
        else:
            exact = False
    return pushed, exact
