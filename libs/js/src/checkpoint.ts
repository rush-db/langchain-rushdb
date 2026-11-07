/** RushDB-backed LangGraph checkpointer. */
import type { RunnableConfig } from '@langchain/core/runnables'
import {
  BaseCheckpointSaver,
  type ChannelVersions,
  type Checkpoint,
  type CheckpointListOptions,
  type CheckpointMetadata,
  type CheckpointPendingWrite,
  type CheckpointTuple,
  type PendingWrite,
  type SerializerProtocol,
  TASKS,
  WRITES_IDX_MAP,
  compareWritesSortKeys,
  copyCheckpoint,
  maxChannelVersion,
  writesSortKey
} from '@langchain/langgraph-checkpoint'

import {
  type Client,
  type Row,
  b64,
  clientFromEnv,
  deleteIds,
  findAll,
  isScalar,
  mirror,
  transact,
  unb64
} from './connection.js'

export const CHECKPOINT = 'LG_CHECKPOINT'
export const BLOB = 'LG_CHECKPOINT_BLOB'
export const WRITE = 'LG_CHECKPOINT_WRITE'
const METADATA_PREFIX = 'metadata'

export interface RushDBSaverConfig {
  /** A RushDB client; built from the environment when omitted. */
  client?: Client
  /** The serializer; LangGraph's default when omitted. */
  serde?: SerializerProtocol
}

/**
 * LangGraph checkpointer on RushDB, Cloud or self-hosted alike.
 *
 * Checkpoints, channel values and pending writes are records of
 * `LG_CHECKPOINT`, `LG_CHECKPOINT_BLOB` and `LG_CHECKPOINT_WRITE` (see the
 * repository's `spec/README.md`); serialized bytes are stored base64-encoded.
 * Each `put` and `putWrites` is one RushDB transaction.
 */
export class RushDBSaver extends BaseCheckpointSaver {
  readonly client: Client

  constructor(config: RushDBSaverConfig = {}) {
    super(config.serde)
    this.client = config.client ?? clientFromEnv()
  }

  // ── reads ──────────────────────────────────────────────────────────────

  async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
    const configurable = config.configurable ?? {}
    const where: Row = {
      thread_id: String(configurable.thread_id),
      checkpoint_ns: configurable.checkpoint_ns ?? ''
    }
    if (configurable.checkpoint_id) where.checkpoint_id = configurable.checkpoint_id
    const page = await this.client.records.find({
      labels: [CHECKPOINT],
      where,
      orderBy: { checkpoint_id: 'desc' },
      limit: 1
    } as any)
    if (!page.data.length) return undefined
    return this.tuple(page.data[0].data as Row)
  }

  async *list(config: RunnableConfig, options: CheckpointListOptions = {}): AsyncGenerator<CheckpointTuple> {
    const { filter, before, limit } = options
    const configurable = config?.configurable ?? {}
    const where: Row = {}
    if (configurable.thread_id !== undefined) where.thread_id = String(configurable.thread_id)
    if (configurable.checkpoint_ns !== undefined && configurable.checkpoint_ns !== null) {
      where.checkpoint_ns = configurable.checkpoint_ns
    }
    if (configurable.checkpoint_id) where.checkpoint_id = configurable.checkpoint_id
    // The engine compares numbers and datetimes only: `before` (a string id)
    // is applied here, on rows already ordered by id, newest first.
    const beforeId: string | undefined = before?.configurable?.checkpoint_id
    const [pushed, pushedAll] = metadataPushdown(filter)
    Object.assign(where, pushed)
    const exact = pushedAll && beforeId === undefined
    const search: Row = { labels: [CHECKPOINT], orderBy: { checkpoint_id: 'desc' } }
    if (Object.keys(where).length) search.where = where
    let rows: Row[]
    if (exact && limit !== undefined) {
      if (limit <= 0) return
      const page = await this.client.records.find({ ...search, limit } as any)
      rows = page.data.map((r: any) => r.data as Row)
    } else {
      rows = await findAll(this.client, search)
    }
    let count = 0
    for (const row of rows) {
      if (limit !== undefined && count >= limit) return
      if (beforeId !== undefined && row.checkpoint_id >= beforeId) continue
      const checkpointTuple = await this.tuple(row)
      if (
        filter &&
        !Object.entries(filter).every(
          ([k, v]) => (checkpointTuple.metadata as Record<string, unknown> | undefined)?.[k] === v
        )
      ) {
        continue
      }
      count += 1
      yield checkpointTuple
    }
  }

  private async tuple(row: Row): Promise<CheckpointTuple> {
    const { thread_id, checkpoint_ns, checkpoint_id } = row
    const checkpoint: Checkpoint = await this.serde.loadsTyped(row.checkpoint_type, unb64(row.checkpoint))
    const metadata: CheckpointMetadata = await this.serde.loadsTyped(row.metadata_type, unb64(row.metadata))
    const loaded: Checkpoint = {
      ...checkpoint,
      channel_values: await this.loadBlobs(thread_id, checkpoint_ns, checkpoint.channel_versions)
    }
    if (checkpoint.v < 4 && row.parent_checkpoint_id) {
      await this.migratePendingSends(loaded, thread_id, checkpoint_ns, row.parent_checkpoint_id)
    }
    const tuple: CheckpointTuple = {
      config: { configurable: { thread_id, checkpoint_ns, checkpoint_id } },
      checkpoint: loaded,
      metadata,
      pendingWrites: await this.loadWrites(thread_id, checkpoint_ns, checkpoint_id)
    }
    if (row.parent_checkpoint_id) {
      tuple.parentConfig = {
        configurable: { thread_id, checkpoint_ns, checkpoint_id: row.parent_checkpoint_id }
      }
    }
    return tuple
  }

  /** Checkpoints before format v4 kept pending sends as writes of the parent. */
  private async migratePendingSends(
    checkpoint: Checkpoint,
    threadId: string,
    checkpointNs: string,
    parentId: string
  ): Promise<void> {
    const sends = (await this.loadWrites(threadId, checkpointNs, parentId))
      .filter(([, channel]) => channel === TASKS)
      .map(([, , value]) => value)
    checkpoint.channel_values ??= {}
    checkpoint.channel_values[TASKS] = sends
    checkpoint.channel_versions ??= {}
    const versions = Object.values(checkpoint.channel_versions)
    checkpoint.channel_versions[TASKS] = versions.length
      ? maxChannelVersion(...versions)
      : this.getNextVersion(undefined)
  }

  private async loadBlobs(
    threadId: string,
    checkpointNs: string,
    versions: ChannelVersions
  ): Promise<Record<string, unknown>> {
    const channels = Object.keys(versions ?? {})
    if (!channels.length) return {}
    const wanted = new Set(channels.map((c) => JSON.stringify([c, String(versions[c])])))
    const rows = await findAll(this.client, {
      labels: [BLOB],
      where: { thread_id: threadId, checkpoint_ns: checkpointNs, channel: { $in: channels.sort() } }
    })
    const values: Record<string, unknown> = {}
    for (const row of rows) {
      if (!wanted.has(JSON.stringify([row.channel, row.version])) || row.blob_type === 'empty') continue
      values[row.channel] = await this.serde.loadsTyped(row.blob_type, unb64(row.blob ?? ''))
    }
    return values
  }

  private async loadWrites(
    threadId: string,
    checkpointNs: string,
    checkpointId: string
  ): Promise<CheckpointPendingWrite[]> {
    const rows = await findAll(this.client, {
      labels: [WRITE],
      where: { thread_id: threadId, checkpoint_ns: checkpointNs, checkpoint_id: checkpointId }
    })
    rows.sort((a, b) =>
      compareWritesSortKeys(
        writesSortKey(a.task_path, a.task_id, Number(a.idx)),
        writesSortKey(b.task_path, b.task_id, Number(b.idx))
      )
    )
    return Promise.all(
      rows.map(
        async (r) =>
          [r.task_id, r.channel, await this.serde.loadsTyped(r.blob_type, unb64(r.blob ?? ''))] as CheckpointPendingWrite
      )
    )
  }

  // ── writes ─────────────────────────────────────────────────────────────

  async put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    newVersions: ChannelVersions
  ): Promise<RunnableConfig> {
    const configurable = config.configurable ?? {}
    if (configurable.thread_id === undefined) {
      throw new Error('Failed to put checkpoint: the config is missing "configurable.thread_id".')
    }
    const threadId = String(configurable.thread_id)
    const checkpointNs: string = configurable.checkpoint_ns ?? ''
    const parentId: string | undefined = configurable.checkpoint_id
    const { channel_values: values, ...stored } = copyCheckpoint(checkpoint)

    const channels = Object.keys(newVersions ?? {})
    const present = new Set(
      channels.length
        ? (
            await findAll(this.client, {
              labels: [BLOB],
              where: { thread_id: threadId, checkpoint_ns: checkpointNs, channel: { $in: [...channels].sort() } }
            })
          ).map((r) => JSON.stringify([r.channel, r.version]))
        : []
    )
    const blobs: Row[] = []
    for (const channel of channels) {
      const version = String(newVersions[channel])
      if (present.has(JSON.stringify([channel, version]))) continue
      const row: Row = { thread_id: threadId, checkpoint_ns: checkpointNs, channel, version }
      if (channel in values) {
        const [blobType, data] = await this.serde.dumpsTyped(values[channel])
        row.blob_type = blobType
        row.blob = b64(data)
      } else {
        row.blob_type = 'empty'
      }
      blobs.push(row)
    }

    const [checkpointType, checkpointData] = await this.serde.dumpsTyped(stored)
    const [metadataType, metadataData] = await this.serde.dumpsTyped(metadata)
    const record: Row = {
      thread_id: threadId,
      checkpoint_ns: checkpointNs,
      checkpoint_id: checkpoint.id,
      checkpoint_type: checkpointType,
      checkpoint: b64(checkpointData),
      metadata_type: metadataType,
      metadata: b64(metadataData),
      ...mirror(METADATA_PREFIX, (metadata ?? {}) as Record<string, unknown>)
    }
    if (parentId) record.parent_checkpoint_id = parentId
    const replaced = (
      await findAll(this.client, {
        labels: [CHECKPOINT],
        where: { thread_id: threadId, checkpoint_ns: checkpointNs, checkpoint_id: checkpoint.id }
      })
    ).map((r) => r.__id as string)
    await transact(this.client, replaced, [
      [BLOB, blobs],
      [CHECKPOINT, [record]]
    ])
    return { configurable: { thread_id: threadId, checkpoint_ns: checkpointNs, checkpoint_id: checkpoint.id } }
  }

  async putWrites(config: RunnableConfig, writes: PendingWrite[], taskId: string, taskPath = ''): Promise<void> {
    const configurable = config.configurable ?? {}
    if (configurable.thread_id === undefined) {
      throw new Error('Failed to put writes: the config is missing "configurable.thread_id".')
    }
    if (configurable.checkpoint_id === undefined) {
      throw new Error('Failed to put writes: the config is missing "configurable.checkpoint_id".')
    }
    const threadId = String(configurable.thread_id)
    const checkpointNs: string = configurable.checkpoint_ns ?? ''
    const checkpointId: string = configurable.checkpoint_id
    const existing = new Map<number, string>()
    for (const r of await findAll(this.client, {
      labels: [WRITE],
      where: { thread_id: threadId, checkpoint_ns: checkpointNs, checkpoint_id: checkpointId, task_id: taskId }
    })) {
      existing.set(Number(r.idx), r.__id as string)
    }
    const replaced: string[] = []
    const rows: Row[] = []
    for (const [position, [channel, value]] of writes.entries()) {
      const idx = WRITES_IDX_MAP[channel] ?? position
      const prior = existing.get(idx)
      if (prior !== undefined) {
        if (idx >= 0) continue
        replaced.push(prior)
      }
      const [blobType, data] = await this.serde.dumpsTyped(value)
      rows.push({
        thread_id: threadId,
        checkpoint_ns: checkpointNs,
        checkpoint_id: checkpointId,
        task_id: taskId,
        task_path: taskPath,
        idx,
        channel,
        blob_type: blobType,
        blob: b64(data)
      })
    }
    await transact(this.client, replaced, [[WRITE, rows]])
  }

  async deleteThread(threadId: string): Promise<void> {
    for (const label of [CHECKPOINT, BLOB, WRITE]) {
      await this.client.records.delete({ labels: [label], where: { thread_id: String(threadId) } } as any)
    }
  }

  /** Delete the checkpoints whose metadata `run_id` is one of `runIds`, with their pending writes. */
  async deleteForRuns(runIds: string[]): Promise<void> {
    if (!runIds.length) return
    const rows = await findAll(this.client, {
      labels: [CHECKPOINT],
      where: { [`${METADATA_PREFIX}.run_id`]: { $in: runIds.map(String) } }
    })
    for (const row of rows) {
      await this.client.records.delete({
        labels: [WRITE],
        where: {
          thread_id: row.thread_id,
          checkpoint_ns: row.checkpoint_ns,
          checkpoint_id: row.checkpoint_id
        }
      } as any)
    }
    await deleteIds(
      this.client,
      rows.map((r) => r.__id as string)
    )
  }
}

/** Equality on scalar metadata runs in the engine; the rest in the client. */
function metadataPushdown(filter?: Record<string, unknown>): [Row, boolean] {
  if (!filter || !Object.keys(filter).length) return [{}, true]
  const pushed: Row = {}
  let exact = true
  for (const [key, val] of Object.entries(filter)) {
    if (isScalar(val)) pushed[`${METADATA_PREFIX}.${key}`] = val
    else exact = false
  }
  return [pushed, exact]
}
