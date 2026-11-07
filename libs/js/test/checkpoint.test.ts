/** RushDBSaver: LangGraph's validation suite and a real graph, once per backend. */
import { randomUUID } from 'node:crypto'

import { Annotation, Command, END, START, StateGraph, interrupt } from '@langchain/langgraph'
import { validate } from '@langchain/langgraph-checkpoint-validation'
import { expect, it } from 'vitest'

import type { Client } from '../src/connection.js'
import { BLOB, CHECKPOINT, RushDBSaver, WRITE } from '../src/checkpoint.js'
import { describeBackends } from './backends.js'

async function purge(client: Client): Promise<void> {
  for (const label of [CHECKPOINT, BLOB, WRITE]) await client.records.delete({ labels: [label] } as any)
}

const State = Annotation.Root({
  log: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] })
})

function graph(saver: RushDBSaver) {
  return new StateGraph(State)
    .addNode('plan', () => ({ log: ['plan'] }))
    .addNode('approve', () => {
      const answer = interrupt('approve?')
      return { log: [`approved:${answer}`] }
    })
    .addEdge(START, 'plan')
    .addEdge('plan', 'approve')
    .addEdge('approve', END)
    .compile({ checkpointer: saver })
}

describeBackends('RushDBSaver', (makeClient) => {
  const client = makeClient()

  validate({
    checkpointerName: 'RushDBSaver',
    async createCheckpointer() {
      await purge(client)
      return new RushDBSaver({ client })
    },
    async destroyCheckpointer() {
      await purge(client)
    }
  })

  it('resumes an interrupted thread from a fresh saver', async () => {
    const thread = { configurable: { thread_id: `t-${randomUUID()}` } }
    const out = await graph(new RushDBSaver({ client })).invoke({ log: [] }, thread)
    expect(out).toHaveProperty('__interrupt__')
    const second = graph(new RushDBSaver({ client }))
    expect((await second.getState(thread)).next).toEqual(['approve'])
    const final = await second.invoke(new Command({ resume: 'yes' }), thread)
    expect(final.log).toEqual(['plan', 'approved:yes'])
    const history = []
    for await (const snapshot of second.getStateHistory(thread)) history.push(snapshot)
    expect(history.length).toBeGreaterThanOrEqual(3)
  })

  it('deletes a thread', async () => {
    const saver = new RushDBSaver({ client })
    const threadId = `t-${randomUUID()}`
    await graph(saver).invoke({ log: [] }, { configurable: { thread_id: threadId } })
    await saver.deleteThread(threadId)
    const left = []
    for await (const t of saver.list({ configurable: { thread_id: threadId } })) left.push(t)
    expect(left).toEqual([])
    for (const label of [CHECKPOINT, BLOB, WRITE]) {
      const rows = await client.records.find({ labels: [label], where: { thread_id: threadId } } as any)
      expect(rows.total).toBe(0)
    }
  })
})
