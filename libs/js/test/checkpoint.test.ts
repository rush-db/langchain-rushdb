/** RushDBSaver: LangGraph's validation suite and a real graph, once per backend. */
import { randomUUID } from 'node:crypto'

import { Annotation, Command, END, START, StateGraph, interrupt } from '@langchain/langgraph'
import { validate } from '@langchain/langgraph-checkpoint-validation'
import { expect, it } from 'vitest'

import type { Client } from '../src/connection.js'
import { RushDBSaver } from '../src/checkpoint.js'
import { clearLabels, describeBackends, label } from './backends.js'

/** A saver on this job's checkpoint labels, emptied first. */
async function freshSaver(client: Client): Promise<RushDBSaver> {
  const saver = new RushDBSaver({ client, label: label('LG_CP') })
  await clear(saver)
  return saver
}

async function clear(saver: RushDBSaver): Promise<void> {
  await clearLabels(saver.client, saver.label, saver.blobLabel, saver.writeLabel)
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
      return await freshSaver(client)
    },
    async destroyCheckpointer(saver: RushDBSaver) {
      await clear(saver)
    }
  })

  it('resumes an interrupted thread from a fresh saver', async () => {
    const thread = { configurable: { thread_id: `t-${randomUUID()}` } }
    const first = await freshSaver(client)
    const out = await graph(first).invoke({ log: [] }, thread)
    expect(out).toHaveProperty('__interrupt__')
    const second = graph(new RushDBSaver({ client, label: first.label }))
    expect((await second.getState(thread)).next).toEqual(['approve'])
    const final = await second.invoke(new Command({ resume: 'yes' }), thread)
    expect(final.log).toEqual(['plan', 'approved:yes'])
    const history = []
    for await (const snapshot of second.getStateHistory(thread)) history.push(snapshot)
    expect(history.length).toBeGreaterThanOrEqual(3)
    await clear(first)
  })

  it('deletes a thread', async () => {
    const saver = await freshSaver(client)
    const threadId = `t-${randomUUID()}`
    await graph(saver).invoke({ log: [] }, { configurable: { thread_id: threadId } })
    await saver.deleteThread(threadId)
    const left = []
    for await (const t of saver.list({ configurable: { thread_id: threadId } })) left.push(t)
    expect(left).toEqual([])
    for (const label of [saver.label, saver.blobLabel, saver.writeLabel]) {
      const rows = await client.records.find({ labels: [label], where: { thread_id: threadId } } as any)
      expect(rows.total).toBe(0)
    }
    await clear(saver)
  })
})
