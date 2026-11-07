/** RushDBSaver: LangGraph's validation suite and a real graph, once per backend. */
import { randomUUID } from 'node:crypto'

import { Annotation, Command, END, START, StateGraph, interrupt } from '@langchain/langgraph'
import { validate } from '@langchain/langgraph-checkpoint-validation'
import { expect, it } from 'vitest'

import type { Client } from '../src/connection.js'
import { RushDBSaver } from '../src/checkpoint.js'
import { describeBackends, dropLabels, unique } from './backends.js'

/** A saver on labels of its own, so concurrent runs on one project never meet. */
function freshSaver(client: Client): RushDBSaver {
  return new RushDBSaver({ client, label: `LG_CP_${unique()}` })
}

async function drop(saver: RushDBSaver): Promise<void> {
  await dropLabels(saver.client, saver.label, saver.blobLabel, saver.writeLabel)
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
      return freshSaver(client)
    },
    async destroyCheckpointer(saver: RushDBSaver) {
      await drop(saver)
    }
  })

  it('resumes an interrupted thread from a fresh saver', async () => {
    const thread = { configurable: { thread_id: `t-${randomUUID()}` } }
    const first = freshSaver(client)
    const out = await graph(first).invoke({ log: [] }, thread)
    expect(out).toHaveProperty('__interrupt__')
    const second = graph(new RushDBSaver({ client, label: first.label }))
    expect((await second.getState(thread)).next).toEqual(['approve'])
    const final = await second.invoke(new Command({ resume: 'yes' }), thread)
    expect(final.log).toEqual(['plan', 'approved:yes'])
    const history = []
    for await (const snapshot of second.getStateHistory(thread)) history.push(snapshot)
    expect(history.length).toBeGreaterThanOrEqual(3)
    await drop(first)
  })

  it('deletes a thread', async () => {
    const saver = freshSaver(client)
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
    await drop(saver)
  })
})
