"""RushDBSaver: LangGraph's conformance suite and a real graph, once per backend."""

import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_rushdb import RushDBSaver
from langchain_rushdb.checkpoint import BLOB, CHECKPOINT, WRITE


def _purge(client):
    for label in (CHECKPOINT, BLOB, WRITE):
        client.records.delete({"labels": [label]})


async def test_conformance(client):
    _purge(client)

    @checkpointer_test(name="RushDBSaver")
    async def saver():
        yield RushDBSaver(client=client)
        _purge(client)

    report = await validate(saver)
    report.print_report()
    assert report.passed_all_base()
    assert report.passed_all()


class State(TypedDict):
    log: Annotated[list[str], operator.add]


def _graph(saver):
    def plan(state):
        return {"log": ["plan"]}

    def approve(state):
        answer = interrupt("approve?")
        return {"log": [f"approved:{answer}"]}

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("approve", approve)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approve")
    builder.add_edge("approve", END)
    return builder.compile(checkpointer=saver)


def test_interrupt_and_resume_across_saver_instances(client):
    thread = {"configurable": {"thread_id": f"t-{uuid.uuid4()}"}}
    first = _graph(RushDBSaver(client=client))
    out = first.invoke({"log": []}, thread)
    assert "__interrupt__" in out
    # A fresh saver (new process, same database) resumes the thread.
    second = _graph(RushDBSaver(client=client))
    assert second.get_state(thread).next == ("approve",)
    final = second.invoke(Command(resume="yes"), thread)
    assert final["log"] == ["plan", "approved:yes"]
    history = list(second.get_state_history(thread))
    assert len(history) >= 3


def test_delete_thread(client):
    saver = RushDBSaver(client=client)
    thread_id = f"t-{uuid.uuid4()}"
    graph = _graph(saver)
    graph.invoke({"log": []}, {"configurable": {"thread_id": thread_id}})
    saver.delete_thread(thread_id)
    assert list(saver.list({"configurable": {"thread_id": thread_id}})) == []
    for label in (CHECKPOINT, BLOB, WRITE):
        rows = client.records.find({"labels": [label], "where": {"thread_id": thread_id}})
        assert rows.total == 0
