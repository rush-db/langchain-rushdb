"""RushDBSaver: LangGraph's conformance suite and a real graph, once per backend."""

import operator
import uuid
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_rushdb import RushDBSaver
from tests.conftest import clear_labels, label


@pytest.fixture
def saver_label(client):
    name = label("LG_CP")
    _clear(client, name)
    yield name
    _clear(client, name)


def _clear(client, name):
    clear_labels(client, name, f"{name}_BLOB", f"{name}_WRITE")


async def test_conformance(client):
    name = label("LG_CF")

    @checkpointer_test(name="RushDBSaver")
    async def saver():
        # Some conformance tests list every checkpoint: start each one empty.
        _clear(client, name)
        yield RushDBSaver(client=client, label=name)
        _clear(client, name)

    report = await validate(saver)
    report.print_report()
    failures = [
        f"{capability}: {failure}"
        for capability, result in report.results.items()
        for failure in result.failures
    ]
    assert report.passed_all(), "\n".join(failures)


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


def test_interrupt_and_resume_across_saver_instances(client, saver_label):
    thread = {"configurable": {"thread_id": f"t-{uuid.uuid4()}"}}
    first = _graph(RushDBSaver(client=client, label=saver_label))
    out = first.invoke({"log": []}, thread)
    assert "__interrupt__" in out
    # A fresh saver (new process, same database) resumes the thread.
    second = _graph(RushDBSaver(client=client, label=saver_label))
    assert second.get_state(thread).next == ("approve",)
    final = second.invoke(Command(resume="yes"), thread)
    assert final["log"] == ["plan", "approved:yes"]
    history = list(second.get_state_history(thread))
    assert len(history) >= 3


def test_delete_thread(client, saver_label):
    saver = RushDBSaver(client=client, label=saver_label)
    thread_id = f"t-{uuid.uuid4()}"
    graph = _graph(saver)
    graph.invoke({"log": []}, {"configurable": {"thread_id": thread_id}})
    saver.delete_thread(thread_id)
    assert list(saver.list({"configurable": {"thread_id": thread_id}})) == []
    for name in (saver.label, saver.blob_label, saver.write_label):
        rows = client.records.find({"labels": [name], "where": {"thread_id": thread_id}})
        assert rows.total == 0
