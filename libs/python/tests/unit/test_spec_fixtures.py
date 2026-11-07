"""The shared spec fixtures; the JavaScript package runs the same cases."""

import pytest

from langchain_rushdb._connection import mirror
from langchain_rushdb.filters import FilterError, translate_filter
from langchain_rushdb.store import namespace_string, prefix_where
from langchain_rushdb.vectorstores import relevance_fn
from tests.fixtures import load

FILTERS = load("filters.json")
MIRRORS = load("mirrors.json")
NAMESPACES = load("namespaces.json")


@pytest.mark.parametrize("case", FILTERS["cases"], ids=lambda c: c["name"])
def test_filter_translation(case):
    assert translate_filter(case["filter"], FILTERS["prefix"]) == case["where"]


@pytest.mark.parametrize("case", FILTERS["errors"], ids=lambda c: c["name"])
def test_filter_errors(case):
    with pytest.raises(FilterError):
        translate_filter(case["filter"], FILTERS["prefix"])


@pytest.mark.parametrize("case", MIRRORS["cases"], ids=lambda c: c["name"])
def test_mirrors(case):
    assert mirror(MIRRORS["prefix"], case["object"]) == case["properties"]


@pytest.mark.parametrize("case", NAMESPACES["cases"], ids=lambda c: c["ns"])
def test_namespace_string(case):
    assert namespace_string(tuple(case["namespace"])) == case["ns"]


@pytest.mark.parametrize("case", NAMESPACES["prefixWhere"], ids=lambda c: ".".join(c["prefix"]))
def test_prefix_where(case):
    assert prefix_where(tuple(case["prefix"])) == case["where"]


@pytest.mark.parametrize(
    "case", NAMESPACES["relevance"], ids=lambda c: f"{c['similarity']}-{c['score']}"
)
def test_relevance(case):
    assert relevance_fn(case["similarity"])(case["score"]) == pytest.approx(case["relevance"])
