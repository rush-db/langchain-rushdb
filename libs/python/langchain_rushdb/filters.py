"""Translation of LangChain-style filters to a RushDB ``where`` (spec: Filters)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._connection import is_scalar

COMPARISON = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}
MEMBERSHIP = {"$in", "$nin"}
STRING_MATCH = {"$contains", "$startsWith", "$endsWith"}
LOGIC = {"$and", "$or", "$not"}


class FilterError(ValueError):
    """A filter that cannot be translated; never silently ignored."""


def translate_filter(filter: Mapping[str, Any] | None, prefix: str) -> dict[str, Any]:
    """Rename every field to ``<prefix>.<field>`` and validate the operators."""
    if not filter:
        return {}
    if not isinstance(filter, Mapping):
        raise FilterError(f"A filter is an object, got {type(filter).__name__}.")
    where: dict[str, Any] = {}
    for key, value in filter.items():
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise FilterError(f"{key} takes a non-empty list of filters.")
            where[key] = [translate_filter(item, prefix) for item in value]
        elif key == "$not":
            where[key] = translate_filter(value, prefix)
        elif key.startswith("$"):
            raise FilterError(f"Unsupported logic operator {key!r}.")
        else:
            where[f"{prefix}.{key}"] = _condition(key, value)
    return where


def _condition(field: str, value: Any) -> Any:
    if is_scalar(value):
        return value
    if not isinstance(value, Mapping) or not value:
        raise FilterError(
            f"The condition on {field!r} must be a string, number, boolean or an "
            f"operator object, got {value!r}."
        )
    out: dict[str, Any] = {}
    for op, operand in value.items():
        if op in COMPARISON:
            if not is_scalar(operand):
                raise FilterError(f"{op} on {field!r} takes a string, number or boolean.")
        elif op in MEMBERSHIP:
            if not isinstance(operand, list) or not all(is_scalar(v) for v in operand):
                raise FilterError(f"{op} on {field!r} takes a list of scalars.")
        elif op == "$exists":
            if not isinstance(operand, bool):
                raise FilterError(f"$exists on {field!r} takes a boolean.")
        elif op in STRING_MATCH:
            if not isinstance(operand, str):
                raise FilterError(f"{op} on {field!r} takes a string.")
        else:
            raise FilterError(f"Unsupported operator {op!r} on {field!r}.")
        out[op] = operand
    return out
