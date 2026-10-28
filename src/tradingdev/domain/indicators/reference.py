"""Discoverable indicator references derived from the installed public facade."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.domain import indicators
from tradingdev.domain.indicators import technical

if TYPE_CHECKING:
    from types import FunctionType


def _functions() -> dict[str, FunctionType]:
    return {
        name: value
        for name in indicators.__all__
        if inspect.isfunction(value := getattr(indicators, name))
    }


def _summary(name: str, function: FunctionType) -> dict[str, str]:
    return {
        "name": name,
        "signature": f"{name}{inspect.signature(function)}",
        "summary": (inspect.getdoc(function) or "").split("\n", 1)[0],
    }


def _supporting_source(function: FunctionType) -> list[dict[str, str]]:
    """Include actual local helper calls, including their transitive validators.

    Read source only. Never execute the selected indicator, evaluate a user name,
    import a user-selected module or present third-party internals as our code.
    """
    visited = {function.__name__}
    sources: dict[str, str] = {}

    def visit(current: FunctionType) -> None:
        source = textwrap.dedent(inspect.getsource(current))
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            value = current.__globals__.get(name)
            if (
                name not in visited
                and inspect.isfunction(value)
                and value.__module__ == current.__module__
            ):
                visited.add(name)
                sources[name] = textwrap.dedent(inspect.getsource(value))
                visit(value)

    visit(function)
    return [{"name": name, "source": sources[name]} for name in sorted(sources)]


def indicator_reference(name: str | None = None) -> dict[str, Any]:
    """List public functions, or return source and statically named local helpers."""
    functions = _functions()
    if name is not None and name not in functions:
        raise ValueError(
            f"Unknown indicator {name!r}; request section='indicators' without "
            "indicator to discover the installed public functions."
        )
    detail = None
    if name is not None:
        function = functions[name]
        annotation = inspect.signature(function).return_annotation
        result = (
            function.__globals__.get(annotation)
            if isinstance(annotation, str)
            else annotation
        )
        result_fields = (
            [item.name for item in fields(result)]
            if inspect.isclass(result) and is_dataclass(result)
            else []
        )
        detail = {
            **_summary(name, function),
            "documentation": inspect.getdoc(function) or "",
            "source": textwrap.dedent(inspect.getsource(function)),
            "supporting_source": _supporting_source(function),
            "result_fields": result_fields,
        }
    return {
        "namespace": indicators.__name__,
        "module_contract": inspect.getdoc(technical) or "",
        "catalog": [_summary(key, value) for key, value in functions.items()],
        "selected": detail,
    }
