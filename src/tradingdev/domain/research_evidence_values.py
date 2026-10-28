"""Bounded, immutable JSON containers used only by research evidence snapshots."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Never

from tradingdev.shared.utils.json_values import strict_json_value

if TYPE_CHECKING:
    from pydantic import JsonValue

MAX_RESEARCH_EVIDENCE_BYTES = 512 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _immutable(*args: object, **kwargs: object) -> Never:
    raise TypeError("Research evidence snapshots are immutable")


class FrozenDict[K, V](dict[K, V]):
    """Keep the JSON object shape while refusing normal mutation operations."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class FrozenList[T](list[T]):
    """Keep the JSON array shape while refusing normal mutation operations."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def bounded_json(value: object) -> JsonValue:
    """Reject cyclic/nonfinite JSON and cap depth, text and total node counts."""
    nodes = 0
    active: set[int] = set()

    def check(node: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 16 or nodes > 4096:
            raise ValueError("Research evidence JSON exceeds depth or node limits")
        if isinstance(node, str) and len(node) > 4000:
            raise ValueError("Research evidence JSON text exceeds 4000 characters")
        if isinstance(node, dict | list):
            if id(node) in active:
                raise ValueError("Research evidence JSON cannot contain cycles")
            active.add(id(node))
            try:
                if isinstance(node, dict):
                    for key, item in node.items():
                        if not isinstance(key, str) or len(key) > 512:
                            raise ValueError("Research evidence JSON key is invalid")
                        check(item, depth + 1)
                else:
                    for item in node:
                        check(item, depth + 1)
            finally:
                active.remove(id(node))

    check(value, 0)
    return strict_json_value(value)


def freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return FrozenDict((key, freeze_json(item)) for key, item in value.items())
    if isinstance(value, list):
        return FrozenList(freeze_json(item) for item in value)
    return value
