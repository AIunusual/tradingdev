"""Exact constructor leaves and search overlays shared by evidence and display."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.strategies.execution import merge_strategy_parameters

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest


def effective_candidates(
    manifest: ExecutionManifest,
) -> list[tuple[int, int, str, object]]:
    """Resolve each search-group candidate exactly as execution merges its values."""
    if manifest.optimization is None:
        return []
    kwargs = manifest.strategy_execution.constructor_kwargs
    bundled = manifest.strategy_execution.kind == "bundled"
    captured = kwargs.get("config") if bundled else kwargs
    if not isinstance(captured, dict):
        raise ValueError("Strategy does not expose searchable parameters")
    result: list[tuple[int, int, str, object]] = []
    for group, (name, candidates) in enumerate(
        manifest.optimization.param_ranges.items(), 1
    ):
        token = name.replace("~", "~0").replace("/", "~1")
        pointer = ("/config" if bundled else "") + "/" + token
        for candidate_index, candidate in enumerate(candidates, 1):
            effective = merge_strategy_parameters(captured, {name: candidate})
            result.append((group, candidate_index, pointer, effective[name]))
    return result


def parameter_leaves(value: object, pointer: str = "") -> dict[str, object]:
    """Preserve explicit zero, null and empty containers as effective leaves."""
    if isinstance(value, dict) and value:
        result: dict[str, object] = {}
        for key in sorted(value):
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.update(parameter_leaves(value[key], pointer + "/" + token))
        return result
    if isinstance(value, list) and value:
        result = {}
        for index, item in enumerate(value):
            result.update(parameter_leaves(item, f"{pointer}/{index}"))
        return result
    return {pointer: value} if pointer else {}
