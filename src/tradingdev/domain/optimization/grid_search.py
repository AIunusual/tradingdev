"""Generic grid-search primitives used by strategies and workers."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tradingdev.domain.performance.catalog import METRIC_CATALOG, MetricScalar

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GridSearchResult[T = object]:
    """One evaluated grid-search result."""

    params: dict[str, T]
    metric_value: float | None
    metrics: dict[str, MetricScalar]


def parameter_grid[T](param_ranges: Mapping[str, Sequence[T]]) -> list[dict[str, T]]:
    """Expand named parameter ranges into dictionaries."""
    names = list(param_ranges.keys())
    values = [list(param_ranges[name]) for name in names]
    return [
        dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)
    ]


def tuple_grid[T](*ranges: Iterable[T]) -> list[tuple[T, ...]]:
    """Expand positional parameter ranges into tuples."""
    return list(itertools.product(*ranges))


type OptimizationDirection = Literal["maximize", "minimize"]


def metric_direction(metric_id: str) -> OptimizationDirection:
    """Resolve an eligible objective from the catalog for a new search."""
    definition = METRIC_CATALOG.get(metric_id)
    if definition is None or definition.optimization_direction is None:
        raise ValueError(f"Metric '{metric_id}' is not an optimization objective")
    return definition.optimization_direction


def finite_metric_value(value: object) -> float | None:
    """Keep unavailable and nonfinite observations out of objective selection."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def is_better_metric(
    value: float | None,
    incumbent: float | None,
    direction: OptimizationDirection,
) -> bool:
    """Compare finite values while preserving first-candidate ties."""
    if value is None or not math.isfinite(value):
        return False
    if incumbent is None or not math.isfinite(incumbent):
        return True
    return value < incumbent if direction == "minimize" else value > incumbent


def best_result[T](
    results: Iterable[GridSearchResult[T]],
    *,
    direction: OptimizationDirection,
) -> GridSearchResult[T]:
    """Select the best finite objective, failing if no trial can be ranked."""
    selected: GridSearchResult[T] | None = None
    for candidate in results:
        value = finite_metric_value(candidate.metric_value)
        if is_better_metric(
            value, selected.metric_value if selected else None, direction
        ):
            selected = candidate
    if selected is None:
        raise ValueError("No parameter combination has a finite objective value")
    return selected
