"""Objective ordering and missing-metric regression tests."""

from __future__ import annotations

import pytest

from tradingdev.domain.optimization.grid_search import (
    GridSearchResult,
    best_result,
    finite_metric_value,
)


@pytest.mark.parametrize(
    ("direction", "expected"), [("maximize", 0.4), ("minimize", 0.1)]
)
def test_best_result_respects_direction_and_excludes_invalid_values(
    direction: str, expected: float
) -> None:
    results = [
        GridSearchResult({"candidate": index}, value, {"max_drawdown": value})
        for index, value in enumerate([None, float("inf"), 0.4, float("nan"), 0.1])
    ]
    if direction == "minimize":
        result = best_result(results, direction="minimize")
    else:
        result = best_result(results, direction="maximize")
    assert result.metric_value == expected
    assert result.metrics == {"max_drawdown": expected}


@pytest.mark.parametrize("values", [[], [None], [float("nan"), float("-inf")]])
def test_no_finite_candidate_is_an_explicit_failure(values: list[float | None]) -> None:
    with pytest.raises(ValueError, match="No parameter combination.*finite"):
        best_result(
            (GridSearchResult({}, value, {}) for value in values),
            direction="minimize",
        )


def test_equal_candidates_preserve_search_order() -> None:
    first = GridSearchResult({"window": 5}, 0.1, {})
    second = GridSearchResult({"window": 10}, 0.1, {})
    assert best_result([first, second], direction="minimize") is first


@pytest.mark.parametrize(
    "value", [None, True, "1.0", float("nan"), float("inf"), 10**400]
)
def test_unavailable_metric_is_not_coerced_to_zero(value: object) -> None:
    assert finite_metric_value(value) is None


def test_zero_objective_remains_eligible() -> None:
    zero = GridSearchResult({}, 0.0, {})
    assert best_result([zero], direction="maximize") is zero
