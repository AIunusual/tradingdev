"""Threshold rules: which columns a rule may read and how it takes its legs."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from tradingdev.domain.ml.rules import ThresholdRule


def _rule(direction: str = "below", threshold: float = 30.0) -> ThresholdRule:
    return ThresholdRule.create(
        column="rsi_14", threshold=threshold, direction=direction
    )


class TestConstruction:
    def test_a_rule_names_its_column_side_and_level(self) -> None:
        rule = _rule()
        assert rule.name == "rsi_14 < 30"
        assert rule.parameters == {"threshold": 30.0}

    def test_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _rule(direction="across")

    @pytest.mark.parametrize(
        "column", ["forward_return", "target", "timestamp", "label_end", "symbol"]
    )
    def test_the_label_and_row_bookkeeping_are_not_readable(self, column: str) -> None:
        with pytest.raises(ValueError, match="cannot be read"):
            ThresholdRule.create(column=column, threshold=0.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_level_is_refused(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            _rule(threshold=value)

    def test_a_non_numeric_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match="number"):
            ThresholdRule(column="rsi_14", threshold=cast("float", "30"))


class TestSignal:
    def test_above_takes_the_long_leg_over_the_level(self) -> None:
        rule = ThresholdRule.create(
            column="macd_hist", threshold=0.0, direction="above"
        )
        assert rule.signal([0.5, 0.0, -0.5]).tolist() == [1.0, -1.0, -1.0]

    def test_below_takes_the_long_leg_under_the_level(self) -> None:
        assert _rule().signal([20.0, 30.0, 40.0]).tolist() == [1.0, -1.0, -1.0]

    def test_a_level_outside_the_range_is_refused(self) -> None:
        """A rule that never changes side would be recorded with no positions."""
        with pytest.raises(ValueError, match="never changes"):
            _rule().signal([40.0, 50.0, 60.0])

    def test_a_non_finite_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _rule().signal([20.0, float("nan")])

    def test_an_empty_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _rule().signal([])

    def test_two_dimensions_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _rule().signal(np.array([[20.0, 40.0]]))
