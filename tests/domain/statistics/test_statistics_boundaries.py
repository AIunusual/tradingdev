"""Input and defensive boundaries the ported suites leave uncovered.

Expected values here are derived from the declared contracts (a Series index
is required, confidence is a real number strictly between 0 and 1, a capture
ratio with a zero denominator is undefined), not read from either module.

The zero-mean capture case is a defensive branch: ``compare_to_benchmark``
masks strictly positive and strictly negative periods before calling
``_capture``, so a zero mean cannot arise from that path without underflow.
It is exercised directly so the guard itself stays verified.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.statistics.benchmark import _capture
from tradingdev.domain.statistics.comparison import _tail, align_returns


def _daily(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"),
    )


class TestAlignReturnsBoundaries:
    @pytest.mark.parametrize(
        "bad",
        [
            [0.01, 0.02],
            np.array([0.01, 0.02]),
            "returns",
            None,
            1.5,
        ],
        ids=["list", "ndarray", "str", "None", "float"],
    )
    def test_rejects_non_series_values(self, bad: object) -> None:
        payload = cast("dict[str, pd.Series]", {"a": bad})

        with pytest.raises(ValueError, match="must be a pandas Series"):
            align_returns(payload)

    def test_rejects_a_second_non_series_after_a_valid_one(self) -> None:
        payload = cast(
            "dict[str, pd.Series]", {"a": _daily([0.01, 0.02]), "b": [0.01, 0.02]}
        )

        with pytest.raises(ValueError, match="returns 'b' must be a pandas Series"):
            align_returns(payload)


class TestTailBoundaries:
    @pytest.mark.parametrize(
        "confidence",
        [True, False, "0.95", None, 0.95 + 0j, object()],
        ids=["True", "False", "str", "None", "complex", "object"],
    )
    def test_rejects_non_real_or_boolean(self, confidence: object) -> None:
        with pytest.raises(ValueError, match="confidence must be strictly"):
            _tail(cast("Any", confidence))

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, math.nan, math.inf])
    def test_rejects_out_of_range_real(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence must be strictly"):
            _tail(confidence)

    def test_midpoint_of_the_two_tails(self) -> None:
        # A 90% interval leaves 5% in each tail.
        assert _tail(0.9) == pytest.approx(0.05)


class TestCaptureBoundaries:
    def test_empty_periods_are_undefined(self) -> None:
        assert _capture(np.array([]), np.array([])) is None

    def test_zero_mean_benchmark_is_undefined(self) -> None:
        strategy = np.array([0.01, 0.02, 0.03])
        benchmark = np.array([1.0, -1.0, 0.0])

        assert float(benchmark.mean()) == 0.0
        assert _capture(strategy, benchmark) is None

    def test_capture_ratio_uses_means_of_each_side(self) -> None:
        strategy = np.array([0.02, 0.04])
        benchmark = np.array([0.01, 0.03])

        # mean(strategy) / mean(benchmark) == 0.03 / 0.02
        assert _capture(strategy, benchmark) == pytest.approx(1.5)
