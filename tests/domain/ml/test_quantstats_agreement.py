"""Our return metrics agree with the QuantStats calls the backtest path makes.

``domain/backtest/metrics.py`` asks QuantStats for the same statistic names that
``performance.return_metrics`` computes by hand for an ML book. Two
implementations of one definition is a drift risk, so the agreement is pinned on
fixed inputs here.

If this fails after a QuantStats upgrade, the two have diverged: decide which is
authoritative and change both. Do not paste the new QuantStats output into the
expectations below, or the test stops meaning anything.

``beta`` and ``alpha`` are out of scope. QuantStats 0.0.86 exposes no
``stats.beta`` or ``stats.alpha``, and ``stats.greeks`` raises on a
timezone-aware index, so there is nothing to compare against.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from quantstats import stats as qs

from tradingdev.domain.ml.evaluation.performance import return_metrics

_PERIODS = 252.0
# Annualisation bases an hourly crypto book actually uses.
_PERIODS_CASES = [252.0, 8760.0, 4380.0]


def _series(values: np.ndarray) -> pd.Series:
    """QuantStats needs a DatetimeIndex; the metrics themselves do not.

    Both implementations reduce positionally, so handing the same object to
    each keeps the comparison free of any index-alignment question.
    """
    index = pd.date_range("2024-01-01", periods=values.size, freq="B", tz="UTC")
    return pd.Series(values, index=index)


def _scalar(value: float | pd.Series) -> float:
    """QuantStats' union annotation also covers frame inputs we never pass."""
    assert not isinstance(value, pd.Series), "expected a scalar statistic"
    return float(value)


@pytest.fixture(scope="module")
def book() -> pd.Series:
    """A deterministic book with one real drawdown in the middle."""
    rng = np.random.default_rng(7)
    values = rng.normal(0.0006, 0.011, 500)
    values[120:135] -= 0.012
    return _series(values)


@pytest.fixture(scope="module")
def market() -> pd.Series:
    return _series(np.random.default_rng(11).normal(0.0003, 0.009, 500))


class TestAgreement:
    @pytest.mark.parametrize("periods", _PERIODS_CASES)
    def test_sharpe_matches_quantstats(
        self, book: pd.Series, market: pd.Series, periods: float
    ) -> None:
        ours = return_metrics(book, market, periods_per_year=periods)

        theirs = _scalar(qs.sharpe(book, rf=0.0, periods=int(periods)))

        assert ours.sharpe == pytest.approx(theirs, rel=1e-12)

    def test_max_drawdown_matches_quantstats(
        self, book: pd.Series, market: pd.Series
    ) -> None:
        ours = return_metrics(book, market, periods_per_year=_PERIODS)

        # QuantStats reports drawdown as a negative number.
        theirs = abs(_scalar(qs.max_drawdown(book)))

        assert ours.max_drawdown == pytest.approx(theirs, rel=1e-12)

    def test_annual_return_matches_quantstats_cagr(
        self, book: pd.Series, market: pd.Series
    ) -> None:
        ours = return_metrics(book, market, periods_per_year=_PERIODS)

        theirs = _scalar(qs.cagr(book, periods=int(_PERIODS)))

        assert ours.annual_return == pytest.approx(theirs, rel=1e-12)

    def test_a_rising_book_has_no_drawdown_in_either(self) -> None:
        rising = _series(np.abs(np.random.default_rng(3).normal(0.001, 0.004, 500)))

        ours = return_metrics(rising, rising, periods_per_year=_PERIODS)

        assert ours.max_drawdown == pytest.approx(0.0)
        assert abs(_scalar(qs.max_drawdown(rising))) == pytest.approx(0.0)


class TestAssumptions:
    def test_the_risk_free_rate_is_assumed_zero(
        self, book: pd.Series, market: pd.Series
    ) -> None:
        """The equality above is only meaningful at rf=0; pin the assumption."""
        ours = return_metrics(book, market, periods_per_year=_PERIODS)

        at_zero = _scalar(qs.sharpe(book, rf=0.0, periods=252))
        at_five_percent = _scalar(qs.sharpe(book, rf=0.05, periods=252))

        assert ours.sharpe == pytest.approx(at_zero, rel=1e-12)
        assert ours.sharpe != pytest.approx(at_five_percent, rel=1e-6)

    def test_a_zero_variance_book_is_unavailable_rather_than_nan(self) -> None:
        """We report ``None`` where QuantStats reports ``nan``.

        A stored metric must never be ``nan``, so the two representations have
        to stay distinguishable rather than silently agreeing.
        """
        flat = _series(np.zeros(500))

        ours = return_metrics(flat, flat, periods_per_year=_PERIODS)

        assert ours.sharpe is None
        # QuantStats divides by a zero standard deviation and lets the
        # resulting ``nan`` escape; that is why it cannot agree here.
        with pytest.warns(RuntimeWarning, match="invalid value"):
            theirs = _scalar(qs.sharpe(flat, rf=0.0, periods=252))
        assert math.isnan(theirs)
