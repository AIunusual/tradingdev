"""Tests for buy-and-hold benchmark returns and the benchmark regression."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import quantstats as qs

from tradingdev.domain.statistics.benchmark import (
    buy_and_hold_returns,
    compare_to_benchmark,
)
from tradingdev.domain.statistics.periods import period_returns
from tradingdev.domain.statistics.sharpe import sharpe_ratio

_DAILY = 365.0


def _daily(values: npt.ArrayLike, start: str = "2024-01-01") -> pd.Series:
    array = np.asarray(values, dtype=np.float64)
    index = pd.date_range(start, periods=len(array), freq="D", tz="UTC")
    return pd.Series(array, index=index)


def _benchmark(n: int = 250, seed: int = 7) -> pd.Series:
    return _daily(np.random.default_rng(seed).normal(0.0005, 0.02, n))


def _strategy_on(benchmark: pd.Series, seed: int = 11) -> pd.Series:
    """Half exposure plus a small edge and independent noise."""
    noise = np.random.default_rng(seed).normal(0.0, 0.005, len(benchmark))
    return 0.5 * benchmark + 0.0004 + noise


def _hourly_prices(
    days: int = 10, seed: int = 3, volatility: float = 0.004
) -> tuple[npt.NDArray[np.float64], pd.DatetimeIndex]:
    timestamps = pd.date_range("2024-03-01", periods=days * 24, freq="h")
    steps = np.random.default_rng(seed).normal(0.0, volatility, len(timestamps))
    return 100.0 * np.exp(np.cumsum(steps)), timestamps


class TestBuyAndHoldReturns:
    def test_compounding_reproduces_the_price_ratio(self) -> None:
        prices, timestamps = _hourly_prices()

        returns = buy_and_hold_returns(prices, timestamps=timestamps, frequency="daily")

        assert float(np.prod(1.0 + returns.to_numpy())) == pytest.approx(
            prices[-1] / prices[0], rel=1e-12
        )

    def test_shares_the_utc_periods_of_strategy_returns(self) -> None:
        prices, timestamps = _hourly_prices()
        equity = 10_000.0 + np.linspace(0.0, 50.0, len(timestamps))

        benchmark = buy_and_hold_returns(
            prices, timestamps=timestamps, frequency="daily"
        )
        strategy = period_returns(equity, timestamps, "daily")

        assert isinstance(benchmark.index, pd.DatetimeIndex)
        assert benchmark.index.equals(strategy.index)
        assert str(benchmark.index.tz) == "UTC"
        assert len(benchmark) == 10

    def test_converts_aware_timestamps_to_utc(self) -> None:
        prices = np.array([100.0, 110.0, 99.0])
        timestamps = pd.date_range(
            "2024-01-01 08:00", periods=3, freq="D", tz="Asia/Taipei"
        )

        returns = buy_and_hold_returns(prices, timestamps=timestamps, frequency="daily")

        assert list(returns.index) == list(
            pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        )
        assert returns.to_numpy() == pytest.approx([0.0, 0.1, -0.1])

    def test_accepts_aware_timestamps_with_different_offsets(self) -> None:
        # Local Berlin times either side of the 2024-03-31 DST change.
        timestamps = ["2024-03-30T12:00:00+01:00", "2024-03-31T12:00:00+02:00"]

        returns = buy_and_hold_returns(
            [100.0, 110.0], timestamps=timestamps, frequency="daily"
        )

        assert list(returns.index) == list(
            pd.date_range("2024-03-30", periods=2, freq="D", tz="UTC")
        )
        assert returns.to_numpy() == pytest.approx([0.0, 0.1])

    @pytest.mark.parametrize(
        ("prices", "timestamps", "match"),
        [
            ([100.0, 101.0], ["2024-01-01"], "does not match"),
            ([100.0, math.nan], ["2024-01-01", "2024-01-02"], "finite"),
            ([100.0, math.inf], ["2024-01-01", "2024-01-02"], "finite"),
            ([100.0, 101.0], ["2024-01-02", "2024-01-01"], "strictly increasing"),
            ([100.0, 101.0], ["2024-01-01", "2024-01-01"], "strictly increasing"),
            ([100.0, 0.0], ["2024-01-01", "2024-01-02"], "strictly positive"),
            ([], [], "non-empty"),
        ],
    )
    def test_rejects_invalid_inputs(
        self, prices: list[float], timestamps: list[str], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            buy_and_hold_returns(
                prices, timestamps=pd.DatetimeIndex(timestamps), frequency="daily"
            )


class TestRegression:
    def test_strategy_equal_to_benchmark(self) -> None:
        benchmark = _benchmark()

        result = compare_to_benchmark(
            benchmark.copy(), benchmark, periods_per_year=_DAILY
        )

        assert result.n_periods == 250
        assert result.beta == pytest.approx(1.0, abs=1e-12)
        assert result.alpha_annualised == pytest.approx(0.0, abs=1e-12)
        assert result.active_return_annualised == 0.0
        assert result.tracking_error_annualised == 0.0
        assert result.information_ratio is None
        assert result.up_capture == pytest.approx(1.0)
        assert result.down_capture == pytest.approx(1.0)

    @pytest.mark.parametrize("k", [2.0, 0.5, 0.0, -1.0])
    def test_fixed_multiple_of_the_benchmark_has_that_beta(self, k: float) -> None:
        benchmark = _benchmark()

        result = compare_to_benchmark(k * benchmark, benchmark, periods_per_year=_DAILY)

        assert result.beta == pytest.approx(k, abs=1e-12)
        assert result.alpha_annualised == pytest.approx(0.0, abs=1e-12)

    def test_formulas_on_known_data(self) -> None:
        benchmark = _benchmark()
        strategy = _strategy_on(benchmark)
        s, b = strategy.to_numpy(), benchmark.to_numpy()

        result = compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)

        slope, intercept = np.polyfit(b, s, 1)
        assert result.beta == pytest.approx(slope, rel=1e-10)
        assert result.alpha_annualised == pytest.approx(intercept * _DAILY, rel=1e-8)
        assert result.active_return_annualised == pytest.approx(
            (s.mean() - b.mean()) * _DAILY
        )
        assert result.tracking_error_annualised == pytest.approx(
            np.std(s - b, ddof=1) * math.sqrt(_DAILY)
        )
        assert result.information_ratio == pytest.approx(
            sharpe_ratio(s - b, periods_per_year=_DAILY)
        )

    def test_beta_and_alpha_agree_with_quantstats(self) -> None:
        benchmark = _benchmark()
        strategy = _strategy_on(benchmark)

        result = compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)
        # QuantStats strips the benchmark's timezone, so compare naive series.
        # greeks has no annotations, unlike the QuantStats functions in src.
        greeks = qs.stats.greeks(  # type: ignore[no-untyped-call]
            strategy.tz_localize(None),
            benchmark.tz_localize(None),
            periods=_DAILY,
            prepare_returns=False,
        )

        assert result.beta is not None
        assert result.alpha_annualised is not None
        assert result.beta == pytest.approx(float(greeks["beta"]), abs=1e-9)
        # QuantStats annualises the intercept arithmetically, as we do.
        assert result.alpha_annualised == pytest.approx(
            float(greeks["alpha"]), abs=1e-9
        )

    def test_uses_only_the_common_periods(self) -> None:
        benchmark = _benchmark(n=30)
        strategy = _strategy_on(benchmark).iloc[5:]
        shifted = benchmark.iloc[:25]

        result = compare_to_benchmark(strategy, shifted, periods_per_year=_DAILY)
        expected = compare_to_benchmark(
            strategy.iloc[:20], shifted.iloc[5:], periods_per_year=_DAILY
        )

        assert result.n_periods == 20
        assert result == expected

    def test_single_period_leaves_dispersion_figures_undefined(self) -> None:
        result = compare_to_benchmark(
            _daily([0.02]), _daily([0.01]), periods_per_year=_DAILY
        )

        assert result.n_periods == 1
        assert result.beta is None
        assert result.alpha_annualised is None
        assert result.tracking_error_annualised is None
        assert result.information_ratio is None
        assert result.active_return_annualised == pytest.approx(0.01 * _DAILY)

    def test_constant_benchmark_has_no_beta(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, -0.02, 0.03]),
            _daily([0.001, 0.001, 0.001]),
            periods_per_year=_DAILY,
        )

        assert result.beta is None
        assert result.alpha_annualised is None
        assert result.tracking_error_annualised is not None


class TestRoundingNoise:
    def test_constant_edge_has_zero_tracking_error(self) -> None:
        benchmark = _benchmark()

        result = compare_to_benchmark(
            benchmark + 0.0004, benchmark, periods_per_year=_DAILY
        )

        assert result.tracking_error_annualised == 0.0
        assert result.information_ratio is None
        assert result.active_return_annualised == pytest.approx(0.0004 * _DAILY)

    def test_small_real_tracking_error_is_kept(self) -> None:
        benchmark = _benchmark()
        noise = np.random.default_rng(5).normal(0.0, 1e-8, len(benchmark))
        strategy = benchmark + noise

        result = compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)

        assert result.tracking_error_annualised == pytest.approx(
            np.std(noise, ddof=1) * math.sqrt(_DAILY), rel=1e-6
        )
        assert result.information_ratio is not None

    def test_benchmark_varying_by_one_ulp_has_no_beta(self) -> None:
        benchmark = _daily([0.001, np.nextafter(0.001, 1.0), 0.001])

        result = compare_to_benchmark(
            _daily([0.01, -0.02, 0.03]), benchmark, periods_per_year=_DAILY
        )

        assert result.beta is None
        assert result.alpha_annualised is None

    def test_underflowing_benchmark_variance_has_no_beta(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, 0.02]), _daily([1e-170, 2e-170]), periods_per_year=_DAILY
        )

        assert result.beta is None


class TestCapture:
    def test_up_and_down_capture_on_known_values(self) -> None:
        strategy = _daily([0.01, 0.03, -0.01, 0.005, 0.02])
        benchmark = _daily([0.02, 0.04, -0.02, -0.03, 0.0])

        result = compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)

        assert result.up_capture == pytest.approx(0.02 / 0.03)
        assert result.down_capture == pytest.approx(-0.0025 / -0.025)
        # The zero-return period counts towards neither side.
        assert result.n_up_periods == 2
        assert result.n_down_periods == 2

    def test_rising_benchmark_has_no_down_capture(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, -0.01, 0.02]),
            _daily([0.01, 0.02, 0.03]),
            periods_per_year=_DAILY,
        )

        assert result.up_capture == pytest.approx(0.02 / 3 / 0.02)
        assert result.down_capture is None
        assert result.n_down_periods == 0

    def test_falling_benchmark_has_no_up_capture(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, -0.01, 0.02]),
            _daily([-0.01, -0.02, -0.03]),
            periods_per_year=_DAILY,
        )

        assert result.up_capture is None
        assert result.down_capture == pytest.approx((0.02 / 3) / -0.02)
        assert result.n_up_periods == 0

    def test_flat_benchmark_has_no_capture(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, -0.01]), _daily([0.0, 0.0]), periods_per_year=_DAILY
        )

        assert result.up_capture is None
        assert result.down_capture is None


class TestValidation:
    def test_disjoint_periods_raise(self) -> None:
        with pytest.raises(ValueError, match="share no period"):
            compare_to_benchmark(
                _daily([0.01, 0.02], start="2024-01-01"),
                _daily([0.01, 0.02], start="2024-02-01"),
                periods_per_year=_DAILY,
            )

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("side", ["strategy", "benchmark"])
    def test_non_finite_returns_raise(self, bad: float, side: str) -> None:
        good = _daily([0.01, 0.02, 0.03])
        broken = _daily([0.01, bad, 0.03])
        strategy, benchmark = (broken, good) if side == "strategy" else (good, broken)

        with pytest.raises(ValueError, match="must be finite"):
            compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)

    def test_naive_and_utc_indexes_raise_a_timezone_mismatch(self) -> None:
        utc = _daily([0.01, 0.02])
        naive = pd.Series(
            utc.to_numpy(), index=pd.date_range("2024-01-01", periods=2, freq="D")
        )

        with pytest.raises(ValueError, match="timezone mismatch"):
            compare_to_benchmark(naive, utc, periods_per_year=_DAILY)

    def test_duplicate_labels_raise(self) -> None:
        returns = _daily([0.01, 0.02])
        duplicated = pd.Series([0.01, 0.02], index=[returns.index[0]] * 2)

        with pytest.raises(ValueError, match="duplicate index labels"):
            compare_to_benchmark(duplicated, returns, periods_per_year=_DAILY)

    @pytest.mark.parametrize(
        "periods_per_year", [0.0, -365.0, math.nan, math.inf, True, "365"]
    )
    def test_invalid_periods_per_year_raise(self, periods_per_year: object) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            compare_to_benchmark(
                _daily([0.01, 0.02]),
                _daily([0.01, 0.03]),
                periods_per_year=periods_per_year,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("days", "volatility"),
    # A quiet instrument has tiny returns but the same rounding noise.
    [(40, 0.004), (365, 2e-5)],
)
def test_fully_invested_strategy_matches_buy_and_hold(
    days: int, volatility: float
) -> None:
    prices, timestamps = _hourly_prices(days=days, volatility=volatility)
    equity = 10_000.0 * prices / prices[0]

    strategy = period_returns(equity, timestamps, "daily")
    benchmark = buy_and_hold_returns(prices, timestamps=timestamps, frequency="daily")
    result = compare_to_benchmark(strategy, benchmark, periods_per_year=_DAILY)

    assert result.n_periods == days
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.alpha_annualised == pytest.approx(0.0, abs=1e-9)
    # The returns differ from buy-and-hold only by rounding.
    assert result.tracking_error_annualised == 0.0
    assert result.information_ratio is None
