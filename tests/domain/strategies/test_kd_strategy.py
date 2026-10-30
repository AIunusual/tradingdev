"""Tests for the KD crossover strategy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

from tradingdev.domain.backtest.signal_engine import (
    SignalBacktestEngine,
)
from tradingdev.domain.strategies.bundled.kd_strategy.config import (
    KDFitConfig,
    KDStrategyConfig,
)
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class TestKDStrategy:
    def setup_method(self) -> None:
        self.config = KDStrategyConfig()
        self.strategy = KDStrategy(config=self.config)

    def test_signal_values_valid(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        unique_signals = set(result["signal"].unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_indicator_columns_present(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_get_parameters(self) -> None:
        params = self.strategy.get_parameters()
        assert params["k_period"] == 14
        assert params["overbought"] == 80.0
        assert params["oversold"] == 20.0

    @pytest.mark.parametrize(
        ("overbought", "oversold", "expected"),
        [
            (80.0, 20.0, [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1]),
            (90.0, 10.0, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1]),
        ],
        ids=["default-thresholds", "custom-thresholds"],
    )
    def test_custom_thresholds_change_signals(
        self, overbought: float, oversold: float, expected: list[int]
    ) -> None:
        # With this price range and unsmoothed one-bar K, K equals close and
        # D is the mean of two closes. Crossovers occur at K=15, 5, 85 and 95.
        frame = pd.DataFrame(
            {
                "high": [100.0] * 12,
                "low": [0.0] * 12,
                "close": [
                    15.0,
                    5.0,
                    15.0,
                    5.0,
                    3.0,
                    5.0,
                    85.0,
                    95.0,
                    85.0,
                    95.0,
                    97.0,
                    95.0,
                ],
            }
        )
        config = KDStrategyConfig(
            k_period=1,
            d_period=2,
            smooth_k=1,
            overbought=overbought,
            oversold=oversold,
        )
        strategy = KDStrategy(config=config)
        result = strategy.generate_signals(frame)
        assert result["signal"].tolist() == expected

    def test_no_look_ahead_bias(
        self,
        sample_ohlcv_with_kd: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        assert_no_look_ahead(
            self.strategy,
            sample_ohlcv_with_kd,
            check_points=[30, 75, 120, 199],
        )

    def test_signals_cover_long_short_and_flat(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        config = KDStrategyConfig(overbought=50.0, oversold=50.0)
        strategy = KDStrategy(config=config)

        signals = strategy.generate_signals(sample_ohlcv_with_kd)["signal"]

        assert not signals.isna().any()
        assert set(signals.unique()) == {-1, 0, 1}


class TestKDStrategyFit:
    @pytest.mark.parametrize("first", [0.5, None, float("nan"), float("inf")])
    def test_fit_minimizes_drawdown_and_excludes_missing_metrics(
        self,
        sample_ohlcv_with_kd: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        first: float | None,
    ) -> None:
        from tradingdev.domain.strategies.bundled.kd_strategy import strategy as module

        scores = iter([first, 0.1])
        engine = cast(
            "BaseBacktestEngine",
            SimpleNamespace(
                run=lambda frame: SimpleNamespace(
                    metrics={"max_drawdown": next(scores)}
                )
            ),
        )
        monkeypatch.setattr(module, "estimate_n_jobs", lambda *args, **kwargs: 1)
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=KDFitConfig(
                k_period_range=[9, 14],
                d_period_range=[3],
                smooth_k_range=[3],
                overbought_range=[80.0],
                oversold_range=[20.0],
                target_metric="max_drawdown",
            ),
            backtest_engine=engine,
        )
        strategy.fit(sample_ohlcv_with_kd)
        assert strategy.get_parameters()["k_period"] == 14

    def test_fit_all_missing_metrics_fails(
        self, sample_ohlcv_with_kd: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tradingdev.domain.strategies.bundled.kd_strategy import strategy as module

        engine = cast(
            "BaseBacktestEngine",
            SimpleNamespace(
                run=lambda frame: SimpleNamespace(metrics={"sharpe_ratio": None})
            ),
        )
        monkeypatch.setattr(module, "estimate_n_jobs", lambda *args, **kwargs: 1)
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=KDFitConfig(
                k_period_range=[9],
                d_period_range=[3],
                smooth_k_range=[3],
                overbought_range=[80.0],
                oversold_range=[20.0],
            ),
            backtest_engine=engine,
        )
        with pytest.raises(ValueError, match="No parameter combination.*finite"):
            strategy.fit(sample_ohlcv_with_kd)

    def test_fit_without_fit_config_is_noop(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        config = KDStrategyConfig()
        strategy = KDStrategy(config=config)
        original = strategy.get_parameters().copy()
        strategy.fit(sample_ohlcv_with_kd)
        assert strategy.get_parameters() == original

    def test_fit_updates_parameters(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
            overbought_range=[80.0],
            oversold_range=[20.0],
            target_metric="sharpe_ratio",
        )
        engine = SignalBacktestEngine(
            init_cash=10_000.0, fees=0.0, slippage=0.0, periods_per_year=365
        )
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
            backtest_engine=engine,
        )
        strategy.fit(sample_ohlcv_with_kd)
        params = strategy.get_parameters()
        assert params["k_period"] in [9, 14]

    def test_fit_requires_engine(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
        )
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
        )
        with pytest.raises(RuntimeError, match="backtest_engine"):
            strategy.fit(sample_ohlcv_with_kd)

    def test_fit_generates_valid_signals_after(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
            overbought_range=[80.0],
            oversold_range=[20.0],
        )
        engine = SignalBacktestEngine(
            init_cash=10_000.0, fees=0.0, slippage=0.0, periods_per_year=365
        )
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
            backtest_engine=engine,
        )
        strategy.fit(sample_ohlcv_with_kd)
        result = strategy.generate_signals(sample_ohlcv_with_kd)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})
