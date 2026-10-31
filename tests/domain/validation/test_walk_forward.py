"""Tests for the walk-forward validation framework."""

from typing import Any

import pandas as pd
import pytest

from tradingdev.domain.backtest.signal_engine import (
    SignalBacktestEngine,
)
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
    ExplicitWalkForwardConfig,
)
from tradingdev.domain.validation.report import (
    format_walk_forward_report,
    summarize_results,
)
from tradingdev.domain.validation.walk_forward import (
    WalkForwardValidator,
)


def _explicit_config(df: pd.DataFrame) -> ExplicitWalkForwardConfig:
    """Split ``df`` at its midpoint, leaving the boundary days disjoint.

    Boundaries are inclusive UTC days, so the test window starts the day
    after the train window ends instead of sharing the midpoint's day.
    """
    ts = df["timestamp"]
    boundary = ts.iloc[len(ts) // 2]
    return ExplicitWalkForwardConfig(
        mode="explicit",
        train_start=ts.iloc[0].to_pydatetime(),
        train_end=boundary.to_pydatetime(),
        test_start=(boundary + pd.Timedelta(days=1)).to_pydatetime(),
        test_end=ts.iloc[-1].to_pydatetime(),
    )


class _TracingKDStrategy(KDStrategy):
    """Count ``generate_signals`` calls and report the count as its trace."""

    def __init__(self) -> None:
        super().__init__(config=KDStrategyConfig())
        self.calls = 0
        self.trace: dict[str, Any] = {"call": 0, "nested": {"call": 0}}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        self.trace["call"] = self.calls
        self.trace["nested"]["call"] = self.calls
        return super().generate_signals(df)

    def rolling_trace(self) -> dict[str, Any] | None:
        return self.trace


class TestWalkForwardValidator:
    def setup_method(self) -> None:
        self.engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)

    def test_each_fold_frame_keeps_its_own_rolling_trace(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """The trace describes the frame just scored, not the fold as a whole.

        Both frames call ``generate_signals``, so a single stored value would
        attribute the training replay's schedule to the evaluation frame.
        """
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = _TracingKDStrategy()
        validator = WalkForwardValidator(config=config, engine=self.engine)

        results = validator.validate(strategy, walk_forward_ohlcv_df)

        train_meta = results[0].train_backtest
        test_meta = results[0].test_backtest
        assert train_meta is not None
        assert test_meta is not None
        assert train_meta.metric_metadata["rolling_trace"] == {
            "call": 1,
            "nested": {"call": 1},
        }
        assert test_meta.metric_metadata["rolling_trace"] == {
            "call": 2,
            "nested": {"call": 2},
        }

    def test_single_explicit_split(self, walk_forward_ohlcv_df: pd.DataFrame) -> None:
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert len(results) == 1

    def test_result_contains_metrics(self, walk_forward_ohlcv_df: pd.DataFrame) -> None:
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        r = results[0]
        assert "total_return" in r.train_metrics
        assert "total_return" in r.test_metrics
        assert "sharpe_ratio" in r.test_metrics

    def test_strategy_params_recorded(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert "k_period" in results[0].strategy_params

    @pytest.mark.parametrize("expanding", [False, True])
    def test_auto_split_reaches_last_observation(
        self, walk_forward_ohlcv_df: pd.DataFrame, expanding: bool
    ) -> None:
        config = AutomaticWalkForwardConfig(
            mode="automatic", n_splits=2, train_ratio=0.7, expanding=expanding
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert len(results) == 2
        assert results[-1].test_end == walk_forward_ohlcv_df.timestamp.iloc[-1]
        assert results[-1].test_backtest is not None
        assert results[-1].test_backtest.timestamps is not None
        assert (
            pd.Timestamp(results[-1].test_backtest.timestamps[-1])
            == walk_forward_ohlcv_df.timestamp.iloc[-1]
        )

    def test_summary_aggregation(self, walk_forward_ohlcv_df: pd.DataFrame) -> None:
        config = AutomaticWalkForwardConfig(
            mode="automatic", n_splits=2, train_ratio=0.7, expanding=False
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        summary = summarize_results(results)
        assert summary["n_folds"] == 2
        assert "total_return" in summary
        assert "mean" in summary["total_return"]

    def test_result_contains_full_backtest(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        r = results[0]
        assert r.train_backtest is not None
        assert r.test_backtest is not None
        assert r.train_backtest.metrics == r.train_metrics
        assert r.test_backtest.metrics == r.test_metrics
        assert len(r.train_backtest.equity_curve) > 0
        assert len(r.test_backtest.equity_curve) > 0

    def test_format_report(self, walk_forward_ohlcv_df: pd.DataFrame) -> None:
        config = _explicit_config(walk_forward_ohlcv_df)
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(config=config, engine=self.engine)
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        report = format_walk_forward_report(results)
        assert "Walk-Forward" in report
        assert "Fold 0" in report
