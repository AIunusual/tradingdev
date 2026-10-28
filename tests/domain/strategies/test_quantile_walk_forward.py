"""Quantile scoring through public training and real walk-forward consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)
from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
)
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tradingdev.domain.backtest.result import BacktestResult


class _RecordingQuantileStrategy(quantile.QuantileStrategy):
    """Observe the real strategy's input/output at the validator boundary."""

    def __init__(
        self, config: QuantileStrategyConfig, engine: VolumeBacktestEngine
    ) -> None:
        super().__init__(config=config, backtest_engine=engine)
        self.scored_frames: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        original = df.copy(deep=True)
        result = super().generate_signals(df)
        pd.testing.assert_frame_equal(df, original)
        self.scored_frames.append((original, result))
        return result


def _assert_aligned_scoring(
    original: pd.DataFrame, signals: pd.DataFrame, backtest: BacktestResult
) -> None:
    """Both strategy output and the engine must retain every input bar."""
    pd.testing.assert_frame_equal(signals[original.columns], original)
    assert len(signals) == len(original)
    assert len(backtest.equity_curve) == len(original)
    assert np.isfinite(backtest.equity_curve).all()
    assert set(signals["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert np.isfinite(signals["size_weight"]).all()
    assert (signals["size_weight"] > 0).all()


def test_real_model_walk_forward_keeps_train_and_test_rows(
    large_ohlcv_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay survives its first 200-bar slice with no mature target labels."""
    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)
    engine = VolumeBacktestEngine(position_size=50000, signal_as_position=True)
    strategy = _RecordingQuantileStrategy(
        QuantileStrategyConfig(
            horizon=200,
            retrain_interval=200,
            n_estimators=1,
            max_depth=1,
            dynamic_sizing=False,
            min_monthly_pnl=-1e9,
        ),
        engine,
    )
    train = large_ohlcv_df.iloc[:700].copy()
    test = large_ohlcv_df.iloc[700:].copy()
    train.index = pd.Index(np.arange(len(train)) * 3 + 7)
    test.index = pd.Index(np.arange(len(test)) * 5 + 11)
    original_train = train.copy(deep=True)
    original_test = test.copy(deep=True)

    validator = WalkForwardValidator(
        AutomaticWalkForwardConfig(
            mode="automatic",
        ),
        engine,
    )
    result = validator.run_fold(0, strategy, train, test)

    assert len(strategy.scored_frames) == 2
    assert result.train_backtest is not None
    assert result.test_backtest is not None
    _assert_aligned_scoring(*strategy.scored_frames[0], result.train_backtest)
    _assert_aligned_scoring(*strategy.scored_frames[1], result.test_backtest)
    assert (strategy.scored_frames[0][1]["signal"].iloc[:30] == 0).all()
    assert result.strategy_params["best_horizon"] == 200
    assert result.train_metrics == result.train_backtest.metrics
    assert result.test_metrics == result.test_backtest.metrics
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(test, original_test)


@pytest.mark.parametrize("index_history", [False, True])
def test_public_fit_normalizes_mixed_time_representations(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    index_history: bool,
) -> None:
    """Column/index and timezone changes must not change fitted OOS signals."""
    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)
    engine = VolumeBacktestEngine(position_size=50000, signal_as_position=True)
    config = QuantileStrategyConfig(
        horizon=3,
        retrain_interval=200,
        n_estimators=1,
        max_depth=1,
        dynamic_sizing=False,
        min_monthly_pnl=-1e9,
    )
    train = large_ohlcv_df.iloc[:600].copy()
    scoring = large_ohlcv_df.iloc[600:800].copy()
    baseline = quantile.QuantileStrategy(config=config, backtest_engine=engine)
    baseline.fit(train)
    expected = baseline.generate_signals(scoring)

    if index_history:
        train = train.set_index("timestamp")
        train.index = pd.DatetimeIndex(train.index).tz_convert("Asia/Taipei")
        scoring["timestamp"] = scoring["timestamp"].dt.tz_localize(None)
    else:
        scoring = scoring.set_index("timestamp")
        scoring.index = pd.DatetimeIndex(scoring.index).tz_convert("Asia/Taipei")
    original_train = train.copy(deep=True)
    original_scoring = scoring.copy(deep=True)
    strategy = quantile.QuantileStrategy(config=config, backtest_engine=engine)

    strategy.fit(train)
    actual = strategy.generate_signals(scoring)
    backtest = engine.run(actual)

    _assert_aligned_scoring(scoring, actual, backtest)
    pd.testing.assert_frame_equal(
        actual[["signal", "size_weight"]].reset_index(drop=True),
        expected[["signal", "size_weight"]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(scoring, original_scoring)


def test_parameter_search_scores_source_gaps_and_unlabelled_tail(
    large_ohlcv_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting parameters and replaying them must see the same flat gap."""

    class _RecordingEngine(VolumeBacktestEngine):
        def __init__(self) -> None:
            super().__init__(position_size=50000, signal_as_position=True)
            self.frames: list[pd.DataFrame] = []

        def run(self, df: pd.DataFrame) -> BacktestResult:
            self.frames.append(df.copy())
            return super().run(df)

    class _GapProbabilities:
        def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
            # Feature warm-up removes bars 0..29, and the flat candle removes 60.
            assert len(features) == 289
            probabilities = np.zeros((len(features), 4), dtype=np.float64)
            probabilities[:30, 0] = 1.0
            probabilities[30:-20, 3] = 1.0
            probabilities[-20:, 0] = 1.0
            return probabilities

    monkeypatch.setattr(quantile, "estimate_n_jobs", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        quantile,
        "_train_regime_classifier",
        lambda *args, **kwargs: _GapProbabilities(),
    )
    source = large_ohlcv_df.iloc[:320].copy()
    source.loc[60, ["open", "high", "low"]] = source.loc[60, "close"]
    source.index = pd.Index(np.arange(len(source)) * 3 + 7)
    original = source.copy(deep=True)
    engine = _RecordingEngine()
    strategy = quantile.QuantileStrategy(
        QuantileStrategyConfig(
            horizon=200,
            retrain_interval=500,
            train_window=1,
            min_holding_bars=1,
            strategy_sl=0.0,
            dynamic_sizing=False,
            min_monthly_pnl=-1e9,
        ),
        backtest_engine=engine,
    )

    strategy.fit(source)
    replay = strategy.generate_signals(source)

    assert len(engine.frames) == 1
    evaluated = engine.frames[0]
    assert len(evaluated) == len(source)
    assert (evaluated["signal"].iloc[30:60] == 1.0).all()
    assert (evaluated["signal"].iloc[60:300] == 0.0).all()
    assert (evaluated["signal"].iloc[300:] == 1.0).all()
    pd.testing.assert_frame_equal(
        evaluated[["signal", "size_weight"]],
        replay[["signal", "size_weight"]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(source, original)
