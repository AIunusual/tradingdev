"""Feature consumers retain source bars through warm-up and interior gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.ml.threshold_optimizer import ThresholdOptimizer


def _gapped_frame() -> pd.DataFrame:
    close = 100.0 + np.arange(180) % 7
    close[70:110] = 100.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=180, freq="h"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(180, 1000.0),
        }
    )


class _PredictionRecorder(XGBoostDirectionModel):
    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self.frames.append(df.copy(deep=True))
        return pd.DataFrame({1: 1.0}, index=df.index)


class _InputRecorder(BaseBacktestEngine):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[pd.DataFrame] = []

    def run(self, df: pd.DataFrame) -> BacktestResult:
        self.frames.append(df.copy(deep=True))
        return BacktestResult(
            metrics={"total_pnl": 0.0, "total_trades": 1},
            equity_curve=np.zeros(len(df), dtype=np.float64),
        )


@pytest.mark.parametrize("cooldown", [1, 5])
@pytest.mark.parametrize("candidate", [0.0, 0.6])
def test_threshold_search_maps_features_to_all_source_bars(
    cooldown: int, candidate: float
) -> None:
    frame = _gapped_frame()
    frame.index = pd.Index(np.arange(len(frame)) * 3 + 7)
    original = frame.copy(deep=True)
    model = _PredictionRecorder()
    engine = _InputRecorder()

    threshold = ThresholdOptimizer(engine, cooldown).search(
        frame, model, [candidate], 0.9, feature_engineer=FeatureEngineer(lookback=24)
    )

    assert threshold == candidate
    assert len(model.frames) == 1
    assert len(engine.frames) == 1
    scored = engine.frames[0]
    pd.testing.assert_frame_equal(scored[frame.columns], original)
    # Warm-up ends at 33; the flat region has zero Bollinger width at 89:110.
    ready = np.ones(len(frame), dtype=bool)
    ready[:33] = False
    ready[89:110] = False
    expected = np.zeros(len(frame), dtype=int)
    last_trade = -cooldown
    for position in np.flatnonzero(ready):
        if position - last_trade >= cooldown:
            expected[position] = 1
            last_trade = int(position)
    np.testing.assert_array_equal(scored["signal"], expected)
    pd.testing.assert_frame_equal(frame, original)


def test_threshold_search_without_complete_features_keeps_default() -> None:
    model = _PredictionRecorder()
    engine = _InputRecorder()
    threshold = ThresholdOptimizer(engine).search(
        _gapped_frame().iloc[:20],
        model,
        [0.6],
        0.9,
        feature_engineer=FeatureEngineer(lookback=24),
    )
    assert threshold == 0.9
    assert model.frames == []
    assert engine.frames == []


@pytest.mark.parametrize("position,expected", [(20, 0), (60, 1), (100, 0), (115, 1)])
def test_rolling_prediction_requires_the_current_source_row(
    position: int, expected: int
) -> None:
    frame = _gapped_frame()
    original = frame.copy(deep=True)
    model = _PredictionRecorder()
    retrainer = RollingRetrainer(XGBoostModelConfig(), 1000, 0.6, 0, 24)

    signal = retrainer._predict_bar(frame, position, FeatureEngineer(24), model)

    assert signal == expected
    assert len(model.frames) == expected
    if expected:
        pd.testing.assert_frame_equal(
            model.frames[0][frame.columns], frame.iloc[[position]]
        )
    pd.testing.assert_frame_equal(frame, original)
