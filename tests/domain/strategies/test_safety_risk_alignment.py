"""Risk probabilities must describe the source bar they are assigned to."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)


def _market_frame() -> pd.DataFrame:
    positions = np.arange(180, dtype=float)
    close = 1000.0 + positions + 0.5 * np.sin(positions)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=180, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + positions,
        },
        index=np.arange(180) * 7 + 3,
    )


class _SafeModel(XGBoostDirectionModel):
    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())
        self.scored_times: list[pd.Timestamp] = []

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self.scored_times.extend(pd.DatetimeIndex(df["timestamp"]))
        return pd.DataFrame({0: 0.0, 1: 1.0}, index=df.index)


@pytest.mark.parametrize("include_target", [False, True])
def test_risk_features_preserve_source_identity_after_dropping_rows(
    include_target: bool,
) -> None:
    frame = _market_frame()
    frame.loc[frame.index[170], "open"] = np.nan
    original = frame.copy(deep=True)

    features = RiskFeatureEngineer(80, 3).transform(
        frame, include_target=include_target
    )

    assert not features.empty
    assert frame.index[170] not in features.index
    pd.testing.assert_series_equal(
        features["timestamp"], frame.loc[features.index, "timestamp"]
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("time_format", ["column", "index", "mixed"])
def test_public_risk_signals_do_not_reuse_a_previous_bars_features(
    time_format: str,
) -> None:
    frame = _market_frame()
    history = frame.iloc[:160].copy()
    scored = frame.iloc[160:].copy()
    scored.loc[scored.index[10], "open"] = np.nan
    if time_format == "index":
        history = history.set_index("timestamp")
    if time_format in {"index", "mixed"}:
        scored = scored.set_index("timestamp")
    original_history = history.copy(deep=True)
    original_scored = scored.copy(deep=True)
    model = _SafeModel()
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            risk_threshold=0.8,
            sma_fast=3,
            sma_slow=5,
            min_holding_bars=1,
            max_holding_bars=100,
            lookback_candidates=[80],
            retrain_interval=1000,
        )
    )
    strategy._risk_model = model
    strategy._risk_fe = RiskFeatureEngineer(80, 1000)
    strategy._best_lookback = 80
    strategy._train_data = history

    result = strategy.generate_signals(scored)

    assert result["signal"].iloc[9:12].tolist() == [1, 0, 1]
    expected_times = frame["timestamp"].iloc[160:].drop(frame.index[170])
    assert model.scored_times == expected_times.tolist()
    pd.testing.assert_index_equal(result.index, scored.index)
    pd.testing.assert_frame_equal(result.drop(columns="signal"), original_scored)
    pd.testing.assert_frame_equal(history, original_history)
    pd.testing.assert_frame_equal(scored, original_scored)
