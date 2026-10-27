"""Public SafetyVolume scoring preserves prefixes with optional ML direction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _DirectionProbe(XGBoostDirectionModel):
    """A deterministic predictor that consumes the actual engineered features."""

    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())
        self.frames: list[pd.DataFrame] = []

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        self.frames.append(df.copy(deep=True))
        return pd.Series(np.sign(df["log_return"]), index=df.index)


@pytest.mark.parametrize("use_ml_direction", [False, True], ids=["sma", "ml"])
def test_public_signals_are_invariant_to_future_rows(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    use_ml_direction: bool,
) -> None:
    config = SafetyVolumeStrategyConfig(
        risk_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
        direction_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
        lookback_candidates=[24],
        use_ml_direction=use_ml_direction,
        target_holding_bars=3,
        min_holding_bars=1,
        max_holding_bars=7,
        sma_fast=3,
        sma_slow=10,
    )
    strategy = SafetyVolumeStrategy(config)
    history = large_ohlcv_df.iloc[:400].copy()
    history_original = history.copy(deep=True)
    scored = large_ohlcv_df.iloc[400:580].copy()
    scored.index = pd.Index(np.arange(len(scored)) * 3 + 7)
    # A later flat interval removes feature rows without changing earlier prices.
    gap_index = scored.index[110:150]
    scored.loc[gap_index, ["open", "close"]] = 100.0
    scored.loc[gap_index, "high"] = 101.0
    scored.loc[gap_index, "low"] = 99.0
    scored.loc[gap_index, "volume"] = 1000.0
    original = scored.copy(deep=True)
    strategy.fit(history)
    assert history["timestamp"].iloc[-1] < scored["timestamp"].iloc[0]
    predictor = _DirectionProbe()
    strategy._dir_model = predictor
    risk_calls: list[int] = []

    def allow_entries(
        _strategy: SafetyVolumeStrategy, frame: pd.DataFrame
    ) -> NDArray[np.float64]:
        risk_calls.append(len(frame))
        return np.ones(len(frame), dtype=np.float64)

    monkeypatch.setattr(SafetyVolumeStrategy, "_predict_risk_rolling", allow_entries)
    full = strategy.generate_signals(scored)
    prefix = strategy.generate_signals(scored.iloc[:90])
    changed_future = scored.copy(deep=True)
    for column in ("open", "high", "low", "close", "volume"):
        changed_future.loc[changed_future.index[90:], column] *= np.linspace(
            1.5, 3.0, len(scored) - 90
        )
    perturbed = strategy.generate_signals(changed_future)

    assert risk_calls == [180, 90, 180]
    assert len(predictor.frames) == (3 if use_ml_direction else 0)
    assert prefix["signal"].ne(0).any(), "the causal comparison must exercise entries"
    pd.testing.assert_series_equal(full["signal"].iloc[:90], prefix["signal"])
    pd.testing.assert_series_equal(
        full["signal"].iloc[:90], perturbed["signal"].iloc[:90]
    )
    pd.testing.assert_frame_equal(full[scored.columns], original)
    pd.testing.assert_frame_equal(scored, original)
    pd.testing.assert_frame_equal(history, history_original)
    if use_ml_direction:
        assert strategy._dir_fe is not None
        assert all(not frame.empty for frame in predictor.frames)
        assert all("log_return" in frame.columns for frame in predictor.frames)
