"""With the rolling switch on, the new engine must not reintroduce issue #24.

The off-path file next to this one pins the classic loop.  This file pins the
engine on the same reproductions: a frame scored after ``fit()`` saw it, frames
that disagree about how time is represented, and a bar whose features are
missing.  The expected series is the same arithmetic the off-path test uses, so
the two paths can be compared against one hand-computed answer instead of
against each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)

_FIT_BARS = 150
_NEUTRAL_P_SAFE = 0.5


def _market_frame(rows: int) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    close = 1000.0 + positions + 0.5 * np.sin(positions)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + positions,
        }
    )


def _context_complaints(contexts: list[pd.DataFrame]) -> list[str]:
    """Name every context whose bars are duplicated or out of order."""
    complaints: list[str] = []
    for position, context in enumerate(contexts):
        stamps = pd.DatetimeIndex(context["timestamp"])
        if not stamps.is_unique:
            complaints.append(f"context {position}: a bar appears twice")
        if not stamps.is_monotonic_increasing:
            complaints.append(f"context {position}: time runs backwards")
    return complaints


def _stamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Time identity of a frame, whichever way the caller carried it."""
    if "timestamp" in frame.columns:
        return pd.DatetimeIndex(frame["timestamp"])
    return pd.DatetimeIndex(frame.index)


class _RecordingFeatureEngineer(FeatureEngineer):
    """Record scoring contexts and hand the engine a labelled training frame.

    The engine only trains when a window holds enough labelled rows, so this
    stub supplies an alternating target instead of running the real indicators.
    """

    def __init__(self) -> None:
        super().__init__(lookback=24, target_horizon=1)
        self.contexts: list[pd.DataFrame] = []
        self.windows: list[pd.DataFrame] = []

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        result = df.copy()
        # One numeric column, so a real model has something to fit on.
        result["stub_feature"] = np.arange(len(result), dtype=float)
        if include_target:
            result["target"] = np.where(np.arange(len(result)) % 2 == 0, 1, -1)
            self.windows.append(result)
            return result
        self.contexts.append(result)
        return result


class _AlwaysLong(XGBoostDirectionModel):
    """Deterministic starting model; retrained models replace it."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)


def _fitted_xgboost(
    history: pd.DataFrame,
    recorder: FeatureEngineer,
    *,
    scheme: str,
) -> XGBoostStrategy:
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
            lookback_candidates=[24],
            retrain_interval=50,
            target_horizon=1,
            min_bars_between_trades=1,
            rolling_gen={"enabled": True, "scheme": scheme},
        )
    )
    assert strategy._config.rolling_gen.enabled is True
    strategy._model = _AlwaysLong()
    strategy._feature_engineer = recorder
    strategy._train_data = history.copy()
    strategy._best_lookback = 24
    return strategy


def test_full_replay_keeps_the_context_chronological() -> None:
    """Issue #24: scoring the fitted frame must not replay it into itself."""
    frame = _market_frame(200)
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(frame, recorder, scheme="roll_sd")

    result = strategy.generate_signals(frame)

    # Bar 0 has no earlier history; the next scheduled fit is bar 50. No model
    # fitted on the full frame may score the preceding 50 replay bars.
    assert len(recorder.contexts) == len(frame) - 50
    assert (result["signal"].iloc[:50] == 0).all()
    trace = strategy.rolling_trace()
    assert trace is not None
    assert trace["initial_model_available"] is False
    assert trace["unscored_bars"] == list(range(50))
    assert _context_complaints(recorder.contexts) == []
    for bar, context in enumerate(recorder.contexts, start=50):
        assert pd.Timestamp(_stamps(context)[-1]) == pd.Timestamp(
            _stamps(frame)[bar]
        ), f"bar {bar}: the context ends on a different bar"
    assert set(result["signal"].unique()) <= {-1, 0, 1}


def test_partial_replay_keeps_only_the_earlier_history() -> None:
    """History that genuinely precedes the scored frame must survive."""
    frame = _market_frame(200)
    history = frame.iloc[:150].copy()
    scored = frame.iloc[100:].copy()
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(history, recorder, scheme="roll_ex")

    strategy.generate_signals(scored)

    assert _context_complaints(recorder.contexts) == []
    # The expanding training window starts at the first retained history bar and
    # stops at the bar being scored; the 50 overlapping history bars are gone.
    window = recorder.windows[0]
    assert pd.Timestamp(_stamps(window)[0]) == pd.Timestamp(_stamps(history)[0])
    assert pd.Timestamp(_stamps(window)[-1]) < pd.Timestamp(_stamps(scored)[0])
    assert len(window) == 100
    assert pd.Timestamp(_stamps(recorder.contexts[0])[-1]) == pd.Timestamp(
        _stamps(scored)[0]
    )


@pytest.mark.parametrize("scored_time", ["column", "index"], ids=["column", "index"])
def test_mixed_time_representations_keep_every_prediction(scored_time: str) -> None:
    """Issue #24: "10 valid predictions became 0" when the frames disagreed."""
    frame = _market_frame(200)
    history = frame.iloc[:_FIT_BARS].copy()
    scored = frame.iloc[_FIT_BARS:].copy()
    if scored_time == "index":
        scored = scored.set_index("timestamp")
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(history, recorder, scheme="roll_ex")

    strategy.generate_signals(scored)

    assert len(recorder.contexts) == len(scored) == 50
    assert _context_complaints(recorder.contexts) == []
    for bar, context in enumerate(recorder.contexts):
        assert pd.Timestamp(_stamps(context)[-1]) == pd.Timestamp(_stamps(scored)[bar])


class _LabelEchoModel(XGBoostDirectionModel):
    """Report P(safe) from the bar's own row label; never return the neutral 0.5."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        p_safe = 0.10 + (int(df.index[0]) % 8) / 100.0
        return pd.DataFrame({-1: 1.0 - p_safe, 1: p_safe}, index=df.index)


class _DroppingRiskFeatureEngineer(RiskFeatureEngineer):
    """Drop one bar from every scoring context, as a missing feature would."""

    def __init__(self, drop_label: int) -> None:
        super().__init__(lookback=24, target_holding_bars=3)
        self._drop_label = drop_label

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        if include_target:
            return df.copy()
        return df.drop(index=self._drop_label, errors="ignore").copy()


def test_risk_gate_neither_shifts_nor_reuses_a_score() -> None:
    """Issue #24: the engine must leave a hole at the missing bar, not fill it.

    The expected series is the same arithmetic as the off-path test: bar ``i``
    carries the concatenated label ``150 + i``, so its score is
    ``0.10 + label % 8 / 100``, except at the hole, which stays neutral.
    """
    frame = _market_frame(200)
    history = frame.iloc[:_FIT_BARS].copy()
    scored = frame.iloc[_FIT_BARS:].copy()
    hole = 10
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            risk_model=XGBoostModelConfig(n_estimators=2, n_jobs=1),
            lookback_candidates=[24],
            retrain_interval=10_000,
            target_holding_bars=3,
            min_holding_bars=1,
            max_holding_bars=7,
            rolling_gen={"enabled": True, "scheme": "roll_ex"},
        )
    )
    assert strategy._config.rolling_gen.enabled is True
    strategy._risk_model = _LabelEchoModel()
    strategy._risk_fe = _DroppingRiskFeatureEngineer(_FIT_BARS + hole)
    strategy._best_lookback = 10_000
    strategy._train_data = history

    scores = strategy._predict_risk_rolling(scored)

    expected = np.array(
        [0.10 + ((_FIT_BARS + bar) % 8) / 100.0 for bar in range(len(scored))]
    )
    expected[hole] = _NEUTRAL_P_SAFE
    np.testing.assert_allclose(scores, expected)
