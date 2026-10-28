"""Scoring a window that overlaps the training frame must not replay it.

``RollingRetrainer`` and ``SafetyVolumeStrategy._predict_risk_rolling`` build a
prediction context by concatenating saved training rows with the frame being
scored.  When the scored frame is the training frame itself -- which is exactly
what ``fit(train)`` followed by ``generate_signals(train)`` does for training
metrics -- the naive concatenation puts the tail of one copy in front of the
head of the next.  The resulting context is not chronological, so the features
at the start of scoring are computed across an artificial jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.ml.scoring_history import align_scoring_history
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)


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


class _AlwaysLong(XGBoostDirectionModel):
    """Deterministic predictor: the assertions only care about the context."""

    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)


class _RecordingFeatureEngineer(FeatureEngineer):
    """Record the prediction contexts without computing real indicators."""

    def __init__(self) -> None:
        super().__init__(lookback=24, target_horizon=1)
        self.contexts: list[pd.DataFrame] = []

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        if not include_target:
            self.contexts.append(df.copy(deep=True))
        return df.copy()


class _RecordingRiskFeatureEngineer(RiskFeatureEngineer):
    def __init__(self) -> None:
        super().__init__(lookback=24, target_holding_bars=3)
        self.contexts: list[pd.DataFrame] = []

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        if not include_target:
            self.contexts.append(df.copy(deep=True))
        return df.copy()


def _assert_chronological(contexts: list[pd.DataFrame]) -> None:
    assert contexts, "the scored window must produce at least one prediction"
    for context in contexts:
        stamps = pd.DatetimeIndex(context["timestamp"])
        assert stamps.is_unique, (
            f"duplicated bar in context at {_first_break(stamps, unique=True)}"
        )
        assert stamps.is_monotonic_increasing, (
            f"context is not chronological at {_first_break(stamps, unique=False)}"
        )


def _first_break(stamps: pd.DatetimeIndex, *, unique: bool) -> str:
    """Describe the first offending row instead of dumping the whole index."""
    if unique:
        seen: dict[pd.Timestamp, int] = {}
        for position, stamp in enumerate(stamps):
            if stamp in seen:
                return f"row {position} repeats row {seen[stamp]}: {stamp}"
            seen[stamp] = position
        return "the final row"
    for position in range(1, len(stamps)):
        if stamps[position] < stamps[position - 1]:
            return f"row {position}: {stamps[position - 1]} -> {stamps[position]}"
    return "the final row"


def test_retrainer_replay_context_stays_chronological() -> None:
    """``run(train, train, ...)`` must not concatenate the frame with itself."""
    frame = _market_frame(60)
    recorder = _RecordingFeatureEngineer()
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=1000,
    )

    signals = retrainer.run(frame, frame, _AlwaysLong(), recorder)

    assert len(signals) == len(frame)
    _assert_chronological(recorder.contexts)


def test_retrainer_keeps_history_that_genuinely_precedes_scoring() -> None:
    """The overlap fix must not throw away earlier training rows."""
    frame = _market_frame(120)
    history = frame.iloc[:60].copy()
    scored = frame.iloc[60:].copy()
    recorder = _RecordingFeatureEngineer()
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=1000,
    )

    retrainer.run(scored, history, _AlwaysLong(), recorder)

    first = recorder.contexts[0]
    assert pd.Timestamp(first["timestamp"].iloc[-1]) == pd.Timestamp(
        scored["timestamp"].iloc[0]
    )
    assert len(first) == len(history) + 1
    _assert_chronological(recorder.contexts)


def test_retrainer_refuses_frames_without_a_time_identity() -> None:
    """The identity is required for every non-empty scoring call.

    It is not limited to frames that overlap: without timestamps nothing can
    show that the training rows precede the scored window, so the call refuses
    instead of assuming it.
    """
    scored = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    history = pd.DataFrame({"close": [99.0, 99.5]})
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=1000,
    )

    with pytest.raises(ValueError, match="timestamp"):
        retrainer.run(scored, history, _AlwaysLong(), _RecordingFeatureEngineer())


def test_retrainer_scores_an_empty_frame_without_validating_it() -> None:
    """An empty scored frame produces no signals, so there is nothing to order."""
    empty = pd.DataFrame({"close": pd.Series(dtype=float)})
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=1000,
    )

    signals = retrainer.run(empty, empty, _AlwaysLong(), _RecordingFeatureEngineer())

    assert signals.tolist() == []


def test_retrainer_refuses_duplicated_history_timestamps() -> None:
    """A history that repeats a bar cannot be ordered against the scored frame."""
    frame = _market_frame(6)
    # Non-decreasing, so only the uniqueness rule can reject it.
    repeated = pd.concat(
        [frame.iloc[:3], frame.iloc[[2]], frame.iloc[3:]], ignore_index=True
    )
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=1000,
    )

    with pytest.raises(ValueError, match="unique"):
        retrainer.run(frame, repeated, _AlwaysLong(), _RecordingFeatureEngineer())


def _safety_strategy(
    train: pd.DataFrame,
) -> tuple[SafetyVolumeStrategy, _RecordingRiskFeatureEngineer]:
    """Build a fitted-looking strategy whose retraining is switched off."""
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(
            risk_model=XGBoostModelConfig(n_estimators=2, max_depth=2, n_jobs=1),
            lookback_candidates=[24],
            target_holding_bars=3,
            min_holding_bars=1,
            max_holding_bars=7,
        )
    )
    recorder = _RecordingRiskFeatureEngineer()
    strategy._risk_fe = recorder
    strategy._risk_model = _AlwaysLong()
    strategy._train_data = train.copy()
    # A lookback this large keeps ``window`` below the retraining threshold, so
    # only the prediction context is exercised.
    strategy._best_lookback = 10_000
    return strategy, recorder


def test_safety_risk_replay_context_stays_chronological() -> None:
    """``fit(train)`` then ``generate_signals(train)`` must not replay itself."""
    frame = _market_frame(300)
    strategy, recorder = _safety_strategy(frame)

    scores = strategy._predict_risk_rolling(frame)

    assert len(scores) == len(frame)
    _assert_chronological(recorder.contexts)


def test_safety_risk_scoring_keeps_preceding_history() -> None:
    frame = _market_frame(400)
    history = frame.iloc[:300].copy()
    scored = frame.iloc[300:].copy()
    strategy, recorder = _safety_strategy(history)

    scores = strategy._predict_risk_rolling(scored)

    assert len(scores) == len(scored)
    first = recorder.contexts[0]
    assert pd.Timestamp(first["timestamp"].iloc[-1]) == pd.Timestamp(
        scored["timestamp"].iloc[0]
    )
    assert len(first) == len(history) + 1
    _assert_chronological(recorder.contexts)


@pytest.mark.parametrize("history_index", [False, True])
@pytest.mark.parametrize("scored_index", [False, True])
def test_mixed_time_representations_keep_real_features_and_signals(
    history_index: bool, scored_index: bool
) -> None:
    frame = _market_frame(170)
    history = frame.iloc[:160].copy()
    scored = frame.iloc[160:].copy()
    # Equivalent instants with different time zones must share one chronology.
    history["timestamp"] = history["timestamp"].dt.tz_convert("Asia/Taipei")
    if history_index:
        history = history.set_index("timestamp")
    if scored_index:
        scored = scored.set_index("timestamp")
    original_history = history.copy(deep=True)
    original_scored = scored.copy(deep=True)

    aligned, normalized = align_scoring_history(history, scored)
    np.testing.assert_array_equal(
        aligned["timestamp"].to_numpy(), frame["timestamp"].iloc[:160].to_numpy()
    )
    np.testing.assert_array_equal(
        normalized["timestamp"].to_numpy(), frame["timestamp"].iloc[160:].to_numpy()
    )
    retrainer = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=1000,
        threshold=0.6,
        cooldown=0,
        lookback=80,
    )
    # Real feature filtering; a long horizon leaves no mature retraining rows.
    signals = retrainer.run(scored, history, _AlwaysLong(), FeatureEngineer(80, 1000))
    np.testing.assert_array_equal(signals, np.ones(len(scored), dtype=int))
    pd.testing.assert_frame_equal(history, original_history)
    pd.testing.assert_frame_equal(scored, original_scored)


def test_partial_replay_excludes_overlapping_and_future_training_rows() -> None:
    frame = _market_frame(170)
    history = frame.set_index("timestamp")
    scored = frame.iloc[80:100].copy()

    aligned, normalized = align_scoring_history(history, scored)

    assert len(aligned) == 80
    assert aligned["timestamp"].max() < normalized["timestamp"].min()
    pd.testing.assert_frame_equal(normalized, scored)
