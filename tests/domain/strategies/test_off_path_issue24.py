"""With the rolling switch off, issue #24 must not be reproducible.

Each test takes one reproduction from issue #24, runs it through the classic
path with ``rolling_gen.enabled`` false, and asserts the recorded symptom is
absent.  The premise is asserted rather than assumed: if a later change flipped
the default, these scenarios would move onto the new path and this file could
stay green while checking something else.

Expected values are arithmetic on the fixture (bar offset plus a per-label
rule), never copied from a run.  The issue's two hard numbers are reproduced
here as assertions: a missing bar must keep the documented neutral ``0.5``, and
no prediction may move to a different bar.
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
"""History length used by the replay and mixed-representation tests."""

_NEUTRAL_P_SAFE = 0.5
"""The documented fallback when a bar's risk features are missing."""


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


class _RecordingFeatureEngineer(FeatureEngineer):
    """Record each scoring context instead of computing real indicators."""

    def __init__(self) -> None:
        super().__init__(lookback=24, target_horizon=1)
        self.contexts: list[pd.DataFrame] = []

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        if not include_target:
            self.contexts.append(df.copy())
        return df.copy()


class _AlwaysLong(XGBoostDirectionModel):
    """Deterministic predictor, so a count of predictions is a count of bars."""

    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)


def _fitted_xgboost(
    history: pd.DataFrame,
    recorder: FeatureEngineer,
) -> XGBoostStrategy:
    """A strategy that looks fitted but never retrains during these tests.

    A lookback this large keeps every retraining window below the classic
    ``lookback + 10`` threshold, so the stub predictor stays in charge and the
    assertions describe contexts rather than model quality.
    """
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            model=XGBoostModelConfig(n_estimators=2, n_jobs=1),
            lookback_candidates=[24],
            retrain_interval=10_000,
            target_horizon=1,
            min_bars_between_trades=1,
            rolling_gen={"enabled": False},
        )
    )
    assert strategy._config.rolling_gen.enabled is False
    strategy._model = _AlwaysLong()
    strategy._feature_engineer = recorder
    strategy._train_data = history.copy()
    strategy._best_lookback = 10_000
    return strategy


def test_full_replay_keeps_the_context_chronological() -> None:
    """Issue #24: "49 rows with inverted time" must not come back.

    Scoring the frame that ``fit()`` already saw used to concatenate the frame
    with itself, putting the tail of one copy in front of the head of the next.
    """
    frame = _market_frame(200)
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(frame, recorder)

    result = strategy.generate_signals(frame)

    assert len(recorder.contexts) == len(frame)
    assert _context_complaints(recorder.contexts) == []
    for bar, context in enumerate(recorder.contexts):
        assert pd.Timestamp(context["timestamp"].iloc[-1]) == pd.Timestamp(
            frame["timestamp"].iloc[bar]
        ), f"bar {bar}: the context ends on a different bar"
    assert set(result["signal"].unique()) <= {-1, 0, 1}


def test_partial_replay_drops_only_the_overlapping_history() -> None:
    """History that genuinely precedes the scored frame must survive."""
    frame = _market_frame(200)
    history = frame.iloc[:150].copy()
    scored = frame.iloc[100:].copy()
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(history, recorder)

    strategy.generate_signals(scored)

    assert _context_complaints(recorder.contexts) == []
    # 100 of the 150 history bars precede the first scored bar; the other 50
    # overlap it and are dropped, then the scored bar itself is appended.
    assert len(recorder.contexts[0]) == 100 + 1
    assert pd.Timestamp(recorder.contexts[0]["timestamp"].iloc[-1]) == pd.Timestamp(
        scored["timestamp"].iloc[0]
    )


def test_the_chronology_check_flags_a_replayed_context() -> None:
    """The check above is only worth its runtime if it can fail."""
    frame = _market_frame(10)
    replayed = pd.concat([frame.iloc[5:], frame.iloc[:3]], ignore_index=True)

    complaints = _context_complaints([replayed])

    assert complaints, "a frame concatenated with itself must be reported"


@pytest.mark.parametrize("scored_time", ["column", "index"], ids=["column", "index"])
def test_mixed_time_representations_keep_every_prediction(scored_time: str) -> None:
    """Issue #24: "10 valid predictions became 0" when the frames disagreed.

    A training frame carrying a ``timestamp`` column and a scored frame carrying
    a ``DatetimeIndex`` used to lose their shared time identity at the seam.
    """
    frame = _market_frame(200)
    history = frame.iloc[:_FIT_BARS].copy()
    scored = frame.iloc[_FIT_BARS:].copy()
    if scored_time == "index":
        scored = scored.set_index("timestamp")
    recorder = _RecordingFeatureEngineer()
    strategy = _fitted_xgboost(history, recorder)

    result = strategy.generate_signals(scored)

    assert len(recorder.contexts) == len(scored) == 50
    assert int((result["signal"] != 0).sum()) == len(scored)
    assert _context_complaints(recorder.contexts) == []


def test_the_off_path_is_the_classic_retrainer() -> None:
    """Switching the feature off must run the original loop, not a lookalike.

    ``RollingRetrainer`` is byte-identical to the base branch, so matching its
    output proves the disabled path is the code that was already reviewed and
    tested, rather than a re-implementation that merely looks similar.
    """
    frame = _market_frame(200)
    history = frame.iloc[:150].copy()
    scored = frame.iloc[150:].copy()
    strategy = _fitted_xgboost(history, _RecordingFeatureEngineer())

    result = strategy.generate_signals(scored)

    lookback = strategy._best_lookback
    model = strategy._model
    fe = strategy._feature_engineer
    assert lookback is not None
    assert model is not None
    assert fe is not None
    classic = RollingRetrainer(
        model_config=strategy._config.model,
        retrain_interval=strategy._config.retrain_interval,
        threshold=strategy._best_threshold,
        cooldown=strategy._config.min_bars_between_trades,
        lookback=lookback,
    ).run(scored, history, model, fe)
    assert result["signal"].tolist() == classic.tolist()


class _LabelEchoModel(XGBoostDirectionModel):
    """Report P(safe) from the bar's own row label; never return the neutral 0.5.

    Encoding the label into the value makes both issue #24 symptoms visible at
    once: a shifted prediction reads the wrong label, and a reused prediction
    reports a neighbour's value instead of the neutral fallback.
    """

    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())

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
    """Issue #24: a missing bar shifted later scores and reused the previous one.

    The expected series is arithmetic: bar ``i`` of the scored frame carries the
    concatenated label ``150 + i``, so its score is ``0.10 + label % 8 / 100``,
    except at the hole, which must stay at the documented neutral value.
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
            rolling_gen={"enabled": False},
        )
    )
    assert strategy._config.rolling_gen.enabled is False
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
    assert _NEUTRAL_P_SAFE not in expected[np.arange(len(expected)) != hole]
