"""Regression tests for the train-replay context (issue #24 follow-up 1).

``walk_forward`` calls ``fit(train_df)`` and then ``generate_signals(train_df)``
to report training metrics. ``generate_signals`` prepends ``self._train_df`` to
the frame it scores, so that call concatenated the training window with itself:
the 60-bar prediction context spanned both copies, the timestamp mask kept the
duplicate rows, and the state machine started writing at bar 0 while reading
from bar 940.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    import pandas as pd
    import pytest
    import xgboost as xgb


class _RecordingModel:
    """Stand-in classifier that records how many rows it was asked to score."""

    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.calls.append(len(features))
        probabilities = np.zeros((len(features), 4), dtype=np.float64)
        probabilities[:, 0] = 0.6
        probabilities[:, 2] = 0.1
        probabilities[:, 3] = 0.3
        return probabilities


def _fitted_strategy(
    frame: pd.DataFrame,
    calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
    *,
    retrain_interval: int,
) -> quantile.QuantileStrategy:
    """A strategy that already ran ``fit(frame)``."""
    config = QuantileStrategyConfig(
        horizon=3,
        retrain_interval=retrain_interval,
        min_entry_edge=0.0015,
    )
    feature_eng = QuantileFeatureEngineer(
        horizon=3,
        profit_target=config.profit_target,
    )
    feature_eng.transform(frame, include_target=True, target_type="regime")
    feature_names = [
        name
        for name in feature_eng.get_feature_names()
        if name not in ("target_regime", "target_long", "target_short", "target")
    ]

    model = _RecordingModel(calls)
    monkeypatch.setattr(
        quantile, "_train_regime_classifier", lambda *args, **kwargs: model
    )

    strategy = quantile.QuantileStrategy(config=config)
    strategy._train_df = frame.copy()  # noqa: SLF001
    strategy._feature_eng = feature_eng  # noqa: SLF001
    strategy._feature_names = feature_names  # noqa: SLF001
    strategy._best_horizon = 3  # noqa: SLF001
    strategy._regime_model = cast("xgb.XGBClassifier", model)  # noqa: SLF001
    return strategy


def test_replaying_the_training_frame_scores_each_bar_once(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring the training frame must not score the duplicated context too."""
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:320]
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame, calls, monkeypatch, retrain_interval=retrain_interval
    )

    signals = strategy.generate_signals(frame)

    # Independent expectation: how many bars of this frame have features at all.
    expected = len(
        QuantileFeatureEngineer(
            horizon=3,
            profit_target=strategy._config.profit_target,  # noqa: SLF001
        ).transform(frame, include_target=False)
    )
    assert sum(calls) == expected, (
        "each bar must be scored exactly once; the difference is duplicated context"
    )
    assert max(calls) <= retrain_interval, (
        "a chunk must never be handed more rows than it covers"
    )
    assert len(signals) == len(frame)


def test_scoring_the_training_frame_keeps_the_output_aligned(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first chunk must start at bar 0, not at the end of the frame.

    The duplicated copy made the state machine consume bars 260..319 before
    writing output row 0, so a warm-up bar could inherit a late signal.
    """
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:320]
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame, calls, monkeypatch, retrain_interval=retrain_interval
    )

    signals = strategy.generate_signals(frame)

    # Bar 0 has no usable history, so the context transform drops it and the
    # chunk only covers the bars that survived. The chunk therefore reports
    # strictly fewer rows than a full interval, never more.
    assert calls[0] < retrain_interval
    assert signals["signal"].iloc[0] == 0.0
