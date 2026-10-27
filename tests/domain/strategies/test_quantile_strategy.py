"""Regression tests for the untrained quantile fallback (issue #24 C).

``generate_signals`` used to train a regime classifier on the whole frame it
was about to score, so early signals came from a model that had already seen
later bars. The strategy now refuses the call instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def _untrained_strategy() -> quantile.QuantileStrategy:
    """A strategy that never completed ``fit`` (no engine, so it no-ops)."""
    return quantile.QuantileStrategy(
        config=QuantileStrategyConfig(horizon=3, min_entry_edge=0.0015)
    )


def test_generate_signals_refuses_an_untrained_strategy(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    strategy = _untrained_strategy()

    with pytest.raises(RuntimeError, match="no trained regime model"):
        strategy.generate_signals(sample_ohlcv_df)


def test_untrained_call_never_trains_on_the_frame_it_evaluates(
    sample_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak was training on the evaluation frame; assert that never happens.

    The call is captured rather than wrapped in ``pytest.raises`` so the
    training assertion is evaluated first: a regression that trains and then
    fails for an unrelated reason still reports the real problem.
    """
    trained: list[pd.DataFrame] = []

    class _RecordingTrainer:
        """Stands in for the classifier so the old flow could still finish."""

        def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.zeros((len(features), 4), dtype=np.float64)
            probabilities[:, 0] = 1.0
            return probabilities

    def record(feat_df: pd.DataFrame, *args: object, **kwargs: object) -> object:
        trained.append(feat_df)
        return _RecordingTrainer()

    monkeypatch.setattr(quantile, "_train_regime_classifier", record)
    strategy = _untrained_strategy()

    error: RuntimeError | None = None
    try:
        strategy.generate_signals(sample_ohlcv_df)
    except RuntimeError as exc:
        error = exc

    assert trained == [], "the evaluation frame must not be used for training"
    assert error is not None, "an untrained strategy must refuse the call"
    assert "no trained regime model" in str(error)


def test_untrained_call_installs_no_hidden_state(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    """A refused call must not stash a model for the next call to reuse."""
    strategy = _untrained_strategy()

    with pytest.raises(RuntimeError, match="no trained regime model"):
        strategy.generate_signals(sample_ohlcv_df)

    assert strategy._regime_model is None  # noqa: SLF001
    assert strategy._feature_eng is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="no trained regime model"):
        strategy.generate_signals(sample_ohlcv_df)


def test_fit_without_an_engine_still_counts_as_untrained(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    """``fit`` skips training without an engine, so the call must still refuse."""
    strategy = _untrained_strategy()

    strategy.fit(sample_ohlcv_df)

    assert strategy._regime_model is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="call fit"):
        strategy.generate_signals(sample_ohlcv_df)


@pytest.mark.parametrize("missing_state", ["model", "features", "history", None])
def test_model_feature_engineer_and_history_are_required(
    large_ohlcv_df: pd.DataFrame,
    missing_state: str | None,
) -> None:
    """Only the complete fitted state can produce full-length signals."""
    feature_eng = QuantileFeatureEngineer(horizon=3)
    feature_eng.transform(large_ohlcv_df, include_target=True, target_type="regime")
    feature_names = [
        name
        for name in feature_eng.get_feature_names()
        if name not in ("target_regime", "target_long", "target_short", "target")
    ]

    class _FixedProbabilities:
        """Stand-in classifier: long-only regime dominates every bar."""

        def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.zeros((len(features), 4), dtype=np.float64)
            probabilities[:, 0] = 0.6
            probabilities[:, 2] = 0.1
            probabilities[:, 3] = 0.3
            return probabilities

    strategy = quantile.QuantileStrategy(
        config=QuantileStrategyConfig(horizon=3, min_entry_edge=0.0015, train_window=1)
    )
    strategy._feature_eng = None if missing_state == "features" else feature_eng
    strategy._feature_names = feature_names  # noqa: SLF001
    strategy._best_horizon = 3  # noqa: SLF001
    strategy._regime_model = None if missing_state == "model" else _FixedProbabilities()
    strategy._train_df = None if missing_state == "history" else large_ohlcv_df.copy()
    if missing_state is not None:
        with pytest.raises(RuntimeError, match="call fit"):
            strategy.generate_signals(large_ohlcv_df)
        return

    signals = strategy.generate_signals(large_ohlcv_df)

    assert "signal" in signals.columns
    assert "size_weight" in signals.columns
    assert len(signals) == len(large_ohlcv_df)
    assert signals.index.equals(large_ohlcv_df.index)
    assert set(signals["signal"].unique()).issubset({-1.0, 0.0, 1.0})
