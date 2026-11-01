"""Threshold search scores candidates under the direction mode ``fit`` declares.

Issue #94: the search read whatever direction model an earlier fit had left on
the strategy.  A second fit therefore compared candidates under ML direction
while the first had used the SMA crossover, and the risk gate's validation
matrix carried a second copy of every engineered column into the direction
pipeline, which XGBoost rejects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

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
    _MLDirection,
)
from tradingdev.domain.validation.config import AutomaticWalkForwardConfig
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine

_CANDIDATES = [0.2, 0.5, 0.8]
_LOOKBACK = 24
_VALIDATION_RATIO = 0.2


class _ExposureEngine:
    """Score a candidate by the exposure its signals take.

    The defect lives in which direction model scored the candidates, so the
    engine only has to turn those directions into a comparable number.  Running
    the bar-by-bar engine would add minutes per fit without changing that.
    """

    def run(self, df: pd.DataFrame) -> SimpleNamespace:
        return SimpleNamespace(
            metrics={
                "total_pnl": float(np.sign(df["signal"]).sum()),
                "total_volume": float((df["signal"] != 0).sum()),
            },
            metric_metadata={},
        )


class _DirectionProbe:
    """Record what the direction pipeline was trained on and scored against."""

    def __init__(self) -> None:
        self.trained: list[pd.DataFrame] = []
        self.scored: list[pd.DataFrame] = []
        self.sma_calls = 0

    def reset(self) -> None:
        self.trained.clear()
        self.scored.clear()
        self.sma_calls = 0


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _DirectionProbe:
    recorded = _DirectionProbe()
    original_transform = FeatureEngineer.transform
    original_predict = XGBoostDirectionModel.predict
    original_sma = SafetyVolumeStrategy._compute_sma_directions

    def transform(
        engineer: FeatureEngineer,
        frame: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        if include_target:
            recorded.trained.append(frame.copy(deep=True))
        return original_transform(engineer, frame, include_target=include_target)

    def predict(model: XGBoostDirectionModel, frame: pd.DataFrame) -> pd.Series[float]:
        recorded.scored.append(frame.copy(deep=True))
        return original_predict(model, frame)

    def sma(strategy: SafetyVolumeStrategy, frame: pd.DataFrame) -> np.ndarray:
        recorded.sma_calls += 1
        return original_sma(strategy, frame)

    monkeypatch.setattr(FeatureEngineer, "transform", transform)
    monkeypatch.setattr(XGBoostDirectionModel, "predict", predict)
    monkeypatch.setattr(SafetyVolumeStrategy, "_compute_sma_directions", sma)
    return recorded


def _config(
    *,
    use_ml_direction: bool,
    candidates: list[float] | None = _CANDIDATES,
    max_acceptable_loss_pct: float = 0.0005,
    risk_threshold: float = 0.5,
) -> SafetyVolumeStrategyConfig:
    return SafetyVolumeStrategyConfig(
        risk_model=XGBoostModelConfig(n_estimators=3, max_depth=2, n_jobs=1),
        direction_model=XGBoostModelConfig(n_estimators=3, max_depth=2, n_jobs=1),
        lookback_candidates=[_LOOKBACK],
        risk_threshold=risk_threshold,
        risk_threshold_candidates=candidates,
        use_ml_direction=use_ml_direction,
        validation_ratio=_VALIDATION_RATIO,
        target_holding_bars=3,
        max_acceptable_loss_pct=max_acceptable_loss_pct,
        min_holding_bars=2,
        max_holding_bars=10,
        sma_fast=3,
        sma_slow=10,
        retrain_interval=1000,
    )


def _validation_start(df: pd.DataFrame) -> pd.Timestamp:
    split = int(len(df) * (1 - _VALIDATION_RATIO))
    return pd.Timestamp(df["timestamp"].iloc[split])


def test_repeated_fit_keeps_the_declared_direction_mode(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """A second fit must search under the same mode as the first, not crash."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True), cast("BaseBacktestEngine", _ExposureEngine())
    )

    strategy.fit(large_ohlcv_df)
    first = strategy.get_parameters()
    assert probe.scored, "the search must score directions with the ML model"
    assert probe.sma_calls == 0, "ML direction is declared, so SMA must not score"

    probe.reset()
    strategy.fit(large_ohlcv_df)
    second = strategy.get_parameters()

    assert probe.scored, "the second search must score directions itself"
    assert probe.sma_calls == 0
    assert second["best_threshold"] == first["best_threshold"]
    assert second["best_lookback"] == first["best_lookback"]


def test_threshold_search_trains_direction_before_the_validation_window(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """Only the live-signal refit may train on the rows the search scores."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True), cast("BaseBacktestEngine", _ExposureEngine())
    )

    strategy.fit(large_ohlcv_df)

    validation_start = _validation_start(large_ohlcv_df)
    full_end = large_ohlcv_df["timestamp"].iloc[-1]
    endings = [frame["timestamp"].iloc[-1] for frame in probe.trained]

    assert sum(end < validation_start for end in endings) == 1, (
        "the search model is trained on the training split alone"
    )
    assert endings.count(full_end) == 1, "live signals keep the full-frame refit"


def test_direction_inputs_stay_unique_for_a_risk_feature_frame(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    """The risk gate's matrix already carries ``log_return`` and its lags."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True, candidates=None),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )
    strategy.fit(large_ohlcv_df)

    risk_features = RiskFeatureEngineer(
        lookback=_LOOKBACK,
        target_holding_bars=3,
        fee_rate=0.0011,
        max_acceptable_loss_pct=0.0005,
    )
    frame = risk_features.transform(large_ohlcv_df, include_target=False)
    assert "log_return" in frame.columns, "the fixture must collide with a feature"
    ohlcv_only = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]

    actual = strategy._compute_ml_directions(frame)
    expected = strategy._compute_ml_directions(ohlcv_only)

    assert np.count_nonzero(actual) > 0, "the comparison must cover real predictions"
    np.testing.assert_array_equal(actual, expected)


def test_sma_mode_never_consults_a_direction_model(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """With ML direction off the search keeps the crossover and trains no model."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=False),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )

    strategy.fit(large_ohlcv_df)

    assert probe.trained == []
    assert probe.scored == []
    assert probe.sma_calls >= 1
    assert strategy.get_parameters()["best_threshold"] in _CANDIDATES


def test_without_candidates_only_the_signal_model_is_fitted(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """No search runs, so the direction model is trained once on the full frame."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True, candidates=None),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )

    strategy.fit(large_ohlcv_df)

    assert len(probe.trained) == 1
    trained_end = probe.trained[0]["timestamp"].iloc[-1]
    assert trained_end == large_ohlcv_df["timestamp"].iloc[-1]
    assert probe.scored == []


def test_unqualified_lookback_search_skips_the_search_direction_model(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """A loss ceiling nothing can breach leaves no model to search a threshold."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True, max_acceptable_loss_pct=1.0),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )

    strategy.fit(large_ohlcv_df)

    assert probe.scored == [], "no qualified risk model means no threshold search"
    assert len(probe.trained) == 1, "only the live-signal model is fitted"
    assert strategy.get_parameters()["best_threshold"] == 0.5


def test_two_walk_forward_folds_refit_the_same_strategy(
    large_ohlcv_df: pd.DataFrame,
    probe: _DirectionProbe,
) -> None:
    """Folds share one strategy object, so each fit must reset the search mode."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True), cast("BaseBacktestEngine", _ExposureEngine())
    )
    validator = WalkForwardValidator(
        AutomaticWalkForwardConfig(
            mode="automatic",
            n_splits=2,
            train_ratio=0.6,
        ),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )

    results = validator.validate(strategy, large_ohlcv_df)

    assert [result.fold_index for result in results] == [0, 1]
    assert results[0].train_end != results[1].train_end, "folds must differ"
    assert probe.sma_calls == 0
    for result in results:
        assert result.strategy_params["best_threshold"] in _CANDIDATES


@pytest.mark.parametrize("index_offset", [0, 1000], ids=["range", "offset"])
def test_scored_validation_bars_keep_their_directions(
    large_ohlcv_df: pd.DataFrame,
    index_offset: int,
) -> None:
    """The search must not spend the validation window on feature warm-up."""
    frame = large_ohlcv_df.copy()
    frame.index = frame.index + index_offset
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True, candidates=None),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )
    strategy.fit(frame)
    assert strategy._dir_fe is not None
    assert strategy._dir_model is not None
    source = _MLDirection(features=strategy._dir_fe, model=strategy._dir_model)

    split = int(len(frame) * (1 - _VALIDATION_RATIO))
    risk_features = RiskFeatureEngineer(
        lookback=_LOOKBACK,
        target_holding_bars=3,
        fee_rate=0.0011,
        max_acceptable_loss_pct=0.0005,
    )
    feat_val = risk_features.transform(frame.iloc[split:], include_target=False)

    sliced = strategy._compute_ml_directions(feat_val)
    with_history = strategy._validation_directions(feat_val, frame, source)

    assert np.count_nonzero(sliced) < len(sliced), "the slice loses its warm-up"
    assert np.count_nonzero(with_history) > np.count_nonzero(sliced)
    positions = frame.index.get_indexer(feat_val.index)
    np.testing.assert_array_equal(
        with_history,
        strategy._compute_ml_directions(frame)[positions],
    )


def test_fit_rejects_duplicate_index_labels_in_the_search(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    """Scored rows must map back to one bar each, or the search cannot align."""
    frame = pd.concat([large_ohlcv_df.iloc[:500], large_ohlcv_df.iloc[:500]])
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=False),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )

    with pytest.raises(ValueError, match="unique index labels"):
        strategy.fit(frame)


def test_a_fit_without_a_qualified_model_resets_the_threshold(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fold that cannot search must not trade on the previous fold's value."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=True, risk_threshold=0.99),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )
    strategy.fit(large_ohlcv_df)
    assert strategy.get_parameters()["best_threshold"] in _CANDIDATES

    monkeypatch.setattr(
        strategy,
        "_search_lookback",
        lambda train_df, val_df: (_LOOKBACK, None, None),
    )
    strategy.fit(large_ohlcv_df)

    assert strategy.get_parameters()["best_threshold"] == 0.99


def test_search_without_any_direction_keeps_the_configured_threshold(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates cannot be compared when no bar may take a position."""
    strategy = SafetyVolumeStrategy(
        _config(use_ml_direction=False, risk_threshold=0.99),
        cast("BaseBacktestEngine", _ExposureEngine()),
    )
    monkeypatch.setattr(
        strategy,
        "_compute_sma_directions",
        lambda frame: np.zeros(len(frame), dtype=np.float64),
    )

    strategy.fit(large_ohlcv_df)

    assert strategy.get_parameters()["best_threshold"] == 0.99
