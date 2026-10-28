"""Training entry points only receive outcomes realised in their own windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.flaml_model import FLAMLAutoMLDirectionModel
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
    GLFTMLStrategy,
)
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


def _market_frame(rows: int, direction: int = 1) -> pd.DataFrame:
    positions = np.arange(rows)
    close = 1000.0 + direction * positions + 0.05 * np.sin(positions)
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


def _assert_mature_training_rows(
    trained: pd.DataFrame,
    source: pd.DataFrame,
    horizon: int,
    expected_label: int,
) -> None:
    assert not trained.empty
    last_realised = source["timestamp"].iloc[-horizon - 1]
    assert trained["timestamp"].max() == last_realised
    assert trained["timestamp"].le(last_realised).all()
    assert trained["target"].eq(expected_label).all()
    assert pd.api.types.is_integer_dtype(trained["target"])


@dataclass
class _TrainingCall:
    model: XGBoostDirectionModel
    frame: pd.DataFrame
    validation: pd.DataFrame | None


def _capture_xgboost_training(monkeypatch: pytest.MonkeyPatch) -> list[_TrainingCall]:
    calls: list[_TrainingCall] = []

    def train(
        model: XGBoostDirectionModel,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        calls.append(
            _TrainingCall(
                model,
                df.copy(deep=True),
                eval_df.copy(deep=True) if eval_df is not None else None,
            )
        )

    def predict(_model: XGBoostDirectionModel, df: pd.DataFrame) -> pd.Series[float]:
        return pd.Series(1.0, index=df.index)

    def predict_proba(_model: XGBoostDirectionModel, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)

    monkeypatch.setattr(XGBoostDirectionModel, "train", train)
    monkeypatch.setattr(XGBoostDirectionModel, "predict", predict)
    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict_proba)
    return calls


@pytest.mark.parametrize("direction", [-1, 1], ids=["falling", "rising"])
def test_xgboost_fit_matures_search_validation_and_final_refit_separately(
    monkeypatch: pytest.MonkeyPatch, direction: int
) -> None:
    frame = _market_frame(500, direction)
    original = frame.copy(deep=True)
    calls = _capture_xgboost_training(monkeypatch)
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            lookback_candidates=[12], target_horizon=5, validation_ratio=0.2
        )
    )

    strategy.fit(frame)

    assert len(calls) == 2
    search, final = calls
    assert search.validation is not None
    _assert_mature_training_rows(search.frame, frame.iloc[:400], 5, direction)
    _assert_mature_training_rows(search.validation, frame.iloc[400:], 5, direction)
    _assert_mature_training_rows(final.frame, frame, 5, direction)
    assert final.validation is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("direction", [-1, 1], ids=["falling", "rising"])
def test_rolling_retraining_uses_only_mature_outcomes_before_each_prediction(
    monkeypatch: pytest.MonkeyPatch, direction: int
) -> None:
    frame = _market_frame(510, direction)
    original = frame.copy(deep=True)
    train = frame.iloc[:500].copy()
    test = frame.iloc[500:].copy()
    calls = _capture_xgboost_training(monkeypatch)
    initial_model = XGBoostDirectionModel(XGBoostModelConfig())
    retrainer = RollingRetrainer(
        XGBoostModelConfig(), retrain_interval=4, threshold=0.5, cooldown=0, lookback=24
    )

    signals = retrainer.run(test, train, initial_model, FeatureEngineer(24, 5))

    assert len(signals) == len(test)
    assert len(calls) == 3
    for call, start in zip(calls, (0, 4, 8), strict=True):
        source = frame.iloc[start : 500 + start]
        _assert_mature_training_rows(call.frame, source, 5, direction)
        assert call.validation is None
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(train, original.iloc[:500])
    pd.testing.assert_frame_equal(test, original.iloc[500:])


def test_rolling_retraining_keeps_existing_model_without_mature_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _market_frame(85)
    original = frame.copy(deep=True)
    calls = _capture_xgboost_training(monkeypatch)
    initial_model = XGBoostDirectionModel(XGBoostModelConfig())
    prediction_models: list[XGBoostDirectionModel] = []

    def predict_proba(model: XGBoostDirectionModel, df: pd.DataFrame) -> pd.DataFrame:
        prediction_models.append(model)
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)

    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict_proba)
    retrainer = RollingRetrainer(
        XGBoostModelConfig(), retrain_interval=2, threshold=0.5, cooldown=0, lookback=24
    )

    signals = retrainer.run(
        frame.iloc[80:], frame.iloc[:80], initial_model, FeatureEngineer(24, 100)
    )

    assert calls == []
    assert len(prediction_models) == 5
    assert all(model is initial_model for model in prediction_models)
    np.testing.assert_array_equal(signals, np.ones(5, dtype=int))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("direction", [-1, 1], ids=["falling", "rising"])
def test_safety_volume_fit_matures_its_optional_direction_training(
    monkeypatch: pytest.MonkeyPatch, direction: int
) -> None:
    frame = _market_frame(500, direction)
    original = frame.copy(deep=True)
    calls = _capture_xgboost_training(monkeypatch)
    strategy = SafetyVolumeStrategy(
        SafetyVolumeStrategyConfig(use_ml_direction=True, lookback_candidates=[12])
    )

    strategy.fit(frame)

    direction_calls = [call for call in calls if call.model is strategy._dir_model]
    assert len(direction_calls) == 1
    _assert_mature_training_rows(direction_calls[0].frame, frame, 1, direction)
    assert direction_calls[0].validation is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("direction", [-1, 1], ids=["falling", "rising"])
def test_glft_fit_only_trains_on_mature_direction_outcomes(
    monkeypatch: pytest.MonkeyPatch, direction: int
) -> None:
    frame = _market_frame(500, direction)
    original = frame.copy(deep=True)
    trained_frames: list[pd.DataFrame] = []

    def train(
        _model: FLAMLAutoMLDirectionModel,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        assert eval_df is None
        trained_frames.append(df.copy(deep=True))

    def predict_proba(
        _model: FLAMLAutoMLDirectionModel, df: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame({0: 0.0, 1: 1.0}, index=df.index)

    def skip_parameter_search(
        _strategy: GLFTMLStrategy,
        search_frame: pd.DataFrame,
    ) -> None:
        pd.testing.assert_frame_equal(search_frame, original)

    monkeypatch.setattr(FLAMLAutoMLDirectionModel, "train", train)
    monkeypatch.setattr(FLAMLAutoMLDirectionModel, "predict_proba", predict_proba)
    monkeypatch.setattr(GLFTMLStrategy, "_grid_search_glft", skip_parameter_search)
    strategy = GLFTMLStrategy(
        GLFTMLStrategyConfig(prediction_horizon=5),
        backtest_engine=SignalBacktestEngine(init_cash=10_000.0),
    )

    strategy.fit(frame)

    assert len(trained_frames) == 1
    _assert_mature_training_rows(trained_frames[0], frame, 5, int(direction > 0))
    pd.testing.assert_frame_equal(frame, original)
