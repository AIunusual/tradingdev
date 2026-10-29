"""Missing ML features cannot compress the price/holding-period timeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.direction_features import DirectionFeatureEngineer
from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
    GLFTMLStrategy,
)
from tradingdev.domain.strategies.signals import generate_checked_signals


def test_missing_dvol_blocks_entries_even_with_an_ml_direction(
    sample_ohlcv_with_dvol_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = GLFTMLStrategy(
        GLFTMLStrategyConfig(gamma=0.0, kappa=1000.0, ema_window=2, vol_type="implied")
    )
    strategy._ml_trained = True
    monkeypatch.setattr(
        strategy, "_predict_ml_directions", lambda frame, _: np.ones(len(frame))
    )
    complete = generate_checked_signals(strategy, sample_ohlcv_with_dvol_df)
    assert complete["signal"].iloc[:50].ne(0).any()

    frame = sample_ohlcv_with_dvol_df.copy()
    frame.loc[frame.index[:50], "dvol"] = np.nan
    before = frame.copy(deep=True)
    result = generate_checked_signals(strategy, frame)

    assert result["signal"].iloc[:50].eq(0).all()
    assert result["signal"].iloc[50:].ne(0).any()
    assert not result["signal"].isna().any()
    pd.testing.assert_frame_equal(result[frame.columns], before)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("include_target", [False, True])
def test_direction_features_keep_original_row_identity_after_dropna(
    sample_ohlcv_df: pd.DataFrame, include_target: bool
) -> None:
    frame = sample_ohlcv_df.assign(dvol=np.linspace(40, 60, len(sample_ohlcv_df)))
    frame.index = pd.Index(np.arange(len(frame)) * 7 + 1000, name="source_row")
    frame.loc[frame.index[80], "dvol"] = np.nan
    features = DirectionFeatureEngineer().transform(
        frame, include_target=include_target
    )
    assert not features.empty and len(features) < len(frame)
    assert features.index.name == frame.index.name
    assert features.index.isin(frame.index).all()
    assert frame.index[80] not in features.index
    pd.testing.assert_series_equal(
        features["timestamp"], frame.loc[features.index, "timestamp"]
    )


@pytest.mark.parametrize("positions", [[1, 4], []])
def test_ml_probabilities_are_placed_on_the_full_original_price_timeline(
    monkeypatch: pytest.MonkeyPatch, positions: list[int]
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "close": [100.0, 98.0, 99.0, 101.0, 102.0, 103.0],
            "high": [104.0] * 6,
            "low": [97.0] * 6,
            "dvol": [50.0] * 6,
        },
        index=pd.Index(range(100, 106), name="source_row"),
    )
    before = frame.copy(deep=True)
    strategy = GLFTMLStrategy(GLFTMLStrategyConfig())
    strategy._ml_trained = True
    features = Mock()
    features.transform.side_effect = lambda df, **kw: df.iloc[positions].copy()
    model = Mock()
    model.predict_proba.side_effect = lambda df: pd.DataFrame(
        {0: [0.1, 0.9], 1: [0.9, 0.1]}, index=df.index
    )
    observed: list[dict[str, Any]] = []

    def states(**kwargs: Any) -> Any:
        observed.append(kwargs)
        return kwargs["ml_dir"]

    monkeypatch.setattr(strategy, "_feature_eng", features)
    monkeypatch.setattr(strategy, "_ml_model", model)
    monkeypatch.setattr(strategy, "_run_ml_glft_state_machine", states)
    result = generate_checked_signals(strategy, frame)
    expected = [0, 1, 0, 0, -1, 0] if positions else [0] * 6
    assert result["signal"].tolist() == expected
    np.testing.assert_array_equal(observed[0]["close"], before["close"].to_numpy())
    pd.testing.assert_frame_equal(result[frame.columns], before)
    pd.testing.assert_frame_equal(frame, before)
    assert model.predict_proba.call_count == bool(positions)


def test_ml_probability_row_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = GLFTMLStrategy(GLFTMLStrategyConfig())
    strategy._ml_trained = True
    features = Mock()
    features.transform.return_value = pd.DataFrame({"close": [100.0]}, index=[4])
    model = Mock()
    model.predict_proba.return_value = pd.DataFrame({0: [0.1], 1: [0.9]}, index=[0])
    monkeypatch.setattr(strategy, "_feature_eng", features)
    monkeypatch.setattr(strategy, "_ml_model", model)
    with pytest.raises(ValueError, match="probabilities must preserve"):
        strategy.generate_signals(pd.DataFrame({"close": [100.0] * 6}))


@pytest.mark.parametrize("tune", [False, True])
def test_missing_predictions_do_not_pause_holding_age_in_search_or_execution(
    monkeypatch: pytest.MonkeyPatch,
    tune: bool,
) -> None:
    frame = pd.DataFrame({"close": [100.0, 98.0, 98.0, 98.0, 102.0, 102.0]}).assign(
        high=104.0, low=97.0, dvol=50.0
    )
    engine = Mock()
    engine.run.return_value.metrics = {"total_volume": 1.0}
    strategy = GLFTMLStrategy(
        GLFTMLStrategyConfig(
            min_holding_bars=1,
            max_holding_bars=2,
            profit_target_ratio=0,
            strategy_sl=0,
            gamma_candidates=[0.0],
            kappa_candidates=[1000.0],
            ema_window_candidates=[2],
            max_holding_bars_candidates=[2],
            min_entry_edge_candidates=[0.0012],
            profit_target_ratio_candidates=[0.0],
            confidence_threshold_candidates=[0.55],
        ),
        backtest_engine=engine,
    )
    strategy._ml_trained = True
    features = Mock()
    features.transform.side_effect = lambda df, **kw: df.iloc[[1, 4]].copy()
    features.get_feature_names.return_value = ["close"]
    model = Mock()
    model.predict_proba.side_effect = lambda df: pd.DataFrame(
        {0: [0.1, 0.9], 1: [0.9, 0.1]}, index=df.index
    )
    monkeypatch.setattr(strategy, "_feature_eng", features)
    monkeypatch.setattr(strategy, "_ml_model", model)
    monkeypatch.setattr(strategy, "_compute_ema", lambda *args: np.full(6, 100.0))
    monkeypatch.setattr(strategy, "_compute_volatility", lambda *a, **kw: np.zeros(6))
    if tune:
        strategy.fit(frame)
    result = generate_checked_signals(strategy, frame)
    # The position opened at 1 expires at 3, despite neither bar having a prediction.
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    if tune:
        engine.run.assert_called_once()
        pd.testing.assert_frame_equal(engine.run.call_args.args[0], result)
        model.train.assert_called_once()
