"""Tests for the XGBoost direction prediction strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestXGBoostStrategy:
    def _make_config(self) -> XGBoostStrategyConfig:
        return XGBoostStrategyConfig(
            model=XGBoostModelConfig(
                n_estimators=20,
                max_depth=3,
            ),
            lookback_candidates=[6, 12],
            retrain_interval=50,
            validation_ratio=0.2,
        )

    def test_generate_signals_before_fit_raises(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """generate_signals() before fit() should raise."""
        strategy = XGBoostStrategy(config=self._make_config())
        with pytest.raises(RuntimeError, match="not fitted"):
            strategy.generate_signals(large_ohlcv_df)

    def test_fit_selects_lookback(self, large_ohlcv_df: pd.DataFrame) -> None:
        """After fit(), best_lookback should be from candidates."""
        strategy = XGBoostStrategy(config=self._make_config())
        strategy.fit(large_ohlcv_df)
        params = strategy.get_parameters()
        assert params["best_lookback"] in [6, 12]

    def test_generate_signals_valid(self, large_ohlcv_df: pd.DataFrame) -> None:
        """After fit(), generate_signals produces valid signal column."""
        config = self._make_config()
        strategy = XGBoostStrategy(config=config)

        # Use first 800 for fit, last 200 for test
        fit_data = large_ohlcv_df.iloc[:800].copy()
        test_data = large_ohlcv_df.iloc[800:].copy()
        test_data.index = pd.RangeIndex(1000, 1600, 3)
        original = test_data.copy(deep=True)

        strategy.fit(fit_data)
        result = strategy.generate_signals(test_data)

        assert "signal" in result.columns
        unique = set(result["signal"].unique())
        assert unique.issubset({-1, 0, 1})
        pd.testing.assert_frame_equal(result[test_data.columns], original)
        pd.testing.assert_frame_equal(test_data, original)

    def test_get_parameters(self) -> None:
        """get_parameters() should include config fields."""
        strategy = XGBoostStrategy(config=self._make_config())
        params = strategy.get_parameters()
        assert "lookback_candidates" in params
        assert "retrain_interval" in params
        assert params["best_lookback"] is None

    @pytest.mark.parametrize("horizon", [40, 300])
    def test_fit_rejects_no_eligible_mature_training_windows(
        self, large_ohlcv_df: pd.DataFrame, horizon: int
    ) -> None:
        strategy = XGBoostStrategy(
            XGBoostStrategyConfig(
                model=XGBoostModelConfig(n_estimators=2, n_jobs=1),
                lookback_candidates=[5, 12],
                target_horizon=horizon,
                validation_ratio=0.2,
            )
        )

        with pytest.raises(ValueError, match="No lookback candidate.*mature"):
            strategy.fit(large_ohlcv_df.iloc[:300])

        assert strategy.get_parameters()["best_lookback"] is None
        assert strategy._model is None
        assert strategy._feature_engineer is None
        assert strategy._train_data is None
        with pytest.raises(RuntimeError, match="not fitted"):
            strategy.generate_signals(large_ohlcv_df.iloc[300:350])

    def test_rejected_fit_preserves_previous_training(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        strategy = XGBoostStrategy(
            XGBoostStrategyConfig(
                model=XGBoostModelConfig(n_estimators=2, n_jobs=1),
                lookback_candidates=[5, 12],
                target_horizon=40,
                validation_ratio=0.2,
            )
        )
        strategy.fit(large_ohlcv_df)
        before_model = strategy._model
        before_features = strategy._feature_engineer
        before_train = strategy._train_data
        before_parameters = strategy.get_parameters()

        with pytest.raises(ValueError, match="No lookback candidate.*mature"):
            strategy.fit(large_ohlcv_df.iloc[:300])

        assert strategy._model is before_model
        assert strategy._feature_engineer is before_features
        assert strategy._train_data is before_train
        assert strategy.get_parameters() == before_parameters

    def test_fit_uses_an_eligible_candidate_after_an_insufficient_one(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        strategy = XGBoostStrategy(
            XGBoostStrategyConfig(
                model=XGBoostModelConfig(n_estimators=2, n_jobs=1),
                lookback_candidates=[500, 6],
                target_horizon=5,
                validation_ratio=0.2,
            )
        )

        strategy.fit(large_ohlcv_df.iloc[:300])

        assert strategy.get_parameters()["best_lookback"] == 6
        assert strategy._model is not None

    def test_no_look_ahead_bias(
        self,
        large_ohlcv_df: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        """Rolling prediction may only use bars at or before each bar."""
        strategy = XGBoostStrategy(config=self._make_config())
        fit_data = large_ohlcv_df.iloc[:800].copy()
        test_data = large_ohlcv_df.iloc[800:].reset_index(drop=True)

        strategy.fit(fit_data)
        assert_no_look_ahead(strategy, test_data, check_points=[60, 120, 199])


class _StubFeatureEngineer(FeatureEngineer):
    """Feature engineer stand-in that passes bars through unchanged."""

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        return df


class _StubDirectionModel(XGBoostDirectionModel):
    """Direction model emitting a scripted probability per prediction."""

    def __init__(self, probabilities: list[tuple[float, float]]) -> None:
        super().__init__(XGBoostModelConfig())
        self._probabilities = probabilities
        self._calls = 0

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        p_long, p_short = self._probabilities[self._calls]
        self._calls += 1
        return pd.DataFrame({1: p_long, -1: p_short}, index=df.index)


class TestRollingRetrainerSignalMapping:
    def test_probabilities_map_to_long_short_and_flat(self) -> None:
        """The tri-state mapping must emit exactly -1, 0, and 1."""
        model = _StubDirectionModel([(0.9, 0.1), (0.1, 0.9), (0.5, 0.5)])
        retrainer = RollingRetrainer(
            model_config=XGBoostModelConfig(),
            retrain_interval=1000,
            threshold=0.6,
            cooldown=0,
            lookback=1000,
        )
        test_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-02", periods=3, freq="h", tz="UTC"),
                "close": [100.0, 101.0, 102.0],
            }
        )
        train_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
                "close": [99.0, 99.5],
            }
        )

        signals = retrainer.run(
            test_df,
            train_df,
            model,
            _StubFeatureEngineer(),
        )

        assert signals.tolist() == [1, -1, 0]
