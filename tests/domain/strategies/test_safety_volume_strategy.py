"""Tests for the safety-first volume strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel

from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.strategy import (
    SafetyVolumeStrategy,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_config() -> SafetyVolumeStrategyConfig:
    """Create a lightweight config for fast tests."""
    return SafetyVolumeStrategyConfig(
        risk_model=XGBoostModelConfig(
            n_estimators=20,
            max_depth=3,
        ),
        lookback_candidates=[6, 12],
        retrain_interval=50,
        validation_ratio=0.2,
        target_holding_bars=3,
        # Use a tiny threshold so that small moves also count as unsafe,
        # ensuring both classes (0, 1) appear in the synthetic data.
        max_acceptable_loss_pct=0.0001,
        fee_rate=0.0011,
        min_holding_bars=3,
        max_holding_bars=15,
        sma_fast=3,
        sma_slow=10,
    )


# ── RiskFeatureEngineer tests ────────────────────────────────


class TestRiskFeatureEngineer:
    def test_transform_produces_features(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """transform() should add feature columns."""
        fe = RiskFeatureEngineer(
            lookback=12,
            target_holding_bars=3,
            fee_rate=0.0011,
            max_acceptable_loss_pct=0.003,
        )
        result = fe.transform(sample_ohlcv_df, include_target=True)

        assert "target" in result.columns
        assert "realized_vol_5" in result.columns
        assert "body_ratio" in result.columns
        assert "log_return" in result.columns
        assert len(result) > 0

    def test_transform_no_target(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """transform(include_target=False) should omit target."""
        fe = RiskFeatureEngineer(lookback=12)
        result = fe.transform(
            sample_ohlcv_df,
            include_target=False,
        )
        assert "target" not in result.columns

    def test_safe_target_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Target should only contain 0 and 1."""
        fe = RiskFeatureEngineer(
            lookback=12,
            target_holding_bars=3,
        )
        result = fe.transform(
            sample_ohlcv_df,
            include_target=True,
        )
        unique = set(result["target"].unique())
        assert unique.issubset({0, 1})

    def test_feature_names_populated(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """get_feature_names() should return non-empty list."""
        fe = RiskFeatureEngineer(lookback=12)
        fe.transform(sample_ohlcv_df)
        names = fe.get_feature_names()
        assert len(names) > 10
        assert "timestamp" not in names
        assert "close" not in names


# ── State machine tests ──────────────────────────────────────


class TestStateMachine:
    def test_min_holding_enforced(self) -> None:
        """Position should not exit before min_holding_bars."""
        risk = np.array([0.8, 0.8, 0.1, 0.1, 0.1, 0.8])
        dirs = np.array([1, 1, 1, 1, 1, 1])

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=4,
            max_hold=100,
        )

        # Enter at bar 0 (p_safe=0.8 >= 0.5)
        assert signals[0] == 1
        # Bars 1-3: still within min_hold → must stay
        assert signals[1] == 1
        assert signals[2] == 1
        assert signals[3] == 1
        # Bar 4: past min_hold and p_safe=0.1 < 0.5 → exit
        assert signals[4] == 0

    def test_max_holding_exit(self) -> None:
        """Position should exit at max_holding_bars."""
        risk = np.full(10, 0.9)
        dirs = np.full(10, 1)

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=2,
            max_hold=5,
        )

        assert signals[0] == 1  # enter
        assert signals[4] == 1  # still in (bars_in_pos=4)
        assert signals[5] == 0  # exit (bars_in_pos >= max_hold)

    def test_risk_exit(self) -> None:
        """Position should exit when risk goes below threshold."""
        risk = np.array([0.8, 0.8, 0.8, 0.2, 0.2])
        dirs = np.array([1, 1, 1, 1, 1])

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=2,
            max_hold=100,
        )

        assert signals[0] == 1
        assert signals[1] == 1
        assert signals[2] == 1
        # Bar 3: bars_in_pos=3 >= min_hold=2, p_safe=0.2 < 0.5
        assert signals[3] == 0

    def test_direction_reversal(self) -> None:
        """Direction change should trigger position reversal."""
        risk = np.full(10, 0.9)
        dirs = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=2,
            max_hold=100,
        )

        assert signals[0] == 1  # enter long
        assert signals[1] == 1  # hold (min_hold)
        assert signals[2] == 1  # hold (bars_in_pos=2 >= min_hold)
        assert signals[3] == -1  # reversal to short

    def test_no_entry_when_unsafe(self) -> None:
        """Should not enter position when risk is below threshold."""
        risk = np.full(5, 0.3)
        dirs = np.full(5, 1)

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=2,
            max_hold=100,
        )

        assert all(s == 0 for s in signals)

    def test_no_entry_without_direction(self) -> None:
        """Should not enter when direction is 0."""
        risk = np.full(5, 0.9)
        dirs = np.zeros(5, dtype=int)

        signals = SafetyVolumeStrategy._run_state_machine_with_threshold(
            risk,
            dirs,
            threshold=0.5,
            min_hold=2,
            max_hold=100,
        )

        assert all(s == 0 for s in signals)


# ── VolumeBacktestEngine new flags ───────────────────────────


class TestVolumeEngineNewFlags:
    def _make_df(self, signals: list[int]) -> pd.DataFrame:
        """Build a minimal df for the engine."""
        n = len(signals)
        close = np.full(n, 100.0)
        return pd.DataFrame(
            {
                "open": close,
                "close": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "signal": signals,
            }
        )

    def test_signal_as_position_closes_on_zero(self) -> None:
        """signal=0 should close position when signal_as_position=True."""
        engine = VolumeBacktestEngine(
            fees=0.0,
            slippage=0.0,
            position_size=100.0,
            signal_as_position=True,
        )
        # Shift by 1 is done in engine, so signal [0,1,1,0,0]
        # means engine sees [0,0,1,1,0] after shift
        df = self._make_df([0, 1, 1, 0, 0])
        result = engine.run(df)

        # Should have exactly 1 trade (enter on bar 2, exit on bar 4)
        assert result.metrics["total_trades"] == 1

    def test_no_re_entry_after_sl(self) -> None:
        """After SL hit, should not re-enter immediately."""
        engine = VolumeBacktestEngine(
            fees=0.0,
            slippage=0.0,
            position_size=100.0,
            stop_loss=0.001,
            re_entry_after_sl=False,
        )
        # Create data where SL will hit
        n = 10
        close = np.full(n, 100.0)
        high = np.full(n, 100.1)
        low = np.full(n, 99.8)  # low enough to trigger 0.1% SL
        df = pd.DataFrame(
            {
                "open": close,
                "close": close,
                "high": high,
                "low": low,
                "signal": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = engine.run(df)

        # With re_entry=False, after SL exit it should NOT
        # immediately re-enter on the same bar
        trades = result.trades
        assert len(trades) >= 1
        # First trade should be a SL exit
        assert trades[0]["exit_price"] < trades[0]["entry_price"]


# ── Strategy integration tests ───────────────────────────────


class TestSafetyVolumeStrategy:
    def test_generate_signals_before_fit_raises(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """generate_signals() before fit() should raise."""
        strategy = SafetyVolumeStrategy(config=_make_config())
        with pytest.raises(RuntimeError, match="not fitted"):
            strategy.generate_signals(large_ohlcv_df)

    def test_fit_selects_lookback(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """After fit(), best_lookback should be from candidates."""
        strategy = SafetyVolumeStrategy(config=_make_config())
        strategy.fit(large_ohlcv_df)
        params = strategy.get_parameters()
        assert params["best_lookback"] in [6, 12]

    def test_generate_signals_valid(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """After fit(), generate_signals produces valid signals."""
        config = _make_config()
        strategy = SafetyVolumeStrategy(config=config)

        fit_data = large_ohlcv_df.iloc[:800].copy()
        test_data = large_ohlcv_df.iloc[800:].copy()

        strategy.fit(fit_data)
        result = strategy.generate_signals(test_data)

        assert "signal" in result.columns
        unique = set(result["signal"].unique())
        assert unique.issubset({-1, 0, 1})

    def test_get_parameters(self) -> None:
        """get_parameters() should include config fields."""
        strategy = SafetyVolumeStrategy(config=_make_config())
        params = strategy.get_parameters()
        assert "lookback_candidates" in params
        assert "risk_threshold" in params

    def test_no_look_ahead_bias(
        self,
        large_ohlcv_df: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        """Rolling risk prediction may only use bars at or before each bar."""
        strategy = SafetyVolumeStrategy(config=_make_config())
        fit_data = large_ohlcv_df.iloc[:300].copy()
        test_data = large_ohlcv_df.iloc[300:340].reset_index(drop=True)

        strategy.fit(fit_data)
        assert_no_look_ahead(strategy, test_data, check_points=[15, 39])


# ── ML direction alignment ───────────────────────────────────


class TestMLDirectionAlignment:
    """Predictions stay on their source bars despite warm-up or interior gaps."""

    def test_predictions_land_on_the_bars_their_features_came_from(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        df = large_ohlcv_df.iloc[:200].reset_index(drop=True)
        fe = FeatureEngineer(lookback=12, target_horizon=1)
        feat = fe.transform(df, include_target=False)
        offset = len(df) - len(feat)
        assert offset > 0, "fixture must exercise a warm-up gap"

        class _EchoDirectionModel:
            """One distinct value per feature row, so a shift is visible."""

            def predict(self, frame: pd.DataFrame) -> pd.Series[float]:
                return pd.Series(
                    np.arange(len(frame), dtype=float) + 100.0, index=frame.index
                )

        strategy = SafetyVolumeStrategy(config=_make_config())
        strategy._dir_fe = fe
        strategy._dir_model = cast("XGBoostDirectionModel", _EchoDirectionModel())

        directions = strategy._compute_ml_directions(df)

        assert len(directions) == len(df)
        np.testing.assert_array_equal(directions[:offset], 0.0)
        np.testing.assert_array_equal(
            directions[offset:], np.arange(len(feat), dtype=float) + 100.0
        )
        # The first predicted bar is the first row that survived the drop.
        assert df["timestamp"].iloc[offset] == feat["timestamp"].iloc[0]

    @pytest.mark.parametrize("mismatch", ["count", "order"])
    def test_predictions_must_match_feature_rows(
        self,
        large_ohlcv_df: pd.DataFrame,
        mismatch: str,
    ) -> None:
        df = large_ohlcv_df.iloc[:100].reset_index(drop=True)

        class _MismatchedDirectionModel:
            def predict(self, frame: pd.DataFrame) -> pd.Series[float]:
                index = frame.index[:-1] if mismatch == "count" else frame.index[::-1]
                return pd.Series(1.0, index=index)

        strategy = SafetyVolumeStrategy(config=_make_config())
        strategy._dir_fe = FeatureEngineer(lookback=12, target_horizon=1)
        strategy._dir_model = cast("XGBoostDirectionModel", _MismatchedDirectionModel())
        with pytest.raises(RuntimeError, match="do not match the feature rows"):
            strategy._compute_ml_directions(df)

    @pytest.mark.parametrize("rows", [0, 15])
    def test_no_complete_features_returns_zero_without_calling_model(
        self,
        large_ohlcv_df: pd.DataFrame,
        rows: int,
    ) -> None:
        class _UnexpectedDirectionModel:
            def predict(self, frame: pd.DataFrame) -> pd.Series[float]:
                pytest.fail("No feature rows should reach the direction model")

        strategy = SafetyVolumeStrategy(config=_make_config())
        strategy._dir_fe = FeatureEngineer(lookback=12, target_horizon=1)
        strategy._dir_model = cast("XGBoostDirectionModel", _UnexpectedDirectionModel())

        directions = strategy._compute_ml_directions(large_ohlcv_df.iloc[:rows])
        np.testing.assert_array_equal(directions, np.zeros(rows))

    @pytest.mark.parametrize("time_layout", ["column", "datetime_index", "none"])
    def test_interior_missing_features_do_not_change_past_directions(
        self, time_layout: str
    ) -> None:
        close = 100.0 + np.arange(180) % 7
        close[70:110] = 100.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(180, 1000.0),
            }
        )
        timestamps = pd.date_range("2024-01-01", periods=len(df), freq="h")
        if time_layout == "column":
            df["timestamp"] = timestamps
        elif time_layout == "datetime_index":
            df.index = timestamps
        else:
            df.index = pd.RangeIndex(1000, 1180)
        original = df.copy(deep=True)

        class _CloseDirectionModel:
            def predict(self, frame: pd.DataFrame) -> pd.Series[float]:
                return pd.Series(
                    np.where(frame["close"] >= 103.0, 1.0, -1.0), index=frame.index
                )

        strategy = SafetyVolumeStrategy(config=_make_config())
        strategy._dir_fe = FeatureEngineer(lookback=12, target_horizon=1)
        strategy._dir_model = cast("XGBoostDirectionModel", _CloseDirectionModel())

        directions = strategy._compute_ml_directions(df)
        prefix = strategy._compute_ml_directions(df.iloc[:60])

        # Warm-up ends at 33; a later flat period makes Bollinger width zero
        # and removes feature rows 89:110, without changing earlier rows.
        expected = np.where(close >= 103.0, 1.0, -1.0)
        expected[:33] = 0.0
        expected[89:110] = 0.0
        np.testing.assert_array_equal(directions, expected)
        np.testing.assert_array_equal(directions[:60], prefix)
        pd.testing.assert_frame_equal(df, original)
