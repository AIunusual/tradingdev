"""Tests for the GLFT market-making strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.strategies.bundled.glft_strategy.config import (
    GLFTStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_strategy.strategy import GLFTStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

# ── Helpers ──────────────────────────────────────────────────


def _make_config(**overrides: object) -> GLFTStrategyConfig:
    """Create a lightweight config for fast tests."""
    defaults: dict[str, object] = {
        "gamma": 0.01,
        "kappa": 1.5,
        "ema_window": 10,
        "vol_window": 10,
        "min_holding_bars": 3,
        "max_holding_bars": 15,
        "gamma_candidates": [0.005, 0.01, 0.05],
        "kappa_candidates": [1.0, 1.5],
        "ema_window_candidates": [10, 21],
        "max_holding_bars_candidates": [10, 15],
        "vol_window_candidates": [10, 20],
    }
    defaults.update(overrides)
    return GLFTStrategyConfig(**defaults)


# ── Config validation tests ──────────────────────────────────


class TestGLFTStrategyConfig:
    def test_default_values(self) -> None:
        """Default config should have valid parameter values."""
        config = GLFTStrategyConfig()
        assert config.gamma == 500.0
        assert config.kappa == 1000.0
        assert config.ema_window == 21
        assert config.min_holding_bars == 5
        assert config.max_holding_bars == 30

    def test_gamma_must_be_non_negative(self) -> None:
        """gamma < 0 should raise validation error; 0 is allowed."""
        GLFTStrategyConfig(gamma=0.0)  # should NOT raise
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(gamma=-0.01)

    def test_kappa_must_be_positive(self) -> None:
        """kappa <= 0 should raise validation error."""
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(kappa=0.0)
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(kappa=-1.0)

    def test_max_holding_gt_min_holding(self) -> None:
        """max_holding_bars must be > min_holding_bars."""
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(
                min_holding_bars=10,
                max_holding_bars=5,
            )
        with pytest.raises(ValidationError):
            GLFTStrategyConfig(
                min_holding_bars=10,
                max_holding_bars=10,
            )

    def test_implied_vol_type_accepted(self) -> None:
        """vol_type='implied' is valid without strategy path parameters."""
        config = GLFTStrategyConfig(vol_type="implied")
        assert config.vol_type == "implied"


# ── State machine tests ──────────────────────────────────────


class TestGLFTStateMachine:
    def test_min_holding_enforced(self) -> None:
        """Cannot exit before min_holding_bars."""
        n = 20
        # Price starts well below EMA, then jumps above
        close = np.concatenate(
            [
                np.full(5, 99.0),  # below EMA → should enter long
                np.full(15, 102.0),  # above EMA → would want to exit
            ]
        )
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.005)

        signals, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=5,
            max_hold=30,
        )

        # Find first entry
        entry_idx = -1
        for i in range(n):
            if signals[i] != 0:
                entry_idx = i
                break

        assert entry_idx >= 0, "Should have entered a position"
        # Must hold for at least min_hold bars after entry
        for j in range(entry_idx, min(entry_idx + 5, n)):
            assert signals[j] != 0, f"Exited too early at bar {j}"

    @pytest.mark.parametrize(
        ("price", "position"), [(99.5, 1), (100.5, -1)], ids=["long", "short"]
    )
    def test_max_holding_forced_exit(self, price: float, position: int) -> None:
        """Position must exit at max_holding_bars."""
        max_hold = 10
        n = max_hold + 1
        # A constant deviation beyond the entry spread opens a position and
        # cannot trigger either the profit target or the adverse stop-loss.
        close = np.full(n, price)
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)

        signals, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=2,
            max_hold=max_hold,
        )

        assert signals[0] == position, "Must enter before checking the deadline"
        np.testing.assert_array_equal(signals[:max_hold], position)
        assert signals[max_hold] == 0

    def test_flat_when_within_spread(self) -> None:
        """No entry when price is near EMA (within spread)."""
        n = 50
        close = np.full(n, 100.0)  # price == EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.01)

        signals, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.1,
            kappa=1.5,
            min_hold=5,
            max_hold=30,
        )

        # Should stay flat
        assert np.all(signals == 0)

    def test_long_entry_when_cheap(self) -> None:
        """Price below EMA by more than spread triggers long."""
        n = 20
        close = np.full(n, 95.0)  # 5% below EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)  # low vol → tight spread

        signals, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=2,
            max_hold=15,
        )

        # Should enter long
        assert signals[0] == 1

    def test_short_entry_when_expensive(self) -> None:
        """Price above EMA by more than spread triggers short."""
        n = 20
        close = np.full(n, 105.0)  # 5% above EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)

        signals, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=2,
            max_hold=15,
        )

        assert signals[0] == -1


# ── Signal generation tests ──────────────────────────────────


class TestGLFTSignalGeneration:
    def test_signal_values_in_valid_range(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """All signals must be -1, 0, or 1."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert set(result["signal"].unique()).issubset(
            {-1.0, 0.0, 1.0},
        )

    def test_no_nan_in_signals(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Signals should not contain NaN."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert not result["signal"].isna().any()

    def test_signal_column_added(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """generate_signals() should add a signal column."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert "signal" in result.columns
        assert len(result) == len(sample_ohlcv_df)

    def test_original_df_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """generate_signals() should not modify the input df."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        cols_before = set(sample_ohlcv_df.columns)
        strategy.generate_signals(sample_ohlcv_df)
        assert set(sample_ohlcv_df.columns) == cols_before

    def test_no_look_ahead_bias(
        self,
        sample_ohlcv_df: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        """Signals must be identical when future bars are truncated away."""
        strategy = GLFTStrategy(config=_make_config())
        assert_no_look_ahead(
            strategy,
            sample_ohlcv_df,
            check_points=[20, 45, 70, 99],
        )

    def test_signals_cover_long_short_and_flat(self) -> None:
        """A mean-reverting series must produce all three signal states."""
        n = 240
        t = np.arange(n)
        close = 100.0 + 3.0 * np.sin(t / 6.0)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.full(n, 100.0),
            }
        )
        config = _make_config(
            gamma=0.0,
            kappa=1000.0,
            min_entry_edge=0.005,
            min_holding_bars=2,
            max_holding_bars=8,
        )
        strategy = GLFTStrategy(config=config)

        signals = strategy.generate_signals(df)["signal"]

        assert not signals.isna().any()
        assert set(signals.unique()) == {-1.0, 0.0, 1.0}


# ── fit() tests ──────────────────────────────────────────────


class TestGLFTFit:
    def test_fit_selects_from_candidates(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """After fit(), best params should be from candidates."""
        engine = VolumeBacktestEngine(
            fees=0.0,
            slippage=0.0,
            position_size=100.0,
            signal_as_position=True,
        )
        config = _make_config()
        strategy = GLFTStrategy(
            config=config,
            backtest_engine=engine,
        )
        strategy.fit(large_ohlcv_df)
        params = strategy.get_parameters()
        assert params["best_gamma"] in config.gamma_candidates
        assert params["best_kappa"] in config.kappa_candidates
        assert params["best_ema_window"] in config.ema_window_candidates
        assert params["best_max_holding_bars"] in config.max_holding_bars_candidates
        assert params["best_vol_window"] in config.vol_window_candidates

    def test_fit_without_engine_uses_defaults(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """fit() without engine should use default params."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        strategy.fit(large_ohlcv_df)  # should not raise
        params = strategy.get_parameters()
        assert params["best_gamma"] == config.gamma
        assert params["best_kappa"] == config.kappa


# ── get_parameters() tests ───────────────────────────────────


class TestGLFTGetParameters:
    def test_includes_config_fields(self) -> None:
        """get_parameters() should include config fields."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        params = strategy.get_parameters()
        assert "gamma" in params
        assert "kappa" in params
        assert "ema_window" in params
        assert "min_holding_bars" in params

    def test_includes_best_params(self) -> None:
        """get_parameters() should include best_* fields."""
        config = _make_config()
        strategy = GLFTStrategy(config=config)
        params = strategy.get_parameters()
        assert "best_gamma" in params
        assert "best_kappa" in params
        assert "best_ema_window" in params
        assert "best_max_holding_bars" in params
        assert "best_vol_window" in params


# ── Integration tests ────────────────────────────────────────


class TestGLFTIntegration:
    def test_end_to_end_with_volume_engine(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Full pipeline: fit -> generate_signals -> engine.run."""
        engine = VolumeBacktestEngine(
            fees=0.0006,
            slippage=0.0005,
            position_size=100.0,
            stop_loss=0.03,
            signal_as_position=True,
            re_entry_after_sl=False,
        )
        config = _make_config()
        strategy = GLFTStrategy(
            config=config,
            backtest_engine=engine,
        )
        strategy.fit(large_ohlcv_df.iloc[:800])
        signals = strategy.generate_signals(
            large_ohlcv_df.iloc[800:],
        )
        result = engine.run(signals)

        assert result.metrics["total_trades"] >= 0
        assert "total_return" in result.metrics


# ── Implied volatility (DVOL) tests ─────────────────────────


def _make_implied_config(
    **overrides: object,
) -> GLFTStrategyConfig:
    """Config with vol_type='implied' for fast tests."""
    defaults: dict[str, object] = {
        "gamma": 0.01,
        "kappa": 1.5,
        "ema_window": 10,
        "vol_window": 0,
        "vol_type": "implied",
        "min_holding_bars": 3,
        "max_holding_bars": 15,
        "gamma_candidates": [0.005, 0.01],
        "kappa_candidates": [1.0, 1.5],
        "ema_window_candidates": [10],
        "max_holding_bars_candidates": [15],
        "vol_window_candidates": [10, 20],
    }
    defaults.update(overrides)
    return GLFTStrategyConfig(**defaults)


class TestGLFTImpliedVolatility:
    def test_compute_volatility_implied_conversion(
        self,
    ) -> None:
        """sigma = dvol / 100 / sqrt(525960)."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        dvol = np.array([45.0, 60.0, 30.0])
        close = np.array([40000.0, 40000.0, 40000.0])
        high = close.copy()
        low = close.copy()

        sigma = strategy._compute_volatility(close, high, low, dvol=dvol)

        expected = dvol / 100.0 / np.sqrt(525_960.0)
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_compute_volatility_implied_no_dvol_raises(
        self,
    ) -> None:
        """Calling implied vol without dvol array raises."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        close = np.array([40000.0])
        high = close.copy()
        low = close.copy()

        with pytest.raises(ValueError, match="dvol array"):
            strategy._compute_volatility(close, high, low)

    def test_generate_signals_with_dvol(
        self,
        sample_ohlcv_with_dvol_df: pd.DataFrame,
    ) -> None:
        """End-to-end signal generation with implied vol."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_with_dvol_df)

        assert "signal" in result.columns
        assert not result["signal"].isna().any()
        assert set(result["signal"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_generate_signals_implied_missing_dvol_raises(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Missing dvol column raises ValueError."""
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config)

        with pytest.raises(ValueError, match="dvol"):
            strategy.generate_signals(sample_ohlcv_df)

    def test_leading_missing_dvol_blocks_entries_until_a_value_is_available(
        self,
        sample_ohlcv_with_dvol_df: pd.DataFrame,
    ) -> None:
        strategy = GLFTStrategy(
            config=_make_implied_config(
                gamma=0.0, kappa=1000.0, ema_window=2, momentum_guard=False
            )
        )
        complete = strategy.generate_signals(sample_ohlcv_with_dvol_df)
        assert complete["signal"].iloc[:50].ne(0).any()

        frame = sample_ohlcv_with_dvol_df.copy()
        frame.loc[frame.index[:50], "dvol"] = np.nan
        before = frame.copy(deep=True)
        result = strategy.generate_signals(frame)

        assert result["signal"].iloc[:50].eq(0).all()
        assert result["signal"].iloc[50:].ne(0).any()
        assert not result["signal"].isna().any()
        pd.testing.assert_frame_equal(result[frame.columns], before)
        pd.testing.assert_frame_equal(frame, before)

    def test_fit_implied_ignores_vol_window(
        self,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        """fit() with implied vol should use [0] for vol_window."""
        # Add a dvol column so generate_signals works
        df = large_ohlcv_df.copy()
        rng = np.random.default_rng(seed=99)
        df["dvol"] = rng.uniform(30.0, 80.0, size=len(df))

        engine = VolumeBacktestEngine(
            fees=0.0,
            slippage=0.0,
            position_size=100.0,
            signal_as_position=True,
        )
        config = _make_implied_config()
        strategy = GLFTStrategy(config=config, backtest_engine=engine)
        strategy.fit(df)
        params = strategy.get_parameters()
        # vol_window should be 0 (the sentinel for implied)
        assert params["best_vol_window"] == 0


# ── Dynamic position sizing tests ─────────────────────────


class TestGLFTDynamicSizing:
    def test_size_weight_column_present(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """dynamic_sizing=True adds size_weight column."""
        config = _make_config(dynamic_sizing=True)
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert "size_weight" in result.columns
        assert len(result["size_weight"]) == len(sample_ohlcv_df)

    def test_size_weight_all_ones_when_disabled(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """dynamic_sizing=False -> all weights are 1.0."""
        config = _make_config(dynamic_sizing=False)
        strategy = GLFTStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_df)
        assert "size_weight" in result.columns
        np.testing.assert_array_equal(
            result["size_weight"].values,
            np.ones(len(result)),
        )

    def test_size_weight_range(self) -> None:
        """Weights must be in [min_weight, 1.0]."""
        n = 100
        close = np.full(n, 95.0)  # 5% below EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)
        min_w = 500.0 / 3000.0

        _, weights = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.001,
            kappa=1.0,
            min_hold=2,
            max_hold=15,
            dynamic_sizing=True,
            min_weight=min_w,
            edge_for_full_size=0.005,
        )

        # In-position bars should have weight in [min_w, 1.0]
        in_pos = weights[weights != 1.0]
        if len(in_pos) > 0:
            assert np.all(in_pos >= min_w - 1e-10)
            assert np.all(in_pos <= 1.0 + 1e-10)

    def test_size_weight_proportional(self) -> None:
        """Larger deviation -> larger weight."""
        n = 20
        sigma = np.full(n, 0.001)
        ema = np.full(n, 100.0)
        min_w = 500.0 / 3000.0
        efs = 0.10  # 10% for full size

        # Use gamma=0 so spread_const=1/kappa is small
        # Small deviation: 2% below EMA
        close_small = np.full(n, 98.0)
        _, w_small = GLFTStrategy._run_glft_state_machine(
            close=close_small,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=2,
            max_hold=15,
            min_entry_edge=0.001,
            dynamic_sizing=True,
            min_weight=min_w,
            edge_for_full_size=efs,
        )

        # Large deviation: 8% below EMA
        close_large = np.full(n, 92.0)
        _, w_large = GLFTStrategy._run_glft_state_machine(
            close=close_large,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=2,
            max_hold=15,
            min_entry_edge=0.001,
            dynamic_sizing=True,
            min_weight=min_w,
            edge_for_full_size=efs,
        )

        # First entry bar weight should be larger for bigger deviation
        assert w_large[0] > w_small[0]

    def test_size_weight_carries_through_hold(self) -> None:
        """Weight stays constant during holding period."""
        n = 30
        close = np.full(n, 95.0)  # 5% below EMA
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.001)
        min_w = 500.0 / 3000.0

        signals, weights = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.001,
            kappa=1.0,
            min_hold=2,
            max_hold=15,
            dynamic_sizing=True,
            min_weight=min_w,
            edge_for_full_size=0.10,
        )

        # Find first position span
        entry_w = None
        for i in range(n):
            if signals[i] != 0:
                if entry_w is None:
                    entry_w = weights[i]
                else:
                    assert weights[i] == entry_w
            elif entry_w is not None:
                break  # position ended

    def test_min_position_le_max_validator(self) -> None:
        """min_position_size > position_size raises."""
        with pytest.raises(ValidationError):
            _make_config(
                dynamic_sizing=True,
                min_position_size=5000.0,
                position_size=3000.0,
            )


# ── Indicator warm-up tests ─────────────────────────────────


class TestIndicatorWarmup:
    def test_compute_ema_is_nan_until_window_completes(self) -> None:
        close = np.linspace(100.0, 110.0, 50)
        ema = GLFTStrategy._compute_ema(close, ema_window=10, agg_minutes=1)
        assert np.isnan(ema[:9]).all()
        assert np.isfinite(ema[9:]).all()

    def test_compute_ema_aggregated_is_nan_until_enough_bars(self) -> None:
        close = np.linspace(100.0, 110.0, 120)
        ema = GLFTStrategy._compute_ema(close, ema_window=5, agg_minutes=5)
        # Five aggregated 5-minute bars need 25 one-minute bars; bar 24 is the
        # first whose most recently completed window has a finite EMA.
        assert np.isnan(ema[:24]).all()
        assert np.isfinite(ema[24:]).all()

    def test_trend_filter_blocks_entries_while_trend_is_unknown(self) -> None:
        n = 20
        close = np.full(n, 99.0)  # below EMA → wants long every bar
        ema = np.full(n, 100.0)
        sigma = np.full(n, 0.005)

        blocked, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=1,
            max_hold=30,
            trend_dir=np.full(n, np.nan),
            momentum_guard=False,
        )
        assert not blocked.any()

        allowed, _ = GLFTStrategy._run_glft_state_machine(
            close=close,
            ema=ema,
            sigma=sigma,
            gamma=0.0,
            kappa=1000.0,
            min_hold=1,
            max_hold=30,
            trend_dir=np.ones(n),
            momentum_guard=False,
        )
        assert allowed.any()

    def test_generate_signals_stays_flat_during_trend_ema_warmup(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """A trend EMA longer than the data never warms up, so nothing is entered."""
        strategy = GLFTStrategy(config=_make_config(trend_ema_window=200, ema_window=5))
        result = strategy.generate_signals(sample_ohlcv_df)
        assert (result["signal"] == 0).all()

    def test_aggregated_ema_never_uses_a_future_aggregate(self) -> None:
        """Regression: negative aggregate indices were clipped to the first one."""
        close = np.arange(100.0, 120.0)
        ema = GLFTStrategy._compute_ema(close, ema_window=2, agg_minutes=5)
        # The seed requires two completed aggregates (closes 104 and 109).
        assert np.isnan(ema[:9]).all()
        np.testing.assert_array_equal(ema[9:14], np.full(5, 106.5))
        np.testing.assert_array_equal(ema[14:19], np.full(5, 111.5))

    def test_aggregated_ema_with_fewer_bars_than_one_aggregate(self) -> None:
        close = np.arange(100.0, 103.0)
        ema = GLFTStrategy._compute_ema(close, ema_window=2, agg_minutes=5)
        assert ema.shape == (3,)
        assert np.isnan(ema).all()

    @pytest.mark.parametrize("ema_window", [2, 3])
    def test_aggregated_ema_is_truncation_invariant(self, ema_window: int) -> None:
        """A prefix of the series must reproduce the prefix of the result."""
        rng = np.random.default_rng(7)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, 60))
        full = GLFTStrategy._compute_ema(close, ema_window=ema_window, agg_minutes=5)
        for m in (3, 4, 5, 9, 10, 14, 15, 23, 60):
            partial = GLFTStrategy._compute_ema(
                close[:m], ema_window=ema_window, agg_minutes=5
            )
            np.testing.assert_array_equal(partial, full[:m])

    def test_no_look_ahead_with_aggregated_ema(
        self,
        sample_ohlcv_df: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        strategy = GLFTStrategy(
            config=_make_config(ema_window=2, signal_agg_minutes=5),
        )
        assert_no_look_ahead(
            strategy,
            sample_ohlcv_df,
            check_points=[3, 4, 5, 9, 10, 14, 15, 20, 45, 99],
        )
