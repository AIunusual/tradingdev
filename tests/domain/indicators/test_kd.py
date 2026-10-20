"""Tests for the KD Stochastic Oscillator indicator."""

from typing import TYPE_CHECKING

from tradingdev.domain.indicators.kd import KDIndicator

if TYPE_CHECKING:
    import pandas as pd


class TestKDIndicator:
    def setup_method(self) -> None:
        self.indicator = KDIndicator()

    def test_output_columns_exist(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        result = self.indicator.calculate(sample_ohlcv_with_kd)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_values_in_range(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        result = self.indicator.calculate(sample_ohlcv_with_kd)
        valid_k = result["stoch_k"].dropna()
        valid_d = result["stoch_d"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_initial_nan_values(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        """KD requires a warmup period; initial values should be NaN."""
        result = self.indicator.calculate(sample_ohlcv_with_kd)
        assert result["stoch_k"].iloc[0] != result["stoch_k"].iloc[0]  # NaN check

    def test_get_parameters(self) -> None:
        indicator = KDIndicator(k_period=9, d_period=5, smooth_k=5)
        params = indicator.get_parameters()
        assert params == {"k_period": 9, "d_period": 5, "smooth_k": 5}

    def test_custom_parameters(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        indicator = KDIndicator(k_period=9, d_period=5, smooth_k=5)
        result = indicator.calculate(sample_ohlcv_with_kd)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_preserves_original_columns(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        original_cols = set(sample_ohlcv_with_kd.columns)
        result = self.indicator.calculate(sample_ohlcv_with_kd)
        assert original_cols.issubset(set(result.columns))
