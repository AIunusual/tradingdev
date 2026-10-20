"""Tests for the feature engineering module."""

from typing import TYPE_CHECKING

from tradingdev.domain.ml.features.features import FeatureEngineer

if TYPE_CHECKING:
    import pandas as pd


class TestFeatureEngineer:
    def test_transform_produces_features(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(lookback=12)
        result = fe.transform(large_ohlcv_df, include_target=True)
        assert "log_return" in result.columns
        assert "ret_lag_1" in result.columns
        assert "ret_lag_12" in result.columns
        assert "target" in result.columns

    def test_no_nan_in_output(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(lookback=12)
        result = fe.transform(large_ohlcv_df, include_target=True)
        feature_names = fe.get_feature_names()
        for col in feature_names:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_target_values(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(lookback=6)
        result = fe.transform(large_ohlcv_df, include_target=True)
        assert set(result["target"].unique()).issubset({-1, 0, 1})

    def test_no_target_when_disabled(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(lookback=6)
        result = fe.transform(large_ohlcv_df, include_target=False)
        assert "target" not in result.columns

    def test_feature_names_cached(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(lookback=6)
        fe.transform(large_ohlcv_df, include_target=True)
        names = fe.get_feature_names()
        assert len(names) > 0
        assert "target" not in names
        assert "close" not in names

    def test_different_lookback_sizes(self, large_ohlcv_df: pd.DataFrame) -> None:
        fe_small = FeatureEngineer(lookback=6)
        fe_large = FeatureEngineer(lookback=24)
        r_small = fe_small.transform(large_ohlcv_df)
        r_large = fe_large.transform(large_ohlcv_df)
        # Larger lookback → more lag features → more columns
        assert len(r_large.columns) > len(r_small.columns)
        # Larger lookback → more warm-up rows dropped → fewer rows
        assert len(r_large) <= len(r_small)
