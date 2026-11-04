"""Tests for the feature engineering module."""

import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.technical_features import TA_CONTEXT_ROWS


class TestFeatureEngineer:
    def test_transform_keeps_source_identity_without_changing_features(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        df = large_ohlcv_df.set_index("timestamp")
        original = df.copy(deep=True)
        fe = FeatureEngineer(lookback=12)
        indexed = fe.transform(df, include_target=False)
        indexed_names = fe.get_feature_names()
        positional = fe.transform(df.reset_index(drop=True), include_target=False)

        assert 0 < len(indexed) < len(df)
        pd.testing.assert_index_equal(indexed.index, df.index[-len(indexed) :])
        pd.testing.assert_index_equal(positional.index, pd.RangeIndex(33, len(df)))
        pd.testing.assert_frame_equal(
            indexed.reset_index(drop=True), positional.reset_index(drop=True)
        )
        assert indexed_names == fe.get_feature_names()
        pd.testing.assert_frame_equal(df, original)

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


class TestContextRows:
    """The warm-up requirement belongs to the features, not to ``lookback``."""

    def test_the_slowest_fixed_indicator_sets_the_short_lookback_floor(
        self,
    ) -> None:
        # MACD is the last fixed feature to appear: TA-Lib defines its first
        # histogram value at index ``slow + signal - 2``, so the bar needs
        # ``26 + 9 - 1`` rows. The deepest lag of lookback=12 asks for 14.
        assert TA_CONTEXT_ROWS == 26 + 9 - 1
        assert FeatureEngineer(lookback=12).context_rows > TA_CONTEXT_ROWS

    def test_the_seeded_indicators_are_given_time_to_settle(self) -> None:
        # RSI recurses with Wilder's 1/14, MACD's slow EMA with 2/27. A seed
        # below 1e-3 needs ceil(ln(1e-3) / ln(1 - alpha)) bars: 94 and 90.
        assert FeatureEngineer(lookback=12).context_rows == 26 + 9 - 1 + 90
        assert FeatureEngineer(lookback=24).context_rows == 26 + 9 - 1 + 90

    def test_the_deepest_lag_takes_over_once_it_passes_the_indicators(
        self,
    ) -> None:
        # ``ret_lag_L`` shifts a log return by L, and that return reads the
        # previous bar, so the bar itself needs L + 2 rows.
        assert FeatureEngineer(lookback=168).context_rows == 168 + 2

    @pytest.mark.parametrize("lookback", [6, 12, 24])
    def test_the_bare_warmup_leaves_exactly_the_scored_bar(
        self, large_ohlcv_df: pd.DataFrame, lookback: int
    ) -> None:
        # Beyond lookback 32 the deepest lag, not the indicators, sets the
        # requirement, so a bare indicator warm-up is already too short there.
        window = large_ohlcv_df.iloc[-TA_CONTEXT_ROWS:]

        result = FeatureEngineer(lookback=lookback).transform(
            window, include_target=False
        )

        assert result.index.tolist() == [large_ohlcv_df.index[-1]]

    @pytest.mark.parametrize("lookback", [6, 12, 24, 48])
    def test_the_declared_rows_end_on_the_scored_bar(
        self, large_ohlcv_df: pd.DataFrame, lookback: int
    ) -> None:
        fe = FeatureEngineer(lookback=lookback)

        result = fe.transform(
            large_ohlcv_df.iloc[-fe.context_rows :], include_target=False
        )

        assert result.index[-1] == large_ohlcv_df.index[-1]

    @pytest.mark.parametrize("lookback", [12, 24])
    def test_one_row_short_of_the_indicator_warmup_leaves_no_row(
        self, large_ohlcv_df: pd.DataFrame, lookback: int
    ) -> None:
        fe = FeatureEngineer(lookback=lookback)
        window = large_ohlcv_df.iloc[: TA_CONTEXT_ROWS - 1]

        assert fe.transform(window, include_target=False).empty

    def test_one_row_short_of_the_deepest_lag_leaves_no_row(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """The lag term is exact in both directions, not just a lower bound.

        ``lookback=168`` asks for 170 rows, so 169 must still leave nothing: a
        requirement that were merely inflated would otherwise pass unnoticed.
        """
        fe = FeatureEngineer(lookback=168)
        assert fe.context_rows == 168 + 2

        assert fe.transform(
            large_ohlcv_df.iloc[: fe.context_rows - 1], include_target=False
        ).empty

    def test_a_bare_warmup_keeps_the_bar_but_not_its_values(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """Existing rows are not enough: the seeded ones must also agree.

        ``TA_CONTEXT_ROWS`` rows produce the bar, but its MACD histogram still
        carries most of a seed an unlimited history would not have.  The
        declared ``context_rows`` brings it back to the same value, so the test
        would fail if the settling margin were dropped again.
        """
        fe = FeatureEngineer(lookback=12)
        bar = large_ohlcv_df.index[-1]
        reference = fe.transform(large_ohlcv_df, include_target=False).loc[bar]
        bare = fe.transform(
            large_ohlcv_df.iloc[-TA_CONTEXT_ROWS:], include_target=False
        ).loc[bar]
        settled = fe.transform(
            large_ohlcv_df.iloc[-fe.context_rows :], include_target=False
        ).loc[bar]
        scale = float(
            fe.transform(large_ohlcv_df, include_target=False)["macd_hist"].std()
        )

        assert abs(settled["macd_hist"] - reference["macd_hist"]) < 0.01 * scale
        assert abs(bare["macd_hist"] - reference["macd_hist"]) > 0.01 * scale
