"""Tests for the feature-level technical indicator helpers."""

import numpy as np
import pandas as pd
import pytest
import talib

from tradingdev.domain.ml.features.direction_features import DirectionFeatureEngineer
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_ta_indicators,
    compute_volume_features,
)


def _expected_indicators(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Independent oracle: %B with population std and the MACD histogram."""
    middle = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    pctb = (close - (middle - 2 * std)) / (4 * std).replace(0, np.nan)
    _, _, histogram = talib.MACD(close.to_numpy(dtype=np.float64))
    return pctb, pd.Series(histogram, index=close.index)


class TestRatioFeatures:
    def test_sma_ratios(self, sample_ohlcv_df: pd.DataFrame) -> None:
        close = sample_ohlcv_df["close"].astype(float)
        features = compute_sma_ratios(close, [7, 14])
        assert set(features) == {"close_sma_ratio_7", "close_sma_ratio_14"}
        expected = close / close.rolling(7).mean()
        np.testing.assert_allclose(
            features["close_sma_ratio_7"].dropna(), expected.dropna()
        )

    def test_volume_features(self, sample_ohlcv_df: pd.DataFrame) -> None:
        volume = sample_ohlcv_df["volume"].astype(float)
        features = compute_volume_features(volume, [7])
        assert set(features) == {"volume_change", "vol_sma_ratio_7"}
        expected = volume / volume.rolling(7).mean()
        np.testing.assert_allclose(
            features["vol_sma_ratio_7"].dropna(), expected.dropna()
        )

    def test_volume_change_keeps_missing_endpoints_unknown(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        volume = pd.Series(
            [100.0, 110.0, np.nan, 121.0, 133.1],
            index=timestamps,
            name="volume",
        )
        actual = compute_volume_features(volume, [2])["volume_change"]
        expected = pd.Series(
            [np.nan, 0.1, np.nan, np.nan, 0.1],
            index=timestamps,
            name="volume",
        )
        pd.testing.assert_series_equal(actual, expected)


class TestComputeTaIndicators:
    def test_bb_pctb_matches_percent_b_with_population_std(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        close = large_ohlcv_df["close"].astype(float)
        expected_pctb, _ = _expected_indicators(close)
        features = compute_ta_indicators(close)
        pd.testing.assert_series_equal(
            features["bb_pctb"], expected_pctb, check_names=False
        )

    def test_bb_pctb_above_half_when_close_above_middle_band(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """Regression: positional band selection produced ``1 - %B``."""
        close = large_ohlcv_df["close"].astype(float)
        middle = close.rolling(20).mean()
        pctb = compute_ta_indicators(close)["bb_pctb"]
        above = (close > middle) & pctb.notna()
        assert above.any()
        assert (pctb[above] > 0.5).all()

    def test_macd_hist_matches_histogram_not_signal(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        close = large_ohlcv_df["close"].astype(float)
        _, expected_hist = _expected_indicators(close)
        features = compute_ta_indicators(close)
        pd.testing.assert_series_equal(
            features["macd_hist"], expected_hist, check_names=False
        )
        _, signal, _ = talib.MACD(np.asarray(close, dtype=np.float64))
        valid = features["macd_hist"].notna()
        assert not np.allclose(features["macd_hist"][valid], signal[valid])

    def test_warmup_preserves_index_without_filling_values(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        close = sample_ohlcv_df.set_index("timestamp")["close"].astype(float)
        features = compute_ta_indicators(close)
        for name, warmup in {"rsi_14": 14, "bb_pctb": 19, "macd_hist": 33}.items():
            series = features[name]
            pd.testing.assert_index_equal(series.index, close.index)
            assert series.iloc[:warmup].isna().all()
            assert series.iloc[warmup:].notna().all()

    def test_short_series_keeps_keys_with_nan_values(self) -> None:
        close = pd.Series(np.linspace(100.0, 110.0, 10))
        features = compute_ta_indicators(close)
        assert set(features) == {"rsi_14", "macd_hist", "bb_pctb"}
        for series in features.values():
            assert len(series) == 10
            assert series.isna().all()


def _expected_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    expected_pctb, expected_hist = _expected_indicators(close)
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "expected_pctb": expected_pctb.to_numpy(),
            "expected_hist": expected_hist.to_numpy(),
        }
    )


class TestFeatureEngineersUseIndicatorSemantics:
    @pytest.mark.parametrize(
        ("engineer", "changes"),
        [
            (
                DirectionFeatureEngineer(),
                (("dvol_change_5", 5), ("dvol_change_15", 15)),
            ),
            (QuantileFeatureEngineer(), (("dvol_change", 5),)),
        ],
        ids=["DirectionFeatureEngineer", "QuantileFeatureEngineer"],
    )
    def test_dvol_gap_drops_unknown_endpoints_and_recovers(
        self,
        engineer: DirectionFeatureEngineer | QuantileFeatureEngineer,
        changes: tuple[tuple[str, int], ...],
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        frame = sample_ohlcv_df.assign(
            dvol=np.linspace(40.0, 60.0, len(sample_ohlcv_df))
        )
        baseline = engineer.transform(frame, include_target=False).set_index(
            "timestamp"
        )
        gap = 80
        unavailable = frame["timestamp"].take(
            [gap, *(gap + period for _, period in changes)]
        )
        assert unavailable.isin(baseline.index).all()

        with_gap = frame.copy()
        with_gap.loc[gap, "dvol"] = np.nan
        actual = engineer.transform(with_gap, include_target=False).set_index(
            "timestamp"
        )

        expected = baseline.loc[~baseline.index.isin(unavailable)]
        pd.testing.assert_frame_equal(actual, expected)
        for column, period in changes:
            recovered = gap + period + 1
            timestamp = frame["timestamp"].iloc[recovered]
            expected_change = (
                frame["dvol"].iloc[recovered] / frame["dvol"].iloc[recovered - period]
                - 1
            )
            assert actual.loc[timestamp, column] == pytest.approx(expected_change)

    @pytest.mark.parametrize(
        "engineer",
        [
            FeatureEngineer(lookback=24),
            DirectionFeatureEngineer(lookback=60, prediction_horizon=5),
        ],
        ids=["FeatureEngineer", "DirectionFeatureEngineer"],
    )
    def test_bb_pctb_and_macd_hist_semantics(
        self,
        engineer: FeatureEngineer | DirectionFeatureEngineer,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        result = engineer.transform(large_ohlcv_df, include_target=False)
        joined = result[["timestamp", "bb_pctb", "macd_hist"]].merge(
            _expected_by_timestamp(large_ohlcv_df), on="timestamp"
        )
        assert len(joined) == len(result)
        np.testing.assert_allclose(joined["bb_pctb"], joined["expected_pctb"])
        np.testing.assert_allclose(joined["macd_hist"], joined["expected_hist"])

    def test_risk_features_bollinger_width_is_positive(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """Regression: swapped bands produced a negative bandwidth."""
        result = RiskFeatureEngineer(lookback=24).transform(
            large_ohlcv_df, include_target=False
        )
        assert (result["bb_width_20"] > 0).all()
        close = large_ohlcv_df["close"].astype(float)
        expected = large_ohlcv_df[["timestamp"]].assign(
            expected_width=4 * close.rolling(20).std(ddof=0) / close
        )
        joined = result[["timestamp", "bb_width_20"]].merge(expected, on="timestamp")
        assert len(joined) == len(result)
        np.testing.assert_allclose(joined["bb_width_20"], joined["expected_width"])

    @pytest.mark.parametrize(
        ("engineer", "warmup"),
        [
            (FeatureEngineer(lookback=24), 33),
            (DirectionFeatureEngineer(lookback=60, prediction_horizon=5), 74),
            (RiskFeatureEngineer(lookback=24), 60),
        ],
        ids=["FeatureEngineer", "DirectionFeatureEngineer", "RiskFeatureEngineer"],
    )
    def test_feature_rows_start_after_all_indicators_warm_up(
        self,
        engineer: FeatureEngineer | DirectionFeatureEngineer | RiskFeatureEngineer,
        warmup: int,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        before = sample_ohlcv_df.copy(deep=True)
        insufficient = engineer.transform(
            sample_ohlcv_df.iloc[:warmup], include_target=False
        )
        assert insufficient.empty

        first_ready = engineer.transform(
            sample_ohlcv_df.iloc[: warmup + 1], include_target=False
        )
        assert len(first_ready) == 1
        assert (
            first_ready["timestamp"].iloc[0]
            == sample_ohlcv_df["timestamp"].iloc[warmup]
        )
        assert first_ready.notna().all().all()
        pd.testing.assert_frame_equal(sample_ohlcv_df, before)
