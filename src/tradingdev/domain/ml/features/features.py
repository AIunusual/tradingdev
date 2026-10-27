"""Feature engineering for ML trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_ta_indicators,
    compute_volume_features,
)
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_ROLL_WINDOWS = [6, 12, 24]
_SMA_WINDOWS = [7, 14, 21]
_DENSE_THRESHOLD = 32

_EXCLUDE_COLS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target",
}


def _sampled_lags(lookback: int) -> list[int]:
    """Return lag positions: dense up to 32, then exp-spaced."""
    dense_end = min(_DENSE_THRESHOLD, lookback)
    lags = list(range(1, dense_end + 1))

    lag = _DENSE_THRESHOLD * 2
    while lag <= lookback:
        lags.append(lag)
        lag *= 2

    if lags[-1] != lookback:
        lags.append(lookback)
    return lags


class FeatureEngineer:
    """Generate feature matrices from OHLCV data."""

    def __init__(self, lookback: int = 24, target_horizon: int = 1) -> None:
        self._lookback = lookback
        self._target_horizon = target_horizon
        self._feature_names: list[str] = []

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def target_horizon(self) -> int:
        return self._target_horizon

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Transform OHLCV and retain source index labels after dropping NaNs."""
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        log_ret = close.apply(np.log).diff()

        features: dict[str, pd.Series[float]] = {"log_return": log_ret}

        for lag in _sampled_lags(self._lookback):
            features[f"ret_lag_{lag}"] = log_ret.shift(lag)

        for w in _ROLL_WINDOWS:
            features[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            features[f"ret_std_{w}"] = log_ret.rolling(w).std()

        features.update(compute_sma_ratios(close, _SMA_WINDOWS))
        features.update(compute_volume_features(volume, _SMA_WINDOWS))
        features.update(compute_ta_indicators(close))

        if include_target:
            next_ret = close.shift(-self._target_horizon) / close - 1
            # The last ``target_horizon`` rows have no realised outcome yet.
            # Leave them unlabelled so the shared ``dropna`` removes them;
            # filling them with 0 would invent a flat label that the bars
            # arriving later contradict.
            features["target"] = np.sign(next_ret)

        result = pd.concat(
            [df, pd.DataFrame(features, index=df.index)],
            axis=1,
        )
        result = result.dropna()
        if include_target:
            result["target"] = result["target"].astype(int)
        self._feature_names = [c for c in result.columns if c not in _EXCLUDE_COLS]
        return result

    def get_feature_names(self) -> list[str]:
        return list(self._feature_names)
