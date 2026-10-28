"""Feature engineering for Risk Gate model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_volume_features,
)
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_VOL_WINDOWS = [5, 15, 30, 60]
_ATR_WINDOWS = [14, 21]
_SMA_WINDOWS = [7, 14, 21]
_ROLL_WINDOWS = [6, 12, 24]

_EXCLUDE_COLS = frozenset(
    {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target",
    }
)


class RiskFeatureEngineer:
    """Generate feature matrices for the Risk Gate model.

    Focuses on volatility, regime, and market microstructure
    features rather than directional prediction.
    """

    def __init__(
        self,
        lookback: int = 24,
        target_holding_bars: int = 5,
        fee_rate: float = 0.0011,
        max_acceptable_loss_pct: float = 0.003,
    ) -> None:
        self._lookback = lookback
        self._target_holding_bars = target_holding_bars
        self._fee_rate = fee_rate
        self._max_loss_pct = max_acceptable_loss_pct
        self._feature_names: list[str] = []

    @property
    def lookback(self) -> int:
        return self._lookback

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Transform OHLCV while retaining surviving source index labels."""
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        log_ret = close.apply(np.log).diff()

        features: dict[str, pd.Series] = {}

        # --- Realized volatility at multiple horizons ---
        for w in _VOL_WINDOWS:
            features[f"realized_vol_{w}"] = log_ret.rolling(w).std()

        # --- Volatility change (trend in vol) ---
        for w in [5, 15]:
            vol_now = log_ret.rolling(w).std()
            vol_prev = log_ret.shift(w).rolling(w).std()
            features[f"vol_change_{w}"] = vol_now / vol_prev.replace(0, np.nan) - 1

        # --- ATR (normalized by close) ---
        for w in _ATR_WINDOWS:
            features[f"atr_norm_{w}"] = (
                indicators.atr(high, low, close, length=w) / close
            )

        # --- Bollinger bandwidth ---
        bands = indicators.bollinger_bands(close, length=20)
        features["bb_width_20"] = (bands.upper - bands.lower) / close

        # --- ADX (trend strength) ---
        features["adx_14"] = indicators.adx(high, low, close, length=14).adx

        # --- Volume anomaly ---
        features.update(compute_volume_features(volume, _SMA_WINDOWS))

        # --- Candle microstructure ---
        body = (close - df["open"].astype(float)).abs()
        full_range = (high - low).replace(0, np.nan)
        features["body_ratio"] = body / full_range
        features["upper_wick_ratio"] = (high - close.clip(upper=high)) / full_range
        features["lower_wick_ratio"] = (close.clip(lower=low) - low) / full_range

        # --- Return distribution moments ---
        features["ret_skew_24"] = log_ret.rolling(24).skew()
        features["ret_kurt_24"] = log_ret.rolling(24).kurt()

        # --- Lagged returns & rolling stats (reuse pattern) ---
        features["log_return"] = log_ret
        for lag in _sampled_lags(self._lookback):
            features[f"ret_lag_{lag}"] = log_ret.shift(lag)
        for w in _ROLL_WINDOWS:
            features[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            features[f"ret_std_{w}"] = log_ret.rolling(w).std()

        # --- SMA ratios ---
        features.update(compute_sma_ratios(close, _SMA_WINDOWS))

        # --- Target ---
        if include_target:
            features["target"] = self._compute_safe_target(df)

        result = pd.concat(
            [df, pd.DataFrame(features, index=df.index)],
            axis=1,
        )
        result = result.dropna()

        self._feature_names = [c for c in result.columns if c not in _EXCLUDE_COLS]
        return result

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names."""
        return list(self._feature_names)

    def _compute_safe_target(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Compute safe/unsafe binary target.

        safe=1: at least one direction yields net P&L above
                the negative loss threshold over the holding period.
        safe=0: even the best direction loses more than threshold.
        """
        close = df["close"].astype(float).values
        holding = self._target_holding_bars
        fee_rt = 2 * self._fee_rate  # round-trip fee

        n = len(close)
        targets = np.full(n, np.nan)

        for t in range(n - holding):
            exit_p = close[t + holding]
            entry_p = close[t]
            if entry_p == 0:
                continue
            long_pnl = (exit_p / entry_p - 1) - fee_rt
            short_pnl = (1 - exit_p / entry_p) - fee_rt
            best_pnl = max(long_pnl, short_pnl)
            targets[t] = 1 if best_pnl > -self._max_loss_pct else 0

        return pd.Series(targets, index=df.index, name="target")


_DENSE_THRESHOLD = 32


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
