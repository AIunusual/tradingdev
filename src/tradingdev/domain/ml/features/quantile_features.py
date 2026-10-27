"""Feature engineering for the quantile / path-opportunity strategy.

Produces features suitable for predicting whether a profitable price
move will occur within a configurable horizon window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_volume_features,
)
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_RETURN_WINDOWS = [1, 5, 10, 30]
_ROLL_STD_WINDOWS = [5, 10, 30]
_SMA_WINDOWS = [7, 14, 21]

_EXCLUDE_COLS = frozenset(
    {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dvol",
        "dvol_open",
        "dvol_high",
        "dvol_low",
        "dvol_close",
        "target",
        "target_long",
        "target_short",
        "target_regime",
        "funding_rate_raw",
    }
)


class QuantileFeatureEngineer:
    """Generate features for path-opportunity prediction.

    Features include multi-period log returns, rolling volatility,
    candle structure ratios, ATR, cyclical time encodings, and
    optional DVOL (implied volatility) features.

    Supports two target modes:

    * ``"endpoint"`` — log return at exactly ``t + horizon``
      (original quantile regression target).
    * ``"path"`` — binary labels indicating whether price reaches
      a profit threshold at any point within the horizon window.
    """

    def __init__(
        self,
        horizon: int = 30,
        profit_target: float = 0.001,
    ) -> None:
        self._horizon = horizon
        self._profit_target = profit_target
        self._feature_names: list[str] = []

    @property
    def horizon(self) -> int:
        """Prediction horizon in bars."""
        return self._horizon

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
        target_type: str = "path",
    ) -> pd.DataFrame:
        """Transform OHLCV (+ optional DVOL) into a feature DataFrame.

        Args:
            df: DataFrame with ``close``, ``high``, ``low``,
                ``open``, ``volume``, ``timestamp``, and optionally
                ``dvol``.
            include_target: Whether to add target columns.
            target_type: ``"path"`` for binary path targets
                (``target_long``, ``target_short``), or
                ``"endpoint"`` for log return at horizon.

        Returns:
            DataFrame with original + feature + target columns.
            DVOL changes remain unknown when either comparison endpoint is missing.
            Rows with NaN features are dropped; surviving source index labels
            are preserved so predictions can be aligned with the input.
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        log_close = close.replace(0, np.nan).apply(np.log)
        log_ret = log_close.diff()

        features: dict[str, pd.Series[float]] = {}

        # --- Multi-period log returns ---
        for w in _RETURN_WINDOWS:
            features[f"log_return_{w}"] = log_close.diff(w)

        # --- Rolling volatility ---
        for w in _ROLL_STD_WINDOWS:
            features[f"rolling_std_{w}"] = log_ret.rolling(w).std()

        # --- Candle structure (ratio form) ---
        full_range = (high - low).replace(0, np.nan)
        body = (close - open_).abs()
        features["body_ratio"] = body / full_range
        co_max = pd.concat([close, open_], axis=1).max(axis=1)
        co_min = pd.concat([close, open_], axis=1).min(axis=1)
        features["upper_shadow_ratio"] = (high - co_max) / full_range
        features["lower_shadow_ratio"] = (co_min - low) / full_range
        features["close_position"] = (close - low) / full_range

        # --- ATR (normalized) ---
        features["atr_14"] = indicators.atr(high, low, close, length=14) / close

        # --- Rolling range (normalized) ---
        for w in [10, 30]:
            roll_high = high.rolling(w).max()
            roll_low = low.rolling(w).min()
            features[f"rolling_range_{w}"] = (roll_high - roll_low) / close

        # --- SMA ratios ---
        features.update(compute_sma_ratios(close, _SMA_WINDOWS))

        # --- Volume features ---
        features.update(compute_volume_features(volume, _SMA_WINDOWS))

        # --- RSI ---
        features["rsi_14"] = indicators.rsi(close, length=14)

        # --- DVOL features (optional) ---
        if "dvol" in df.columns:
            dvol = df["dvol"].astype(float)
            features["dvol_level"] = dvol
            features["dvol_change"] = dvol.pct_change(5, fill_method=None)

        # --- Funding rate features (optional) ---
        if "funding_rate" in df.columns:
            fr = df["funding_rate"].astype(float)
            features["funding_rate"] = fr
            # Cumulative rate over recent periods (sentiment pressure)
            features["funding_rate_ma3"] = fr.rolling(3).mean()
            # Extreme positive = longs crowded, negative = shorts crowded
            features["funding_rate_abs"] = fr.abs()

        # --- Time features (cyclical encoding) ---
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour_frac = np.asarray(
                ts.dt.hour + ts.dt.minute / 60.0,
                dtype=np.float64,
            )
            features["hour_sin"] = pd.Series(
                np.sin(2 * np.pi * hour_frac / 24),
                index=df.index,
            )
            features["hour_cos"] = pd.Series(
                np.cos(2 * np.pi * hour_frac / 24),
                index=df.index,
            )
            dow = np.asarray(ts.dt.dayofweek, dtype=np.float64)
            features["dow_sin"] = pd.Series(
                np.sin(2 * np.pi * dow / 7),
                index=df.index,
            )
            features["dow_cos"] = pd.Series(
                np.cos(2 * np.pi * dow / 7),
                index=df.index,
            )

        # --- Targets ---
        if include_target:
            if target_type == "regime":
                features["target_regime"] = self._compute_regime_target(
                    close,
                    high,
                    low,
                )
            elif target_type == "path":
                tl, ts_ = self._compute_path_targets(
                    close,
                    high,
                    low,
                )
                features["target_long"] = tl
                features["target_short"] = ts_
            else:
                features["target"] = log_close.shift(-self._horizon) - log_close

        result = pd.concat(
            [df, pd.DataFrame(features, index=df.index)],
            axis=1,
        )
        result = result.dropna()

        self._feature_names = [c for c in result.columns if c not in _EXCLUDE_COLS]
        return result

    def _compute_path_targets(
        self,
        close: pd.Series[float],
        high: pd.Series[float],
        low: pd.Series[float],
    ) -> tuple[pd.Series[float], pd.Series[float]]:
        """Compute path-based binary targets.

        For each bar t, look forward h bars and check whether:
        - ``max(high[t+1:t+h])`` exceeds ``close[t] * (1 + profit_target)``
        - ``min(low[t+1:t+h])`` drops below ``close[t] * (1 - profit_target)``

        Returns:
            Tuple of (target_long, target_short) Series with values 0/1.
        """
        h = self._horizon
        pt = self._profit_target
        n = len(close)

        close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64))
        high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64))
        low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64))

        target_long = np.full(n, np.nan)
        target_short = np.full(n, np.nan)

        for i in range(n - h):
            c = close_arr[i]
            future_high = np.max(high_arr[i + 1 : i + h + 1])
            future_low = np.min(low_arr[i + 1 : i + h + 1])
            target_long[i] = 1.0 if future_high >= c * (1 + pt) else 0.0
            target_short[i] = 1.0 if future_low <= c * (1 - pt) else 0.0

        return (
            pd.Series(target_long, index=close.index),
            pd.Series(target_short, index=close.index),
        )

    def _compute_regime_target(
        self,
        close: pd.Series[float],
        high: pd.Series[float],
        low: pd.Series[float],
    ) -> pd.Series[float]:
        """Compute 4-class regime target.

        Classes:
        - 0: long_only — only upside opportunity exists
        - 1: short_only — only downside opportunity exists
        - 2: both — both directions reach profit target (dangerous)
        - 3: neither — no opportunity in either direction

        Returns:
            Series with values 0/1/2/3 (NaN for insufficient future data).
        """
        h = self._horizon
        pt = self._profit_target
        n = len(close)

        close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64))
        high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64))
        low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64))

        target = np.full(n, np.nan)
        for i in range(n - h):
            c = close_arr[i]
            future_high = np.max(high_arr[i + 1 : i + h + 1])
            future_low = np.min(low_arr[i + 1 : i + h + 1])
            can_long = future_high >= c * (1 + pt)
            can_short = future_low <= c * (1 - pt)

            if can_long and not can_short:
                target[i] = 0.0
            elif can_short and not can_long:
                target[i] = 1.0
            elif can_long and can_short:
                target[i] = 2.0
            else:
                target[i] = 3.0

        return pd.Series(target, index=close.index)

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names."""
        return list(self._feature_names)
