"""Feature engineering for multi-minute direction prediction.

Produces a feature matrix suitable for tabular learners (FLAML AutoML,
LightGBM, XGBoost, CatBoost) to predict the direction of price movement
over configurable horizons
(e.g. 5, 15, 30 minutes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_ta_indicators,
    compute_volume_features,
)
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_RETURN_LAG_BARS = [1, 2, 3, 5, 10, 15, 30, 60]
_ROLL_WINDOWS = [5, 15, 30, 60]
_SMA_WINDOWS = [7, 14, 21, 50]
_EMA_WINDOWS = [5, 15, 30, 75]
_VOL_WINDOWS = [5, 15, 30, 60]

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
    }
)


class DirectionFeatureEngineer:
    """Generate feature matrices for direction prediction.

    Features include price returns, volatility, volume, technical
    indicators, DVOL (if available), and cyclical time encodings.
    """

    def __init__(
        self,
        lookback: int = 60,
        prediction_horizon: int = 5,
    ) -> None:
        self._lookback = lookback
        self._prediction_horizon = prediction_horizon
        self._feature_names: list[str] = []

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_horizon

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Transform OHLCV (+ optional DVOL) into a feature DataFrame.

        Args:
            df: DataFrame with columns ``close``, ``high``, ``low``,
                ``open``, ``volume``, ``timestamp``, and optionally
                ``dvol``.
            include_target: Whether to add a ``target`` column
                (binary: 1=up, 0=down based on prediction_horizon).

        Returns:
            DataFrame with original columns + feature columns + target.
            DVOL changes remain unknown when either comparison endpoint is missing.
            Rows with NaN features are dropped; retained source indices identify
            their original rows and are never model input features.
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        log_ret = close.apply(np.log).diff()

        features: dict[str, pd.Series] = {}

        # --- Return features ---
        features["log_return"] = log_ret
        for lag in _RETURN_LAG_BARS:
            if lag <= self._lookback:
                features[f"ret_lag_{lag}"] = log_ret.shift(lag)
        for w in _ROLL_WINDOWS:
            features[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            features[f"ret_std_{w}"] = log_ret.rolling(w).std()

        # --- Price structure: SMA ratios ---
        features.update(compute_sma_ratios(close, _SMA_WINDOWS))

        # --- EMA deviations (directly relevant to GLFT) ---
        for w in _EMA_WINDOWS:
            ema = indicators.ema(close, w)
            features[f"ema_dev_{w}"] = (close - ema) / ema

        # --- Volatility ---
        for w in _VOL_WINDOWS:
            features[f"realized_vol_{w}"] = log_ret.rolling(w).std()
        # Parkinson volatility (normalized)
        log_hl = np.log(high / low.replace(0, np.nan))
        parkinson_var = log_hl**2 / (4.0 * np.log(2))
        features["parkinson_vol_14"] = np.sqrt(
            parkinson_var.rolling(14, min_periods=1).mean()
        )
        # Volatility change
        for w in [5, 15]:
            vol_now = log_ret.rolling(w).std()
            vol_prev = log_ret.shift(w).rolling(w).std()
            features[f"vol_change_{w}"] = vol_now / vol_prev.replace(0, np.nan) - 1

        # --- ATR (normalized) ---
        features["atr_norm_14"] = indicators.atr(high, low, close, length=14) / close

        # --- Volume features ---
        features.update(compute_volume_features(volume, _SMA_WINDOWS))

        # --- Technical indicators ---
        features.update(compute_ta_indicators(close))

        features["adx_14"] = indicators.adx(high, low, close, length=14).adx

        # --- Candle microstructure ---
        body = (close - open_).abs()
        full_range = (high - low).replace(0, np.nan)
        features["body_ratio"] = body / full_range

        # --- Higher moments ---
        features["ret_skew_24"] = log_ret.rolling(24).skew()
        features["ret_kurt_24"] = log_ret.rolling(24).kurt()

        # --- DVOL features (optional) ---
        if "dvol" in df.columns:
            dvol = df["dvol"].astype(float)
            features["dvol_level"] = dvol
            for w in [5, 15]:
                features[f"dvol_change_{w}"] = dvol.pct_change(w, fill_method=None)

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

        # --- Target ---
        if include_target:
            h = self._prediction_horizon
            fwd_return = close.shift(-h) / close - 1
            # The last ``h`` rows have no realised outcome yet. Keep them
            # unlabelled so the shared ``dropna`` removes them; ``NaN > 0``
            # would otherwise read as a down move.
            features["target"] = (
                (fwd_return > 0).astype(float).where(fwd_return.notna())
            )

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
        """Return the list of feature column names."""
        return list(self._feature_names)
