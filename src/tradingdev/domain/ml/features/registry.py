"""Named feature sets with declared columns and OHLCV builders.

Builders preserve input rows, including NaN during indicator warm-up. The
caller drops incomplete rows while maintaining feature/label alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradingdev.domain.ml.features.technical_features import (
    compute_sma_ratios,
    compute_ta_indicators,
    compute_volume_features,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FeatureSet:
    """One named, enumerable collection of features and its output contract."""

    name: str
    description: str
    columns: tuple[str, ...]
    build: Callable[[pd.DataFrame], pd.DataFrame]
    tags: tuple[str, ...] = ()

    @property
    def n_features(self) -> int:
        """How many columns this set promises to emit."""
        return len(self.columns)


_REGISTRY: dict[str, FeatureSet] = {}


def register(feature_set: FeatureSet) -> FeatureSet:
    """Add a feature set to the registry, rejecting duplicate names."""
    if feature_set.name in _REGISTRY:
        raise ValueError(f"feature set {feature_set.name!r} is already registered")
    _REGISTRY[feature_set.name] = feature_set
    return feature_set


def get_feature_set(name: str) -> FeatureSet:
    """Return a registered set by name; raise ``KeyError`` when unknown."""
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(
            f"unknown feature set {name!r}; known: {sorted(_REGISTRY)}"
        ) from error


def list_feature_sets() -> tuple[FeatureSet, ...]:
    """All registered sets, sorted by name for stable tool output."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def feature_set_names() -> tuple[str, ...]:
    """The names a tool can offer an LLM as a menu."""
    return tuple(sorted(_REGISTRY))


def _build_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log return, lagged returns and rolling moments of the log return."""
    close = df["close"].astype(float)
    log_return = close.apply(np.log).diff()
    out = pd.DataFrame(index=df.index)
    out["log_return"] = log_return
    for lag in (1, 2, 3):
        out[f"ret_lag_{lag}"] = log_return.shift(lag)
    for window in (6, 12):
        out[f"ret_mean_{window}"] = log_return.rolling(window).mean()
        out[f"ret_std_{window}"] = log_return.rolling(window).std()
    return out


def _build_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Close price divided by its moving averages (momentum style)."""
    return pd.DataFrame(
        compute_sma_ratios(df["close"].astype(float), [7, 14, 21]),
        index=df.index,
    )


def _build_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Volume change and volume relative to its moving averages."""
    return pd.DataFrame(
        compute_volume_features(df["volume"].astype(float), [7, 14, 21]),
        index=df.index,
    )


def _build_technical(df: pd.DataFrame) -> pd.DataFrame:
    """RSI(14), the MACD histogram and Bollinger %B(20)."""
    return pd.DataFrame(
        compute_ta_indicators(df["close"].astype(float)),
        index=df.index,
    )


def _build_comprehensive(df: pd.DataFrame) -> pd.DataFrame:
    """Everything above, concatenated into one row-preserving frame."""
    return pd.concat(
        [
            _build_returns(df),
            _build_price_ratios(df),
            _build_volume(df),
            _build_technical(df),
        ],
        axis=1,
    )


def _register_builtin_sets() -> None:
    """Populate the registry with the audited, named feature sets."""
    register(
        FeatureSet(
            name="returns",
            description=(
                "Log return plus lagged log returns and rolling mean/std of the "
                "log return. Pure time-series momentum and volatility."
            ),
            columns=(
                "log_return",
                "ret_lag_1",
                "ret_lag_2",
                "ret_lag_3",
                "ret_mean_6",
                "ret_std_6",
                "ret_mean_12",
                "ret_std_12",
            ),
            build=_build_returns,
            tags=("momentum", "volatility"),
        )
    )
    register(
        FeatureSet(
            name="price_ratios",
            description=(
                "Close price divided by its 7/14/21-bar simple moving "
                "averages; a slow momentum signal."
            ),
            columns=(
                "close_sma_ratio_7",
                "close_sma_ratio_14",
                "close_sma_ratio_21",
            ),
            build=_build_price_ratios,
            tags=("momentum",),
        )
    )
    register(
        FeatureSet(
            name="volume",
            description=(
                "Volume change rate and volume relative to its 7/14/21-bar "
                "moving average; participation / liquidity context."
            ),
            columns=(
                "volume_change",
                "vol_sma_ratio_7",
                "vol_sma_ratio_14",
                "vol_sma_ratio_21",
            ),
            build=_build_volume,
            tags=("volume",),
        )
    )
    register(
        FeatureSet(
            name="technical",
            description=(
                "Standardised technical indicators: RSI(14), MACD histogram "
                "and Bollinger %B(20)."
            ),
            columns=("rsi_14", "macd_hist", "bb_pctb"),
            build=_build_technical,
            tags=("technical",),
        )
    )
    register(
        FeatureSet(
            name="comprehensive",
            description=(
                "Union of returns, price_ratios, volume and technical: the "
                "full single-instrument feature surface."
            ),
            columns=(
                "log_return",
                "ret_lag_1",
                "ret_lag_2",
                "ret_lag_3",
                "ret_mean_6",
                "ret_std_6",
                "ret_mean_12",
                "ret_std_12",
                "close_sma_ratio_7",
                "close_sma_ratio_14",
                "close_sma_ratio_21",
                "volume_change",
                "vol_sma_ratio_7",
                "vol_sma_ratio_14",
                "vol_sma_ratio_21",
                "rsi_14",
                "macd_hist",
                "bb_pctb",
            ),
            build=_build_comprehensive,
            tags=("momentum", "volatility", "volume", "technical"),
        )
    )


_register_builtin_sets()
