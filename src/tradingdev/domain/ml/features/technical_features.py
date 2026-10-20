"""Technical indicator feature extraction functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain import indicators

if TYPE_CHECKING:
    import pandas as pd


def compute_sma_ratios(
    close: pd.Series,
    windows: list[int],
) -> dict[str, pd.Series]:
    """Compute close/SMA(N) ratios."""
    return {f"close_sma_ratio_{w}": close / indicators.sma(close, w) for w in windows}


def compute_volume_features(
    volume: pd.Series,
    windows: list[int],
) -> dict[str, pd.Series]:
    """Compute volume change rate and volume/SMA ratios.

    Volume change remains unknown when either comparison endpoint is missing.
    """
    features: dict[str, pd.Series] = {
        "volume_change": volume.pct_change(fill_method=None)
    }
    for w in windows:
        features[f"vol_sma_ratio_{w}"] = volume / indicators.sma(volume, w)
    return features


def compute_ta_indicators(
    close: pd.Series,
) -> dict[str, pd.Series]:
    """Compute RSI(14), the MACD histogram, and Bollinger %B(20).

    Every key is always present; values are NaN while an indicator warms up
    or when *close* is shorter than its window.
    """
    bands = indicators.bollinger_bands(close, length=20)
    band_width = (bands.upper - bands.lower).replace(0, float("nan"))
    return {
        "rsi_14": indicators.rsi(close, length=14),
        "macd_hist": indicators.macd(close).histogram,
        "bb_pctb": (close - bands.lower) / band_width,
    }
