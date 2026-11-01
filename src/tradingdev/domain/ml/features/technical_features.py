"""Technical indicator feature extraction functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tradingdev.domain import indicators

if TYPE_CHECKING:
    import pandas as pd

RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_LENGTH = 20

_EMA_SEED_TOLERANCE = 1e-3
"""Residual weight an exponential seed may still carry at the scored bar.

TA-Lib seeds its EMAs and Wilder smoothing from the first window it sees, so a
short context starts those recursions from a value an unlimited history would
not have.  A context is only long enough once the seed has decayed below this
weight; measured against a full history, that leaves the seeded features within
about 0.1% of a standard deviation.
"""


def _settle_bars(alpha: float) -> int:
    """Bars after which an exponential seed keeps less than the tolerance."""
    return math.ceil(math.log(_EMA_SEED_TOLERANCE) / math.log1p(-alpha))


TA_CONTEXT_ROWS = max(
    RSI_LENGTH + 1,
    # TA-Lib defines the first MACD histogram at index ``slow + signal - 2``.
    MACD_SLOW + MACD_SIGNAL - 1,
    BOLLINGER_LENGTH,
)
"""Rows, including the bar itself, that every indicator needs to appear.

The slowest of the three decides it, so a caller that hands a feature engineer
fewer rows than this gets no complete row at all -- independent of the
strategy's ``lookback``.
"""

TA_SETTLED_CONTEXT_ROWS = max(
    RSI_LENGTH + 1 + _settle_bars(1.0 / RSI_LENGTH),
    MACD_SLOW + MACD_SIGNAL - 1 + _settle_bars(2.0 / (MACD_SLOW + 1)),
    BOLLINGER_LENGTH,
)
"""Rows, including the bar itself, for the seeded indicators to settle.

:data:`TA_CONTEXT_ROWS` makes the values exist; this makes them agree with what
an unlimited history would produce, to within :data:`_EMA_SEED_TOLERANCE`.  The
difference is not academic: at :data:`TA_CONTEXT_ROWS` the MACD histogram is
still off by more than a standard deviation on some bars, which would hand the
model a feature it was never trained on.
"""


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
    or when *close* is shorter than its window.  The periods are the module
    constants above so that :data:`TA_CONTEXT_ROWS` describes this output.
    """
    bands = indicators.bollinger_bands(close, length=BOLLINGER_LENGTH)
    band_width = (bands.upper - bands.lower).replace(0, float("nan"))
    return {
        f"rsi_{RSI_LENGTH}": indicators.rsi(close, length=RSI_LENGTH),
        "macd_hist": indicators.macd(
            close,
            fast=MACD_FAST,
            slow=MACD_SLOW,
            signal=MACD_SIGNAL,
        ).histogram,
        "bb_pctb": (close - bands.lower) / band_width,
    }
