"""Versioned raw market evidence and retained evaluation row coordinates."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tradingdev.domain.data.cache import DataIdentity

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
RAW_MARKET_VERSION = 1


class RawMarketSeries(TypedDict):
    """One symbol's input before filtering, labelling or calendar alignment."""

    source: str
    market_type: str | None
    symbol: str
    timeframe: str
    first_utc: str
    last_utc: str
    rows: int
    fingerprint: str


class RawMarketEvidence(TypedDict):
    """Format version and one series receipt per symbol, in symbol order."""

    version: Literal[1]
    series: list[RawMarketSeries]


class EvidenceCheck(TypedDict):
    """Outcome across the complete selected candidate set."""

    status: Literal["verified", "mismatch", "unknown"]
    reasons: list[str]


class MarketComparison(TypedDict):
    """Raw inputs and evaluated row coordinates are independent checks."""

    raw_baseline: EvidenceCheck
    evaluation_rows: EvidenceCheck


def _timestamps(values: pd.Series) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    if index.hasnans:
        raise ValueError("market evidence timestamps must be present")
    return index


def raw_market_series(bars: pd.DataFrame, identity: DataIdentity) -> RawMarketSeries:
    """Hash the ordered input rows using the stable version 1 binary format."""
    times = _timestamps(bars["timestamp"])
    if times.empty or times.has_duplicates:
        raise ValueError("raw market timestamps must be nonempty and unique")

    # 48-byte rows: UTC int64 nanoseconds, then five IEEE-754 float64s.
    # Fixed little endian and NaN bits make the bytes portable.
    encoded = np.empty(len(bars), dtype=[("time", "<i8"), ("ohlcv", "<f8", (5,))])
    encoded["time"] = times.to_numpy(dtype="datetime64[ns]").view("int64")
    values = encoded["ohlcv"]
    values[:] = bars.loc[:, list(OHLCV_COLUMNS[1:])].to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    values.view("<u8")[np.isnan(values)] = 0x7FF8000000000000
    digest = hashlib.sha256(b"tradingdev.ml.raw_market.v1\0")
    digest.update(encoded.tobytes())
    return {
        "source": identity.source,
        "market_type": identity.market_type,
        "symbol": identity.symbol,
        "timeframe": identity.timeframe,
        "first_utc": times.min().isoformat(),
        "last_utc": times.max().isoformat(),
        "rows": len(times),
        "fingerprint": digest.hexdigest(),
    }


def evaluated_rows_fingerprint(
    frame: pd.DataFrame, *, series: RawMarketSeries, horizon: int
) -> str:
    """Validate retained coordinates and hash ordered timestamps and origins.

    The raw ordinal names the bar at label_start; the label_end ordinal is
    raw_row + horizon. Exclude the end from row equality so a label change can
    be distinguished from a change to the rows that were evaluated.
    """
    evaluated = _timestamps(frame["timestamp"])
    starts = _timestamps(frame["label_start"])
    ends = _timestamps(frame["label_end"])
    first = pd.Timestamp(series["first_utc"])
    last = pd.Timestamp(series["last_utc"])
    positions = frame["raw_row"]
    if (
        len(evaluated) == 0
        or evaluated.has_duplicates
        or not pd.api.types.is_integer_dtype(positions)
        or positions.isna().any()
        or not positions.is_monotonic_increasing
        or (positions < 0).any()
        or (positions >= series["rows"] - horizon).any()
        or (starts > evaluated).any()
        or (starts == ends).any()
        or starts.min() < first
        or ends.min() < first
        or starts.max() > last
        or ends.max() > last
    ):
        raise ValueError("invalid dataset evaluation row mapping")
    coordinates = np.column_stack(
        (
            evaluated.to_numpy(dtype="datetime64[ns]").view("int64"),
            starts.to_numpy(dtype="datetime64[ns]").view("int64"),
            positions.to_numpy(dtype=np.int64),
        )
    )
    return hashlib.sha256(coordinates.astype("<i8").tobytes()).hexdigest()
