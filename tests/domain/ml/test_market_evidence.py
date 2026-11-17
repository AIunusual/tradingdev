"""The durable raw hash follows the published binary format."""

import hashlib
import struct

import numpy as np
import pandas as pd

from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.ml.market_evidence import raw_market_series


def test_raw_hash_has_fixed_endianness_missing_values_and_order() -> None:
    times = pd.to_datetime(
        ["2024-01-01T00:00:00.000000001Z", "2024-01-01T00:00:00.000000003Z"]
    )
    frame = pd.DataFrame(
        {
            "volume": [5, 6],
            "close": [4, 5],
            "low": [3, 4],
            "open": [1, 2],
            "high": [np.nan, 3.0],
            "timestamp": times,
        }
    )
    identity = DataIdentity(
        source="binance_vision", market_type="spot", symbol="BTC/USDT", timeframe="1h"
    )
    expected = hashlib.sha256(
        b"tradingdev.ml.raw_market.v1\0"
        + struct.pack("<qdQddd", 1704067200000000001, 1, 0x7FF8000000000000, 3, 4, 5)
        + struct.pack("<qddddd", 1704067200000000003, 2, 3, 4, 5, 6)
    ).hexdigest()

    evidence = raw_market_series(frame, identity)
    assert evidence["fingerprint"] == expected
    assert evidence["first_utc"] == "2024-01-01T00:00:00.000000001+00:00"
    assert evidence["last_utc"] == "2024-01-01T00:00:00.000000003+00:00"
    assert raw_market_series(frame.iloc[::-1], identity)["fingerprint"] != expected
    converted = frame.assign(timestamp=times.tz_convert("Europe/Oslo"))
    assert raw_market_series(converted, identity) == evidence
