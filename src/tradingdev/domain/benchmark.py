"""Canonical buy-and-hold baselines chosen by asset class, never by the caller.

A benchmark answers one question: what would the same capital have earned by
holding the reference asset over the same bars? The mapping from a saved data
requirement to a benchmark is fixed here, so two people running the same market
always receive the same baseline. Nothing in this module fetches data; the
application layer supplies the closes and records how it obtained them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

# Data sources whose market is a crypto pair quoted in a stablecoin.
CRYPTO_SOURCES = frozenset({"binance_vision", "binance_api"})

# Yahoo lists non-US venues with an exchange suffix and derivates with ``=``.
# Indices are accepted only from this explicit list; an unfamiliar ``^`` symbol
# stays undefined instead of silently borrowing the US large-cap baseline.
US_INDEX_SYMBOLS = frozenset({"^GSPC", "^SPX", "^DJI", "^IXIC", "^NDX", "^RUT"})

BenchmarkStatus = Literal["available", "unavailable", "undefined"]


@dataclass(frozen=True)
class BenchmarkDefinition:
    """The fixed reference asset and the weighting rule applied to it."""

    id: str
    asset_class: str
    symbol: str
    sources: tuple[str, ...]
    label: str
    weighting: str
    rationale: str


BENCHMARKS = MappingProxyType(
    {
        "crypto": BenchmarkDefinition(
            id="crypto_btc_buy_and_hold",
            asset_class="crypto",
            symbol="BTC/USDT",
            sources=("binance_vision", "binance_api"),
            label="BTC/USDT 買進持有",
            weighting="buy_and_hold",
            rationale=(
                "加密貨幣策略的機會成本基準是持有比特幣。使用與回測相同的資料源、"
                "市場類型與 K 線週期，讓比較只反映策略進出場，不混入換交易所或"
                "換週期造成的差異。"
            ),
        ),
        "us_equity": BenchmarkDefinition(
            id="us_equity_sp500_buy_and_hold",
            asset_class="us_equity",
            symbol="^GSPC",
            sources=("yahoo_finance",),
            label="S&P 500（^GSPC）買進持有",
            weighting="buy_and_hold",
            rationale=(
                "美國股票策略的機會成本基準是持有 S&P 500 指數。使用 ^GSPC 的"
                "價格報酬，未計入成分股現金股利，因此不是含息總報酬指數。"
            ),
        ),
    }
)


def asset_class_for(source: str, symbol: str) -> str | None:
    """Classify a saved market requirement; unknown markets stay unclassified."""
    if source in CRYPTO_SOURCES and re.fullmatch(r"[A-Z0-9]+/?USDT", symbol.upper()):
        return "crypto"
    if source == "yahoo_finance":
        return "us_equity" if is_us_equity_symbol(symbol) else None
    return None


def is_us_equity_symbol(symbol: str) -> bool:
    """Accept US listings only, using Yahoo's documented symbol conventions.

    ``AAPL`` and ``SPY`` carry no venue suffix. A ``.XX`` suffix names another
    exchange, ``=F``/``=X`` are futures/FX, and an index must be one of the
    known US indices. Anything else is refused rather than assumed.
    """
    value = symbol.strip().upper()
    if not value or value.endswith(("=X", "=F")):
        return False
    if value.startswith("^"):
        return value in US_INDEX_SYMBOLS
    if "." in value:
        if not value.endswith(".US"):
            return False
        value = value[:-3]
    # Share classes such as BRK-B are supported; currency pairs such as BTC-USD
    # are not evidence of a US equity listing.
    return re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z])?", value) is not None


def resolve_benchmark(source: str, symbol: str) -> BenchmarkDefinition | None:
    """Return the fixed baseline for a market, or ``None`` when undefined."""
    asset_class = asset_class_for(source, symbol)
    return None if asset_class is None else BENCHMARKS.get(asset_class)


def benchmark_source(definition: BenchmarkDefinition, run_source: str) -> str:
    """Prefer the run's own provider so both series share a bar calendar."""
    return run_source if run_source in definition.sources else definition.sources[0]


@dataclass(frozen=True)
class BenchmarkEvidence:
    """One run's baseline identity and the closes obtained for it.

    ``status`` distinguishes "no baseline is defined for this asset class"
    from "a baseline exists but its data could not be used"; both stay visible
    instead of becoming a zero return.
    """

    status: BenchmarkStatus
    asset_class: str | None = None
    definition: BenchmarkDefinition | None = None
    reason: str | None = None
    detail: str | None = None
    source: str | None = None
    market_type: str | None = None
    dataset_id: str | None = None
    reused_run_frame: bool = False
    timestamps: pd.DatetimeIndex | None = None
    closes: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class BuyAndHold:
    """A single full-size entry at the first close, marked to market after."""

    equity: npt.NDArray[np.float64]
    quantity: float
    entry_price: float
    entry_fee: float
    entry_notional: float


def buy_and_hold(
    closes: npt.NDArray[np.float64],
    init_cash: float,
    *,
    fees: float,
    slippage: float,
) -> BuyAndHold:
    """Invest ``init_cash`` once at the first close and hold it to the last bar.

    The entry pays the run's own per-side commission and slippage, so the
    baseline is charged the same way the strategy is. The position is not
    liquidated on the final bar: the strategy's open positions are also marked
    at the closing price without a forced exit, and charging one side only
    would make the comparison asymmetric.
    """
    if not math.isfinite(init_cash) or init_cash <= 0:
        raise ValueError("Benchmark initial capital must be finite and positive")
    prices = np.asarray(closes, dtype=np.float64)
    if prices.ndim != 1 or not len(prices):
        raise ValueError("Benchmark closes must be a non-empty one-dimensional series")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Benchmark closes must be finite and positive")
    for name, value in (("fees", fees), ("slippage", slippage)):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"Benchmark {name} must be a fraction in [0, 1)")
    entry_price = float(prices[0]) * (1.0 + slippage)
    quantity = init_cash / (entry_price * (1.0 + fees))
    return BuyAndHold(
        equity=quantity * prices,
        quantity=quantity,
        entry_price=entry_price,
        entry_fee=quantity * entry_price * fees,
        entry_notional=quantity * entry_price,
    )


def aligned_closes(
    evidence: BenchmarkEvidence, timestamps: pd.DatetimeIndex
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.float64], int]:
    """Restrict a baseline to the observed bars of one result scope.

    Returns the shared timestamps, their closes and how many scope bars the
    baseline is missing. Bars are never filled or shifted: an absent benchmark
    observation is reported, not replaced by the previous close.
    """
    if evidence.timestamps is None or evidence.closes is None:
        raise ValueError("Benchmark evidence has no observations")
    series = pd.Series(evidence.closes, index=evidence.timestamps)
    selected = series.reindex(timestamps)
    present = selected.dropna()
    return (
        pd.DatetimeIndex(present.index),
        np.asarray(present.to_numpy(), dtype=np.float64),
        int(selected.isna().sum()),
    )
