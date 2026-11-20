"""Typed, offline loader for the versioned KD(14,3,3) BTC/USDT 1h 2024 fixture.

The fixture is a historical observation set, not a replay of the current
engine. This module reads only committed Parquet/JSON files. It never
unpickles or executes any file, and never calls an extractor that needs a
private workspace cache.

Every load verifies:

* the SHA-256 of each file against the manifest below, so a silent edit
  cannot change the pinned figures;
* exact column names and dtypes (schema version 1), including UTC-aware
  nanosecond timestamps and float64 values;
* strictly increasing UTC timestamps and the reported observation range;
* the recorded identity (strategy, symbol, timeframe) from ``run.json``,
  with bar and trade counts checked separately against module constants,
  and that every trade is closed with a known status.

A failed check raises ``ValueError`` explaining the reason and, where
applicable, naming the file. Unknown source settings stay unknown; they are
never guessed from equity or timestamps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

FIXTURE_DIR: Final = Path(__file__).parent / "fixtures" / "kd_btcusdt_1h_2024"

SCHEMA_VERSION: Final = 1
_EXPECTED_N_BARS: Final = 8761
_EXPECTED_N_TRADES: Final = 412
_FIRST: Final = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
_LAST: Final = pd.Timestamp("2024-12-31 00:00:00", tz="UTC")

# SHA-256 of the committed bytes. Rebuild procedure is in the fixture README;
# a checksum change requires a deliberate, reviewed fixture version bump.
_CHECKSUMS: Final[dict[str, str]] = {
    "equity.parquet": (
        "5f6c56c529de73a3a841ae30364ce4650c32224d64323389d1ebbff8f14d8f4a"
    ),
    "trades.parquet": (
        "045a0999f17972d1872c35d7ede56bb9c697f9d84f72a72a2a85ad5f83bda2e5"
    ),
    "run.json": "a9dab6e02d72da3243c38fd18514261463ddd645eaf9206b7008d91cb00ab82f",
}
_EQUITY_COLUMNS: Final = {"timestamp": "datetime64[ns, UTC]", "equity": "float64"}
_TRADE_COLUMNS: Final = {
    "size_quote": "float64",
    "entry_price": "float64",
    "exit_price": "float64",
    "net_pnl": "float64",
    "fee": "float64",
    "exit_notional": "float64",
    "status": "str",
}
_REQUIRED_RUN_KEYS: Final = (
    "init_cash",
    "fees",
    "slippage",
    "mode",
    "strategy",
    "symbol",
    "timeframe",
    "parameters",
    "source",
    "engine_source",
    "engine_metrics",
)


@dataclass(frozen=True)
class KDFixture:
    """Raw observations of the pinned KD backtest, as statistics read them.

    ``engine_metrics`` are the current engine run's self-reported figures
    (from ``engine_source``), kept for provenance only: they are never the
    truth for a recomputed statistic. Its ``max_drawdown`` is a positive
    magnitude. The older ``source`` pickle used superseded definitions
    (negative ``max_drawdown``, bars counted as days); those old metrics are
    not part of ``run.json``. Settings absent from the source JSON (notably
    the declared bar frequency and risk-free rate) remain unknown; the tests
    supply them explicitly as settings, never inferred from equity or gaps.
    """

    timestamps: pd.DatetimeIndex
    equity: npt.NDArray[np.float64]
    trades: tuple[dict[str, Any], ...]
    run: dict[str, Any]

    @property
    def init_cash(self) -> float:
        return float(self.run["init_cash"])

    @property
    def fee_rate(self) -> float:
        return float(self.run["fees"])

    @property
    def slippage_rate(self) -> float:
        return float(self.run["slippage"])


def load_kd_fixture(
    directory: Path | None = None, *, checksums: dict[str, str] | None = None
) -> KDFixture:
    """Load and validate the fixture.

    ``directory`` and ``checksums`` are for negative tests only; production
    callers use the committed directory and its pinned manifest.
    """
    root = FIXTURE_DIR if directory is None else directory
    manifest = _CHECKSUMS if checksums is None else checksums
    _verify_checksums(root, manifest)
    bars = _read_parquet(root / "equity.parquet", _EQUITY_COLUMNS)
    trades = _read_parquet(root / "trades.parquet", _TRADE_COLUMNS)
    run = _read_run(root / "run.json")

    timestamps = pd.DatetimeIndex(bars["timestamp"])
    if not (timestamps.is_monotonic_increasing and timestamps.is_unique):
        raise ValueError("equity.parquet timestamps must be strictly increasing")
    if len(bars) != _EXPECTED_N_BARS:
        raise ValueError(
            f"equity.parquet has {len(bars)} bars, expected {_EXPECTED_N_BARS}"
        )
    if timestamps[0] != _FIRST or timestamps[-1] != _LAST:
        raise ValueError(
            f"equity.parquet range is {timestamps[0]}..{timestamps[-1]}, "
            f"expected {_FIRST}..{_LAST}"
        )
    equity = bars["equity"].to_numpy(dtype=np.float64)
    if not np.isfinite(equity).all() or not (equity > 0).all():
        raise ValueError("equity.parquet must hold finite, strictly positive equity")

    if len(trades) != _EXPECTED_N_TRADES:
        raise ValueError(
            f"trades.parquet has {len(trades)} trades, expected {_EXPECTED_N_TRADES}"
        )
    statuses = set(trades["status"].tolist())
    if statuses != {"closed"}:
        raise ValueError(
            f"trades.parquet statuses must all be 'closed', got {statuses}"
        )

    return KDFixture(
        timestamps=timestamps,
        equity=equity,
        trades=tuple(
            {str(key): value for key, value in record.items()}
            for record in trades.to_dict("records")
        ),
        run=run,
    )


def _verify_checksums(root: Path, manifest: dict[str, str]) -> None:
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"fixture file missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"fixture checksum mismatch: {name} (got {digest})")


def _read_parquet(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    actual = {name: str(dtype) for name, dtype in frame.dtypes.items()}
    if actual != columns:
        raise ValueError(
            f"{path.name} schema mismatch: got {actual}, expected {columns}"
        )
    return frame


def _read_run(path: Path) -> dict[str, Any]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError("run.json must hold a JSON object")
    missing = [key for key in _REQUIRED_RUN_KEYS if key not in run]
    if missing:
        raise ValueError(f"run.json missing keys: {missing}")
    for key in ("init_cash", "fees", "slippage"):
        value = run[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"run.json {key} must be a number")
    if (run["strategy"], run["symbol"], run["timeframe"]) != (
        "kd_crossover",
        "BTC/USDT",
        "1h",
    ):
        raise ValueError("run.json identity does not match KD BTC/USDT 1h 2024")
    return run
