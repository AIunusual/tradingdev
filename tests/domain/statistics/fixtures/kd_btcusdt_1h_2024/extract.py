"""Rebuild this fixture from the two local KD pickles it was extracted from.

Run: uv run python tests/domain/statistics/fixtures/kd_btcusdt_1h_2024/extract.py

The pickles live in the unversioned workspace cache; see README.md for what
is taken from each and why. The script refuses to write unless both pickles
have their recorded SHA-256 and agree on every field they share.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CACHE = ROOT / "workspace" / "data" / "processed" / "cache"
SOURCE = CACHE / "cd03bf19459b48af.pkl"
SOURCE_SHA256 = "b9493b6bf5f1f171fee472812a9c6f7aab7a58f5d93fc6644386211d5b6febd0"
ENGINE = CACHE / "cli_5ed5dc0b0579cc80_95a98854a8984f82be488c4bff48902d.pkl"
ENGINE_SHA256 = "af4fc3905570f44f318dc900baa2b754af0892b32829da8f45b80cf1466aacb7"
SHARED_TRADE_FIELDS = ("size_quote", "entry_price", "exit_price", "net_pnl")
ENGINE_TRADE_FIELDS = ("fee", "exit_notional", "status")


def _load(path: Path, digest: str) -> Any:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"{path} has SHA-256 {actual}, expected {digest}")
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - local trusted cache


def main() -> None:
    source = _load(SOURCE, SOURCE_SHA256).backtest_result
    engine_pipeline = _load(ENGINE, ENGINE_SHA256)
    engine = engine_pipeline.backtest_result

    equity = np.asarray(source.equity_curve, dtype=np.float64)
    timestamps = pd.DatetimeIndex(source.timestamps).as_unit("ns")
    if not np.array_equal(equity, np.asarray(engine.equity_curve, dtype=np.float64)):
        raise SystemExit("equity curves differ between the two pickles")
    if not timestamps.equals(pd.DatetimeIndex(engine.timestamps)):
        raise SystemExit("timestamps differ between the two pickles")
    if len(source.trades) != len(engine.trades):
        raise SystemExit("trade counts differ between the two pickles")
    for old, new in zip(source.trades, engine.trades, strict=True):
        if any(old[key] != new[key] for key in SHARED_TRADE_FIELDS):
            raise SystemExit("trade records differ between the two pickles")

    bars = pa.Table.from_pandas(
        pd.DataFrame({"timestamp": timestamps, "equity": equity}),
        preserve_index=False,
    )
    pq.write_table(
        bars,
        HERE / "equity.parquet",
        compression="zstd",
        use_dictionary=["equity"],
        column_encoding={"timestamp": "DELTA_BINARY_PACKED"},
    )
    trades = pd.DataFrame(
        [
            {key: trade[key] for key in SHARED_TRADE_FIELDS + ENGINE_TRADE_FIELDS}
            for trade in engine.trades
        ]
    )
    pq.write_table(
        pa.Table.from_pandas(trades, preserve_index=False),
        HERE / "trades.parquet",
        compression="zstd",
    )

    backtest = engine_pipeline.config_snapshot["backtest"]
    run = {
        "source": {"pickle": SOURCE.name, "sha256": SOURCE_SHA256},
        "engine_source": {"pickle": ENGINE.name, "sha256": ENGINE_SHA256},
        "strategy": engine_pipeline.config_snapshot["strategy"]["id"],
        "parameters": engine_pipeline.config_snapshot["strategy"]["parameters"],
        "symbol": backtest["symbol"],
        "timeframe": backtest["timeframe"],
        "mode": engine.mode,
        "init_cash": engine.init_cash,
        "fees": backtest["fees"],
        "slippage": backtest["slippage"],
        "engine_metrics": dict(engine.metrics),
    }
    (HERE / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
