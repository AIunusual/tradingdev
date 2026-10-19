"""Isolated strategy and market fixtures for preflight boundary tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.preflight import PreflightRequest

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


_CODE = """from __future__ import annotations
from typing import Any
import pandas as pd
from tradingdev.domain.strategies.base import BaseStrategy

class SampleStrategy(BaseStrategy):
    def __init__(
        self, threshold: float = 0.0, backtest_engine: object | None = None
    ) -> None:
        self.threshold = threshold
        self.engine = backtest_engine
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        moves = result["close"].pct_change().fillna(0)
        result["signal"] = 0
        result.loc[moves > self.threshold, "signal"] = 1
        result.loc[moves < -self.threshold, "signal"] = -1
        return result
    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("empty training frame")
    def get_parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold}
"""


def make_preflight_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str = "backtest",
    threshold: float = 0.0,
    failure: str | None = None,
    validation: dict[str, Any] | None = None,
) -> tuple[WorkspacePaths, PreflightRequest]:
    workspace = WorkspacePaths(root / "source")
    strategies = StrategyService(workspace)
    monkeypatch.setattr(strategies, "_quality_gate_diagnostics", lambda _path: [])
    code = _CODE
    if failure is not None:
        statement = (
            'raise RuntimeError("sample failure")'
            if failure == "exception"
            else "while True: pass"
        )
        code = code.replace(
            "        result = df.copy()",
            "        if self.engine is not None:\n"
            f"            {statement}\n        result = df.copy()",
        )
    config: dict[str, Any] = {
        "strategy": {
            "id": "sample_strategy",
            "class_name": "SampleStrategy",
            "parameters": {"threshold": threshold},
        },
        "random_seed": 42,
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "mode": "volume",
            "position_size": 1000.0,
        },
    }
    if kind == "walk_forward":
        config["validation"] = validation or {"n_splits": 2, "train_ratio": 0.5}
    saved = strategies.save_draft("sample_strategy", code, yaml.safe_dump(config))
    assert saved.success
    assert strategies.validate("sample_strategy")["success"]
    assert strategies.dry_run("sample_strategy")["success"]
    rows = 768
    prices = 500 + np.sin(np.arange(rows) / 4)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.full(rows, 1000.0),
        }
    )
    frame.to_parquet(workspace.processed_data / "btcusdt_1h_2024.parquet", index=False)
    arguments: dict[str, Any] = {
        "strategy_id": "sample_strategy",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "revision_id": saved.revision_id,
    }
    if kind == "optimization":
        arguments.update(
            param_ranges={"threshold": [threshold, threshold + 0.01]},
            optimization_metric="total_pnl",
            train_start="2024-01-01",
            train_end="2024-01-14",
            test_start="2024-01-15",
            test_end="2024-01-31",
        )
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-02-01")
    return workspace, PreflightRequest.model_validate(
        {
            "kind": kind,
            "arguments": arguments,
            "minimum_history_bars": 8,
            "sample_bars": 128,
        }
    )
