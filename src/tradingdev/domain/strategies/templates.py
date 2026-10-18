"""Strategy contract templates for generated strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.backtest.schemas import BacktestRunConfig

if TYPE_CHECKING:
    from pathlib import Path


def strategy_contract_payload(package_root: Path) -> dict[str, Any]:
    """Return source and YAML templates for generated strategies."""
    base_path = package_root / "domain" / "strategies" / "base.py"
    base_source = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
    return {
        "base_strategy_source": base_source,
        "example_strategy_code": _EXAMPLE_CODE,
        "example_yaml_config": _EXAMPLE_YAML,
        "config_schema": BacktestRunConfig.model_json_schema(),
        "api_reference": (
            "Generated strategies must inherit BaseStrategy, return a DataFrame "
            "with signal values limited to -1, 0, and 1, and avoid mutating the "
            "input DataFrame. Express configurable settings as constructor "
            "keyword arguments with finite JSON values, including any defaults. "
            "Execution captures these values at submission. backtest_engine and "
            "parallel_config are reserved application injections and cannot be "
            "overridden in strategy.parameters. Use tradingdev.domain.indicators "
            "for indicators. config_schema describes every execution configuration "
            "section: unknown fixed fields are errors, while strategy.parameters "
            "and bundled strategy.fit contain strategy-specific settings. Set "
            "random_seed at the YAML root only, never in backtest. Each validation, "
            "dry-run, normal run, optimization trial and test evaluation starts "
            "isolated Python/NumPy generators with that seed; walk-forward folds "
            "consume a single stream in fold order. Randomized strategies use "
            "tradingdev.domain.randomness.get_random() or get_numpy_rng() inside "
            "their constructor, fit, or generate_signals. Use get_seed() when a "
            "third-party model requires an explicit random_state/seed. Do not read "
            "these functions at module import time. Global random/NumPy RNG state "
            "and arbitrary third-party RNGs are not seeded; null uses independent "
            "entropy. Separate model parameters remain explicit model settings. "
            "The available indicators include "
            "(sma, ema, rsi, "
            "macd, bollinger_bands, atr, adx, stochastic) for standard "
            "indicators and tradingdev.shared.utils.logger for logging. "
            "The indicator facade uses TA-Lib and preserves its warm-up and "
            "NaN behavior; keep signals flat until all required values are "
            "finite, and never backfill indicators from future rows. "
            "Performance uses QuantStats daily return/risk statistics and "
            "VectorBT closed-trade statistics. Set backtest.periods_per_year "
            "explicitly for annualized daily-return metrics (365 for this crypto "
            "example; choose the appropriate trading calendar for other markets). "
            "Without it, annualized metrics are unavailable. Volume mode has no "
            "capital return; use total_pnl and max_drawdown_amount. "
            "Run and job responses show summaries, not all computed metrics. "
            "Use get_metric_catalog to discover definitions and optimization "
            "directions, then get_run_metrics(run_id, metric_ids, scope) to read "
            "saved values, settings, and unavailable reasons. Use the returned "
            "available_scopes for fold/trial results. Missing summary fields "
            "do not mean the backend did not compute them. "
            "talib may be imported directly: pass float64 NumPy arrays to its "
            "Function API, unpack multi-output tuples (MACD: line, signal, "
            "histogram; BBANDS: upper, middle, lower; STOCH: k, d), and align "
            "returned arrays with the input index. Allowed imports are "
            "restricted to a small Python/pandas/numpy/talib/tradingdev "
            "allowlist; pandas_ta is no longer supported. Ruff treats "
            "tradingdev as first-party: separate its imports from third-party "
            "numpy, pandas, and talib imports with a blank line."
        ),
        "lifecycle": (
            "save_strategy creates an immutable source/config revision and returns "
            "revision_id. Pass that revision_id to validate_strategy, "
            "dry_run_strategy, promote_strategy, get_strategy, and execution tools. "
            "Saving a repair creates a new draft revision; earlier revisions and "
            "their validation evidence remain available. Omitting revision_id "
            "selects current once per operation. validate_strategy runs static checks, "
            "restricted import checks, ruff, mypy, inheritance checks, and the "
            "shared signal-contract gate on a short fixture with structured "
            "diagnostics; both checks validate the full execution config before "
            "constructing the strategy and return effective_config with resolved "
            "backtest/parallel defaults and the canonical run seed. Data directory "
            "defaults resolve at job submission. Check this response against the "
            "user's requested settings before proceeding. dry_run_strategy "
            "accepts only validated strategies and "
            "re-runs the same signal-contract gate on a longer fixture, returns "
            "signal_analysis, and marks the strategy runnable; promote_strategy "
            "marks a runnable strategy promoted. validate_strategy and "
            "dry_run_strategy currently execute generated Python code; sandboxed "
            "execution isolation is required future work. Execution accepts only "
            "runnable or promoted strategies and enforces this gate at execution "
            "time on every entry point (MCP jobs, CLI, workers)."
        ),
    }


_EXAMPLE_CODE = '''\
"""Example: Simple moving-average crossover strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class SmaCrossoverStrategy(BaseStrategy):
    """Buy when fast SMA crosses above slow SMA, sell on reverse."""

    def __init__(
        self,
        backtest_engine: BaseBacktestEngine | None = None,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> None:
        self._engine = backtest_engine
        self._fast_period = fast_period
        self._slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = indicators.sma(result["close"], self._fast_period)
        slow = indicators.sma(result["close"], self._slow_period)
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)
        ready = (
            np.isfinite(fast)
            & np.isfinite(slow)
            & np.isfinite(fast_prev)
            & np.isfinite(slow_prev)
        )

        result["signal"] = 0
        result.loc[ready & (fast > slow) & (fast_prev <= slow_prev), "signal"] = 1
        result.loc[ready & (fast < slow) & (fast_prev >= slow_prev), "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
        }
'''

_EXAMPLE_YAML = """\
random_seed: 42

strategy:
  id: "sma_crossover"
  version: "0.1.0"
  class_name: "SmaCrossoverStrategy"
  description: "Simple moving-average crossover"
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  init_cash: 10000.0
  periods_per_year: 365.0
  risk_free_rate: 0.0
  required_return: 0.0
  fees: 0.0006
  slippage: 0.0005
  mode: "signal"

data:
  # market_type: "futures/um"  # binance_vision only; use "spot" for spot
  requirements:
    market:
      # source selects the data crawler: "binance_vision" (default),
      # "binance_api" (ccxt), or "yahoo_finance" (stocks/futures/indices/FX,
      # Yahoo symbols such as "AAPL", "ES=F", "^GSPC")
      source: "binance_vision"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""
