"""Strategy contract templates for generated strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from tradingdev.domain.backtest.schemas import BacktestRunConfig

if TYPE_CHECKING:
    from pathlib import Path


type StrategyContractTopic = Literal["core", "example", "configuration"]

_CONTRACT_SECTIONS = (
    {
        "id": "core",
        "description": "Required Python/signal rules and revision lifecycle.",
    },
    {"id": "example", "description": "Complete Python and YAML example to adapt."},
    {
        "id": "configuration",
        "description": "Full execution YAML schema and metric settings.",
    },
)


def strategy_contract_payload(
    package_root: Path, section: StrategyContractTopic = "core"
) -> dict[str, Any]:
    """Read one complete section; advertise the others without injecting them."""
    content: dict[str, Any]
    if section == "core":
        content = {
            "base_strategy_source": (
                package_root / "domain" / "strategies" / "base.py"
            ).read_text(encoding="utf-8"),
            "requirements": [
                "Inherit BaseStrategy. generate_signals(df) returns a new DataFrame "
                "without mutating its input, with signal in {-1, 0, 1}: short, flat, "
                "long. Preserve aligned rows. Never use future rows or backfill "
                "indicators; keep signals flat until required values are finite.",
                "Expose configurable values as constructor keyword parameters and "
                "get_parameters(). YAML strategy.parameters supplies them. Defaults "
                "must be finite JSON values. backtest_engine and parallel_config "
                "are reserved application injections, not YAML overrides.",
                "Read section=configuration for the executable YAML schema and "
                "section=example when a complete starting example is needed. "
                "Use list_data_sources for source-specific market settings.",
                "Generated Python must pass the repository Ruff and strict Mypy "
                "policy. Allowed import roots: __future__, collections, dataclasses, "
                "datetime, enum, math, statistics, typing, typing_extensions, numpy, "
                "pandas, talib, tradingdev. No network, subprocess, destructive "
                "filesystem access, dynamic imports, eval or exec. These checks "
                "and workspace paths are not a security sandbox.",
                "Prefer tradingdev.domain.indicators: sma, ema, rsi, macd, "
                "bollinger_bands, atr, adx, stochastic. It preserves TA-Lib warm-up, "
                "NaNs and index alignment. Direct talib requires float64 NumPy "
                "arrays; unpack MACD as line/signal/histogram, BBANDS as "
                "upper/middle/lower, STOCH as k/d. pandas_ta is unsupported. "
                "Separate first-party tradingdev imports from third-party imports.",
                "Set random_seed at the YAML root, never in backtest. Use "
                "tradingdev.domain.randomness.get_random() or get_numpy_rng() "
                "inside the constructor, fit or generate_signals; get_seed() "
                "supplies a third-party random_state. Do not call these at module "
                "import. Global RNGs are not seeded; null uses independent entropy.",
            ],
            "lifecycle": [
                "save_strategy creates an immutable source/base-config revision. "
                "Use its revision_id for checks and execution; omitted revision_id "
                "selects current once per operation. Saving repairs creates a new "
                "draft; older validation evidence does not transfer.",
                "validate_strategy runs static, import, Ruff, Mypy and short-fixture "
                "signal/config checks. dry_run_strategy requires validated status "
                "and repeats the contract on a longer fixture. Inspect diagnostics, "
                "signal_analysis and effective_config against the user's request; "
                "passing checks is not proof of every natural-language requirement.",
                "Execution requires runnable or promoted status. Each execution "
                "fixes effective constructor parameters and checks them again; "
                "parameter experiments reuse the runnable revision rather than "
                "rewriting its source or base validation evidence. Save a revision "
                "only when source or base configuration changes.",
                "Each validation, dry run and optimization trial/test starts a "
                "fresh run-local random stream. A backtest or walk-forward run "
                "shares one stream across constructor and folds. Contexts do not "
                "seed arbitrary global or third-party RNGs.",
            ],
        }
    elif section == "example":
        content = {
            "example_strategy_code": _EXAMPLE_CODE,
            "example_yaml_config": _EXAMPLE_YAML,
        }
    elif section == "configuration":
        content = {
            "config_schema": BacktestRunConfig.model_json_schema(),
            "notes": [
                "Fixed configuration fields reject unknown names; "
                "strategy.parameters and bundled strategy.fit are strategy-specific. "
                "Use validation training/test windows for training-required "
                "strategies. Capabilities describe requirements, not fitted state.",
                "Set backtest.periods_per_year for observed UTC daily returns: "
                "365 for continuously traded crypto, an appropriate calendar for "
                "other markets. It is independent of bar timeframe. risk_free_rate "
                "and required_return are annual decimal rates. Missing "
                "annualization makes annualized metrics unavailable.",
                "Performance uses QuantStats return/risk and VectorBT closed-trade "
                "statistics. Volume mode has no capital return; use total_pnl and "
                "max_drawdown_amount. Query saved definitions and unavailable "
                "reasons through get_metric_catalog/get_run_metrics; null is not zero.",
            ],
        }
    else:
        raise ValueError(f"Unknown strategy contract section: {section!r}")
    return {
        "section": section,
        "available_sections": [dict(item) for item in _CONTRACT_SECTIONS],
        "content": content,
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
