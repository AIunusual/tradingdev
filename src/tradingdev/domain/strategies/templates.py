"""Strategy contract templates for generated strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.indicators.reference import indicator_reference
from tradingdev.domain.research import ResearchSpec
from tradingdev.domain.research_policy import research_policy_payload
from tradingdev.domain.strategies.research_contract import research_evidence_contract
from tradingdev.domain.strategies.validator import static_policy_payload

if TYPE_CHECKING:
    from pathlib import Path


type StrategyContractTopic = Literal[
    "core", "research", "evidence", "example", "configuration", "indicators"
]

_CONTRACT_SECTIONS = (
    {
        "id": "core",
        "description": "Required Python/signal rules and revision lifecycle.",
    },
    {
        "id": "research",
        "description": "Required research choices, defaults and explicit inheritance.",
    },
    {
        "id": "evidence",
        "description": "Parameter proposals, source claims and saved trading formulas.",
    },
    {
        "id": "indicators",
        "description": "Installed indicator catalog and selected API/source reference.",
    },
    {"id": "example", "description": "Complete Python and YAML example to adapt."},
    {
        "id": "configuration",
        "description": "Full execution YAML schema and metric settings.",
    },
)


def strategy_contract_payload(
    package_root: Path,
    section: StrategyContractTopic = "core",
    *,
    indicator: str | None = None,
) -> dict[str, Any]:
    """Read one complete section; advertise the others without injecting them."""
    if indicator is not None and section != "indicators":
        raise ValueError("indicator is only valid with section=indicators")
    content: dict[str, Any]
    if section == "core":
        content = {
            "static_policy": static_policy_payload(),
            "base_strategy_source": (
                package_root / "domain" / "strategies" / "base.py"
            ).read_text(encoding="utf-8"),
            "requirements": [
                "Inherit BaseStrategy. generate_signals(df) returns a new DataFrame "
                "without mutating its input, with signal in {-1, 0, 1}. Follow "
                "BaseStrategy's mode-dependent position/zero semantics and "
                "next-bar timing; -1 means short, never sell-only. Preserve every "
                "row, the exact index and timestamp/OHLCV values. Missing signals "
                "are rejected on fixtures and actual execution data. Never use "
                "future rows or backfill indicators; emit zero "
                "until required values are finite.",
                "Expose configurable values as constructor keyword parameters and "
                "get_parameters(). YAML strategy.parameters supplies them. Defaults "
                "must be finite JSON values. backtest_engine and parallel_config "
                "are reserved application injections, not YAML overrides.",
                "Read section=research for the required research input and maintained "
                "simulation policy, section=configuration for the YAML schema and "
                "section=example when a complete starting example is needed. "
                "Use list_data_sources for source-specific market settings.",
                "Generated Python must pass the repository Ruff and strict Mypy "
                "policy. static_policy lists the enforced import/call restrictions. "
                "No network, subprocess, destructive filesystem access, dynamic "
                "imports, eval or exec. These checks and workspace paths are not "
                "a security sandbox.",
                "Prefer tradingdev.domain.indicators for technical indicators. "
                "Read section=indicators for the installed function catalog, then "
                "select indicator by name for its signature, named result fields, "
                "implementation and parameter/input checks. Resolve unavailable "
                "references before coding; do not guess APIs or tuple orders. "
                "Separate first-party tradingdev imports from third-party imports.",
                "Set random_seed at the YAML root, never in backtest. Use "
                "tradingdev.domain.randomness.get_random() or get_numpy_rng() "
                "inside the constructor, fit or generate_signals; get_seed() "
                "supplies a third-party random_state. Do not call these at module "
                "import. Global RNGs are not seeded; null uses independent entropy.",
            ],
            "lifecycle": [
                "save_strategy creates an immutable source/base-config revision. "
                "Omit strategy_id for automatic naming; an explicit ID without "
                "expected_revision_id means new creation and must not replace an "
                "existing strategy. Repairs require the exact expected_revision_id. "
                "Use its revision_id for checks and execution; omitted revision_id "
                "selects current once per operation. Saving repairs creates a new "
                "draft; older validation evidence does not transfer.",
                "validate_strategy runs static, import, Ruff, Mypy and short-fixture "
                "signal/config checks. dry_run_strategy requires validated status "
                "and repeats the contract on a longer fixture. Inspect diagnostics, "
                "signal_analysis and effective_config against the user's request; "
                "passing checks is not proof of every natural-language requirement.",
                "Runtime import, construction and signal checks use a supervised "
                "worker with a 60-second budget. Abnormal exit, timeout or cleanup "
                "failure produces diagnostics and cannot grant passing evidence. "
                "Repair in a new revision; unverified process cleanup retains its "
                "temporary directory for inspection. Cleanup covers the owned "
                "process group, excluding descendants that leave its session. "
                "This is not a security sandbox.",
                "Execution requires runnable or promoted status. Each execution "
                "fixes effective constructor parameters and checks them again; "
                "parameter experiments reuse the runnable revision rather than "
                "rewriting its source or base validation evidence. Save a revision "
                "only when source or base configuration changes.",
                "prepare_comparison captures 1–8 explicitly proposed parameter "
                "sets for an ordinary backtest strategy. Include a label and "
                "selection rationale for each set. All candidates share strategy "
                "revision, research, costs and data; only constructor parameters "
                "vary. Every candidate receives synthetic checks before one whole-"
                "set confirmation. After approval, runs use one captured dataset "
                "with independent engines, strategy instances and random streams. "
                "Training-required and walk-forward comparisons are unsupported; "
                "use the existing training workflow. Fixed comparison does not "
                "authorize adaptive optimization or selection of only successes. "
                "Generate the complete report with comparison_job_id so failed, "
                "cancelled and unexecuted candidates remain visible.",
                "Each validation, dry run and optimization trial/test starts a "
                "fresh run-local random stream. A backtest or walk-forward run "
                "shares one stream across constructor and folds. Contexts do not "
                "seed arbitrary global or third-party RNGs.",
            ],
        }
    elif section == "indicators":
        content = indicator_reference(indicator)
    elif section == "research":
        content = {
            "research_schema": TypeAdapter(ResearchSpec).json_schema(),
            "policy": research_policy_payload(),
            "notes": [
                "Before authoring or preparing, obtain the user's strategy idea, "
                "instrument, date interval and candle timeframe. A vague trading "
                "idea is sufficient; missing choices require clarification, not "
                "values copied from examples or a bundled configuration.",
                "research owns market/date/capital/cost/annualization choices. "
                "The YAML supplies strategy class/parameters, explicit data source, "
                "features, execution controls and validation. When saving, duplicate "
                "research fields in YAML must agree; the backend materializes the "
                "complete immutable base configuration before program checks.",
                "Omit unspecified capital and costs to use the maintained simulation "
                "policy. Preserve the user's explicit valid values, including zero. "
                "Do not supply guessed values and describe them as user instructions.",
                "inherit_selected_revision is an explicit choice to reuse one "
                "identified generated revision; it is not a missing-value fallback. "
                "Supply that exact revision_id when preparing, or expected_revision_id "
                "when repairing. Bundled sample dates cannot supply missing choices.",
                "Research periods select candle timestamps with "
                "start_date <= timestamp <= end_date, including both endpoints "
                "when matching candles exist. Dates without timezone are interpreted "
                "as UTC; explicit offsets convert to UTC. Date-only values mean "
                "00:00:00 UTC, not the whole final day. Review these exact resolved "
                "bounds with the user. No history is read before approval, so actual "
                "available bars and gaps are still unverified.",
                "Use annualization='market' only when a supported market policy "
                "applies. Otherwise propose and disclose a justified positive daily "
                "factor, or obtain an explicit choice to proceed with annualization "
                "unavailable. Do not apply a universal stock-market day count.",
                "Source labels describe what the backend resolved from this request, "
                "the project policy or a selected revision. They do not establish "
                "that a human supplied or approved those choices.",
            ],
        }
    elif section == "evidence":
        content = research_evidence_contract()
    elif section == "example":
        content = {
            "example_strategy_code": _EXAMPLE_CODE,
            "example_yaml_config": _EXAMPLE_YAML,
        }
    elif section == "configuration":
        content = {
            "config_schema": BacktestRunConfig.model_json_schema(),
            "notes": [
                "This is the complete saved execution schema. For save_strategy, "
                "research supplies market/date/capital/cost/annualization fields; "
                "they may be omitted from the authored YAML. The backend adds "
                "research provenance and strategy identity. Do not manufacture them.",
                "Fixed configuration fields reject unknown names; "
                "strategy.parameters and bundled strategy.fit are strategy-specific. "
                "Use validation training/test windows for training-required "
                "strategies. Capabilities describe requirements, not fitted state.",
                "Automatic walk-forward uses train_ratio = initial train / "
                "(initial train + one test window), using the saved decimal ratio "
                "before integer rounding. "
                "It rounds initial training rows down and divides all remaining "
                "rows into consecutive test folds through the final loaded row; "
                "test sizes differ by at most one. Rolling keeps the initial "
                "training length; expanding keeps the first row. Empty train/test "
                "windows are rejected. Explicit and purged modes are unchanged.",
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
"""Example: Long-only moving-average position strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class SmaCrossoverStrategy(BaseStrategy):
    """Hold long while finite fast SMA is above slow SMA; otherwise stay flat."""

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
        ready = np.isfinite(fast) & np.isfinite(slow)

        result["signal"] = 0
        result.loc[ready & (fast > slow), "signal"] = 1
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
  description: "Long-only: hold while fast SMA exceeds slow SMA, otherwise flat"
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
  source: "binance_vision"
  market_type: "futures/um"  # Example market; use "spot" for an explicit spot request.
  requirements:
    market:
      # source selects the data crawler: "binance_vision",
      # "binance_api" (ccxt), or "yahoo_finance" (stocks/futures/indices/FX,
      # Yahoo symbols such as "AAPL", "ES=F", "^GSPC")
      source: "binance_vision"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""
