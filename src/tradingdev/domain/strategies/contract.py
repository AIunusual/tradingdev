"""Runtime signal-contract checks shared by validate and dry-run phases."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from pydantic import ValidationError

from tradingdev.domain.backtest.schemas import BacktestRunConfig, ParallelConfig
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.validator import diagnostic

if TYPE_CHECKING:
    from tradingdev.domain.strategies.execution import StrategyExecution
    from tradingdev.domain.strategies.schemas import (
        StrategyDiagnostic,
        StrategyMetadata,
    )

VALIDATE_FIXTURE_ROWS = 80
DRY_RUN_FIXTURE_ROWS = 240


class SignalContractChecker:
    """Execute a strategy on synthetic data and verify the signal contract.

    The contract is one gate used at two depths: validation runs a short
    fixture, dry-run a longer one. Both must agree on the rules checked here
    (return type, no input mutation, signal column limited to -1/0/1).
    """

    def __init__(self, loader: StrategyLoader | None = None) -> None:
        self._loader = loader or StrategyLoader()

    def check(
        self,
        metadata: StrategyMetadata,
        *,
        fixture_rows: int,
    ) -> dict[str, Any]:
        """Return contract diagnostics and signal analysis for a strategy."""
        try:
            config_path = Path(metadata.config_path)
            config_bytes = config_path.read_bytes()
            if hashlib.sha256(config_bytes).hexdigest() != metadata.config_hash:
                msg = "Strategy revision configuration has changed"
                raise ValueError(msg)
            raw = yaml.safe_load(config_bytes)
            if not isinstance(raw, dict):
                msg = "YAML config must be a mapping"
                raise ValueError(msg)
            strategy_cfg = raw.get("strategy")
            if not isinstance(strategy_cfg, dict):
                msg = "strategy config must be a mapping"
                raise ValueError(msg)
            for field, expected in (
                ("id", metadata.strategy_id),
                ("revision_id", metadata.revision_id),
                ("class_name", metadata.class_name),
                ("source_path", metadata.source_path),
            ):
                if strategy_cfg.get(field) != expected:
                    msg = f"strategy.{field} does not match saved strategy metadata"
                    raise ValueError(msg)
            strategy_cfg["source_hash"] = metadata.source_hash
            return self.check_config(raw, fixture_rows=fixture_rows)
        except Exception as exc:  # noqa: BLE001
            return {
                "diagnostics": [
                    _contract_diagnostic(
                        code="contract_execution_error",
                        message=str(exc),
                        fix="Fix the saved strategy identity or configuration.",
                    )
                ],
                "effective_config": None,
            }

    def check_config(
        self,
        raw_config: dict[str, Any],
        *,
        fixture_rows: int,
        execution: StrategyExecution | None = None,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check effective execution settings without altering revision evidence.

        A captured execution uses exactly its frozen constructor settings. Search
        candidates apply the same overrides used by the real strategy loader.
        """
        diagnostics: list[StrategyDiagnostic] = []
        effective_config: dict[str, Any] | None = None
        try:
            raw = deepcopy(raw_config)
            try:
                run_config = BacktestRunConfig.model_validate(raw)
            except ValidationError as exc:
                diagnostics.append(
                    diagnostic(
                        code="invalid_execution_config",
                        phase="config",
                        message=str(exc),
                        fix=(
                            'Follow get_strategy_contract(section="configuration")'
                            ".content.config_schema. Use finite "
                            "JSON values, string object keys, and no cycles. Put the "
                            "run random_seed at the YAML root and correct unknown "
                            "fields; backtest.random_seed is not supported."
                        ),
                    )
                )
                return {"diagnostics": diagnostics, "effective_config": None}
            effective_config = run_config.model_dump(mode="json")
            effective_config["parallel"] = (
                run_config.parallel or ParallelConfig()
            ).model_dump(mode="json")
            with execution_randomness(run_config.random_seed):
                if execution is None:
                    strategy = self._loader.create_from_config(
                        effective_config, engine=None
                    )
                else:
                    selected = execution
                    if parameter_overrides:
                        selected = self._loader.override_execution(
                            effective_config["strategy"], execution, parameter_overrides
                        )
                    strategy = self._loader.create_from_execution(
                        effective_config["strategy"],
                        selected,
                        None,
                        run_config.parallel or ParallelConfig(),
                    )
                if not isinstance(strategy, BaseStrategy):
                    diagnostics.append(
                        _contract_diagnostic(
                            code="base_strategy_inheritance",
                            message="strategy must inherit BaseStrategy",
                            fix="Make the generated class inherit from BaseStrategy.",
                        )
                    )
                    return {
                        "diagnostics": diagnostics,
                        "effective_config": effective_config,
                    }
                df = _fixture_df(fixture_rows)
                before = df.copy(deep=True)
                result = strategy.generate_signals(df)
            if not isinstance(result, pd.DataFrame):
                diagnostics.append(
                    _contract_diagnostic(
                        code="signals_not_dataframe",
                        message="generate_signals must return DataFrame",
                        fix="Return the copied DataFrame with a signal column.",
                    )
                )
                return {
                    "diagnostics": diagnostics,
                    "effective_config": effective_config,
                }
            if not df.equals(before):
                diagnostics.append(
                    _contract_diagnostic(
                        code="input_mutated",
                        message="generate_signals must not mutate input",
                        fix="Start with result = df.copy() and mutate result only.",
                    )
                )
            if "signal" not in result.columns:
                diagnostics.append(
                    _contract_diagnostic(
                        code="missing_signal_column",
                        message="result must include signal column",
                        fix="Add result['signal'] with values -1, 0, or 1.",
                    )
                )
                return {
                    "diagnostics": diagnostics,
                    "effective_config": effective_config,
                }
            signals = result["signal"]
            values = set(signals.dropna().unique())
            if not values.issubset({-1, 0, 1}):
                diagnostics.append(
                    _contract_diagnostic(
                        code="invalid_signal_values",
                        message="signal values must be limited to -1, 0, and 1",
                        fix=(
                            "Map all generated signals to the project convention: "
                            "-1, 0, 1."
                        ),
                    )
                )
            return {
                "diagnostics": diagnostics,
                "signal_analysis": _signal_analysis(result),
                "effective_config": effective_config,
            }
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                _contract_diagnostic(
                    code="contract_execution_error",
                    message=str(exc),
                    fix="Fix the class name, constructor, imports, or signal logic.",
                )
            )
            return {"diagnostics": diagnostics, "effective_config": effective_config}


def _contract_diagnostic(*, code: str, message: str, fix: str) -> StrategyDiagnostic:
    return diagnostic(code=code, phase="contract", message=message, fix=fix)


def _fixture_df(rows: int) -> pd.DataFrame:
    close: list[float] = []
    price = 100.0
    for i in range(rows):
        if rows <= VALIDATE_FIXTURE_ROWS:
            price += 0.1
        elif i < rows // 3:
            price += 0.12
        elif i < (rows * 2) // 3:
            price -= 0.08
        else:
            price += 0.18 if i % 2 == 0 else -0.11
        close.append(price)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": [value - 0.05 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000.0 + (i % 24) * 10.0 for i in range(rows)],
        }
    )


def _signal_analysis(result: pd.DataFrame) -> dict[str, Any]:
    signals = result["signal"]
    distribution = {
        str(key): int(value)
        for key, value in signals.value_counts(dropna=False).to_dict().items()
    }
    transitions = int(signals.fillna(0).ne(signals.fillna(0).shift()).sum() - 1)
    active = int(signals.isin([-1, 1]).sum())
    return {
        "rows": int(len(result)),
        "signal_distribution": distribution,
        "nan_count": int(signals.isna().sum()),
        "transition_count": max(transitions, 0),
        "active_signal_ratio": active / max(len(result), 1),
        "first_timestamp": (
            str(result["timestamp"].iloc[0])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
        "last_timestamp": (
            str(result["timestamp"].iloc[-1])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
    }
