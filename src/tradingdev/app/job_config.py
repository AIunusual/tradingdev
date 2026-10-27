"""Materialize the exact configuration used by a background job."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.strategies.schemas import StrategyMetadata

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
    from tradingdev.domain.strategies.loader import StrategyLoader
    from tradingdev.domain.strategies.schemas import StrategySpec


def _read_revision_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        msg = f"Cannot read strategy revision file {path}: {exc}"
        raise StrategyNotExecutableError(msg) from exc


def bind_strategy_revision(
    config: dict[str, Any],
    spec: StrategySpec,
) -> None:
    """Check execution identity and bind its trusted source digest in place."""
    strategy = config.get("strategy")
    if not isinstance(strategy, dict) or strategy.get("id") != spec.strategy_id:
        msg = "Config strategy.id does not match the selected strategy"
        raise StrategyNotExecutableError(msg)
    if strategy.get("revision_id") != spec.revision_id:
        msg = "Config revision_id does not match the selected strategy revision"
        raise StrategyNotExecutableError(msg)
    source = strategy.get("source_path")
    declared_source = (
        spec.metadata.get("declared_source_path")
        if spec.kind == "bundled" and isinstance(spec.metadata, dict)
        else None
    )
    # Bundled YAML may name its package-relative source. Its catalog owns the
    # mapping to the installed module; a caller's working directory does not.
    matches_declared_source = (
        isinstance(declared_source, str) and source == declared_source
    )
    if spec.kind == "generated" and not source:
        msg = "Generated strategy config requires its revision source_path"
        raise StrategyNotExecutableError(msg)
    if (
        source
        and not matches_declared_source
        and Path(str(source)).expanduser().resolve()
        != Path(spec.source_path).expanduser().resolve()
    ):
        msg = "Config source_path does not match the selected strategy revision"
        raise StrategyNotExecutableError(msg)
    class_name = strategy.get("class_name")
    if class_name is not None and class_name != spec.class_name:
        msg = "Config class_name does not match the selected strategy revision"
        raise StrategyNotExecutableError(msg)
    if spec.kind == "generated":
        if not spec.revision_id or not isinstance(spec.metadata, StrategyMetadata):
            msg = "Generated strategy requires a verified revision"
            raise StrategyNotExecutableError(msg)
        if class_name != spec.class_name:
            msg = "Generated strategy config requires its revision class_name"
            raise StrategyNotExecutableError(msg)
        base_content = _read_revision_file(Path(spec.config_path))
        if hashlib.sha256(base_content).hexdigest() != spec.metadata.config_hash:
            msg = "Strategy revision config hash changed before execution"
            raise StrategyNotExecutableError(msg)
        base_strategy = yaml.safe_load(base_content)["strategy"]
        ignored = {"source_hash", "source_path", "parameters"}
        actual = {key: value for key, value in strategy.items() if key not in ignored}
        expected = {
            key: value for key, value in base_strategy.items() if key not in ignored
        }
        if actual != expected:
            msg = "Strategy settings do not match the validated revision config"
            raise StrategyNotExecutableError(msg)
        expected_hash = spec.metadata.source_hash
    else:
        source_path = Path(spec.source_path).expanduser().resolve()
        expected_hash = hashlib.sha256(_read_revision_file(source_path)).hexdigest()
        strategy["source_path"] = str(source_path)
        strategy["class_name"] = spec.class_name
    supplied_hash = strategy.get("source_hash")
    if supplied_hash is not None and supplied_hash != expected_hash:
        msg = "Strategy source hash does not match the execution specification"
        raise StrategyNotExecutableError(msg)
    strategy["source_hash"] = expected_hash


def apply_run_overrides(
    raw_config: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Apply request-level market and date choices without editing the strategy."""
    effective_config = deepcopy(raw_config)
    # Programmatic overrides do not carry the research intake's field origins.
    effective_config.pop("research", None)
    backtest = effective_config.get("backtest")
    if not isinstance(backtest, dict):
        msg = "backtest config must be a mapping"
        raise ValueError(msg)
    backtest.update(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    data = effective_config.get("data")
    if isinstance(data, dict):
        requirements = data.get("requirements")
        if isinstance(requirements, dict):
            market = requirements.get("market")
            if isinstance(market, dict):
                market.update(symbol=symbol, timeframe=timeframe)
    return effective_config


def apply_backtest_overrides(
    config: dict[str, Any], overrides: dict[str, Any] | None
) -> None:
    """Apply explicit execution settings without changing revision declarations."""
    if overrides is None:
        return
    allowed = BacktestConfig.model_fields.keys() - {
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
    }
    unknown = overrides.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown backtest overrides: {', '.join(sorted(unknown))}")
    config.pop("research", None)
    config["backtest"].update(deepcopy(overrides))
    BacktestConfig.model_validate(config["backtest"])


def validate_optimization_grid(
    loader: StrategyLoader,
    manifest: ExecutionManifest,
    optimization: OptimizationSpec,
) -> None:
    """Bind every explicit grid candidate using the captured strategy contract."""
    names = list(optimization.param_ranges)
    candidates = (optimization.param_ranges[name] for name in names)
    combinations = (
        dict(zip(names, combination, strict=True))
        for combination in product(*candidates)
    )
    try:
        loader.validate_parameter_grid(
            manifest.config_copy()["strategy"],
            manifest.strategy_execution,
            combinations,
        )
    except Exception as exc:
        raise ManifestError(f"Invalid strategy optimization settings: {exc}") from exc
