"""Prepare an execution and check its program contract under a worker supervisor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.preflight_service import PreflightError
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import (
    PreflightCheck,
    PreflightPayload,
    PreflightReceipt,
    PreflightRequest,
    SignalCheck,
)
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionGuidance,
    StrategyExecutionModeError,
)
from tradingdev.domain.strategies.loader import StrategyLoader


def execute_preflight(
    request: PreflightRequest,
    *,
    source_workspace: WorkspacePaths,
    directory: Path,
) -> PreflightPayload:
    """Capture settings and offline evidence without market, engine or fit calls."""
    started = time.monotonic()
    temporary_workspace = WorkspacePaths(directory / "runtime")
    store = JobStore(workspace=temporary_workspace)
    strategies = StrategyService(source_workspace, store=store.store)
    loader = StrategyLoader(workspace_root=source_workspace.root)
    # Preparation resolves data paths and source declarations only. It never loads
    # cached bars, calls a crawler, or checks whether historical prices exist.
    data = DataService(source_workspace)
    kwargs: dict[str, Any] = dict(request.arguments)
    if request.kind == "optimization":
        prepared = OptimizationService(
            strategy_service=strategies,
            strategy_loader=loader,
            data_service=data,
            job_store=store,
        ).prepare_optimization(**kwargs)
    else:
        jobs = JobService(
            strategy_service=strategies,
            strategy_loader=loader,
            data_service=data,
            job_store=store,
        )
        prepare = (
            jobs.prepare_walk_forward
            if request.kind == "walk_forward"
            else jobs.prepare_backtest
        )
        prepared = prepare(**kwargs)
    if not isinstance(prepared, PreparedExecution):
        guidance = prepared.get("execution_guidance")
        raise PreflightError(
            str(prepared.get("code", "invalid_execution_request")),
            str(prepared.get("message", "Preparation was rejected")),
            execution_guidance=(
                None
                if guidance is None
                else StrategyExecutionGuidance.model_validate(guidance)
            ),
        )
    original = prepared.manifest
    config = original.config_for_execution()
    selected = strategies.resolve_executable(
        config["strategy"]["id"], config["strategy"].get("revision_id")
    )
    training = selected.execution_capabilities.training_requirement
    checks: list[PreflightCheck] = [
        "configuration",
        "execution_capabilities",
        "parameter_binding",
    ]
    signal_check: SignalCheck
    if original.strategy_execution.kind == "generated":
        checks.append("signal_contract")
        signal_check = "synthetic_contract_passed"
    elif training == "required":
        signal_check = "deferred_until_training"
    else:
        signal_check = "bundled_not_executed"
    combinations = None
    if original.optimization is not None:
        checks.append("candidate_binding")
        combinations = original.optimization.total_combinations
    decoded = ExecutionManifest.model_validate_json(original.model_dump_json())
    decoded.verify(expected_hash=original.manifest_hash)
    checks.append("manifest_serialization")
    receipt = PreflightReceipt(
        manifest_hash=original.manifest_hash,
        elapsed_seconds=time.monotonic() - started,
        checked_paths=checks,
        synthetic_fixture_rows=list(prepared.synthetic_fixture_rows),
        signal_check=signal_check,
        training_requirement=training,
        parameter_combinations_checked=combinations,
        not_checked=[
            "historical_data",
            "backtest_engine",
            "training",
            "performance",
        ],
    )
    return PreflightPayload(
        manifest=original,
        original_config_path=str(prepared.original_config_path),
        receipt=receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("result_path", type=Path)
    arguments = parser.parse_args()
    try:
        envelope = json.loads(arguments.request_path.read_text(encoding="utf-8"))
        request = PreflightRequest.model_validate(envelope["request"])
        result = execute_preflight(
            request,
            source_workspace=WorkspacePaths(Path(envelope["source_workspace"])),
            directory=arguments.request_path.parent,
        )
        response: dict[str, Any] = {
            "success": True,
            "result": result.model_dump(mode="json"),
        }
    except Exception as error:
        response = {
            "success": False,
            "code": getattr(error, "code", "preflight_failed"),
            "message": str(error),
        }
        guidance = (
            error.execution_guidance
            if isinstance(error, PreflightError)
            else error.guidance
            if isinstance(error, StrategyExecutionModeError)
            else None
        )
        if guidance is not None:
            response["execution_guidance"] = guidance.model_dump(mode="json")
    arguments.result_path.write_text(
        json.dumps(response, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
