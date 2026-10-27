"""Prepare an execution and check its program contract under a worker supervisor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.execution_requests import (
    BacktestInput,
    ComparisonInput,
    OptimizationInput,
    WalkForwardInput,
)
from tradingdev.app.data_service import DataService
from tradingdev.app.job_store import JobStore
from tradingdev.app.preflight_service import PreflightError
from tradingdev.app.prepared_execution import PreparedComparison, PreparedExecution
from tradingdev.app.research_preparation import ResearchPreparationService
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import (
    ComparisonCandidateReceipt,
    ComparisonPreflightReceipt,
    PreflightCheck,
    PreflightPayload,
    PreflightReceipt,
    PreflightRequest,
    SignalCheck,
)
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionModeError,
)
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from tradingdev.app.prepared_execution import PreparedSubmission


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
    prepared: PreparedSubmission
    try:
        preparation = ResearchPreparationService(
            strategy_service=strategies,
            strategy_loader=loader,
            data_service=data,
        )
        if request.kind == "comparison":
            prepared = preparation.prepare_comparison(
                ComparisonInput.model_validate(request.arguments)
            )
        else:
            arguments = (
                OptimizationInput.model_validate(request.arguments)
                if request.kind == "optimization"
                else WalkForwardInput.model_validate(request.arguments)
                if request.kind == "walk_forward"
                else BacktestInput.model_validate(request.arguments)
            )
            prepared = preparation.prepare(request.kind, arguments)
    except StrategyExecutionModeError:
        raise
    except StrategyNotExecutableError as exc:
        raise PreflightError("strategy_not_executable", str(exc)) from exc
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        code = (
            "invalid_optimization_request"
            if request.kind == "optimization"
            else "invalid_execution_request"
        )
        raise PreflightError(code, str(exc)) from exc
    if isinstance(prepared, PreparedComparison):
        children = [
            ComparisonCandidateReceipt(
                candidate_id=candidate.candidate_id,
                receipt=_execution_receipt(
                    PreparedExecution(
                        candidate.manifest,
                        prepared.original_config_path,
                        prepared.candidate_fixture_rows[candidate.candidate_id],
                    ),
                    strategies,
                    started,
                    comparison_candidate=True,
                ),
            )
            for candidate in prepared.manifest.candidates
        ]
        decoded = ComparisonManifest.model_validate_json(
            prepared.manifest.model_dump_json()
        )
        decoded.verify(expected_hash=prepared.manifest.manifest_hash)
        return PreflightPayload(
            manifest=prepared.manifest,
            original_config_path=str(prepared.original_config_path),
            receipt=ComparisonPreflightReceipt(
                manifest_hash=prepared.manifest.manifest_hash,
                elapsed_seconds=time.monotonic() - started,
                candidates=children,
            ),
        )
    return PreflightPayload(
        manifest=prepared.manifest,
        original_config_path=str(prepared.original_config_path),
        receipt=_execution_receipt(prepared, strategies, started),
    )


def _execution_receipt(
    prepared: PreparedExecution,
    strategies: StrategyService,
    started: float,
    *,
    comparison_candidate: bool = False,
) -> PreflightReceipt:
    """Serialize one exact child and retain evidence from its actual signal checks."""
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
    if comparison_candidate or original.strategy_execution.kind == "generated":
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
    return PreflightReceipt(
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
