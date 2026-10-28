"""Offline readiness checks and bounded worker cleanup precede any real run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tests.integration.execution_fixtures import research_arguments
from tests.preflight_fixtures import make_preflight_fixture

from tradingdev.adapters.execution.preflight_worker import execute_preflight
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.preflight_service import PreflightError, PreflightService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import PreflightReceipt, PreflightRequest
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.strategies.base import BaseStrategy


def _unexpected_execution(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Offline preparation cannot acquire data, fit or run engines")


def _forbid_historical_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    for method in ("load", "load_sample", "load_cached", "ensure"):
        monkeypatch.setattr(DataService, method, _unexpected_execution)
    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", _unexpected_execution
    )
    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.registry.create_crawler", _unexpected_execution
    )
    monkeypatch.setattr(BacktestService, "create_engine", _unexpected_execution)
    monkeypatch.setattr(BacktestService, "run_manifest", _unexpected_execution)
    monkeypatch.setattr(WalkForwardValidator, "run_fold", _unexpected_execution)
    monkeypatch.setattr(WalkForwardValidator, "validate", _unexpected_execution)


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_preflight_checks_effective_program_without_market_training_or_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch, kind=kind)
    assert not list(workspace.processed_data.iterdir())
    request.arguments["parameters"] = {"threshold": 0.001}
    research = request.arguments["research"]
    assert isinstance(research, dict)
    research.update(fees=0.002, slippage=0.003)
    request.arguments["backtest_overrides"] = {"position_size": 500.0}
    if kind == "optimization":
        request.arguments["param_ranges"] = {"threshold": [0.001, 0.02]}
    _forbid_historical_execution(monkeypatch)
    observed: list[tuple[int, float]] = []
    create = StrategyLoader.create_from_execution

    def capture(loader: StrategyLoader, *args: Any, **kwargs: Any) -> BaseStrategy:
        strategy = create(loader, *args, **kwargs)
        generate = strategy.generate_signals
        monkeypatch.setattr(strategy, "fit", _unexpected_execution)

        def checked(frame: pd.DataFrame) -> pd.DataFrame:
            observed.append((len(frame), strategy.get_parameters()["threshold"]))
            return generate(frame)

        monkeypatch.setattr(strategy, "generate_signals", checked)
        return strategy

    monkeypatch.setattr(StrategyLoader, "create_from_execution", capture)
    before = {
        str(path.relative_to(workspace.root)): path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "preflight"
    output.mkdir()
    result = execute_preflight(request, source_workspace=workspace, directory=output)
    assert isinstance(result.manifest, ExecutionManifest)
    assert isinstance(result.receipt, PreflightReceipt)
    assert observed == [(80, 0.001), (240, 0.001)]
    assert result.receipt.synthetic_fixture_rows == [rows for rows, _ in observed]
    assert result.receipt.signal_check == "synthetic_contract_passed"
    assert result.receipt.manifest_hash == result.manifest.manifest_hash
    assert result.manifest.config_copy()["backtest"]["fees"] == 0.002
    assert result.manifest.config_copy()["strategy"]["parameters"]["threshold"] == 0.001
    assert set(result.receipt.not_checked) == {
        "historical_data",
        "backtest_engine",
        "training",
        "performance",
    }
    assert result.receipt.parameter_combinations_checked == (
        2 if kind == "optimization" else None
    )
    assert not list(output.rglob("performance-*.json"))
    assert not list(output.rglob("*.pkl"))
    assert list(workspace.runs.iterdir()) == []
    after = {
        str(path.relative_to(workspace.root)): path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    ("strategy_id", "kind", "expected_signal"),
    [
        ("quantile_volume", "walk_forward", "deferred_until_training"),
        ("xgboost_direction", "walk_forward", "deferred_until_training"),
        ("safety_first_volume", "walk_forward", "deferred_until_training"),
        ("kd_crossover", "backtest", "bundled_not_executed"),
    ],
)
def test_bundled_preparation_does_not_construct_fit_or_score_untrained_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_id: str,
    kind: Any,
    expected_signal: str,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    _forbid_historical_execution(monkeypatch)
    monkeypatch.setattr(StrategyLoader, "create_from_execution", _unexpected_execution)
    request = PreflightRequest(
        kind=kind,
        arguments=research_arguments(
            {
                "strategy_id": strategy_id,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
            }
        ),
    )
    if kind == "walk_forward":
        request.arguments["validation_config"] = {
            "mode": "automatic",
            "n_splits": 2,
            "train_ratio": 0.5,
        }
    result = execute_preflight(
        request, source_workspace=workspace, directory=tmp_path / "preflight"
    )
    assert isinstance(result.receipt, PreflightReceipt)
    assert result.receipt.signal_check == expected_signal
    assert result.receipt.synthetic_fixture_rows == []
    assert "signal_contract" not in result.receipt.checked_paths
    assert "training" in result.receipt.not_checked


@pytest.mark.parametrize("failure", ["exception", "invalid_signal", "input_mutated"])
def test_preflight_rejects_invalid_generated_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path, monkeypatch, failure="exception" if failure == "exception" else None
    )
    create = StrategyLoader.create_from_execution

    def invalid(loader: StrategyLoader, *args: Any, **kwargs: Any) -> BaseStrategy:
        strategy = create(loader, *args, **kwargs)
        generate = strategy.generate_signals

        def signals(frame: pd.DataFrame) -> pd.DataFrame:
            result = generate(frame)
            if failure == "invalid_signal":
                result["signal"] = 2
            elif failure == "input_mutated":
                frame["close"] = 0.0
            return result

        monkeypatch.setattr(strategy, "generate_signals", signals)
        return strategy

    monkeypatch.setattr(StrategyLoader, "create_from_execution", invalid)
    _forbid_historical_execution(monkeypatch)
    with pytest.raises(PreflightError, match="contract"):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )
    assert list(workspace.runs.iterdir()) == []


@pytest.mark.parametrize(
    "invalid", [{"label_horizon": 0}, {"label_horizon": -1}, {"embargo_bars": -1}]
)
def test_invalid_purged_configuration_is_rejected_without_splitting_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: dict[str, int]
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path,
        monkeypatch,
        kind="walk_forward",
    )
    request.arguments["validation_config"] = {
        "mode": "purged",
        "label_horizon": 5,
        "n_splits": 2,
        **invalid,
    }
    _forbid_historical_execution(monkeypatch)
    with pytest.raises(PreflightError, match="label_horizon|embargo"):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )


def test_preflight_requires_actual_fixture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(BacktestService, "check_execution", lambda *_a, **_k: ())
    with pytest.raises(ValueError, match="both contract fixtures"):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )


def test_preflight_requires_manifest_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)

    def reject_serialization(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("manifest serialization failed")

    monkeypatch.setattr(ExecutionManifest, "model_dump_json", reject_serialization)
    with pytest.raises(ValueError, match="manifest serialization"):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )


@pytest.mark.parametrize(
    "outcome", ["success", "exception", "timeout", "invalid_response"]
)
def test_parent_cleans_temporary_files_after_verified_worker_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    directory = tmp_path / "temporary"
    directory.mkdir()
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kwargs: str(directory)
    )

    def run(
        _runner: ProcessRunner, module: str, *args: str, timeout_seconds: float
    ) -> None:
        assert timeout_seconds == 60
        assert json.loads(Path(args[0]).read_text())["source_workspace"] == str(
            workspace.root
        )
        if outcome == "timeout":
            raise BoundedWorkerError(
                "preflight_timeout", "timeout", cleanup_verified=True
            )
        if outcome == "exception":
            Path(args[1]).write_text(
                json.dumps(
                    {"success": False, "code": "sample_error", "message": "bad sample"}
                )
            )
        elif outcome == "invalid_response":
            Path(args[1]).write_text("invalid json")
        else:
            result = execute_preflight(
                request, source_workspace=workspace, directory=directory
            )
            Path(args[1]).write_text(
                json.dumps({"success": True, "result": result.model_dump(mode="json")})
            )

    monkeypatch.setattr(ProcessRunner, "run_module", run)
    if outcome == "success":
        receipt = PreflightService(workspace).prepare(request).receipt
        assert isinstance(receipt, PreflightReceipt)
        assert receipt.synthetic_fixture_rows == [80, 240]
    else:
        with pytest.raises(PreflightError):
            PreflightService(workspace).prepare(request)
    assert not directory.exists()


def test_parent_retains_files_when_worker_cleanup_cannot_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "retained"
    directory.mkdir()
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kwargs: str(directory)
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise BoundedWorkerError(
            "preflight_cleanup_failed",
            "owned child is still alive",
            cleanup_verified=False,
        )

    monkeypatch.setattr(ProcessRunner, "run_module", fail)
    request = PreflightRequest(kind="backtest", arguments={})
    with pytest.raises(PreflightError, match="retained") as caught:
        PreflightService().prepare(request)
    assert caught.value.code == "preflight_cleanup_failed"
    assert directory.is_dir()
    assert (directory / "request.json").is_file()
    assert (directory / "tmp").is_dir()
