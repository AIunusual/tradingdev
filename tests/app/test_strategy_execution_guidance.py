"""Training requirements reject unsupported preparation before data or jobs."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.preflight_service import PreflightError, PreflightService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.strategies.capabilities import StrategyExecutionModeError
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def execution_root(tmp_path: Path) -> Iterator[Path]:
    with TemporaryDirectory(prefix="execution-guidance-", dir=tmp_path) as directory:
        root = Path(directory)
        yield root
    assert not root.exists()


def _arguments(strategy_id: str, kind: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "strategy_id": strategy_id,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if kind == "optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_pnl",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")
    return arguments


def _unexpected_execution(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Unsupported modes must be rejected before data or execution")


def _training_guidance(strategy_id: str = "quantile_volume") -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "capabilities": {
            "training_requirement": "required",
            "supported_execution_modes": ["walk_forward"],
        },
        "recommended_mode": "walk_forward",
        "required_configuration": ["validation"],
    }


@pytest.mark.parametrize(
    "strategy_id", ["quantile_volume", "xgboost_direction", "safety_first_volume"]
)
@pytest.mark.parametrize("kind", ["backtest", "optimization"])
def test_training_requirement_precedes_config_mode_errors_and_job_creation(
    execution_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_id: str,
    kind: str,
) -> None:
    store = JobStore(workspace=WorkspacePaths(execution_root / "workspace"))
    monkeypatch.setattr(DataService, "execution_config", _unexpected_execution)
    monkeypatch.setattr(DataService, "load", _unexpected_execution)
    monkeypatch.setattr(ProcessRunner, "spawn_module", _unexpected_execution)
    arguments = _arguments(strategy_id, kind)
    result = (
        OptimizationService(job_store=store).start_optimization(**arguments)
        if kind == "optimization"
        else JobService(job_store=store).start_backtest(**arguments)
    )
    assert result["code"] == "strategy_requires_training"
    assert result["job_id"] == ""
    guidance = result["execution_guidance"]
    assert guidance["strategy_id"] == strategy_id
    assert guidance["capabilities"]["supported_execution_modes"] == ["walk_forward"]
    assert guidance["recommended_mode"] == "walk_forward"
    assert guidance["required_configuration"] == ["validation"]
    assert store.list_all_jobs() == []
    assert store.list_runs() == []


@pytest.mark.parametrize("keep_validation", [False, True])
def test_caller_config_cannot_override_registered_training_requirement(
    execution_root: Path, monkeypatch: pytest.MonkeyPatch, keep_validation: bool
) -> None:
    entry = BundledStrategyCatalog().get("quantile_volume")
    assert entry is not None
    config = load_config(entry.config_path)
    config["strategy"]["training_requirement"] = "optional"
    if not keep_validation:
        config.pop("validation")
    workspace = WorkspacePaths(execution_root / "workspace")
    service = BacktestService(
        strategy_gate=StrategyService(workspace), data_service=DataService(workspace)
    )
    monkeypatch.setattr(DataService, "execution_config", _unexpected_execution)
    with pytest.raises(StrategyExecutionModeError) as caught:
        service.prepare_execution(config, kind="backtest")
    assert caught.value.guidance.capabilities.training_requirement == "required"


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_optional_training_retains_supported_preparation_modes(
    execution_root: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(execution_root / "workspace"))
    monkeypatch.setattr(DataService, "load", _unexpected_execution)
    monkeypatch.setattr(ProcessRunner, "spawn_module", _unexpected_execution)
    arguments = _arguments("kd_crossover", kind)
    if kind == "optimization":
        result = OptimizationService(job_store=store).prepare_optimization(**arguments)
    else:
        jobs = JobService(job_store=store)
        prepare = (
            jobs.prepare_walk_forward
            if kind == "walk_forward"
            else jobs.prepare_backtest
        )
        result = prepare(**arguments)
    assert isinstance(result, PreparedExecution), result
    assert result.manifest.kind == kind
    assert store.list_all_jobs() == []


def test_required_training_allows_walk_forward_preparation(
    execution_root: Path,
) -> None:
    store = JobStore(workspace=WorkspacePaths(execution_root / "workspace"))
    prepared = JobService(job_store=store).prepare_walk_forward(
        **_arguments("quantile_volume", "walk_forward")
    )
    assert isinstance(prepared, PreparedExecution), prepared
    assert prepared.manifest.kind == "walk_forward"
    assert store.list_all_jobs() == []


@pytest.mark.parametrize(
    "failure_fields",
    [
        pytest.param({"execution_guidance": "unvalidated text"}, id="not_object"),
        pytest.param(
            {"execution_guidance": {"strategy_id": "quantile_volume"}}, id="incomplete"
        ),
        pytest.param(
            {"execution_guidance": {"next_tool": "unsafe"}}, id="wrong_fields"
        ),
        pytest.param({}, id="missing_guidance"),
        pytest.param({"execution_guidance": None}, id="null_guidance"),
        pytest.param(
            {"execution_guidance": _training_guidance("xgboost_direction")},
            id="wrong_strategy",
        ),
        pytest.param(
            {"code": "preflight_failed", "execution_guidance": _training_guidance()},
            id="wrong_code",
        ),
    ],
)
def test_preflight_rejects_malformed_execution_guidance_and_cleans_files(
    execution_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_fields: dict[str, Any],
) -> None:
    workspace = WorkspacePaths(execution_root / "workspace")
    temporary = execution_root / "preflight"
    temporary.mkdir()
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kwargs: str(temporary)
    )

    def run(
        _runner: ProcessRunner, _module: str, *arguments: str, timeout_seconds: float
    ) -> None:
        Path(arguments[1]).write_text(
            json.dumps(
                {
                    "success": False,
                    "code": "strategy_requires_training",
                    "message": "training required",
                    **failure_fields,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ProcessRunner, "run_module", run)
    request = PreflightRequest(
        kind="backtest",
        arguments=_arguments("quantile_volume", "backtest"),
        minimum_history_bars=20,
        sample_bars=128,
    )
    with pytest.raises(PreflightError) as caught:
        PreflightService(workspace).prepare(request)
    assert caught.value.code == "invalid_preflight_response"
    assert caught.value.execution_guidance is None
    assert not temporary.exists()
