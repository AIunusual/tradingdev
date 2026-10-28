"""Preparation services preserve failures before any plan or worker is published."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


class _Preparation:
    """Dispatch to real preparation services without pretending to be MCP wire I/O."""

    def __init__(self, jobs: JobService, optimizer: OptimizationService) -> None:
        self.jobs = jobs
        self.optimizer = optimizer

    def call(
        self, tool: str, arguments: dict[str, Any]
    ) -> PreparedExecution | dict[str, Any]:
        if tool == "prepare_optimization":
            return self.optimizer.prepare_optimization(**arguments)
        if tool == "prepare_walk_forward":
            return self.jobs.prepare_walk_forward(**arguments)
        assert tool == "prepare_backtest"
        return self.jobs.prepare_backtest(**arguments)


def _prepare_arguments(tool: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "prepare_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    elif tool == "prepare_walk_forward":
        arguments.update(start_date="2024-01-01", end_date="2026-01-01")
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")
    return arguments


def _broken_config(path: Path, failure: str) -> Any:
    if failure == "malformed_yaml":
        return yaml.safe_load("backtest: [")
    if failure == "missing_file":
        raise FileNotFoundError("Config disappeared during submission")
    if failure == "unreadable_file":
        raise PermissionError("Config is unreadable during submission")
    if failure == "invalid_encoding":
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "Invalid config encoding")
    if failure == "empty_yaml":
        return None
    if failure == "list_yaml":
        return []
    config = load_config(path)
    if failure == "missing_backtest":
        config.pop("backtest")
    elif failure == "invalid_schema":
        config["parallel"] = {"reserve_cores": "not-an-integer"}
    else:
        raise AssertionError(f"Unknown config failure: {failure}")
    return config


@pytest.mark.parametrize(
    "tool", ["prepare_backtest", "prepare_walk_forward", "prepare_optimization"]
)
@pytest.mark.parametrize(
    "failure", ["malformed_yaml", "invalid_encoding", "unreadable_file", "missing_file"]
)
def test_real_catalog_load_failures_remain_structured_at_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, failure: str
) -> None:
    bundled = tmp_path / "bundled"
    config_path = bundled / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(
        b"\xff" if failure == "invalid_encoding" else b"strategy: ["
    )
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    gate = StrategyService(store.workspace)
    gate._catalog = BundledStrategyCatalog(bundled)
    jobs = JobService(job_store=store, strategy_service=gate)
    service = _Preparation(
        jobs, OptimizationService(job_store=store, strategy_service=gate)
    )
    if failure in {"unreadable_file", "missing_file"}:
        read_text = Path.read_text

        def failing_read(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == config_path:
                if failure == "unreadable_file":
                    raise PermissionError("Bundled config cannot be read")
                raise FileNotFoundError("Bundled config disappeared after discovery")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", failing_read)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Catalog loading failure must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments = {**_prepare_arguments(tool), "strategy_id": "fixture"}
    result = service.call(tool, arguments)
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert payload["code"] == (
        "invalid_optimization_request"
        if tool == "prepare_optimization"
        else "invalid_execution_request"
    )
    assert payload["message"]
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    "tool", ["prepare_backtest", "prepare_walk_forward", "prepare_optimization"]
)
def test_real_catalog_missing_strategy_retains_its_distinct_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    gate = StrategyService(store.workspace)
    gate._catalog = BundledStrategyCatalog(tmp_path / "empty-bundled")
    jobs = JobService(job_store=store, strategy_service=gate)
    service = _Preparation(
        jobs, OptimizationService(job_store=store, strategy_service=gate)
    )

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Missing strategy must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    result = service.call(tool, _prepare_arguments(tool))
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert payload["code"] == "strategy_not_executable"
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    "tool", ["prepare_backtest", "prepare_walk_forward", "prepare_optimization"]
)
@pytest.mark.parametrize(
    "failure",
    [
        "malformed_yaml",
        "missing_file",
        "unreadable_file",
        "invalid_encoding",
        "empty_yaml",
        "list_yaml",
        "missing_backtest",
    ],
)
def test_primary_config_load_errors_are_structured_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, failure: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    jobs = JobService(job_store=store)
    service = _Preparation(jobs, OptimizationService(job_store=store))
    module = "optimization_service" if tool == "prepare_optimization" else "job_service"
    monkeypatch.setattr(
        f"tradingdev.app.{module}.load_config",
        lambda path: _broken_config(path, failure),
    )

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Invalid configuration must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    result = service.call(tool, _prepare_arguments(tool))
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert payload["code"] == (
        "invalid_optimization_request"
        if tool == "prepare_optimization"
        else "invalid_execution_request"
    )
    assert payload["message"]
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    "failure",
    [
        "malformed_yaml",
        "missing_file",
        "unreadable_file",
        "invalid_encoding",
        "empty_yaml",
        "list_yaml",
        "invalid_schema",
    ],
)
def test_fallback_config_load_errors_are_structured_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    service = _Preparation(
        JobService(job_store=store), OptimizationService(job_store=store)
    )
    reads: list[str] = []

    def read(path: Path) -> Any:
        reads.append(path.name)
        if path.name == "walkforward_config.yaml":
            return _broken_config(path, failure)
        return load_config(path)

    monkeypatch.setattr("tradingdev.app.job_service.load_config", read)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Invalid fallback configuration must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    result = service.call(
        "prepare_walk_forward", _prepare_arguments("prepare_walk_forward")
    )
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert payload["code"] == "invalid_execution_request"
    assert reads == ["config.yaml", "walkforward_config.yaml"]
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize("tool", ["prepare_backtest", "prepare_walk_forward"])
def test_valid_configuration_with_wrong_mode_remains_a_mode_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    service = _Preparation(
        JobService(job_store=store), OptimizationService(job_store=store)
    )

    def read(path: Path) -> dict[str, Any]:
        config = load_config(path)
        if tool == "prepare_backtest":
            config["validation"] = {"mode": "automatic"}
        else:
            config.pop("validation", None)
        return config

    monkeypatch.setattr("tradingdev.app.job_service.load_config", read)
    if tool in {"prepare_backtest", "prepare_walk_forward"}:

        def unexpected_spawn(*_args: object) -> None:
            pytest.fail("Wrong run mode must not start a worker")

        monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    result = service.call(tool, _prepare_arguments(tool))
    assert isinstance(result, dict)
    assert result["code"] == "invalid_run_mode"
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize("tool", ["prepare_backtest", "prepare_walk_forward"])
def test_mode_selection_and_submission_use_the_same_config_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    service = _Preparation(
        JobService(job_store=store), OptimizationService(job_store=store)
    )
    reads: list[Path] = []

    def read(path: Path) -> dict[str, Any]:
        assert path not in reads, "Submission reread the selected configuration"
        reads.append(path)
        config = load_config(path)
        config["random_seed"] = 77
        return config

    monkeypatch.setattr("tradingdev.app.job_service.load_config", read)
    monkeypatch.setattr(
        ProcessRunner,
        "spawn_module",
        lambda *_args: WorkerHandle(1234, 100.0, "a" * 32),
    )
    result = service.call(tool, _prepare_arguments(tool))
    assert isinstance(result, PreparedExecution)
    manifest = result.manifest
    assert store.list_all_jobs() == []
    assert manifest.config["random_seed"] == 77
    assert len(reads) == (2 if tool == "prepare_walk_forward" else 1)


@pytest.mark.parametrize(
    "tool", ["prepare_backtest", "prepare_walk_forward", "prepare_optimization"]
)
@pytest.mark.parametrize("invalid", ["nonfinite", "invalid_parallel"])
def test_invalid_execution_config_is_rejected_before_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, invalid: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    job_service = JobService(job_store=store)
    service = _Preparation(job_service, OptimizationService(job_store=store))

    def invalid_config(path: Path) -> dict[str, Any]:
        config = load_config(path)
        if invalid == "nonfinite":
            config["backtest"]["fees"] = float("nan")
        else:
            config["parallel"] = {"reserve_cores": "not-an-integer"}
        if tool == "prepare_walk_forward":
            config["validation"] = {"mode": "automatic"}
        return config

    module = "optimization_service" if tool == "prepare_optimization" else "job_service"
    monkeypatch.setattr(f"tradingdev.app.{module}.load_config", invalid_config)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Rejected execution settings must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "prepare_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    elif tool == "prepare_walk_forward":
        arguments.update(start_date="2024-01-01", end_date="2026-01-01")
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")
    result = service.call(tool, arguments)
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    expected = (
        "invalid_optimization_request"
        if tool == "prepare_optimization"
        else "invalid_execution_request"
    )
    assert payload["code"] == expected
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    "tool", ["prepare_backtest", "prepare_walk_forward", "prepare_optimization"]
)
@pytest.mark.parametrize(
    "failure",
    ["source_mismatch", "stale_revision", "missing_source", "unreadable_source"],
)
def test_strategy_binding_failure_is_structured_before_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, failure: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    gate = StrategyService(store.workspace)
    jobs = JobService(job_store=store, strategy_service=gate)
    optimizer = OptimizationService(job_store=store, strategy_service=gate)
    service = _Preparation(jobs, optimizer)
    original_resolve = gate.resolve_executable
    source_path = Path(original_resolve("kd_crossover").source_path)
    original_read_bytes = Path.read_bytes
    resolutions: list[str] = []
    resolved = False

    def resolve(strategy_id: str, revision_id: str | None = None) -> StrategySpec:
        nonlocal resolved
        resolutions.append(strategy_id)
        if failure == "stale_revision" and len(resolutions) == 2:
            # Simulate a revision invalidated after the request selected it.
            raise StrategyNotExecutableError("Revision changed during submission")
        spec = original_resolve(strategy_id, revision_id)
        resolved = True
        return spec

    def read_bytes(path: Path) -> bytes:
        if resolved and path == source_path:
            if failure == "missing_source":
                raise FileNotFoundError("Source removed during submission")
            if failure == "unreadable_source":
                raise PermissionError("Source inaccessible during submission")
        return original_read_bytes(path)

    def config(path: Path) -> dict[str, Any]:
        raw = load_config(path)
        if failure == "source_mismatch":
            raw["strategy"]["source_path"] = str(tmp_path / "other.py")
        return raw

    monkeypatch.setattr(gate, "resolve_executable", resolve)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    module = "optimization_service" if tool == "prepare_optimization" else "job_service"
    monkeypatch.setattr(f"tradingdev.app.{module}.load_config", config)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Rejected strategy binding must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "prepare_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    elif tool == "prepare_walk_forward":
        arguments.update(start_date="2024-01-01", end_date="2026-01-01")
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")

    result = service.call(tool, arguments)

    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert payload["code"] == (
        "invalid_optimization_request"
        if tool == "prepare_optimization"
        else "invalid_execution_request"
    )
    if failure == "stale_revision":
        assert len(resolutions) == 2
        assert "Revision changed during submission" in payload["message"]
    elif failure == "source_mismatch":
        assert "source_path does not match" in payload["message"]
    else:
        assert "Cannot read strategy revision file" in payload["message"]
        assert "during submission" in payload["message"]
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    ("tool", "loader_method"),
    [
        ("prepare_backtest", "resolve_execution"),
        ("prepare_walk_forward", "resolve_execution"),
        ("prepare_optimization", "resolve_execution"),
        ("prepare_optimization", "validate_parameter_grid"),
    ],
)
@pytest.mark.parametrize("failure_type", [ImportError, RuntimeError, OSError])
def test_strategy_loading_failure_is_structured_before_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    loader_method: str,
    failure_type: type[Exception],
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    job_service = JobService(job_store=store)
    service = _Preparation(job_service, OptimizationService(job_store=store))

    def fail_loading(*_args: object) -> None:
        raise failure_type("Strategy module cannot be loaded at submission")

    monkeypatch.setattr(StrategyLoader, loader_method, fail_loading)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Rejected strategy loading must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "prepare_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    elif tool == "prepare_walk_forward":
        arguments.update(start_date="2024-01-01", end_date="2026-01-01")
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")

    result = service.call(tool, arguments)
    assert isinstance(result, dict)
    payload = result
    assert payload["job_id"] == ""
    assert "Strategy module cannot be loaded at submission" in payload["message"]
    assert payload["code"] == (
        "invalid_optimization_request"
        if tool == "prepare_optimization"
        else "invalid_execution_request"
    )
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())
