"""Sample readiness and parent cleanup evidence remain separate from real runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from tests.preflight_fixtures import make_preflight_fixture

from tradingdev.adapters.execution.preflight_worker import execute_preflight
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.preflight_service import PreflightError, PreflightService
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.validation.walk_forward import WalkForwardValidator

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import BacktestConfig


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_preflight_runs_sample_engine_and_serialization_without_changing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch, kind=kind)
    request.arguments["parameters"] = {"threshold": 0.001}
    request.arguments["backtest_overrides"] = {
        "fees": 0.002,
        "slippage": 0.003,
        "position_size": 500.0,
    }
    if kind == "optimization":
        request.arguments["param_ranges"] = {"threshold": [0.001, 0.02]}
    engine_configs: list[BacktestConfig] = []
    create_engine = BacktestService.create_engine

    def capture_engine(
        service: BacktestService, config: BacktestConfig
    ) -> BaseBacktestEngine:
        engine_configs.append(config)
        return create_engine(service, config)

    monkeypatch.setattr(BacktestService, "create_engine", capture_engine)
    before = {
        str(path.relative_to(workspace.root)): path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "preflight"
    output.mkdir()
    result = execute_preflight(request, source_workspace=workspace, directory=output)
    assert result.receipt.manifest_hash == result.manifest.manifest_hash
    assert result.manifest.config_copy()["backtest"]["start_date"].startswith(
        "2024-01-01"
    )
    assert result.receipt.sample_bars_used <= request.sample_bars
    assert result.manifest.config_copy()["strategy"]["parameters"]["threshold"] == 0.001
    assert engine_configs
    assert all(
        config.fees == 0.002
        and config.slippage == 0.003
        and config.position_size == 500.0
        for config in engine_configs
    )
    assert result.manifest.config_copy()["backtest"]["fees"] == 0.002
    assert {
        "configuration",
        "signal_contract",
        "signals",
        "engine",
        "serialization",
    }.issubset(result.receipt.checked_paths)
    assert list(output.glob("sample/performance-*.json"))
    assert list(output.glob("sample/pipeline-*.pkl"))
    if kind == "walk_forward":
        assert "fit" in result.receipt.checked_paths
        assert result.receipt.tested_fold_count == 1
        assert result.receipt.total_fold_count == 2
    elif kind == "optimization":
        assert "fit" not in result.receipt.checked_paths
        assert result.receipt.tested_candidates == 1
        assert result.receipt.total_candidates == 2
        assert [window.role for window in result.receipt.windows] == ["train", "test"]
    after = {
        str(path.relative_to(workspace.root)): path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_preflight_zero_trades_does_not_claim_trading_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch, threshold=10.0)
    directory = tmp_path / "preflight"
    directory.mkdir()
    result = execute_preflight(request, source_workspace=workspace, directory=directory)
    assert result.receipt.trade_count == 0
    assert result.receipt.trading_path_exercised is False
    assert "no_trades_observed" in result.receipt.warnings


@pytest.mark.parametrize(
    ("purged", "dates", "expected_windows", "total_folds"),
    [
        (False, {}, ((0, 32), (32, 64)), 2),
        (
            False,
            {
                "train_start": "2024-01-08",
                "train_end": "2024-01-10",
                "test_start": "2024-01-20",
                "test_end": "2024-01-22",
            },
            ((168, 232), (456, 520)),
            1,
        ),
        (
            False,
            {
                "train_start": "2024-01-02T03:00:00+08:00",
                "train_end": "2024-01-03T03:00:00+08:00",
                "test_start": "2024-01-04T03:00:00+08:00",
                "test_end": "2024-01-05T03:00:00+08:00",
            },
            ((0, 48), (48, 96)),
            1,
        ),
        (True, {}, ((0, 36), (43, 86)), 2),
        (True, {"train_start": "2024-01-08"}, ((0, 36), (43, 86)), 2),
        (
            True,
            {
                "train_start": "2024-01-08",
                "train_end": "2024-01-20",
                "test_start": "2024-01-20",
                "test_end": "2024-01-22",
            },
            ((0, 36), (43, 86)),
            2,
        ),
        (
            True,
            {
                "train_start": "2024-01-10",
                "train_end": "2024-01-08",
                "test_start": "2024-01-22",
                "test_end": "2024-01-20",
            },
            ((0, 36), (43, 86)),
            2,
        ),
    ],
    ids=[
        "plain_auto",
        "plain_dates",
        "plain_offset_dates",
        "purged_auto",
        "purged_partial_dates",
        "purged_overlapping_dates",
        "purged_reversed_dates",
    ],
)
def test_walk_forward_preflight_uses_selected_splitter_on_bounded_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purged: bool,
    dates: dict[str, str],
    expected_windows: tuple[tuple[int, int], tuple[int, int]],
    total_folds: int,
) -> None:
    validation: dict[str, Any] = {
        "n_splits": 2,
        "train_ratio": 0.5,
        "expanding": False,
        "label_horizon": 5,
        "embargo_bars": 2,
        "purged_split": purged,
        **dates,
    }
    workspace, request = make_preflight_fixture(
        tmp_path, monkeypatch, kind="walk_forward", validation=validation
    )

    def reject_full_validation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Preflight must remain a first-fold sample")

    monkeypatch.setattr(WalkForwardValidator, "validate", reject_full_validation)
    output = tmp_path / "preflight"
    output.mkdir()
    result = execute_preflight(request, source_workspace=workspace, directory=output)
    receipt = result.receipt
    assert receipt.tested_fold_count == 1
    assert receipt.total_fold_count == total_folds
    assert ("single_fold_only" in receipt.warnings) is (total_folds > 1)
    assert "partial_historical_sample" in receipt.warnings
    assert receipt.sample_bars_used == sum(
        end - start for start, end in expected_windows
    )
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    for window, (start, end) in zip(receipt.windows, expected_windows, strict=True):
        assert pd.Timestamp(window.start) == origin + pd.Timedelta(hours=start)
        assert pd.Timestamp(window.end) == origin + pd.Timedelta(hours=end - 1)
        assert window.rows == end - start
    saved = json.loads((output / "sample/performance-0.json").read_text())
    for window in receipt.windows:
        observations = saved["performance"]["scopes"][f"fold/0/{window.role}"][
            "metadata"
        ]["observations"]
        assert pd.Timestamp(observations["start_date"]) == pd.Timestamp(window.start)
        assert pd.Timestamp(observations["end_date"]) == pd.Timestamp(window.end)
        assert observations["bar_count"] == window.rows
    for key, value in validation.items():
        configured = result.manifest.config_copy()["validation"][key]
        if key.endswith(("start", "end")):
            assert configured.startswith(value)
        else:
            assert configured == value


@pytest.mark.parametrize(
    "invalid", [{"label_horizon": 0}, {"label_horizon": -1}, {"embargo_bars": -1}]
)
def test_invalid_purged_preflight_cannot_fall_back_to_explicit_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: dict[str, int]
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path,
        monkeypatch,
        kind="walk_forward",
        validation={
            "purged_split": True,
            "label_horizon": 5,
            "n_splits": 2,
            "train_start": "2024-01-08",
            "train_end": "2024-01-10",
            "test_start": "2024-01-20",
            "test_end": "2024-01-22",
            **invalid,
        },
    )
    output = tmp_path / "preflight"
    output.mkdir()
    with pytest.raises(ValueError, match="label horizon|embargo"):
        execute_preflight(request, source_workspace=workspace, directory=output)
    assert not list((output / "sample").glob("performance-*.json"))


def test_preflight_rejects_insufficient_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path, monkeypatch, kind="walk_forward"
    )
    request = request.model_copy(update={"minimum_history_bars": 40})
    directory = tmp_path / "preflight"
    directory.mkdir()
    with pytest.raises(PreflightError, match="requires at least 40"):
        execute_preflight(request, source_workspace=workspace, directory=directory)


@pytest.mark.parametrize("phase", ["engine", "serialization"])
def test_preflight_requires_actual_execution_and_serialization_to_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path,
        monkeypatch,
        failure="exception" if phase == "engine" else None,
    )
    directory = tmp_path / "preflight"
    directory.mkdir()
    if phase == "serialization":

        def reject_artifacts(*_args: Any, **_kwargs: Any) -> None:
            raise ValueError("invalid observations")

        monkeypatch.setattr(
            "tradingdev.adapters.execution.preflight_worker.bundles_from_pipeline",
            reject_artifacts,
        )
    with pytest.raises(
        (ValueError, RuntimeError), match="sample failure|invalid observations"
    ):
        execute_preflight(request, source_workspace=workspace, directory=directory)
    assert list(workspace.runs.iterdir()) == []


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
        assert (
            PreflightService(workspace).prepare(request).receipt.sample_bars_used == 128
        )
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
    request = PreflightRequest(kind="backtest", arguments={}, minimum_history_bars=1)
    with pytest.raises(PreflightError, match="retained") as caught:
        PreflightService().prepare(request)
    assert caught.value.code == "preflight_cleanup_failed"
    assert directory.is_dir()
    assert (directory / "request.json").is_file()
    assert (directory / "tmp").is_dir()
