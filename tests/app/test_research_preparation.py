"""Actual preparation preserves one research contract through the frozen manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.integration.execution_fixtures import research_input
from tests.preflight_fixtures import make_preflight_fixture

from tradingdev.adapters.execution.preflight_worker import execute_preflight
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.preflight_service import PreflightError
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.preflight import PreflightReceipt, PreflightRequest
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_selected_revision_research_and_origin_reach_captured_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    request.arguments["research"] = {
        "kind": "inherit_selected_revision",
        "fees": 0,
    }
    prepared = execute_preflight(
        request, source_workspace=workspace, directory=tmp_path / "preflight"
    )
    assert isinstance(prepared.manifest, ExecutionManifest)
    config = prepared.manifest.config_copy()
    assert config["backtest"]["fees"] == 0
    assert config["backtest"]["end_date"].startswith("2024-02-01T00:00:00")
    assert (
        config["research"]["selected_revision"]["revision_id"]
        == (request.arguments["revision_id"])
    )
    assert config["research"]["selected_revision"]["config_hash"]
    assert config["research"]["field_sources"]["/backtest/fees"] == "request"
    assert config["research"]["field_sources"]["/backtest/symbol"] == (
        "selected_revision"
    )
    prepared.manifest.verify(prepared.manifest.manifest_hash)


@pytest.mark.parametrize("field", ["init_cash", "fees", "slippage", "periods_per_year"])
def test_second_financial_setting_channel_cannot_replace_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    request.arguments["backtest_overrides"] = {field: 1}
    with pytest.raises(PreflightError, match=field):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )
    assert not list(workspace.processed_data.iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_date", "2023-12-01"),
        ("end_date", "2024-02-02"),
        ("end_date", "2024-01-31"),
        ("end_date", "2024-02-01T00:00:00Z"),
    ],
)
def test_optimization_cannot_rewrite_or_shrink_the_requested_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path, monkeypatch, kind="optimization"
    )
    research = request.arguments["research"]
    assert isinstance(research, dict)
    research[field] = value
    with pytest.raises(PreflightError, match="inclusive UTC calendar days"):
        execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )
    assert not list(workspace.processed_data.iterdir())


@pytest.mark.parametrize("kind", ["backtest", "optimization"])
def test_programmatic_overrides_do_not_relabel_old_research_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    originals = {
        path: path.read_bytes()
        for path in workspace.generated_strategies.rglob("config.yaml")
    }
    store = JobStore(workspace=workspace)
    arguments: dict[str, Any] = {
        "strategy_id": "sample_strategy",
        "revision_id": request.arguments["revision_id"],
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "backtest_overrides": {"fees": 0},
    }
    if kind == "optimization":
        prepared = OptimizationService(job_store=store).prepare_optimization(
            **arguments,
            param_ranges={"threshold": [0, 0.01]},
            optimization_metric="total_pnl",
            train_start="2024-01-02",
            train_end="2024-01-04",
            test_start="2024-01-05",
            test_end="2024-01-08",
        )
    else:
        prepared = JobService(job_store=store).prepare_backtest(
            **arguments, start_date="2024-01-02", end_date="2024-01-08"
        )
    assert isinstance(prepared, PreparedExecution), prepared
    effective = prepared.manifest.config_copy()
    assert effective["backtest"]["fees"] == 0
    assert effective["backtest"]["start_date"].startswith("2024-01-02")
    assert "research" not in effective
    assert originals and {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize("strategy_id", ["kd_crossover", "quantile_volume"])
@pytest.mark.parametrize("selection", ["unchanged", "automatic", "explicit"])
def test_old_bundled_validation_requires_an_explicit_research_window_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_id: str,
    selection: str,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Offline window checks must not execute history or fit")

    monkeypatch.setattr(DataService, "load", forbidden)
    monkeypatch.setattr(BacktestService, "create_engine", forbidden)
    monkeypatch.setattr(StrategyLoader, "create_from_execution", forbidden)
    workspace = WorkspacePaths(tmp_path / "source")
    arguments: dict[str, Any] = {
        "strategy_id": strategy_id,
        "research": research_input(start_date="2026-01-01", end_date="2026-02-01"),
    }
    if selection == "automatic":
        arguments["validation_config"] = {
            "mode": "automatic",
            "n_splits": 2,
            "train_ratio": 0.6,
        }
    elif selection == "explicit":
        arguments["validation_config"] = {
            "mode": "explicit",
            "train_start": "2026-01-01",
            "train_end": "2026-01-14",
            "test_start": "2026-01-15",
            "test_end": "2026-01-31",
        }
    request = PreflightRequest(kind="walk_forward", arguments=arguments)
    if selection == "unchanged":
        with pytest.raises(PreflightError, match="validation_config"):
            execute_preflight(
                request, source_workspace=workspace, directory=tmp_path / "preflight"
            )
    else:
        prepared = execute_preflight(
            request, source_workspace=workspace, directory=tmp_path / "preflight"
        )
        assert isinstance(prepared.manifest, ExecutionManifest)
        assert isinstance(prepared.receipt, PreflightReceipt)
        config = prepared.manifest.config_copy()
        assert config["backtest"]["start_date"].startswith("2026-01-01")
        if selection == "automatic":
            assert "train_start" not in config["validation"]
            assert "test_end" not in config["validation"]
            assert config["validation"]["n_splits"] == 2
        else:
            assert config["validation"]["test_end"].startswith("2026-01-31")
        assert "historical_data" in prepared.receipt.not_checked
    assert not list(workspace.processed_data.iterdir())
