"""Real engine results retain monthly semantics through saved history and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.app.test_comparison_report import _payload
from tests.app.test_submitted_research import approve
from tests.domain.backtest.test_volume_monthly_loss import monthly_market

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.app.job_store import JobStore
from tradingdev.app.report_service import ReportService
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.backtest.engines import create_backtest_engine
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_saved_volume_result_and_policy_match_monthly_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = ExecutionManifest.create(
        kind="backtest",
        strategy_execution=StrategyExecution(kind="bundled", constructor_kwargs={}),
        config={
            "strategy": {"id": "monthly-fixture"},
            "backtest": {
                "mode": "volume",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-30",
                "end_date": "2024-02-02",
                "init_cash": None,
                "position_size": 100,
                "fees": 0,
                "slippage": 0,
                "monthly_max_loss": 5,
            },
        },
    )
    config = manifest.config_copy()
    engine = create_backtest_engine(BacktestConfig.model_validate(config["backtest"]))
    result = engine.run(monthly_market([100, 120, 110, 100]))
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    evidence = approve(manifest)
    jobs.create_job(job_id="monthly", manifest=manifest)
    jobs.update_job("monthly", research_evidence_hash=evidence.content_hash)
    SubmittedResearchStore(jobs.workspace).save("monthly", evidence, manifest)
    RunPublisher(jobs.workspace, jobs.store).publish(
        RunPublicationBinding("monthly", "monthly", manifest),
        result.metrics,
        pipeline=PipelineResult(
            mode="simple",
            execution_manifest=manifest,
            config_snapshot=config,
            backtest_result=result,
        ),
    )
    # History must keep this execution's explanation when future policy changes.
    monkeypatch.setattr(
        "tradingdev.domain.execution_assumptions.EXECUTION_POLICY_VERSION", "future"
    )
    loaded = TradeHistoryService(workspace=jobs.workspace, store=jobs.store).load_scope(
        "monthly"
    )
    assert loaded.observations.equity_curve == [0, 20, 10, 10]
    assert loaded.observations.trades[0]["exit_idx"] == 2
    assert loaded.observations.trades[0]["net_pnl"] == 10
    assert loaded.performance.values["total_pnl"] == 10
    assert loaded.execution_assumptions == evidence.execution_assumptions[0]
    assert loaded.execution_assumptions is not None
    assert loaded.execution_assumptions.policy_version == "execution-v2"
    reports = ReportService(workspace=jobs.workspace, store=jobs.store)
    response = reports.generate_report(
        run_ids=["monthly"], sections=["research", "settings", "trades"]
    )
    html, payload = _payload(response)
    scope = payload["runs"][0]["scopes"][0]
    assert scope["execution_assumptions"]["policy_version"] == "execution-v2"
    assert "上個有行情月份最後收盤權益" in html
    assert "不重計前月損益" in html
