"""Offline fixed-comparison reports retain failures and verify shared evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from mcp_types import CallToolResult

from tradingdev.adapters.cli.reports import main as report_main
from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.reports import ReportResponse
from tradingdev.app.execution_dataset import execution_data_binding
from tradingdev.app.report_service import ReportService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import history

if TYPE_CHECKING:
    from tradingdev.domain.comparison_results import CandidateStatus


@dataclass
class ComparisonFixture:
    workspace: WorkspacePaths
    store: SQLiteStore
    comparisons: ComparisonStore
    manifest: ComparisonManifest
    service: ReportService


def _fixture(
    tmp_path: Path,
    statuses: tuple[CandidateStatus, ...] = ("succeeded", "failed", "not_run"),
    *,
    snapshot: bool = True,
    status: str = "failed",
) -> ComparisonFixture:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    comparisons = ComparisonStore(workspace)
    manifest = ComparisonManifest.create(
        [
            ComparisonCandidate(
                candidate_id=f"candidate_{index}",
                label=f"Window {index}",
                rationale="Fixed before results <script>unsafe()</script>",
                manifest=ExecutionManifest.create(
                    kind="backtest",
                    config={
                        "strategy": {"id": "fixture", "parameters": {"period": index}},
                        "backtest": {
                            "symbol": "BTC/USDT",
                            "timeframe": "1d",
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-03",
                            "init_cash": 100.0,
                            "fees": 0.0,
                            "slippage": 0.0,
                        },
                    },
                    strategy_execution=StrategyExecution(
                        kind="generated", constructor_kwargs={"period": index}
                    ),
                ),
            )
            for index in range(1, len(statuses) + 1)
        ]
    )
    comparisons.initialize("comparison", manifest)
    job: dict[str, Any] = {
        "job_id": "comparison",
        "job_type": "comparison",
        "status": status,
        "created_at": "2024-01-04T00:00:00Z",
        "manifest_hash": manifest.manifest_hash,
        "data_downloaded": snapshot,
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "close": [100.0, 90.0, 110.0],
        },
    )
    if snapshot:
        receipt = comparisons.publish_snapshot(
            "comparison",
            expected_hash=manifest.manifest_hash,
            snapshot_bytes=frame.to_parquet(engine="pyarrow", index=None),
            data_binding_json=execution_data_binding(manifest.candidates[0].manifest),
            source_dataset_id="synthetic-market",
        )
        job.update(
            dataset_id=f"sha256:{receipt.snapshot_sha256}",
            data_binding_json=receipt.data_binding_json,
            source_dataset_id=receipt.source_dataset_id,
        )
    store.upsert_job(job)
    for candidate, candidate_status in zip(manifest.candidates, statuses, strict=True):
        if candidate_status not in {"pending", "not_run"}:
            comparisons.transition(
                "comparison",
                expected_hash=manifest.manifest_hash,
                candidate_id=candidate.candidate_id,
                status="running",
            )
        if candidate_status not in {"pending", "running"}:
            comparisons.transition(
                "comparison",
                expected_hash=manifest.manifest_hash,
                candidate_id=candidate.candidate_id,
                status=candidate_status,
                reason=None
                if candidate_status == "succeeded"
                else "保存原因 <img src=x onerror=unsafe()>",
            )
        if candidate_status != "succeeded":
            continue
        child = candidate.manifest
        config = child.config_copy()
        run_id = f"comparison_{candidate.candidate_id}"
        scope = PerformanceScope(
            mode="signal",
            values={"total_return": 0.1, "win_rate": None},
            parameters=child.strategy_execution.constructor_kwargs,
            metadata={
                "execution_context": {
                    **config["backtest"],
                    "random_seed": config.get("random_seed"),
                },
                "settings": {"initial_cash": 100.0},
                "unavailable": {"win_rate": "no_closed_trades"},
            },
        )
        store.create_run(
            run_id=run_id,
            job_id="comparison",
            strategy_id="fixture",
            manifest_hash=child.manifest_hash,
            dataset_id=job["dataset_id"],
            random_seed=config.get("random_seed"),
            artifact_dir=workspace.runs / run_id,
            metrics=scope.values,
        )
        path = ExecutionManifestStore(workspace).publish(run_id, child)
        store.create_artifact(
            artifact_id=f"{run_id}:execution_manifest",
            run_id=run_id,
            artifact_type="execution_manifest",
            path=path,
            sha256=sha256_file(path),
        )
        observations = ScopeObservations(
            init_cash=100.0,
            equity_curve=[100.0, 90.0, 110.0],
            returns=[0.0, -0.1, 2.0 / 9],
            timestamps=[timestamp.isoformat() for timestamp in frame["timestamp"]],
            trades=[],
        )
        PerformanceStore(workspace, store).publish(
            build_artifacts(
                run_id,
                child.manifest_hash,
                "full",
                {"full": scope},
                {"full": observations},
            )
        )
    return ComparisonFixture(
        workspace,
        store,
        comparisons,
        manifest,
        ReportService(workspace=workspace, store=store),
    )


def _payload(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assert response["success"], response
    text = Path(response["path"]).read_text()
    raw = text.split("<script id='report-data' type='application/json'>", 1)[1].split(
        "</script>", 1
    )[0]
    return text, json.loads(raw)


def test_mixed_report_preserves_order_reasons_navigation_and_frozen_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    before = {
        path: path.read_bytes()
        for path in fixture.workspace.root.rglob("*")
        if path.is_file()
        and (
            path.is_relative_to(fixture.workspace.runs)
            or path.is_relative_to(fixture.workspace.root / "comparisons")
        )
    }
    report = fixture.service.generate_report(
        comparison_job_id="comparison",
        sections=["overview", "settings", "equity", "provenance"],
    )
    text, payload = _payload(report)
    assert ReportResponse.model_validate(report).comparison_job_id == "comparison"
    assert report["run_ids"] == ["comparison_candidate_1"]
    rows = payload["comparison_table"]["rows"]
    assert [row["candidate"] for row in rows] == ["Window 1", "Window 2", "Window 3"]
    assert rows[0]["detail"] == "#result-0-0"
    assert all(row.get("detail") is None for row in rows[1:])
    assert all(row.get("metric:total_return:fraction") is None for row in rows[1:])
    assert "href='#result-0-0'" in text and "id='result-0-0'" in text
    assert "href='#result-1-0'" not in text
    assert "無可用明細" in text and "失敗／無結果" in text and "未執行／無結果" in text
    assert (
        "&lt;script&gt;unsafe()&lt;/script&gt;" in text
        and "<script>unsafe()" not in text
    )
    assert "<img src=x onerror=unsafe()>" not in text
    assert (
        fixture.service.generate_report(
            comparison_job_id="comparison", sections=payload["sections"]
        )
        == report
    )
    manifest = json.loads((Path(report["path"]).parent / "manifest.json").read_text())
    assert manifest["fixed_comparison"] == payload["fixed_comparison"]
    assert (
        payload["fixed_comparison"]["snapshot_sha256"]
        == hashlib.sha256(
            fixture.comparisons.path("comparison", "snapshot.parquet").read_bytes()
        ).hexdigest()
    )
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("sections", [None, []])
@pytest.mark.parametrize(
    "snapshot,statuses,status",
    [
        (True, ("failed", "failed"), "done"),
        (True, ("cancelled", "not_run"), "cancelled"),
        (False, ("not_run", "not_run"), "failed"),
    ],
)
def test_zero_success_reports_are_complete_without_fabricated_metrics(
    tmp_path: Path,
    sections: list[str] | None,
    snapshot: bool,
    statuses: tuple[CandidateStatus, ...],
    status: str,
) -> None:
    fixture = _fixture(tmp_path, statuses, snapshot=snapshot, status=status)
    report = fixture.service.generate_report(
        comparison_job_id="comparison", sections=sections
    )
    text, payload = _payload(report)
    assert report["run_ids"] == [] and report["scope_count"] == 0
    assert len(payload["comparison_table"]["rows"]) == 2
    assert payload["fixed_comparison"]["data_available"] is snapshot
    assert "data-section='comparison_candidates'" in text
    assert "完整有效參數" in text and "比較理由" in text and "單邊手續費率" in text
    assert "id='result-view'" not in text and "href='#result-" not in text
    assert "無結果" in text
    assert not any(
        key.startswith("metric:")
        for row in payload["comparison_table"]["rows"]
        for key in row
    )
    if sections is None:
        assert all(
            f"data-section='{name}'" in text
            for name in ("overview", "settings", "limitations", "provenance")
        )


@pytest.mark.parametrize(
    "change",
    [
        "binding",
        "receipt_missing",
        "snapshot_missing",
        "snapshot_corrupt",
        "ledger_missing",
        "manifest_hash",
        "job_hash",
        "unfinished_job",
        "unfinished_ledger",
        "child_parent",
        "child_dataset",
        "child_metrics",
        "child_parameters",
        "child_timestamps",
        "child_assumptions",
    ],
)
def test_changed_or_missing_evidence_is_rejected(tmp_path: Path, change: str) -> None:
    fixture = _fixture(tmp_path)
    comparisons, store = fixture.comparisons, fixture.store
    if change in {"receipt_missing", "snapshot_missing", "ledger_missing"}:
        filename = {
            "receipt_missing": "receipt.json",
            "snapshot_missing": "snapshot.parquet",
            "ledger_missing": "ledger.json",
        }[change]
        (comparisons.path("comparison").parent / filename).unlink()
    elif change == "snapshot_corrupt":
        comparisons.path("comparison", "snapshot.parquet").write_bytes(b"corrupt")
    elif change == "binding":
        path = comparisons.path("comparison", "receipt.json")
        value = json.loads(path.read_text())
        value["data_binding_json"] = '{"data":{},"market":{}}'
        path.write_text(json.dumps(value))
    elif change in {"job_hash", "unfinished_job"}:
        job = store.get_job("comparison")
        assert job is not None
        job.update(
            {"manifest_hash": "0" * 64}
            if change == "job_hash"
            else {"status": "running_backtest"}
        )
        store.upsert_job(job)
    elif change in {"manifest_hash", "unfinished_ledger"}:
        path = comparisons.path(
            "comparison",
            "manifest.json" if change == "manifest_hash" else "ledger.json",
        )
        value = json.loads(path.read_text())
        if change == "manifest_hash":
            value["candidates"][0]["label"] = "Changed"
        else:
            value["candidates"][2].update(status="pending", reason=None)
        path.write_text(json.dumps(value))
    elif change in {"child_parent", "child_dataset", "child_metrics"}:
        column, value = {
            "child_parent": ("job_id", "another"),
            "child_dataset": ("dataset_id", "sha256:" + "0" * 64),
            "child_metrics": ("metrics", '{"total_return":9}'),
        }[change]
        with store.connect() as connection:
            connection.execute(
                f"update runs set {column}=? where run_id=?",
                (value, "comparison_candidate_1"),
            )
    else:
        name = "observations" if change == "child_timestamps" else "performance"
        artifact_id = f"comparison_candidate_1:{name}_json"
        artifact = store.get_artifact(artifact_id)
        assert artifact is not None
        path = Path(artifact["path"])
        value = json.loads(path.read_text())
        scope = value["scopes"]["full"]
        if change == "child_parameters":
            scope["parameters"]["period"] = 99
        elif change == "child_timestamps":
            scope["timestamps"][0] = "2023-12-31T00:00:00+00:00"
        else:
            scope["metadata"]["execution_context"]["fees"] = 0.5
        path.write_text(json.dumps(value))
        with store.connect() as connection:
            connection.execute(
                "update artifacts set sha256=? where artifact_id=?",
                (sha256_file(path), artifact_id),
            )
    report = fixture.service.generate_report(
        comparison_job_id="comparison", sections=[]
    )
    assert report["success"] is False, report
    assert not (fixture.workspace.root / "reports").exists()


def test_absent_receipt_cannot_hide_parent_data_claims(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, ("not_run",), snapshot=False)
    job = fixture.store.get_job("comparison")
    assert job is not None
    job["dataset_id"] = "sha256:" + "0" * 64
    fixture.store.upsert_job(job)
    assert (
        fixture.service.generate_report(comparison_job_id="comparison")["success"]
        is False
    )


def test_success_navigation_keeps_candidate_order_across_failed_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, ("succeeded", "failed", "succeeded"), status="done")
    report = fixture.service.generate_report(
        comparison_job_id="comparison", sections=["settings", "equity"]
    )
    text, payload = _payload(report)
    rows = payload["comparison_table"]["rows"]
    assert [row.get("detail") for row in rows] == ["#result-0-0", None, "#result-1-0"]
    assert report["run_ids"] == ["comparison_candidate_1", "comparison_candidate_3"]
    assert "href='#result-1-0'" in text and "id='result-1-0'" in text
    assert payload["default_scope_comparison"]["success"]


def test_execution_failure_still_requires_snapshot_when_parent_claims_are_erased(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, ("failed",), status="done")
    fixture.comparisons.path("comparison", "receipt.json").unlink()
    fixture.comparisons.path("comparison", "snapshot.parquet").unlink()
    job = fixture.store.get_job("comparison")
    assert job is not None
    for key in ("dataset_id", "data_binding_json", "source_dataset_id"):
        job.pop(key)
    job["data_downloaded"] = False
    fixture.store.upsert_job(job)
    report = fixture.service.generate_report(
        comparison_job_id="comparison", sections=[]
    )
    assert report["success"] is False


def test_changed_final_ledger_produces_new_report_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, ("failed",), status="done")
    first = fixture.service.generate_report(comparison_job_id="comparison", sections=[])
    path = fixture.comparisons.path("comparison", "ledger.json")
    content = json.loads(path.read_text())
    content["candidates"][0]["reason"] = "Changed retained reason"
    path.write_text(json.dumps(content))
    second = fixture.service.generate_report(
        comparison_job_id="comparison", sections=[]
    )
    assert first["success"] and second["success"]
    assert first["report_id"] != second["report_id"]
    assert "Changed retained reason" not in Path(first["path"]).read_text()


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"run_ids": [], "comparison_job_id": "comparison"},
        {"run_ids": ["x"], "comparison_job_id": "comparison"},
        {"comparison_job_id": ""},
        {"run_ids": ["x", "x"]},
    ],
)
def test_exclusive_selection_is_structured_error(
    tmp_path: Path, arguments: dict[str, Any]
) -> None:
    assert (
        ReportService(workspace=WorkspacePaths(tmp_path)).generate_report(**arguments)[
            "success"
        ]
        is False
    )


def test_comparison_cli_and_mcp_use_same_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path, ("not_run",), snapshot=False)
    report_main(
        [
            "--workspace",
            str(fixture.workspace.root),
            "--comparison-job-id",
            "comparison",
            "--sections",
        ]
    )
    cli_result = json.loads(capsys.readouterr().out)
    server = StrictMCPServer("comparison-report-test")
    history.register(
        server,
        TradeHistoryService(workspace=fixture.workspace, store=fixture.store),
        fixture.service,
    )

    async def check() -> None:
        for arguments in (
            {"comparison_job_id": "comparison", "sections": []},
            {"run_ids": [], "comparison_job_id": "comparison"},
        ):
            result = await server.call_tool("generate_report", arguments)
            assert isinstance(result, CallToolResult)
            assert result.structured_content is not None
            envelope = dict(result.structured_content)
            payload = envelope.get("result", envelope)
            if "run_ids" in arguments:
                assert payload["code"] == "invalid_report_selection"
            else:
                assert ReportResponse.model_validate(
                    payload
                ) == ReportResponse.model_validate(cli_result)

    asyncio.run(check())
    with pytest.raises(SystemExit, match="2"):
        report_main(["--run-id", "x", "--comparison-job-id", "comparison"])
