"""Old saved fold settings remain queryable, but cannot become new execution input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.app.test_submitted_research import approve
from tests.domain.test_research_walk_forward import _EXPLICIT, _artifacts

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.app.job_store import JobStore
from tradingdev.app.report_service import ReportService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import build_artifacts

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "validation",
    [
        {"mode": "automatic", "n_splits": 3},
        _EXPLICIT,
        {"mode": "purged", "n_splits": 3, "label_horizon": 1},
    ],
)
def test_saved_flat_settings_survive_trade_queries_and_html_without_reexecution(
    tmp_path: Path,
    validation: dict[str, Any],
) -> None:
    current, artifacts = _artifacts(validation)
    flat = {
        "train_start": None,
        "train_end": None,
        "test_start": None,
        "test_end": None,
        "n_splits": 3,
        "train_ratio": 0.8,
        "expanding": False,
        "target_metric": "sharpe_ratio",
        "label_horizon": 0,
        "embargo_bars": 0,
        "purged_split": validation["mode"] == "purged",
        **{
            key: value
            for key, value in current.config_copy()["validation"].items()
            if key != "mode"
        },
    }
    if validation["mode"] == "purged":
        # Old purged runs accepted and ignored even reversed/partial dates.
        flat.update(train_start="2030-01-01", train_end="2020-01-01")
    payload = current.model_dump(mode="json", exclude={"manifest_hash"})
    payload["config"]["validation"] = flat
    manifest = ExecutionManifest.model_validate(
        {
            **payload,
            "manifest_hash": ExecutionManifest._digest(payload),
        }
    )
    frozen = manifest.model_dump(mode="json")
    with pytest.raises(ValueError, match="mode"):
        manifest.config_for_execution()
    with pytest.raises(ValueError, match="mode"):
        ExecutionManifest.create(
            kind="walk_forward",
            config=manifest.config_copy(),
            strategy_execution=manifest.strategy_execution,
        )
    workspace = WorkspacePaths(tmp_path / "workspace")
    jobs = JobStore(workspace=workspace)
    evidence = approve(manifest)
    jobs.create_job(job_id="job", manifest=manifest)
    jobs.update_job("job", research_evidence_hash=evidence.content_hash)
    jobs.store.create_run(
        run_id="job",
        job_id="job",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "job",
        metrics={},
        manifest_hash=manifest.manifest_hash,
    )
    ExecutionManifestStore(workspace).publish("job", manifest)
    PerformanceStore(workspace, jobs.store).publish(
        build_artifacts(
            "job",
            manifest.manifest_hash,
            "test_summary",
            artifacts.performance.scopes,
            artifacts.observations.scopes,
        )
    )
    research = SubmittedResearchStore(workspace)
    path = research.publish_run("job", "job", evidence, manifest)
    jobs.store.create_artifact(
        artifact_id="job:research_evidence",
        run_id="job",
        artifact_type="research_evidence",
        path=path,
        sha256=sha256_file(path),
        metadata={"job_id": "job", "research_evidence_hash": evidence.content_hash},
    )
    saved = {
        path: path.read_bytes()
        for path in (workspace.runs / "job").iterdir()
        if path.is_file()
    }
    history = TradeHistoryService(workspace=workspace, store=jobs.store)
    loaded = history.load_run("job")
    assert loaded.manifest is not None
    assert loaded.manifest.model_dump(mode="json") == frozen
    trades = history.get_run_trades("job", scope="fold/0/test")
    assert trades["success"], trades
    report = ReportService(workspace=workspace, store=jobs.store).generate_report(
        ["job"],
        sections=["settings", "research", "trades"],
    )
    assert report["success"], report
    assert all(path.read_bytes() == content for path, content in saved.items())
    assert manifest.model_dump(mode="json") == frozen
