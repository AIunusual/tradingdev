"""Failure evidence reads immutable artifacts without importing generated code."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import yaml

from tests.e2e import novice_diagnostics, test_llm_novice_intake
from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.novice_diagnostics import (
    novice_failure_evidence,
    retain_novice_failure_evidence,
)
from tests.e2e.novice_intake import ObservedIntakeUser
from tests.integration.execution_fixtures import synthetic_preflight_receipt
from tests.integration.mcp_harness import MCPWorkspace, temporary_mcp_workspace
from tests.research_evidence_fixtures import research_evidence_draft
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
from tradingdev.domain.approved_research import ApprovedResearch
from tradingdev.domain.data.requirements import DataRequirement, MarketDataSpec
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.execution_plan import ExecutionPlan
from tradingdev.domain.performance.artifacts import (
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentation,
    build_confirmation_document,
)
from tradingdev.domain.presentation.research_confirmation import (
    enrich_research_confirmation,
)
from tradingdev.domain.research_evidence import ResearchEvidenceRecord
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.schemas import StrategyMetadata, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class EvidenceFixture:
    workspace: MCPWorkspace
    user: ObservedIntakeUser
    store: SQLiteStore
    revision: StrategyMetadata
    code: str
    manifest: ExecutionManifest

    @property
    def paths(self) -> WorkspacePaths:
        return WorkspacePaths(self.workspace.workspace)

    def publish(self, run_id: str = "approved-run") -> None:
        directory = self.paths.runs / run_id
        directory.mkdir(exist_ok=True)
        config = yaml.safe_dump(self.manifest.config_copy())
        (directory / "strategy.py").write_text(self.code, encoding="utf-8")
        (directory / "config.yaml").write_text(config, encoding="utf-8")
        self.store.create_run(
            run_id=run_id,
            job_id=run_id,
            strategy_id=self.revision.strategy_id,
            revision_id=self.revision.revision_id,
            manifest_hash=self.manifest.manifest_hash,
            config_hash=hashlib.sha256(config.encode()).hexdigest(),
            source_hash=self.revision.source_hash,
            artifact_dir=directory,
            metrics={"total_trades": 2},
        )
        ExecutionManifestStore(self.paths).publish(run_id, self.manifest)
        observations = ScopeObservations(
            init_cash=1000.0,
            equity_curve=[1000.0, 1010.0, 1000.0],
            returns=[0.0, 0.01, -0.00990099],
            timestamps=[f"2024-01-01T0{hour}:00:00Z" for hour in range(3)],
            trades=[
                {"entry_idx": 0, "exit_idx": 1, "direction": 1, "status": "closed"},
                {"entry_idx": 1, "exit_idx": 2, "direction": -1, "status": "closed"},
            ],
        )
        PerformanceStore(self.paths, self.store).publish(
            build_artifacts(
                run_id,
                self.manifest.manifest_hash,
                "full",
                {
                    "full": PerformanceScope(
                        mode="signal", values={"total_trades": 2}, metadata={}
                    )
                },
                {"full": observations},
            )
        )


def _fixture(root: Path) -> EvidenceFixture:
    workspace = MCPWorkspace(root)
    paths = WorkspacePaths(workspace.workspace)
    store = SQLiteStore(paths)
    # If any diagnostic imports this source, the marker exposes the execution.
    code = (
        "from pathlib import Path\n"
        f"Path({str(root / 'imported')!r}).touch()\n"
        "raise RuntimeError('must not execute')\n"
    )
    revisions = StrategyRevisionStore(paths)
    revision = revisions.create(
        "strategy_fixture",
        code,
        {
            "strategy": {"class_name": "NeverImported", "parameters": {}},
            "data": {
                **DataConfig().model_dump(mode="json"),
                "requirements": DataRequirement(
                    market=MarketDataSpec(symbol="BTC/USDT", timeframe="1h")
                ).model_dump(mode="json"),
            },
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "init_cash": 1000.0,
            },
        },
    )
    config = yaml.safe_load(Path(revision.config_path).read_text())
    check = ValidationResult(
        revision_id=revision.revision_id,
        checked_at=revision.created_at,
        success=True,
        signal_analysis={"short_signals": 1},
        effective_config=config,
    )
    revision.validation = check
    revision.dry_run = check
    revisions.update(revision)
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=config,
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
    )
    receipt = PreflightReceipt.model_validate(
        synthetic_preflight_receipt(manifest.manifest_hash)
    )
    record = ResearchEvidenceRecord.create(
        research_evidence_draft(
            manifest,
            ConfirmationPresentation(
                title="Fixture", summary="Saved evidence", parameter_descriptions={}
            ),
        )
    )
    research = ApprovedResearch.create(record, manifest)
    document = enrich_research_confirmation(
        build_confirmation_document(
            manifest, record.draft.presentation, receipt, plan_id="a" * 32
        ),
        research.research,
    )
    created = datetime.now(UTC)
    payload = {
        "schema_version": 3,
        "plan_id": "a" * 32,
        "manifest": manifest.model_dump(mode="json"),
        "preflight": receipt.model_dump(mode="json"),
        "research_evidence": research.model_dump(mode="json"),
        "document": document.model_dump(mode="json"),
        "original_config_path": revision.config_path,
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "expires_at": (created + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    plan = ExecutionPlan.model_validate(
        {**payload, "plan_hash": ExecutionPlan.digest(payload)}
    )
    user = ObservedIntakeUser(workspace, set(), plan=plan)
    fixture = EvidenceFixture(workspace, user, store, revision, code, manifest)
    fixture.publish()
    return fixture


@pytest.fixture
def evidence_fixture() -> Iterator[EvidenceFixture]:
    with TemporaryDirectory(prefix="tradingdev-novice-evidence-") as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        yield fixture
        assert not (root / "imported").exists(), "Diagnostics executed strategy source"
    assert not root.exists()


def test_approved_revision_and_raw_pages_survive_a_later_current_revision(
    evidence_fixture: EvidenceFixture,
) -> None:
    fixture = evidence_fixture
    config = yaml.safe_load(Path(fixture.revision.config_path).read_text())
    current = StrategyRevisionStore(fixture.paths).create(
        fixture.revision.strategy_id,
        "# later repaired strategy\n",
        config,
        expected_revision_id=fixture.revision.revision_id,
    )
    calls = [
        ToolCall("get_run_trades", {"run_id": "other-run"}, {"trades": []}),
        ToolCall(
            "get_run_trades",
            {"run_id": "approved-run", "offset": 1},
            {"total": 2, "trades": [{"direction": "short", "raw": "模型取得"}]},
        ),
    ]
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, calls)
    assert fixture.user.turns == []  # A model may fail before completing this turn.
    assert (
        evidence["run"]["revision_id"]
        == fixture.revision.revision_id
        != current.revision_id
    )
    assert evidence["execution_artifacts"]["strategy.py"]["text"] == fixture.code
    assert (
        evidence["execution_artifacts"]["strategy.py"]["sha256"]
        == fixture.revision.source_hash
    )
    checks = evidence["selected_revision_checks"]["value"]
    assert checks["revision_id"] == fixture.revision.revision_id
    assert checks["validation"]["signal_analysis"] == {"short_signals": 1}
    assert checks["dry_run"]["effective_config"] == config
    recorded = evidence["recorded_trade_pages"]
    assert recorded["total_recorded_pages"] == recorded["captured_pages"] == 1
    assert recorded["complete"] is True
    assert recorded["pages"][0]["call_index"] == 1
    assert recorded["pages"][0]["raw_arguments"]["value"] == calls[1].arguments
    assert recorded["pages"][0]["result"]["value"] == calls[1].result
    saved = evidence["saved_trade_queries"]
    assert "not model tool calls" in saved["origin"]
    assert saved["queries"]["all"]["value"]["total"] == 2
    assert (
        saved["queries"]["long"]["value"]["matched"]
        == saved["queries"]["short"]["value"]["matched"]
        == 1
    )
    assert saved["queries"]["short"]["value"]["trades"][0]["direction"] == "short"
    assert calls[1].result["trades"][0]["raw"] == "模型取得"


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_or_missing_run_does_not_select_other_history(
    evidence_fixture: EvidenceFixture,
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = evidence_fixture
    if count == 0:
        with fixture.store.connect() as connection:
            connection.execute("UPDATE runs SET revision_id = 'different'")
    else:
        fixture.publish("duplicate-run")
    reader = Mock(side_effect=AssertionError("must not read ambiguous history"))
    monkeypatch.setattr(novice_diagnostics, "TradeHistoryService", reader)
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, [])
    assert evidence["run_selection"]["available"] is False
    assert evidence["run_selection"]["matching_saved_runs"] == count
    assert "run" not in evidence and "execution_artifacts" not in evidence
    assert evidence["selected_plan"]["manifest"][
        "value"
    ] == fixture.manifest.model_dump(mode="json")
    reader.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "modified", "outside_workspace"])
@pytest.mark.parametrize("filename", ["strategy.py", "config.yaml"])
def test_bad_execution_snapshot_is_unavailable_without_losing_other_evidence(
    evidence_fixture: EvidenceFixture,
    damage: str,
    filename: str,
) -> None:
    fixture = evidence_fixture
    source = fixture.paths.runs / "approved-run" / filename
    source.unlink()
    if damage == "modified":
        source.write_text("# corrupt\n")
    elif damage == "outside_workspace":
        outside = fixture.workspace.root / "outside.py"
        outside.write_text("private test content")
        source.symlink_to(outside)
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, [])
    artifact = evidence["execution_artifacts"][filename]
    assert artifact["available"] is False and artifact["diagnostic_error"]
    assert "private test content" not in json.dumps(evidence)
    assert evidence["execution_artifacts"]["manifest.json"]["available"] is True
    assert evidence["saved_trade_queries"]["queries"]["short"]["value"]["matched"] == 1


@pytest.mark.parametrize("target", ["revision", "manifest", "observations"])
def test_missing_revision_and_corrupt_saved_json_are_explicitly_unavailable(
    evidence_fixture: EvidenceFixture,
    target: str,
) -> None:
    fixture = evidence_fixture
    if target == "revision":
        shutil.rmtree(Path(fixture.revision.source_path).parent)
    else:
        (fixture.paths.runs / "approved-run" / f"{target}.json").write_text("{broken")
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, [])
    assert evidence["run"]["revision_id"] == fixture.revision.revision_id
    if target == "revision":
        assert evidence["selected_revision_checks"]["available"] is False
        assert (
            "Reviewed revision is unavailable"
            in evidence["selected_revision_checks"]["diagnostic_error"]
        )
        assert evidence["execution_artifacts"]["strategy.py"]["text"] == fixture.code
    else:
        query = evidence["saved_trade_queries"]["queries"]["short"]["value"]
        assert query["success"] is False
        assert "trades" not in query
        if target == "manifest":
            assert (
                evidence["execution_artifacts"]["manifest.json"]["available"] is False
            )


@pytest.mark.parametrize(
    ("hash_key", "filename"),
    [("source_hash", "strategy.py"), ("config_hash", "config.yaml")],
)
def test_missing_saved_hash_does_not_claim_a_verified_snapshot(
    evidence_fixture: EvidenceFixture,
    hash_key: str,
    filename: str,
) -> None:
    fixture = evidence_fixture
    with fixture.store.connect() as connection:
        # Column names are the two fixed fixture values above, never external input.
        connection.execute(f"UPDATE runs SET {hash_key} = NULL")
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, [])
    assert evidence["run"][hash_key] is None
    artifact = evidence["execution_artifacts"][filename]
    assert artifact["available"] is False
    assert "verification identity is missing" in artifact["diagnostic_error"]
    assert "text" not in artifact
    assert evidence["execution_artifacts"]["manifest.json"]["available"] is True
    assert (
        evidence["selected_revision_checks"]["value"]["revision_id"]
        == fixture.revision.revision_id
    )
    assert evidence["saved_trade_queries"]["queries"]["short"]["value"]["matched"] == 1


def test_trade_pages_and_unicode_content_report_their_capture_limits(
    evidence_fixture: EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(novice_diagnostics, "_TEXT_LIMIT", 80)
    calls = [
        ToolCall(
            "get_run_trades",
            {"run_id": "approved-run", "extra": "中" * 100},
            {"raw": "文" * 100},
        )
        for _ in range(3)
    ]
    evidence = novice_failure_evidence(
        evidence_fixture.workspace, evidence_fixture.user, calls
    )
    pages = evidence["recorded_trade_pages"]
    assert pages["total_recorded_pages"] == 3 and pages["captured_pages"] == 2
    assert pages["complete"] is False
    for item in pages["pages"]:
        for key in ("raw_arguments", "result"):
            value = item[key]
            assert value["complete"] is False and "value" not in value
            assert len(value["json_prefix"]) == 80
            assert value["total_utf8_bytes"] > value["total_characters"] > 80
    source = evidence["execution_artifacts"]["strategy.py"]
    assert source["complete"] is False and len(source["text"]) == 80
    assert source["sha256"] == evidence_fixture.revision.source_hash
    saved = evidence["saved_trade_queries"]["queries"]["short"]
    assert saved["complete"] is False and len(saved["json_prefix"]) == 80


def test_no_reviewed_plan_does_not_read_or_create_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = Mock(side_effect=AssertionError("No approved evidence to read"))
    monkeypatch.setattr(novice_diagnostics, "SQLiteStore", reader)
    monkeypatch.setattr(novice_diagnostics, "StrategyRevisionStore", reader)
    with TemporaryDirectory(prefix="tradingdev-no-novice-plan-") as temporary:
        workspace = MCPWorkspace(Path(temporary))
        evidence = novice_failure_evidence(
            workspace, ObservedIntakeUser(workspace, set()), []
        )
        assert evidence["selected_plan"] is None
        assert not workspace.workspace.exists()
    reader.assert_not_called()


def test_missing_database_is_unavailable_and_is_not_recreated(
    evidence_fixture: EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = evidence_fixture
    database = fixture.paths.root / "tradingdev.sqlite"
    database.unlink()
    reader = Mock(side_effect=AssertionError("Must not recreate absent database"))
    monkeypatch.setattr(novice_diagnostics, "SQLiteStore", reader)
    evidence = novice_failure_evidence(fixture.workspace, fixture.user, [])
    assert evidence["run_selection"] == {
        "available": False,
        "reason": "No saved run database",
    }
    assert (
        evidence["selected_revision_checks"]["value"]["revision_id"]
        == fixture.revision.revision_id
    )
    assert not database.exists()
    reader.assert_not_called()


@pytest.mark.parametrize(
    "failure", [AssertionError("oracle failed"), TimeoutError("model timed out")]
)
def test_failure_note_survives_normal_workspace_cleanup(failure: Exception) -> None:
    with (
        pytest.raises(type(failure)) as caught,
        temporary_mcp_workspace() as workspace,
    ):
        root = workspace.root
        fixture = _fixture(root)
        with retain_novice_failure_evidence(workspace, fixture.user, []):
            raise failure
    assert caught.value is failure
    assert not root.exists()
    note = json.loads(failure.__notes__[0].split("\n", 1)[1])
    assert note["execution_artifacts"]["strategy.py"]["text"] == fixture.code
    assert note["saved_trade_queries"]["queries"]["short"]["value"]["matched"] == 1


def test_diagnostic_failure_keeps_original_exception_and_cause(
    evidence_fixture: EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = Mock(side_effect=OSError("evidence unreadable"))
    monkeypatch.setattr(novice_diagnostics, "novice_failure_evidence", diagnostic)
    failure = AssertionError("original oracle failure")
    cause = ValueError("original cause")
    with (
        pytest.raises(AssertionError) as caught,
        retain_novice_failure_evidence(
            evidence_fixture.workspace, evidence_fixture.user, []
        ),
    ):
        raise failure from cause
    assert caught.value is failure and caught.value.__cause__ is cause
    note = json.loads(failure.__notes__[0].split("\n", 1)[1])
    assert note["reason"] == "Evidence unavailable"
    assert note["diagnostic_error"] == "OSError: evidence unreadable"
    assert note["complete"] is True


@pytest.mark.parametrize("failure_site", ["artifact", "collector"])
def test_unicode_diagnostic_errors_disclose_truncation_and_preserve_cause(
    evidence_fixture: EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    monkeypatch.setattr(novice_diagnostics, "_TEXT_LIMIT", 40)
    message = "證據無法讀取" * 30
    failing_reader = Mock(side_effect=OSError(message))
    monkeypatch.setattr(
        novice_diagnostics,
        "_artifact" if failure_site == "artifact" else "novice_failure_evidence",
        failing_reader,
    )
    failure = AssertionError("original failure")
    cause = TimeoutError("original timeout")
    with (
        pytest.raises(AssertionError) as caught,
        retain_novice_failure_evidence(
            evidence_fixture.workspace, evidence_fixture.user, []
        ),
    ):
        raise failure from cause
    assert caught.value is failure and caught.value.__cause__ is cause
    note = json.loads(failure.__notes__[0].split("\n", 1)[1])
    error = (
        note["execution_artifacts"]["strategy.py"]
        if failure_site == "artifact"
        else note
    )
    original = "OSError: " + message
    assert error["available"] is False and error["complete"] is False
    assert error["diagnostic_error"] == original[:40]
    assert error["total_characters"] == len(original) > 40
    assert error["total_utf8_bytes"] == len(original.encode("utf-8"))
    assert error["total_utf8_bytes"] > error["total_characters"]


def test_success_does_not_collect_diagnostics(
    evidence_fixture: EvidenceFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = Mock(side_effect=AssertionError("success must not read evidence"))
    monkeypatch.setattr(novice_diagnostics, "novice_failure_evidence", diagnostic)
    with retain_novice_failure_evidence(
        evidence_fixture.workspace, evidence_fixture.user, []
    ):
        pass
    diagnostic.assert_not_called()


@pytest.mark.parametrize("oracle_fails", [True, False])
def test_intake_checks_executed_strategy_before_missing_report(
    evidence_fixture: EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
    oracle_fails: bool,
) -> None:
    fixture = evidence_fixture
    fixture.user.turns = [ModelTurn(1, "No report was delivered", [])]
    oracle = Mock(
        side_effect=AssertionError("wrong executed signals") if oracle_fails else None
    )
    monkeypatch.setattr(test_llm_novice_intake, "verify_generated_strategy", oracle)
    calls = [
        ToolCall("save_strategy", {}, {"strategy_id": fixture.revision.strategy_id}),
        ToolCall("get_job_status", {}, {"status": "done", "run_id": "approved-run"}),
    ]
    message = (
        "wrong executed signals" if oracle_fails else "final report turn must generate"
    )
    with pytest.raises(AssertionError, match=message):
        test_llm_novice_intake._verify_completed_intake(
            fixture.workspace, fixture.user, calls
        )
    oracle.assert_called_once_with(fixture.workspace.root, scenario="novice_intake")
