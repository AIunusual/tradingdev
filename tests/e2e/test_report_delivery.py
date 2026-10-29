"""Report delivery evidence over real saved artifacts, without a model or worker."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.report_delivery import verify_archive_delivery, verify_report_delivery
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.research_notes import NoteExport
from tradingdev.app.report_service import ReportService
from tradingdev.domain.performance.artifacts import (
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)


@pytest.mark.parametrize(
    "reading,accepted",
    [
        ("請完整解壓後開啟 `index.html`，保留 `reports/` 目錄。", True),
        ("[閱讀](</saved/index.html>)", True),
        ("Extract the ZIP and open index.html.", True),
        ("", False),
        ("檔案內含 index.html。", False),
        ("請解壓後開啟報告。", False),
        ("```text\n解壓後開啟 index.html\n```", False),
    ],
)
def test_archive_delivery_accepts_readable_zip_instructions(
    reading: str, accepted: bool
) -> None:
    report = NoteExport(
        success=True,
        archive_id="a" * 64,
        note_id="b" * 32,
        version=2,
        artifact_id="research:" + "a" * 64 + ":zip",
        path="/saved/research.zip",
        html_path="/saved/index.html",
        sha256="c" * 64,
        instructions="Extract and open index.html.",
    )
    text = "[下載](</saved/research.zip>)\n\n" + reading
    if accepted:
        verify_archive_delivery(text, report)
    else:
        with pytest.raises(AssertionError, match="reading entry point"):
            verify_archive_delivery(text, report)
    with pytest.raises(AssertionError, match="actual ZIP"):
        verify_archive_delivery(
            text.replace("/saved/research.zip", "/wrong.zip"), report
        )


@dataclass
class DeliveredReport:
    workspace: WorkspacePaths
    store: SQLiteStore
    result: dict[str, Any]
    turn: ModelTurn

    @property
    def path(self) -> Path:
        return Path(self.result["path"])

    def verify(self, turn: ModelTurn | None = None) -> str:
        return verify_report_delivery(
            self.workspace,
            turn or self.turn,
            "run",
            expected_sections=["overview", "ledger", "trades"],
            expected_commentary=[{"title": "研究評語", "text": "僅為歷史模擬。"}],
        )


@pytest.fixture
def report(tmp_path: Path) -> DeliveredReport:
    workspace = WorkspacePaths(tmp_path / "workspace with spaces")
    store = SQLiteStore(workspace)
    scope = PerformanceScope(
        mode="signal",
        parameters={"period": 20, "stddev": 2},
        values={"total_pnl": 25.0},
        metadata={"execution_context": {"symbol": "BTC/USDT", "timeframe": "1d"}},
    )
    store.create_run(
        run_id="run",
        job_id="job",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "run",
        metrics=scope.model_dump(mode="json")["values"],
    )
    PerformanceStore(workspace, store).publish(
        build_artifacts(
            "run",
            None,
            "full",
            {"full": scope},
            {
                "full": ScopeObservations(
                    init_cash=100.0,
                    equity_curve=[100.0, 125.0],
                    returns=[0.0, 0.25],
                    timestamps=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                    trades=[],
                )
            },
        )
    )
    arguments: dict[str, Any] = {
        "run_ids": ["run"],
        "sections": ["overview", "ledger", "trades"],
        "commentary": [{"title": "研究評語", "text": "僅為歷史模擬。"}],
        "analysis": None,
    }
    result = ReportService(workspace=workspace, store=store).generate_report(
        **arguments
    )
    assert result["success"] is True
    turn = ModelTurn(
        2,
        f"完成 run。\n[閱讀完整回測報告](<{result['path']}>)\n"
        "可由總覽依參數選擇結果，再查看委託與帳戶核對表及配對交易。",
        [ToolCall("generate_report", arguments, result)],
    )
    return DeliveredReport(workspace, store, result, turn)


def test_report_delivery_requires_a_readable_registered_report(
    report: DeliveredReport,
) -> None:
    assert report.verify() == report.path.read_text(encoding="utf-8")
    assert report.path.is_file(), "Delivery verification must not remove the report"


def test_freely_worded_commentary_still_matches_the_delivered_manifest(
    report: DeliveredReport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def verify() -> str:
        return verify_report_delivery(
            report.workspace,
            report.turn,
            "run",
            expected_sections=["overview", "ledger", "trades"],
            expected_commentary=None,
        )

    assert verify()
    logged = capsys.readouterr().out
    evidence = json.loads(logged.split("semantic review: ", 1)[1])
    assert evidence == {
        "run_id": "run",
        "report_id": report.result["report_id"],
        "sections": ["overview", "ledger", "trades"],
        "commentary": [{"title": "研究評語", "text": "僅為歷史模擬。"}],
        "analysis": None,
    }
    report.turn.calls[0].arguments["commentary"] = [
        {"title": "Changed", "text": "Not the saved interpretation"}
    ]
    with pytest.raises(AssertionError, match="manifest commentary"):
        verify()
    assert not capsys.readouterr().out, "Do not log unverified commentary as evidence"


def test_report_delivery_accepts_sdk_encoded_arguments(report: DeliveredReport) -> None:
    report.turn.calls[0].arguments["run_ids"] = '["run"]'
    assert report.verify()


@pytest.mark.parametrize(
    "damage", ["commentary", "prose", "hidden_prose", "fenced_prose"]
)
def test_freely_worded_delivery_requires_interpretation_and_visible_prose(
    report: DeliveredReport, damage: str
) -> None:
    arguments = dict(report.turn.calls[0].arguments)
    if damage == "commentary":
        arguments["commentary"] = []
    result = ReportService(
        workspace=report.workspace, store=report.store
    ).generate_report(**arguments)
    assert result["success"]
    text = f"run [報告](<{result['path']}>)"
    if damage == "commentary":
        text += "。可在總覽選擇參數並查看交易。"
    elif damage == "hidden_prose":
        text += "\n\n<span hidden>可在總覽選擇參數並查看交易。</span>"
    elif damage == "fenced_prose":
        text += "\n```text\n可在總覽選擇參數並查看交易。\n```"
    with pytest.raises(AssertionError, match="interpretation|delivery prose"):
        verify_report_delivery(
            report.workspace,
            ModelTurn(2, text, [ToolCall("generate_report", arguments, result)]),
            "run",
            expected_sections=["overview", "ledger", "trades"],
            expected_commentary=None,
        )


def test_report_delivery_rejects_incomplete_replacement(
    report: DeliveredReport,
) -> None:
    arguments: dict[str, Any] = {"run_ids": ["run"], "sections": ["overview"]}
    replacement = ReportService(
        workspace=report.workspace, store=report.store
    ).generate_report(**arguments)
    assert replacement["success"] is True
    turn = replace(
        report.turn,
        text=f"完成 run。\n[閱讀完整回測報告](<{replacement['path']}>)",
        calls=[
            *report.turn.calls,
            ToolCall("generate_report", arguments, replacement),
        ],
    )
    with pytest.raises(AssertionError, match="sections"):
        report.verify(turn)


def test_report_delivery_checks_the_linked_report_instead_of_the_last_call(
    report: DeliveredReport,
) -> None:
    arguments: dict[str, Any] = {"run_ids": ["run"], "sections": ["overview"]}
    replacement = ReportService(
        workspace=report.workspace, store=report.store
    ).generate_report(**arguments)
    assert replacement["success"] is True
    turn = replace(
        report.turn,
        calls=[
            *report.turn.calls,
            ToolCall("generate_report", arguments, replacement),
        ],
    )
    assert report.verify(turn) == report.path.read_text(encoding="utf-8")
    ambiguous = replace(
        turn,
        text=turn.text + f"\n[另一份報告](<{replacement['path']}>)",
    )
    with pytest.raises(AssertionError, match="exactly one"):
        report.verify(ambiguous)


def test_report_delivery_accepts_an_idempotent_retry_of_the_linked_report(
    report: DeliveredReport,
) -> None:
    turn = replace(report.turn, calls=[*report.turn.calls, *report.turn.calls])
    assert report.verify(turn)


@pytest.mark.parametrize("misleading_arguments", [False, True])
@pytest.mark.parametrize("changed", ["sections", "commentary"])
def test_report_delivery_binds_selected_arguments_and_manifest_to_requirements(
    report: DeliveredReport, changed: str, misleading_arguments: bool
) -> None:
    arguments: dict[str, Any] = {
        **report.turn.calls[0].arguments,
        changed: ["overview"] if changed == "sections" else [],
    }
    replacement = ReportService(
        workspace=report.workspace, store=report.store
    ).generate_report(**arguments)
    assert replacement["success"] is True
    turn = replace(
        report.turn,
        text=f"完成 run。\n[閱讀完整回測報告](<{replacement['path']}>)",
        calls=[
            *report.turn.calls,
            ToolCall(
                "generate_report",
                report.turn.calls[0].arguments if misleading_arguments else arguments,
                replacement,
            ),
        ],
    )
    with pytest.raises(AssertionError, match=changed):
        report.verify(turn)


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "another-run",
        "another_run",
        "run2",
        "run-other",
        "<!-- run -->",
        "<span hidden>run</span>",
    ],
)
def test_report_delivery_requires_the_actual_run_id_in_visible_final_text(
    report: DeliveredReport, identifier: str
) -> None:
    turn = replace(
        report.turn,
        text=f"{identifier}\n\n[閱讀完整回測報告](<{report.path}>)",
    )
    with pytest.raises(AssertionError, match="visible text"):
        report.verify(turn)


@pytest.mark.parametrize("text", ["本次結果為`run`。", "已完成run的回測。"])
def test_report_delivery_accepts_run_id_adjacent_to_chinese(
    report: DeliveredReport, text: str
) -> None:
    turn = replace(
        report.turn,
        text=f"{text}\n\n[閱讀完整回測報告](<{report.path}>)",
    )
    assert report.verify(turn) == report.path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "text",
    [
        "完成 run",
        "完成 run\n{path}",
        "完成 run\n[報告](</another-report.html>)",
        "完成 run\n```markdown\n[報告](<{path}>)\n```",
        "完成 run\n`[報告](<{path}>)`",
        "完成 run\n[](<{path}>)",
        "完成 run\n\n<span hidden>[報告](<{path}>)</span>",
    ],
    ids=[
        "summary-only",
        "bare-path",
        "wrong-report",
        "fenced-code",
        "inline-code",
        "empty-label",
        "hidden-html",
    ],
)
def test_report_delivery_rejects_an_undelivered_link(
    report: DeliveredReport, text: str
) -> None:
    turn = replace(report.turn, text=text.format(path=report.path))
    with pytest.raises(AssertionError, match="clickable link"):
        report.verify(turn)


@pytest.mark.parametrize("damage", ["deleted", "changed", "unregistered"])
def test_report_delivery_rejects_broken_artifacts(
    report: DeliveredReport, damage: str
) -> None:
    if damage == "deleted":
        report.path.unlink()
    elif damage == "changed":
        report.path.write_text("different report", encoding="utf-8")
    else:
        with report.store.connect() as connection:
            connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                (report.result["artifact_id"],),
            )
    with pytest.raises(AssertionError, match="registered artifact|still be readable"):
        report.verify()


def test_report_delivery_rejects_workspace_cleanup_without_recreating_it(
    report: DeliveredReport,
) -> None:
    shutil.rmtree(report.workspace.root)
    with pytest.raises(AssertionError, match="workspace was removed"):
        report.verify()
    assert not report.workspace.root.exists()


@pytest.mark.parametrize("changed", ["request", "response", "registration"])
def test_report_delivery_rejects_another_runs_report(
    report: DeliveredReport, changed: str
) -> None:
    if changed == "request":
        report.turn.calls[0].arguments["run_ids"] = ["other"]
    elif changed == "response":
        report.result["run_ids"] = ["other"]
    else:
        with report.store.connect() as connection:
            connection.execute(
                "UPDATE artifacts SET metadata = ? WHERE artifact_id = ?",
                (json.dumps({"run_ids": ["other"]}), report.result["artifact_id"]),
            )
    with pytest.raises(AssertionError, match="requested run|another run"):
        report.verify()


@pytest.mark.parametrize("field", ["path", "sha256", "artifact_id"])
def test_report_delivery_rejects_mismatched_response_identity(
    report: DeliveredReport, field: str
) -> None:
    report.result[field] = (
        str(report.path.with_name("other.html")) if field == "path" else "f" * 64
    )
    turn = replace(report.turn, text=f"完成 run\n[報告](<{report.result['path']}>)")
    with pytest.raises(
        AssertionError, match="registered artifact|path differs|still be readable"
    ):
        report.verify(turn)


@pytest.mark.parametrize("damage", ["deleted", "changed", "unregistered"])
def test_report_delivery_requires_the_linked_reports_intact_manifest(
    report: DeliveredReport, damage: str
) -> None:
    path = report.path.with_name("manifest.json")
    if damage == "deleted":
        path.unlink()
    elif damage == "changed":
        path.write_text("{}", encoding="utf-8")
    else:
        with report.store.connect() as connection:
            connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                (report.result["manifest_artifact_id"],),
            )
    with pytest.raises(AssertionError, match="manifest"):
        report.verify()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("report_id", "f" * 64),
        ("html_sha256", "f" * 64),
        ("run_ids", ["other"]),
        ("scope_count", 2),
        ("sections", ["overview"]),
        ("commentary", []),
    ],
)
def test_report_delivery_rejects_manifest_identity_and_requirement_mismatches(
    report: DeliveredReport, field: str, value: Any
) -> None:
    path = report.path.with_name("manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest[field] = value
    content = json.dumps(manifest).encode()
    path.write_bytes(content)
    with report.store.connect() as connection:
        connection.execute(
            "UPDATE artifacts SET sha256 = ? WHERE artifact_id = ?",
            (
                hashlib.sha256(content).hexdigest(),
                report.result["manifest_artifact_id"],
            ),
        )
    with pytest.raises(AssertionError, match=f"manifest {field}"):
        report.verify()


@pytest.mark.parametrize("encoded", [False, True])
def test_structured_analysis_verifies_the_actually_linked_report(
    report: DeliveredReport,
    encoded: bool,
) -> None:
    analysis = {
        "summary": "  limited sample  ",
        "observations": "saved results",
        "hypotheses_and_limits": "unverified",
        "next_steps": "study more data",
        "references": [{"run_id": "run", "scope": "full"}],
    }
    arguments = dict(report.turn.calls[0].arguments, analysis=analysis)
    result = ReportService(
        workspace=report.workspace, store=report.store
    ).generate_report(**arguments)
    assert result["success"]
    if encoded:
        arguments["analysis"] = json.dumps(analysis)
    calls = [
        ToolCall(
            "get_run_diagnostics",
            {"run_id": "run"},
            {"success": True, "run_id": "run", "scope": "full"},
        ),
        *report.turn.calls,
        ToolCall("generate_report", arguments, result),
    ]

    def verify(text: str) -> str:
        return verify_report_delivery(
            report.workspace,
            replace(report.turn, calls=calls, text=text),
            "run",
            expected_sections=["overview", "ledger", "trades"],
            expected_commentary=None,
            required_analysis_refs={("run", "full")},
        )

    with pytest.raises(ValueError):
        verify(report.turn.text)
    text = f"完成 run 研究解讀。[報告](<{result['path']}>)"
    assert verify(text)
    # Reusing identical report bytes after reading diagnostics is valid.
    calls.insert(0, calls[-1])
    assert verify(text)
    calls.pop(0)
    calls.pop(0)
    with pytest.raises(AssertionError, match="follow diagnostics"):
        verify(text)


@pytest.mark.parametrize("name", ["prepare_backtest", "submit_execution_decision"])
def test_read_only_research_rejects_failed_execution_attempts(name: str) -> None:
    from tests.e2e.report_delivery import verify_no_execution_attempts

    with pytest.raises(AssertionError, match="must not attempt"):
        verify_no_execution_attempts([ToolCall(name, {}, {"success": False})])
