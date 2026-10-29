"""Saved ML report discovery and delivery evidence, independent of model wording."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt

from tests.e2e.conversation import links_to_file
from tests.e2e.workflow_arguments import effective_workflow_calls
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.reports import ReportAnalysis, ReportResponse
from tradingdev.app.report_service import ReportService
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.registry.leaderboard_store import LeaderboardStore

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace

ML_REPORT_REQUEST = """請找出已儲存、名稱帶「橙海訊號研究」的兩個候選，
把它們不同的評估視窗當作敏感度比較，依平均訊號相關性整理已保存的成果。
請製作可離線開啟的報告，包含兩個候選各自的設定、數值與不可用的分數；
最後提供可點開的報告連結，說明視窗比較與缺值應如何解讀，
以及這份訊號研究與包含交易成本的策略回測之間能作哪些結論。
請依保存證據說明評估資料與訓練資料的關係；沒有記錄的部分請明說未知。
這次只整理既有研究，不需要重新取得行情、訓練或執行策略。
"""

_UNRELATED_OPERATIONS = {
    "build_dataset",
    "build_panel",
    "train_model",
    "predict",
    "evaluate_signal",
    "save_strategy",
    "validate_strategy",
    "dry_run_strategy",
    "promote_strategy",
    "save_research_evidence",
    "submit_execution_decision",
    "cancel_execution_plan",
    "cancel_job",
    "cleanup_strategy_drafts",
    "record_feature_request",
    "list_available_data",
    "list_data_sources",
    "inspect_dataset",
    "ensure_data",
}


def seed_ml_report_candidates(
    workspace: WorkspacePaths,
) -> tuple[LeaderboardEntry, ...]:
    """Store research without any training/evaluation split provenance.

    These records have no dataset or model artifact to establish fitting rows.
    """
    store = LeaderboardStore(workspace.ml_leaderboard)
    entries = tuple(
        LeaderboardEntry(
            entry_id=f"orange-sea-window-{window}",
            label=f"橙海訊號研究／視窗 {window}",
            context=EvaluationContext(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-10",
                horizon=1,
                window=window,
                n_quantiles=2,
                feature_set="returns",
                code_fingerprint="saved-feature-code",
                data_fingerprint="saved-dataset-content",
            ),
            metrics={"ic_mean": score, "hit_rate": 0.6, "ic_std": 0.1},
            n_observations=200,
            model_id="mdl_saved_signal",
            dataset_id="ds_saved_research",
            parameters={"n_estimators": 12, "window": 7},
            signal_fingerprint="saved-signal-content",
        )
        for window, score in ((12, 0.2), (24, None))
    )
    for entry in entries:
        assert not store.record(entry)
    return entries


def verify_ml_report_outcome(
    workspace: MCPWorkspace,
    calls: list[ToolCall],
    turn: ModelTurn,
    entries: tuple[LeaderboardEntry, ...],
) -> dict[str, Any]:
    """Require actual discovery, complete saved evidence and a usable final link.

    Prose checks establish visible coverage of the requested interpretation.
    Printed final wording still needs semantic review; keywords cannot prove
    that a model's financial interpretation is correct.
    """
    effective = effective_workflow_calls(calls)
    assert all(
        call.name not in _UNRELATED_OPERATIONS
        and not call.name.startswith(("prepare_", "execute_"))
        for call in calls
    ), "Saved ML report must not attempt market, training or strategy operations"
    expected_ids = {entry.entry_id for entry in entries}
    discovered: set[str] = set()
    for call in effective:
        if (
            call.name == "leaderboard"
            and isinstance(call.result, dict)
            and call.result.get("success") is True
        ):
            discovered.update(
                row["entry_id"]
                for group in ("ranked", "skipped")
                for row in call.result[group]
            )
        if call.name == "generate_report":
            assert expected_ids <= discovered, (
                "Report candidates must be discovered before report generation"
            )
    assert expected_ids <= discovered, "Missing successful candidate discovery"

    reports = [
        (call, ReportResponse.model_validate(call.result))
        for call in effective
        if call.name == "generate_report"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]
    linked = {
        report.report_id: (call, report)
        for call, report in reports
        if links_to_file(turn.text, Path(report.path))
    }
    assert len(linked) == 1, "Final reply must link exactly one generated ML report"
    call, report = next(iter(linked.values()))
    assert set(call.arguments.get("ml_entry_ids") or []) == expected_ids
    assert call.arguments.get("run_ids") is None
    assert call.arguments.get("comparison_job_id") is None
    assert call.arguments.get("ml_ranking_metric", "ic_mean") == "ic_mean"
    assert call.arguments.get("ml_allow_parameter_axes") is True
    assert call.arguments.get("sections") in (None, ["ml"])
    assert set(report.ml_entry_ids) == expected_ids
    assert report.run_ids == [] and report.comparison_job_id is None
    assert report.scope_count == len(entries)
    assert report.artifact_id == f"report:{report.report_id}:html"

    paths = WorkspacePaths(workspace.workspace)
    assert paths.root.is_dir(), "Workspace must exist during delivery verification"
    store = SQLiteStore(paths)
    registered = store.get_artifact(report.artifact_id)
    assert registered is not None, "Delivered ML HTML must be registered"
    assert registered["metadata"]["ml_entry_ids"] == report.ml_entry_ids
    assert registered["metadata"]["run_ids"] == []
    assert not store.list_runs() and not store.list_jobs(), (
        "Saved ML report must not create a backtest or worker"
    )
    assert LeaderboardStore(paths.ml_leaderboard).entries() == entries, (
        "Reporting must preserve the saved leaderboard"
    )
    for directory in (
        paths.models,
        paths.ml_datasets,
        paths.generated_strategies,
        workspace.data_root,
    ):
        assert not any(path.is_file() for path in directory.rglob("*")), (
            f"Saved ML report created unrelated artifacts in {directory}"
        )
    downloaded = ReportService(workspace=paths, store=store).get_report_download(
        report.report_id, expected_sha256=report.sha256
    )
    assert downloaded.get("success") is True, (
        f"Delivered HTML must remain readable and registered: {downloaded}"
    )
    html = downloaded["content"].decode("utf-8")
    manifest = _read_manifest(store, report)
    assert manifest["report_id"] == manifest["content_sha256"] == report.report_id
    assert manifest["html_sha256"] == report.sha256
    assert manifest["run_ids"] == []
    assert manifest["ml_entry_ids"] == report.ml_entry_ids
    assert manifest["scope_count"] == len(entries)
    assert manifest["sections"] == ["ml"]
    assert manifest["commentary"] == (call.arguments.get("commentary") or [])
    for note in manifest["commentary"]:
        _reject_unsupported_sample_claims(note["title"] + "。" + note["text"])
    assert manifest.get("analysis") == call.arguments.get("analysis")
    assert manifest.get("analysis") is not None, "ML HTML must retain research analysis"
    analysis = ReportAnalysis.model_validate(manifest["analysis"])
    assert analysis.references == []
    for field in (
        analysis.summary,
        analysis.observations,
        analysis.hypotheses_and_limits,
        analysis.next_steps,
    ):
        _reject_unsupported_sample_claims(field)
    assert manifest["sources"] == []
    payload = json.loads(
        html.split("<script id='report-data' type='application/json'>", 1)[1].split(
            "</script>", 1
        )[0]
    )
    assert payload["runs"] == [] and payload["sections"] == ["ml"]
    assert payload["ml"] == manifest["ml"], "HTML and manifest ML evidence differ"
    block = manifest["ml"]
    ordered = {entry.entry_id: entry for entry in entries}
    source = [ordered[entry_id].to_record() for entry_id in block["entry_ids"]]
    encoded = (
        json.dumps(
            source,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert block["source_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert block["entry_ids"] == report.ml_entry_ids
    assert block["kind"] == "ml_signal"
    assert len(block["scopes"]) == len(entries)
    assert [scope["scope_id"] for scope in block["scopes"]] == block["entry_ids"]
    for scope in block["scopes"]:
        entry = ordered[scope["scope_id"]]
        assert scope["values"] == dict(entry.metrics), "Saved metric snapshot changed"
        assert scope["parameters"] == dict(entry.parameters)
        assert scope["metadata"] == {
            **asdict(entry.context),
            "label": entry.label,
            "model_id": entry.model_id,
            "dataset_id": entry.dataset_id,
            "signal_fingerprint": entry.signal_fingerprint,
            "n_observations": entry.n_observations,
        }, "Saved context snapshot changed"
        assert scope["observations"] is None
        assert scope["kind"] == "ml_signal" and scope["mode"] == "signal"
        assert entry.label in html
    assert block["parameter_axes"]["window"] == [12, 24]
    assert block["ranking"]["metric"] == "ic_mean"
    assert block["ranking"]["ranked_scope_ids"] == [entries[0].entry_id]
    skipped = block["ranking"]["skipped"]
    assert len(skipped) == 1
    assert skipped[0]["scope_id"] == entries[1].entry_id
    assert skipped[0]["window"] == 24 and skipped[0]["reason"]
    assert block["caveats"]
    assert set(block["definitions"]) == {"ic_mean", "hit_rate", "ic_std"}
    assert set(block["visible_metrics"]) == {"ic_mean", "hit_rate"}
    assert block["hidden_metrics"] == ["ic_std"]
    for definition in block["definitions"].values():
        assert definition["modes"] == ["signal"]
        assert "requires_annualization" in definition
    _verify_interpretation(turn.text)
    print(
        "[ml-report] Verified final reply for semantic review "
        "(test links expire at cleanup):\n" + turn.text,
        flush=True,
    )
    return manifest


def _read_manifest(store: SQLiteStore, report: ReportResponse) -> dict[str, Any]:
    path = Path(report.path).with_name("manifest.json")
    assert report.manifest_artifact_id == f"report:{report.report_id}:manifest"
    registered = store.get_artifact(report.manifest_artifact_id)
    assert registered is not None, "ML manifest must be registered"
    assert registered["artifact_type"] == "research_report_manifest"
    assert registered["metadata"]["report_id"] == report.report_id
    assert registered["path"] == str(path) and not path.is_symlink()
    assert registered["metadata"]["ml_entry_ids"] == report.ml_entry_ids
    assert registered["metadata"]["run_ids"] == []
    assert path.is_file(), "ML manifest must remain readable"
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == registered["sha256"], (
        "ML manifest bytes differ from registration"
    )
    value = json.loads(content)
    assert isinstance(value, dict)
    return value


def _verify_interpretation(text: str) -> None:
    """Ignore link labels, fenced examples and hidden HTML as interpretation."""
    prose: list[str] = []
    for block in MarkdownIt("commonmark").parse(text):
        children = block.children or []
        if any(child.type == "html_inline" for child in children):
            continue
        in_link = False
        for token in children:
            if token.type == "link_open":
                in_link = True
            elif token.type == "link_close":
                in_link = False
            elif not in_link and token.type in {"text", "code_inline"}:
                prose.append(token.content)
    visible = " ".join(prose)
    _reject_unsupported_sample_claims(visible)
    assert "12" in visible and "24" in visible, (
        "Final reply must interpret both evaluation windows"
    )
    assert re.search(r"訊號|signal", visible, re.IGNORECASE)
    assert re.search(
        r"不可用|缺|無法|未提供|未取得|unavailable|missing", visible, re.IGNORECASE
    ), "Final reply must discuss the unavailable score"
    cost_terms = re.compile(
        r"成本|手續費|滑價|\b(?:costs?|fees?|commissions?|slippage)\b", re.IGNORECASE
    )
    limitation_terms = re.compile(
        r"不|未|無|沒有|需|須|才(?:能|可|足以)|"
        r"\b(?:not|cannot|can't|without|unverified|excludes?|excluded|"
        r"requires?|required|needs?|needed|necessary)\b",
        re.IGNORECASE,
    )
    sentences = re.split(r"[。！？!?；;\n]|\.(?=\s|$)", visible)
    assert re.search(r"回測|backtest", visible, re.IGNORECASE) and any(
        cost_terms.search(sentence) and limitation_terms.search(sentence)
        for sentence in sentences
    ), "Final reply must discuss signal evidence and cost-aware backtest limits"
    _require_unknown_sample_provenance(sentences)


def _require_unknown_sample_provenance(sentences: list[str]) -> None:
    """Require visible provenance coverage, without prescribing one exact reply.

    Unknown scores or unproven profitability do not explain sample provenance.
    These topic/uncertainty checks still require human semantic review.
    """
    unknown = re.compile(
        r"未知|不明|不清楚|不確定|不知道|無法|不能|尚未|未能|未確認|"
        r"(?:未|沒有|未曾)(?:保存|記錄|紀錄|提供|交代)|缺(?:少|乏|失)"
    )
    training = re.compile(r"訓練|擬合")
    relationship = re.compile(r"評估|測試|切分|分割|重疊|參與|是否|曾|保留")
    sample_label = re.compile(r"樣本[內外]")
    identity = re.compile(
        r"是否|與否|歸屬|屬於|來源|關係|判定|確認|認定|區分|身分|身份|"
        r"樣本內[／/或與、 ]*外"
    )
    assert any(
        unknown.search(sentence)
        and (
            (training.search(sentence) and relationship.search(sentence))
            or (sample_label.search(sentence) and identity.search(sentence))
        )
        for sentence in sentences
    ), "Final reply must explain that training/evaluation provenance is unknown"


def _reject_unsupported_sample_claims(text: str) -> None:
    """Catch observed affirmative labels, not every mention of sample provenance.

    This is a regression guard for common assertions, not a semantic classifier.
    Questions, explicit uncertainty and conditional explanations remain valid.
    """
    claim = re.compile(
        r"(?:屬於?|是|為|僅(?:能|可)?支持|只(?:能|可)?支持|可(?:以)?支持)"
        r"\s*(?:已證實的?\s*)?樣本[內外]"
    )
    uncertain = re.compile(
        r"無法|不能|不(?:足以|代表|可|是)|尚未|未能|沒有證據|缺乏證據|是否|(?:不|非|究竟)$"
    )
    conditional = re.compile(
        r"若|如果|假如|倘若|(?:^|[：:,，\s])如[^，,。；;]*(?<!規)則"
    )
    unknown_after = re.compile(r"仍(?:然)?(?:未知|不明)|尚待(?:確認|驗證)|尚未確認")
    for sentence in re.split(r"[。！？!?；;\n]|但|然而|不過", text):
        for match in claim.finditer(sentence):
            before = sentence[: match.start()]
            after = re.split(r"[，,]", sentence[match.end() :], maxsplit=1)[0]
            if (
                uncertain.search(before)
                or conditional.search(before)
                or unknown_after.search(after)
            ):
                continue
            raise AssertionError(
                "Saved leaderboard has no evidence for an affirmative "
                f"in-sample/out-of-sample claim: {match.group()}"
            )
