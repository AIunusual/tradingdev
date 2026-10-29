"""Verify report delivery at the final user-visible model-turn boundary."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt

from tests.e2e.conversation import links_to_file, visible_text
from tests.e2e.workflow_arguments import effective_workflow_calls
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.reports import ReportAnalysis, ReportResponse
from tradingdev.app.report_service import ReportService
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.app.contracts.research_notes import NoteExport


def verify_archive_delivery(text: str, report: NoteExport) -> None:
    """Accept a local reading link or the documented ZIP extraction workflow."""
    assert links_to_file(text, Path(report.path)), (
        "Final reply must link the actual ZIP"
    )
    assert report.html_path is not None
    prose = " ".join(
        visible_text(block.content)
        for block in MarkdownIt("commonmark").parse(text)
        if block.type == "inline"
    )
    assert links_to_file(text, Path(report.html_path)) or (
        "index.html" in prose
        and re.search(r"解壓|extract|unzip", prose, flags=re.IGNORECASE)
    ), "Final reply must explain the archive reading entry point"


def verify_report_delivery(
    workspace: WorkspacePaths,
    turn: ModelTurn,
    run_id: str,
    *,
    expected_sections: list[str],
    expected_commentary: list[dict[str, str]] | None,
    required_analysis_refs: set[tuple[str, str]] | None = None,
) -> str:
    """Require a matching usable report link and return its verified HTML text.

    This checks the completed model reply before an automated test removes its
    workspace. It does not prove that a human opened the report or that an outer
    client will retain it after this conversation.
    """
    reports = [
        (index, call)
        for index, call in enumerate(effective_workflow_calls(turn.calls))
        if call.name == "generate_report"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.arguments.get("run_ids") == [run_id]
    ]
    assert reports, "The final report turn must generate the requested run's report"
    linked: dict[str, tuple[int, ToolCall, ReportResponse]] = {}
    for index, call in reports:
        response = ReportResponse.model_validate(call.result)
        if links_to_file(turn.text, Path(response.path)):
            linked[response.report_id] = (index, call, response)
    assert len(linked) == 1, (
        "The final model reply must contain a clickable link to exactly one "
        "generated report for the requested run"
    )
    call_index, call, report = next(iter(linked.values()))
    assert report.run_ids == [run_id], (
        "Delivered report response belongs to another run"
    )
    assert re.search(
        rf"(?<![\w-]){re.escape(run_id)}(?![\w-])",
        visible_text(turn.text),
        flags=re.ASCII,
    ), "The final model reply must identify the delivered run in visible text"
    selected_sections = call.arguments.get("sections")
    if selected_sections is None:
        selected_sections = list(REPORT_RECIPES["standard"])
    assert selected_sections == expected_sections, (
        "Delivered report sections differ from the requested sections"
    )
    if required_analysis_refs is not None:
        analysis = ReportAnalysis.model_validate(call.arguments.get("analysis"))
        assert {
            (r.run_id, r.scope) for r in analysis.references
        } == required_analysis_refs
        preceding = turn.calls[:call_index]
        for reference_run, reference_scope in required_analysis_refs:
            assert any(
                c.name == "get_run_diagnostics"
                and c.result.get("success") is True
                and c.result.get("run_id") == reference_run
                and c.result.get("scope") == reference_scope
                for c in preceding
            ), "Delivered interpretation must follow diagnostics for its references"
    commentary = call.arguments.get("commentary") or []
    if expected_commentary is not None:
        assert commentary == expected_commentary, (
            "Delivered report commentary differs from the requested commentary"
        )
    else:
        if call.arguments.get("analysis") is not None:
            ReportAnalysis.model_validate(call.arguments["analysis"])
        assert call.arguments.get("analysis") or (
            commentary
            and all(
                item.get("title", "").strip() and item.get("text", "").strip()
                for item in commentary
            )
        ), "A freely worded research report requires interpretation"
        assert _has_delivery_prose(turn.text, run_id), (
            "The final reply must include delivery prose beyond its run ID and links"
        )
    path = Path(report.path)
    assert (workspace.root / "tradingdev.sqlite").is_file(), (
        "Report workspace was removed before final delivery verification"
    )
    store = SQLiteStore(workspace)
    registered = store.get_artifact(report.artifact_id)
    assert registered is not None, "Delivered report is not a registered artifact"
    assert registered["metadata"].get("run_ids") == [run_id], (
        "Delivered report registration belongs to another run"
    )
    assert registered["path"] == report.path, (
        "Delivered report path differs from the registered artifact"
    )
    assert report.artifact_id == f"report:{report.report_id}:html", (
        "Delivered report artifact identity differs from the response"
    )
    downloaded = ReportService(workspace=workspace, store=store).get_report_download(
        report.report_id, expected_sha256=report.sha256
    )
    assert downloaded.get("success") is True, (
        f"Delivered report must still be readable and match its saved identity: "
        f"{downloaded}"
    )
    content = downloaded["content"]
    assert isinstance(content, bytes)
    manifest = _read_manifest(store, report, path)
    expected_manifest = {
        "report_id": report.report_id,
        "content_sha256": report.report_id,
        "html_sha256": report.sha256,
        "run_ids": [run_id],
        "scope_count": report.scope_count,
        "sections": expected_sections,
        "commentary": commentary,
        "analysis": call.arguments.get("analysis"),
    }
    for field, value in expected_manifest.items():
        assert manifest.get(field) == value, (
            f"Delivered report manifest {field} differs from the requested report"
        )
    if expected_commentary is None:
        print(
            "[discovery] Verified report commentary for semantic review: "
            + json.dumps(
                {
                    "run_id": run_id,
                    "report_id": report.report_id,
                    "sections": expected_sections,
                    "commentary": manifest["commentary"],
                    "analysis": manifest.get("analysis"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return content.decode("utf-8")


def _has_delivery_prose(text: str, run_id: str) -> bool:
    """Reject identifier/link-only replies; semantic correctness needs review.

    Do not reward report link labels, hidden HTML or code blocks as explanation,
    and do not prescribe wording or an arbitrary character-count threshold.
    """
    parts: list[str] = []
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
                parts.append(token.content)
    return any(character.isalpha() for character in " ".join(parts).replace(run_id, ""))


def _read_manifest(
    store: SQLiteStore, report: ReportResponse, html_path: Path
) -> dict[str, Any]:
    """Check the selected HTML's registered manifest, not another report's evidence."""
    assert report.manifest_artifact_id == f"report:{report.report_id}:manifest", (
        "Delivered report manifest artifact identity differs from the response"
    )
    registered = store.get_artifact(report.manifest_artifact_id)
    assert registered is not None, "Delivered report manifest is not registered"
    path = html_path.with_name("manifest.json")
    assert (
        registered["artifact_type"] == "research_report_manifest"
        and registered["path"] == str(path)
        and registered["metadata"].get("report_id") == report.report_id
        and registered["metadata"].get("run_ids") == report.run_ids
        and not path.is_symlink()
    ), "Delivered report manifest registration differs from the response"
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise AssertionError(
            "Delivered report manifest must still be readable"
        ) from exc
    assert hashlib.sha256(content).hexdigest() == registered["sha256"], (
        "Delivered report manifest SHA differs from its registration"
    )
    manifest = json.loads(content)
    assert isinstance(manifest, dict), "Delivered report manifest must be an object"
    return manifest


def verify_no_execution_attempts(calls: list[ToolCall]) -> None:
    """Check raw attempts including rejected and failed execution requests."""
    assert not any(
        call.name.startswith("prepare_") or call.name == "submit_execution_decision"
        for call in calls
    ), "Read-only research must not attempt to prepare or submit execution"
