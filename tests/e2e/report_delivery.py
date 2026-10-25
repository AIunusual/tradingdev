"""Verify report delivery at the final user-visible model-turn boundary."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests.e2e.conversation import links_to_file, visible_text
from tests.e2e.workflow_arguments import effective_workflow_calls
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.reports import ReportResponse
from tradingdev.app.report_service import ReportService
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tradingdev.adapters.storage.filesystem import WorkspacePaths


def verify_report_delivery(
    workspace: WorkspacePaths,
    turn: ModelTurn,
    run_id: str,
    *,
    expected_sections: list[str],
    expected_commentary: list[dict[str, str]],
) -> str:
    """Require a matching usable report link and return its verified HTML text.

    This checks the completed model reply before an automated test removes its
    workspace. It does not prove that a human opened the report or that an outer
    client will retain it after this conversation.
    """
    reports = [
        call
        for call in effective_workflow_calls(turn.calls)
        if call.name == "generate_report"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.arguments.get("run_ids") == [run_id]
    ]
    assert reports, "The final report turn must generate the requested run's report"
    linked: dict[str, tuple[ToolCall, ReportResponse]] = {}
    for call in reports:
        response = ReportResponse.model_validate(call.result)
        if links_to_file(turn.text, Path(response.path)):
            linked[response.report_id] = (call, response)
    assert len(linked) == 1, (
        "The final model reply must contain a clickable link to exactly one "
        "generated report for the requested run"
    )
    call, report = next(iter(linked.values()))
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
    assert (call.arguments.get("commentary") or []) == expected_commentary, (
        "Delivered report commentary differs from the requested commentary"
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
        "commentary": expected_commentary,
    }
    for field, value in expected_manifest.items():
        assert manifest.get(field) == value, (
            f"Delivered report manifest {field} differs from the requested report"
        )
    return content.decode("utf-8")


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
