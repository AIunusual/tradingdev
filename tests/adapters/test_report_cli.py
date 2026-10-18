"""CLI delegates report generation to the application service and reports errors."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.cli.reports import main
from tradingdev.app.report_service import ReportService

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("success", [True, False])
def test_report_cli_preserves_service_result_and_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    success: bool,
) -> None:
    response: dict[str, Any] = (
        {"success": True, "path": "報告.html"}
        if success
        else {"success": False, "code": "run_not_found", "error": "Unknown run"}
    )
    received: list[list[str]] = []

    def generate(
        _: ReportService, run_ids: list[str], **options: Any
    ) -> dict[str, Any]:
        received.append(run_ids)
        assert options == {"sections": None, "commentary": None}
        return response

    monkeypatch.setattr(ReportService, "generate_report", generate)
    args = ["--workspace", str(tmp_path), "--run-id", "a", "--run-id", "b"]
    if success:
        main(args)
    else:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 1
    assert received == [["a", "b"]]
    assert json.loads(capsys.readouterr().out) == response


@pytest.mark.parametrize(
    "sections", [[], ["overview", "ledger", "trades", "provenance"]]
)
def test_report_cli_custom_sections_and_commentary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    sections: list[str],
) -> None:
    notes = [{"title": "評語", "text": "測試結果。"}]
    path = tmp_path / "notes.json"
    path.write_text(json.dumps(notes), encoding="utf-8")

    def generate(
        _: ReportService, run_ids: list[str], **options: Any
    ) -> dict[str, Any]:
        assert run_ids == ["a"]
        assert options == {"sections": sections, "commentary": notes}
        return {"success": True}

    monkeypatch.setattr(ReportService, "generate_report", generate)
    main(
        [
            "--workspace",
            str(tmp_path),
            "--run-id",
            "a",
            "--sections",
            *sections,
            "--commentary-json",
            str(path),
        ]
    )
    assert json.loads(capsys.readouterr().out)["success"]


def test_report_cli_invalid_commentary_is_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "notes.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit, match="1"):
        main(
            [
                "--workspace",
                str(tmp_path),
                "--run-id",
                "a",
                "--commentary-json",
                str(path),
            ]
        )
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False and result["code"] == "invalid_report_commentary"
