"""CLI adapter is a thin entry point over the shared statistics service."""

from __future__ import annotations

import json

import pytest

from tradingdev.adapters.cli import statistics
from tradingdev.app.statistics_service import StatisticsService


def test_cli_forwards_one_selection_to_the_shared_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str | None, bool]] = []

    def analyze(
        self: object,
        run_id: str,
        scope: str | None = None,
        *,
        save_artifact: bool = False,
    ) -> dict[str, object]:
        calls.append((run_id, scope, save_artifact))
        return {"success": True, "algorithm_version": "statistics-analysis-v1"}

    monkeypatch.setattr(StatisticsService, "analyze", analyze)
    statistics.main(["--run-id", "run", "--scope", "trial/1/train", "--save-artifact"])
    assert calls == [("run", "trial/1/train", True)]
    assert json.loads(capsys.readouterr().out)["success"] is True


def test_cli_reports_service_failure_with_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def analyze(
        self: object,
        run_id: str,
        scope: str | None = None,
        *,
        save_artifact: bool = False,
    ) -> dict[str, object]:
        return {"success": False, "code": "run_not_found", "error": "Unknown run"}

    monkeypatch.setattr(StatisticsService, "analyze", analyze)
    with pytest.raises(SystemExit) as caught:
        statistics.main(["--run-id", "missing"])
    assert caught.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "run_not_found"
