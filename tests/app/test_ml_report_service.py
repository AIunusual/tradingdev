"""Saved ML evidence reaches immutable reports without acquiring new market data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from tests.app.test_comparison_report import _fixture
from tests.app.test_report_service import _payload, _save

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.registry import LeaderboardStore


@pytest.fixture
def context(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, ReportService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, ReportService(workspace=workspace, store=store)


def _seed(workspace: WorkspacePaths) -> tuple[LeaderboardStore, LeaderboardEntry]:
    store = LeaderboardStore(workspace.ml_leaderboard)
    entry = LeaderboardEntry(
        entry_id="first",
        label="已保存候選",
        context=EvaluationContext(
            "BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1, window=20
        ),
        metrics={"ic_mean": 0.2, "beta": 0.0},
        n_observations=200,
        model_id="model-a",
        dataset_id="dataset-a",
        parameters={"window": 99},
        signal_fingerprint="saved-signal",
    )
    store.record(entry)
    store.record(replace(entry, entry_id="failed", metrics={"ic_mean": None}))
    return store, entry


def _read(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert result["success"], result
    path = Path(result["path"])
    payload, manifest = (
        _payload(path),
        json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    introduction = re.search(
        r"<header>.*?</h1><p>(.*?)</p>", path.read_text(encoding="utf-8"), re.DOTALL
    )
    assert introduction is not None
    assert unescape(introduction[1]) == result["navigation_instructions"]
    assert payload["navigation_instructions"] == manifest["navigation_instructions"]
    assert payload["navigation_instructions"] == result["navigation_instructions"]
    return payload, manifest


@pytest.mark.parametrize("has_run", [False, True])
@pytest.mark.parametrize("show_ml", [False, True])
def test_navigation_matches_selected_ml_and_run_views(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    has_run: bool,
    show_ml: bool,
) -> None:
    workspace, _, service = context
    _seed(workspace)
    sections = ["ml"] if show_ml else []
    if has_run:
        _save(context)
        sections.append("settings")
    result = service.generate_report(
        ["run"] if has_run else None, ml_entry_ids=["first"], sections=sections
    )
    _read(result)
    navigation = result["navigation_instructions"]
    assert ("下載完整 ML 證據 JSON" in navigation) == show_ml
    assert ("未選取 ML 顯示章節" in navigation) != show_ml
    assert ("切換結果" in navigation) == has_run
    assert "離線 HTML" in navigation


def test_ml_only_report_saves_complete_selected_evidence_and_reuses_it(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, database, service = context
    leaderboard, entry = _seed(workspace)
    before = leaderboard.path.read_bytes()
    result = service.generate_report(ml_entry_ids=["first", "failed"])
    payload, manifest = _read(result)
    assert result["run_ids"] == []
    assert result["ml_entry_ids"] == ["first", "failed"]
    assert result["scope_count"] == 2
    assert payload["sections"] == ["ml"]
    assert payload["ml"] == manifest["ml"]
    assert manifest["ml_entry_ids"] == ["first", "failed"]
    block = payload["ml"]
    assert block["ranking"]["ranked_scope_ids"] == ["first"]
    assert block["ranking"]["skipped"][0]["scope_id"] == "failed"
    assert block["scopes"][1]["values"]["ic_mean"] is None
    assert block["scopes"][0]["values"]["beta"] == 0.0
    assert block["scopes"][0]["metadata"]["window"] == 20
    assert block["scopes"][0]["parameters"] == {"window": 99}
    assert block["scopes"][0]["metadata"]["signal_fingerprint"] == "saved-signal"
    records = [
        entry.to_record(),
        replace(entry, entry_id="failed", metrics={"ic_mean": None}).to_record(),
    ]
    expected = (
        json.dumps(
            records,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert block["source_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    artifact = database.get_artifact(result["artifact_id"])
    assert artifact is not None and artifact["run_id"] is None
    assert artifact["metadata"]["ml_entry_ids"] == ["first", "failed"]
    assert service.generate_report(ml_entry_ids=["first", "failed"]) == result
    assert leaderboard.path.read_bytes() == before


def test_ml_snapshot_survives_rerecording_deletion_and_service_restart(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, database, service = context
    leaderboard, entry = _seed(workspace)
    first = service.generate_report(ml_entry_ids=["first"])
    original_html = Path(first["path"]).read_bytes()
    leaderboard.record(replace(entry, metrics={"ic_mean": 0.9}))
    second = service.generate_report(ml_entry_ids=["first"])
    assert first["report_id"] != second["report_id"]
    old, _ = _read(first)
    new, _ = _read(second)
    assert old["ml"]["scopes"][0]["values"]["ic_mean"] == 0.2
    assert new["ml"]["scopes"][0]["values"]["ic_mean"] == 0.9
    leaderboard.path.unlink()
    restarted = ReportService(workspace=workspace, store=database)
    download = restarted.get_report_download(
        first["report_id"], expected_sha256=first["sha256"]
    )
    assert download["success"] and download["content"] == original_html
    missing = restarted.generate_report(ml_entry_ids=["first"])
    assert missing["code"] == "ml_entry_not_found"


def test_ml_and_backtest_are_separate_evidence(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _seed(workspace)
    _save(context)
    result = service.generate_report(
        ["run"], ml_entry_ids=["first"], sections=["provenance", "ml"]
    )
    payload, manifest = _read(result)
    assert result["run_ids"] == ["run"] and result["scope_count"] == 2
    assert payload["runs"][0]["scopes"][0]["kind"] == "backtest"
    assert payload["ml"]["scopes"][0]["kind"] == "ml_signal"
    assert manifest["available_scopes"] == {"run": ["full"]}
    assert "各自獨立" in result["navigation_instructions"]


def test_unselected_ml_section_still_embeds_evidence(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _seed(workspace)
    result = service.generate_report(ml_entry_ids=["first"], sections=[])
    payload, manifest = _read(result)
    assert payload["sections"] == [] and payload["ml"]["scopes"]
    assert "ml" in manifest["omitted_sections"]
    assert "未選取 ML" in result["navigation_instructions"]


@pytest.mark.parametrize("successful_run", [True, False])
def test_fixed_comparison_can_include_independent_ml_evidence(
    tmp_path: Path, successful_run: bool
) -> None:
    fixture = _fixture(
        tmp_path,
        statuses=("succeeded", "failed") if successful_run else ("failed", "not_run"),
    )
    _seed(fixture.workspace)
    result = fixture.service.generate_report(
        comparison_job_id="comparison", ml_entry_ids=["first"], sections=["ml"]
    )
    payload, manifest = _read(result)
    assert result["comparison_job_id"] == "comparison"
    assert result["scope_count"] == 1 + int(successful_run)
    assert len(payload["fixed_comparison"]["candidates"]) == 2
    assert manifest["ml"] == payload["ml"]
    assert payload["ml"]["scopes"][0]["metadata"]["model_id"] == "model-a"


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"ml_entry_ids": []}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": ["first", "first"]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [str(i) for i in range(51)]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": "first"}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [""]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": [1]}, "invalid_report_ml_selection"),
        ({"ml_entry_ids": ["absent"]}, "ml_entry_not_found"),
        ({"ml_allow_parameter_axes": True}, "invalid_report_ml_selection"),
        ({"ml_ranking_metric": "sharpe"}, "invalid_report_ml_selection"),
        (
            {"ml_entry_ids": ["first"], "ml_ranking_metric": "beta"},
            "report_ml_evidence_invalid",
        ),
        (
            {"ml_entry_ids": ["first"], "ml_allow_parameter_axes": 1},
            "invalid_report_ml_selection",
        ),
    ],
)
def test_invalid_ml_selection_does_not_publish_partial_reports(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    options: dict[str, Any],
    code: str,
) -> None:
    workspace, _, service = context
    _seed(workspace)
    result = service.generate_report(**options)
    assert result["success"] is False and result["code"] == code
    assert not list((workspace.root / "reports").glob("**/*"))


def test_surface_requires_explicit_opt_in_and_keeps_the_failed_coordinate(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    leaderboard, entry = _seed(workspace)
    leaderboard.record(
        replace(
            entry,
            entry_id="failed",
            context=replace(entry.context, window=30),
            metrics={"ic_mean": None},
        )
    )
    assert service.generate_report(ml_entry_ids=["first", "failed"])["code"] == (
        "report_ml_evidence_invalid"
    )
    result = service.generate_report(
        ml_entry_ids=["first", "failed"], ml_allow_parameter_axes=True
    )
    payload, _ = _read(result)
    assert payload["ml"]["parameter_axes"]["window"] == [20, 30]
    assert any("evaluation setting" in text for text in payload["ml"]["caveats"])


@pytest.mark.parametrize("corruption", ["malformed", "duplicate", "unknown_metric"])
def test_invalid_saved_evidence_is_not_silently_skipped(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], corruption: str
) -> None:
    workspace, _, service = context
    leaderboard, entry = _seed(workspace)
    if corruption == "malformed":
        leaderboard.path.write_text("{broken", encoding="utf-8")
    elif corruption == "duplicate":
        with leaderboard.path.open("a") as stream:
            stream.write(json.dumps(entry.to_record()) + "\n")
    else:
        leaderboard.record(replace(entry, metrics={"ic_mean": 0.2, "invented": 1.0}))
    result = service.generate_report(ml_entry_ids=["first"])
    assert result["code"] == "report_ml_evidence_invalid"
    assert not list((workspace.root / "reports").glob("**/*"))


def test_no_ml_request_report_bytes_are_pinned(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    # Re-pinned at template 24 when the benchmark section was added; the ML
    # renderer must still be absent from a request that never asks for it.
    _, store, service = context
    _save(context)
    with store.connect() as connection:
        connection.execute("update runs set created_at=?", ("2026-01-01T00:00:00Z",))
    result = service.generate_report(["run"], sections=[])
    payload, manifest = _read(result)
    assert "ml" not in payload and "ml" not in manifest
    assert "ml" not in manifest["omitted_sections"]
    path = Path(result["path"]).parent / "manifest.json"
    # One comparison so a deliberate re-baseline shows every digest at once.
    assert (
        result["report_id"],
        result["sha256"],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ) == (
        "4ff6c967ef3cf6b9cdbbb9c61a0dccd5dc8def8bb42948cb61a971ef3820a44d",
        "b768b19434280d02a7657d2142de50dadc289aae70926fd7a75dbb13471677f9",
        "8727f7b052fa975f2588e7e13d0aa8bbbaa4bd4d68119e07a83003b2a5951ce4",
    )
