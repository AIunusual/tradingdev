"""Offline report presentation, source integrity and write-once publication."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
from dataclasses import asdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from tradingdev.adapters.cli.reports import main as report_cli_main
from tradingdev.adapters.reporting.labels import metric_label, unit_label
from tradingdev.adapters.reporting.quantstats_report import QuantStatsReportError
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    BenchmarkScope,
    FoldStats,
    MetricDefinitionSnapshot,
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    PerformanceScope,
    ScopeObservations,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.presentation.reports import REPORT_TEMPLATE_VERSION
from tradingdev.domain.strategies.execution import StrategyExecution


@pytest.fixture
def context(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, ReportService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, ReportService(workspace=workspace, store=store)


def _scope(**changes: Any) -> PerformanceScope:
    return PerformanceScope.model_validate(
        {
            "mode": "signal",
            "values": {"total_return": 0.25, "win_rate": None},
            "parameters": {"slow": 26},
            "metadata": {
                "unavailable": {"win_rate": "no_closed_trades"},
                "execution_context": {
                    "symbol": "BTC/USDT",
                    "fees": 0.0004,
                    "slippage": 0.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
                "settings": {"initial_cash": 100.0},
                "providers": {"fixture": "1"},
            },
            **changes,
        }
    )


def _save(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    run_id: str = "run",
    *,
    scopes: dict[str, PerformanceScope] | None = None,
    default_scope: str = "full",
    selected_train: str | None = None,
    modern: bool = False,
    volume: bool = False,
    timestamps: tuple[str, ...] | None = (
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
    ),
    trades: list[dict[str, Any]] | None = None,
    definitions: dict[str, MetricDefinitionSnapshot] | None = None,
    benchmarks: dict[str, BenchmarkScope] | None = None,
) -> None:
    workspace, store, _ = context
    scopes = scopes or {"full": _scope()}
    manifest = None
    if modern:
        manifest = ExecutionManifest.create(
            kind="backtest",
            config={
                "strategy": {"id": "fixture", "parameters": {"slow": 26}},
                "backtest": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1d",
                    "init_cash": 100.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            },
            strategy_execution=StrategyExecution(
                kind="generated", constructor_kwargs={"slow": 26, "signal": 9}
            ),
        )
    digest = manifest.manifest_hash if manifest else None
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        revision_id="old-revision",
        manifest_hash=digest,
        artifact_dir=workspace.runs / run_id,
        metrics=scopes[default_scope].model_dump(mode="json")["values"],
        dataset_id="saved-market-hash",
    )
    if manifest is not None:
        path = ExecutionManifestStore(workspace).publish(run_id, manifest)
        store.create_artifact(
            artifact_id=f"{run_id}:execution_manifest",
            run_id=run_id,
            artifact_type="execution_manifest",
            path=path,
            sha256=sha256_file(path),
        )
    definitions = definitions or {
        key: MetricDefinitionSnapshot.model_validate(asdict(METRIC_CATALOG[key]))
        for scope in scopes.values()
        for key in scope.values
    }
    for benchmark in (benchmarks or {}).values():
        for key in benchmark.values:
            definitions.setdefault(
                key,
                MetricDefinitionSnapshot.model_validate(asdict(METRIC_CATALOG[key])),
            )
    observations = ScopeObservations(
        init_cash=None if volume else 100.0,
        equity_curve=[100.0, 90.0, 125.0],
        returns=None if volume else [0.0, -0.1, 0.3888889],
        timestamps=list(timestamps) if timestamps is not None else None,
        trades=trades
        if trades is not None
        else [
            {
                "entry_idx": 0,
                "exit_idx": 2,
                "status": "open",
                "direction": -1,
                "entry_price": 50.0,
                "exit_price": 45.0,
                "net_pnl": 25.0,
                "note": "</script><img src=x onerror=alert(1)>",
            }
        ],
    )
    PerformanceStore(workspace, store).publish(
        PerformanceArtifacts(
            performance=PerformanceBundle(
                run_id=run_id,
                manifest_hash=digest,
                default_scope=default_scope,
                scopes=scopes,
                definitions=definitions,
                selected_train_scope=selected_train,
                benchmarks=benchmarks or {},
            ),
            observations=ObservationsBundle(
                run_id=run_id,
                manifest_hash=digest,
                scopes={
                    key: observations
                    for key, scope in scopes.items()
                    if scope.kind == "backtest"
                },
            ),
        )
    )


def _payload(path: Path) -> dict[str, Any]:
    text = path.read_text()
    content = text.split("<script id='report-data' type='application/json'>", 1)[1]
    parsed: dict[str, Any] = json.loads(content.split("</script>", 1)[0])
    return parsed


def _static_table(text: str, identifier: str) -> tuple[list[str], list[list[str]]]:
    table = text.split(f"<table id='{identifier}-static'>", 1)[1].split("</table>", 1)[
        0
    ]
    headers = re.findall(r"<th>(.*?)</th>", table)
    body = table.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    rows = [
        re.findall(r"<td(?: [^>]*)?>(.*?)</td>", row, flags=re.DOTALL)
        for row in re.findall(r"<tr>(.*?)</tr>", body, flags=re.DOTALL)
    ]
    return headers, rows


def _csv_rows(text: str, identifier: str) -> list[dict[str, str]]:
    encoded = re.search(
        rf"data-table-csv='{re.escape(identifier)}' data-csv='([^']*)'", text
    )
    assert encoded is not None
    return list(
        csv.DictReader(io.StringIO(base64.b64decode(encoded[1]).decode("utf-8-sig")))
    )


class _ExternalResources(HTMLParser):
    """Inspect actual resource elements, not URLs in bundled library comments."""

    def __init__(self) -> None:
        super().__init__()
        self.resources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"script", "link", "img", "iframe"}:
            return
        for name, value in attrs:
            if (
                name in {"src", "href"}
                and value
                and value.startswith(("https://", "http://", "//"))
            ):
                self.resources.append(value)


def test_quantstats_analysis_is_scoped_derived_evidence_with_versioned_identity(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _, service = context
    _save(context, scopes={"train": _scope(), "test": _scope()}, default_scope="test")
    before = {p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()}
    provider_version = "one"
    calls = []

    def analysis(scope: dict[str, Any]) -> dict[str, Any]:
        calls.append(scope["scope_id"])
        return {
            "available": True,
            "reason": None,
            "html": f"<!doctype html><p>Native {scope['scope_id']}</p>",
            "tables_html": "<table><tr><td>Native metric 12.34%</td></tr></table>",
            "metadata": {"providers": {"quantstats": provider_version}},
        }

    monkeypatch.setattr(
        "tradingdev.app.report_service.build_quantstats_report", analysis
    )
    first = service.generate_report(["run"], sections=["metrics"])
    assert first["success"], first
    assert set(calls) == {"train", "test"}
    path = Path(first["path"])
    content = path.read_text()
    scopes = _payload(path)["runs"][0]["scopes"]
    for scope in scopes:
        assert scope["values"]["total_return"] == 0.25
        assert scope["quantstats"]["html"].endswith(f"{scope['scope_id']}</p>")
    assert "sandbox=''" in content and "frame-src 'self'" in content
    assert "data-quantstats='0:0'" in content and "data-quantstats='0:1'" in content
    assert "下載 QuantStats 完整績效 HTML" in content
    assert "不是逐筆交易統計" in content
    # Text readers can access native metrics without evaluating the iframe loader.
    body = content.split("<script id='report-data'", 1)[0]
    assert "<td>Native metric 12.34%</td>" in body
    assert body.count("<td>Native metric 12.34%</td>") == 2
    no_script = body.split("<noscript>", 1)[1].split("</noscript>", 1)[0]
    assert ".scope[hidden],#result-view[hidden],#overview-view[hidden]" in no_script
    assert "display:block!important" in no_script
    assert ".bokeh-table,.quantstats-frame,[data-quantstats-download]" in no_script
    assert "以下展開全部結果範圍" in no_script
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert set(manifest["derived_analytics"]["run"]) == {"train", "test"}
    assert "html" not in manifest["derived_analytics"]["run"]["train"]
    assert "tables_html" not in manifest["derived_analytics"]["run"]["train"]
    assert service.generate_report(["run"], sections=["metrics"]) == first

    provider_version = "two"
    second = service.generate_report(["run"], sections=["metrics"])
    assert second["success"] and second["report_id"] != first["report_id"]
    assert path.read_text() == content
    assert {
        p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()
    } == before


def test_omitted_metrics_do_not_generate_quantstats_analysis(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, service = context
    _save(context)
    monkeypatch.setattr(
        "tradingdev.app.report_service.build_quantstats_report",
        lambda _scope: pytest.fail("omitted analytics must not run"),
    )
    response = service.generate_report(["run"], sections=["trades", "equity"])
    assert response["success"], response
    scope = _payload(Path(response["path"]))["runs"][0]["scopes"][0]
    assert "quantstats" not in scope and scope["observations"]["returns"]


@pytest.mark.parametrize(
    "failure",
    [ValueError("invalid native output"), QuantStatsReportError("provider failure")],
)
def test_quantstats_generation_failure_does_not_publish_or_change_sources(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    workspace, _, service = context
    _save(context)
    before = {p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()}

    def fail(_scope: dict[str, Any]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr("tradingdev.app.report_service.build_quantstats_report", fail)
    response = service.generate_report(["run"])
    assert response["code"] == "report_generation_failed"
    with pytest.raises(SystemExit) as error:
        report_cli_main(["--workspace", str(workspace.root), "--run-id", "run"])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == response
    assert not (workspace.root / "reports").exists()
    assert {
        p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()
    } == before


def test_report_uses_immutable_saved_evidence_and_deterministic_retry(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, store, service = context
    _save(context, modern=True)
    before = {p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()}
    monkeypatch.setattr("pickle.load", lambda *_: pytest.fail("must not unpickle"))
    monkeypatch.setattr(
        "tradingdev.app.run_lineage.read_strategy_snapshot",
        lambda *_args, **_kwargs: pytest.fail("must not read current strategy"),
    )
    first = service.generate_report(["run"])
    assert first["success"], first
    path = Path(first["path"])
    original = path.read_bytes()
    assert service.generate_report(["run"]) == first
    assert path.read_bytes() == original
    assert {
        p: p.read_bytes() for p in workspace.runs.rglob("*") if p.is_file()
    } == before
    assert first["sha256"] == hashlib.sha256(original).hexdigest()
    registered = store.get_artifact(first["artifact_id"])
    assert registered is not None
    assert registered["artifact_type"] == "research_report_html"
    payload = _payload(path)
    scope = payload["runs"][0]["scopes"][0]
    assert scope["parameters"] == {"slow": 26, "signal": 9}
    assert scope["values"]["total_return"] == 0.25
    assert scope["observations"]["trades"][0]["exit_price"] == 45.0
    text = path.read_text()
    assert "25.00%" in text and "沒有已平倉交易" in text
    assert "期末估值時間（UTC）" in text and "期末估值價格" in text
    assert "不是實際平倉" in text and "暖機" in text
    assert "&lt;/script&gt;&lt;img" in text
    assert "</script><img" not in text and "\\u003c/script\\u003e" in text
    resources = _ExternalResources()
    resources.feed(text)
    assert resources.resources == []
    assert "<svg" in text
    assert not list((workspace.root / "reports").rglob(".report-*"))
    assert not list((workspace.root / "reports").glob("*.publishing"))


@pytest.mark.parametrize("via_cli", [False, True])
@pytest.mark.parametrize(
    ("status", "direction", "status_label", "direction_label"),
    [
        ("closed", 1, "已平倉", "做多"),
        ("open", -1, "未平倉", "做空"),
        (None, None, "未知", "未知"),
        ("unrecognized", 0, "未識別狀態：unrecognized", "未識別方向：0"),
    ],
)
def test_trade_table_has_readable_labels_without_changing_saved_records(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    capsys: pytest.CaptureFixture[str],
    via_cli: bool,
    status: str | None,
    direction: int | None,
    status_label: str,
    direction_label: str,
) -> None:
    workspace, _, service = context
    trade = {
        "status": status,
        "direction": direction,
        "entry_idx": 0,
        "exit_idx": 2,
        "entry_price": 50.0,
        "exit_price": 45.0,
        "size": 2.0,
        "net_pnl": 9.9,
        "fee": 0.1,
        "note": "<script>unsafe()</script>",
        "custom_details": {"unrounded": 1.23456789012345, "flag": False},
    }
    _save(context, trades=[trade])
    if via_cli:
        report_cli_main(
            [
                "--workspace",
                str(workspace.root),
                "--run-id",
                "run",
                "--sections",
                "trades",
            ]
        )
        response = json.loads(capsys.readouterr().out)
    else:
        response = service.generate_report(["run"], sections=["trades"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    headers, rows = _static_table(text, "trades-0-0")
    assert headers == [
        "序號",
        "進場時間（UTC）",
        "平倉時間（UTC）",
        "期末估值時間（UTC）",
        "交易方向",
        "持倉狀態",
        "進場價格（報價貨幣／標的單位）",
        "平倉價格（報價貨幣／標的單位）",
        "期末估值價格（報價貨幣／標的單位）",
        "數量（標的單位）",
        "進場手續費（報價貨幣）",
        "平倉手續費（報價貨幣）",
        "手續費合計（報價貨幣）",
        "毛損益（報價貨幣）",
        "淨損益（報價貨幣）",
        "進場 K 線索引（從 0 起算）",
        "平倉／期末估值 K 線索引（從 0 起算）",
        "原始資料（JSON）",
    ]
    assert len(rows) == 1
    cells = rows[0]
    final_time = "2024-01-03T00:00:00+00:00"
    missing = "—"
    assert cells[:-1] == [
        "1",
        "2024-01-01T00:00:00+00:00",
        final_time if status == "closed" else missing,
        final_time if status == "open" else missing,
        direction_label,
        status_label,
        "50.00",
        "45.00" if status == "closed" else missing,
        "45.00" if status == "open" else missing,
        "2.00",
        missing,
        missing,
        "0.10",
        missing,
        "9.90",
        "0",
        "2",
    ]
    assert "&lt;script&gt;unsafe()&lt;/script&gt;" in cells[-1]
    assert json.loads(unescape(cells[-1])) == trade
    saved_scope = _payload(path)["runs"][0]["scopes"][0]
    assert saved_scope["observations"]["trades"] == [trade]
    table = saved_scope["tables"]["trades"]
    assert [column["title"] for column in table["columns"]] == headers
    assert table["filter_field"] == "filter_status"
    assert table["rows"][0]["filter_status"] == status
    assert json.loads(table["rows"][0]["raw"]) == trade
    assert "data-table-json='0:0:trades'" in text
    csv_rows = _csv_rows(text, "trades-0-0")
    assert len(csv_rows) == 1
    assert list(csv_rows[0]) == headers
    assert csv_rows[0]["淨損益（報價貨幣）"] == "9.9"
    assert csv_rows[0]["手續費合計（報價貨幣）"] == "0.1"
    assert csv_rows[0]["平倉價格（報價貨幣／標的單位）"] == (
        "45.0" if status == "closed" else ""
    )
    assert csv_rows[0]["期末估值價格（報價貨幣／標的單位）"] == (
        "45.0" if status == "open" else ""
    )
    assert json.loads(csv_rows[0]["原始資料（JSON）"]) == trade
    assert "下載全部紀錄 CSV（中文欄名、完整精度）" in text


@pytest.mark.parametrize(
    ("value", "display"),
    [
        (123.456, "123.46"),
        (-123.456, "-123.46"),
        (0.0, "0.00"),
        (-0.0, "0.00"),
        (0.0049, "0.00"),
        (-0.0049, "0.00"),
        (1e-8, "0.00"),
        (12, "12.00"),
        (12.3451, "12.35"),
        (12.3452, "12.35"),
    ],
)
def test_trade_table_rounds_display_but_preserves_numeric_precision(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    value: int | float,
    display: str,
) -> None:
    _, _, service = context
    trades = [
        {
            "status": status,
            "direction": 1,
            "entry_price": value,
            "exit_price": value,
            "size": value,
            "net_pnl": value,
            "fee": value,
        }
        for status in ("closed", "open")
    ]
    _save(context, trades=trades)
    response = service.generate_report(["run"], sections=["trades"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    headers, rows = _static_table(text, "trades-0-0")
    assert len(rows) == 2
    scope = _payload(path)["runs"][0]["scopes"][0]
    spec = scope["tables"]["trades"]
    csv_rows = _csv_rows(text, "trades-0-0")
    for index, (cells, trade) in enumerate(zip(rows, trades, strict=True)):
        assert cells[0] == str(index + 1)
        for field in ("entry_price", "size", "fee", "net_pnl") + (
            "exit_price" if trade["status"] == "closed" else "mark_price",
        ):
            column_index = next(
                i
                for i, column in enumerate(spec["columns"])
                if column["field"] == field
            )
            assert cells[column_index] == display
            assert spec["rows"][index][field] == value
            assert spec["full_precision_values"][field][index] == str(value)
            assert csv_rows[index][headers[column_index]] == str(value)
        assert json.loads(unescape(cells[-1])) == trade
        assert json.loads(csv_rows[index]["原始資料（JSON）"]) == trade
    assert "四捨五入至小數點後兩位" in text
    assert "CSV／JSON 保留全部欄位、全部紀錄與完整精度" in text
    assert scope["observations"]["trades"] == trades


@pytest.mark.parametrize("volume", [False, True])
def test_trade_table_preserves_missing_values_and_empty_history(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], volume: bool
) -> None:
    _, _, service = context
    _save(
        context,
        timestamps=None,
        trades=[{"status": "open", "direction": -1}],
        volume=volume,
        scopes={"full": _scope(mode="volume" if volume else "signal")},
    )
    response = service.generate_report(["run"], sections=["trades"])
    assert response["success"], response
    text = Path(response["path"]).read_text()
    _, rows = _static_table(text, "trades-0-0")
    assert len(rows) == 1
    cells = rows[0]
    assert cells[:4] == ["1"] + ["—"] * 3
    assert cells[4:6] == ["做空", "未平倉"]
    assert cells[6:-1] == ["—"] * 11
    csv_rows = _csv_rows(text, "trades-0-0")
    assert csv_rows[0]["進場時間（UTC）"] == ""
    assert csv_rows[0]["淨損益（報價貨幣）"] == ""

    _save(context, "empty", trades=[])
    empty = service.generate_report(["empty"], sections=["trades"])
    assert empty["success"], empty
    text = Path(empty["path"]).read_text()
    empty_headers, empty_rows = _static_table(text, "trades-0-0")
    assert empty_headers and empty_rows == []
    assert "共 0 筆；全部欄位與資料均保留。" in text
    assert _csv_rows(text, "trades-0-0") == []
    assert (
        _payload(Path(empty["path"]))["runs"][0]["scopes"][0]["tables"]["trades"][
            "rows"
        ]
        == []
    )


def test_csv_downloads_bind_every_run_and_scope_to_its_own_complete_records(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(
        context, "first", trades=[{"net_pnl": 1e-12}, {"net_pnl": -2.345678901234567}]
    )
    _save(
        context,
        "second",
        trades=[{"net_pnl": 9007199254740993}],
        scopes={"full": _scope(), "test": _scope(split="test")},
    )
    response = service.generate_report(["first", "second"], sections=["trades"])
    assert response["success"], response
    path = Path(response["path"])
    downloads = re.findall(
        r"data-table-csv='trades-(\d+)-(\d+)' data-csv='([^']*)'",
        path.read_text(),
    )
    payload = _payload(path)
    assert {(ri, si) for ri, si, _ in downloads} == {("0", "0"), ("1", "0"), ("1", "1")}
    for ri, si, encoded in downloads:
        run_index, scope_index = int(ri), int(si)
        rows = list(
            csv.DictReader(io.StringIO(base64.b64decode(encoded).decode("utf-8-sig")))
        )
        source = payload["runs"][run_index]["scopes"][scope_index]["observations"][
            "trades"
        ]
        assert [row["淨損益（報價貨幣）"] for row in rows] == [
            str(trade["net_pnl"]) for trade in source
        ]
        assert [json.loads(row["原始資料（JSON）"]) for row in rows] == source
        assert f"data-table-json='{ri}:{si}:trades'" in path.read_text()
    json_downloads = re.findall(
        r"data-table-json='(\d+):(\d+):trades' data-json='([^']*)'", path.read_text()
    )
    assert {(ri, si) for ri, si, _ in json_downloads} == {
        ("0", "0"),
        ("1", "0"),
        ("1", "1"),
    }
    for ri, si, encoded in json_downloads:
        exported = json.loads(base64.b64decode(encoded))
        scope = payload["runs"][int(ri)]["scopes"][int(si)]
        assert exported == scope["tables"]["trades"]
        assert [row["net_pnl"] for row in exported["rows"]] == [
            trade["net_pnl"] for trade in scope["observations"]["trades"]
        ]
    # Downloading the prepared bytes avoids JavaScript parsing/re-serializing the
    # integer 9007199254740993 as the neighboring representable value.
    assert "atob(button.dataset.json)" in path.read_text()


@pytest.mark.parametrize(
    "previous_version", ["6", "7", "8", "9", "10", "11", "12", "17"]
)
def test_template_upgrade_publishes_new_report_and_keeps_previous_report(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    previous_version: str,
) -> None:
    _, _, service = context
    _save(context)
    with monkeypatch.context() as previous:
        previous.setattr(
            "tradingdev.app.report_service.REPORT_TEMPLATE_VERSION", previous_version
        )
        old = service.generate_report(["run"], sections=["trades"])
    assert old["success"], old
    old_path = Path(old["path"])
    old_content = old_path.read_bytes()
    assert _payload(old_path)["template_version"] == previous_version

    new = service.generate_report(["run"], sections=["trades"])
    assert new["success"], new
    assert new["report_id"] != old["report_id"]
    new_path = Path(new["path"])
    assert _payload(new_path)["template_version"] == REPORT_TEMPLATE_VERSION
    manifest = json.loads((new_path.parent / "manifest.json").read_text())
    assert manifest["template_version"] == REPORT_TEMPLATE_VERSION
    assert old_path.read_bytes() == old_content
    assert (
        service.get_report_download(old["report_id"], expected_sha256=old["sha256"])[
            "content"
        ]
        == old_content
    )
    assert service.generate_report(["run"], sections=["trades"]) == new


@pytest.mark.parametrize("via_cli", [False, True])
def test_report_tables_use_readable_names_units_and_settings(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    capsys: pytest.CaptureFixture[str],
    via_cli: bool,
) -> None:
    workspace, _, service = context
    _save(context)
    if via_cli:
        report_cli_main(["--workspace", str(workspace.root), "--run-id", "run"])
        response = json.loads(capsys.readouterr().out)
    else:
        response = service.generate_report(["run"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    for label in (
        "已選取章節",
        "未選取章節",
        "執行識別碼",
        "預設結果範圍",
        "選定的訓練範圍",
        "策略識別碼",
        "執行設定完整性",
        "結果範圍",
        "結果類型",
        "回測模式",
        "實際 K 線數",
        "實際開始時間（UTC）",
        "初始資金（報價貨幣）",
        "單邊手續費率",
        "參數來源",
        "參數是否完整",
        "累積報酬率",
        "已平倉交易勝率",
    ):
        assert f"<th>{label}</th>" in text
    assert "<td>0.04%</td>" in text
    assert "<td>訊號回測模式</td>" in text
    assert "<td>保存範圍的參數快照</td>" in text
    assert "<td>否</td>" in text
    assert ">百分比</td>" in text
    assert "沒有已平倉交易" in text
    assert "Cumulative return on initial capital." in text
    assert "自訂參數（slow）" in text
    assert _payload(path)["runs"][0]["scopes"][0]["values"]["total_return"] == 0.25


def test_report_preserves_unknown_codes_and_historical_metric_definitions(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    code = "custom_<img>"
    definition = MetricDefinitionSnapshot.model_validate(
        asdict(METRIC_CATALOG["total_return"])
    )
    historical = definition.model_copy(
        update={
            "unit": "amount",
            "description": "Saved historical definition <not current>",
        }
    )
    custom = definition.model_copy(update={"id": code, "unit": "unknown_<unit>"})
    _save(
        context,
        scopes={
            "custom_<scope>": _scope(
                values={"total_return": 0.25, code: None},
                metadata={"unavailable": {code: "new_<reason>"}},
                parameters={"<param>": 1.2345},
            )
        },
        default_scope="custom_<scope>",
        definitions={"total_return": historical, code: custom},
    )
    response = service.generate_report(["run"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    assert "未識別指標：custom_&lt;img&gt;" in text
    assert "未識別單位：unknown_&lt;unit&gt;" in text
    assert "未識別原因：new_&lt;reason&gt;" in text
    assert "自訂結果範圍（custom_&lt;scope&gt;）" in text
    assert "自訂參數（&lt;param&gt;）" in text
    assert "<td>1.23</td>" in text
    assert "<td class='metric'>0.25</td><td>報價貨幣金額</td>" in text
    assert "Saved historical definition &lt;not current&gt;" in text
    saved = _payload(path)["runs"][0]["scopes"][0]
    assert saved["definitions"]["total_return"] == historical.model_dump(mode="json")
    assert saved["parameters"] == {"<param>": 1.2345}


def test_builtin_metrics_have_readable_labels_and_units() -> None:
    for key, definition in METRIC_CATALOG.items():
        assert not metric_label(key).startswith("未識別")
        assert not unit_label(definition.unit).startswith("未識別")


def test_overview_keeps_historical_metric_units_in_separate_columns(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    current = MetricDefinitionSnapshot.model_validate(
        asdict(METRIC_CATALOG["total_return"])
    )
    historical = current.model_copy(
        update={"unit": "amount", "description": "Historical amount definition"}
    )
    _save(context, "fraction", scopes={"full": _scope(values={"total_return": 0.25})})
    _save(
        context,
        "amount",
        scopes={"full": _scope(values={"total_return": 0.25})},
        definitions={"total_return": historical},
    )
    response = service.generate_report(["fraction", "amount"], sections=["overview"])
    assert response["success"], response
    payload = _payload(Path(response["path"]))
    table = payload["comparison_table"]
    columns = {column["field"]: column for column in table["columns"]}
    assert columns["metric:total_return:fraction"]["kind"] == "percent"
    assert columns["metric:total_return:amount"]["kind"] == "number"
    fractional, amount = table["rows"]
    assert fractional["metric:total_return:fraction"] == 0.25
    assert fractional.get("metric:total_return:amount") is None
    assert amount["metric:total_return:amount"] == 0.25
    assert amount.get("metric:total_return:fraction") is None
    assert payload["runs"][1]["scopes"][0]["definitions"]["total_return"] == (
        historical.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    "sections",
    [
        [],
        ["overview"],
        ["provenance"],
        ["limitations"],
        ["overview", "provenance"],
        ["overview", "limitations"],
        ["provenance", "limitations"],
        ["overview", "provenance", "limitations"],
        ["provenance", "overview", "limitations"],
        ["limitations", "provenance", "overview"],
    ],
)
def test_reports_without_details_keep_selected_sections_in_main_view(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], sections: list[str]
) -> None:
    _, _, service = context
    _save(context, scopes={"full": _scope(), "other": _scope()})
    response = service.generate_report(["run"], sections=sections)
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    payload = _payload(path)
    comparison = payload["comparison_table"]
    assert len(comparison["rows"]) == 2
    assert comparison["navigate_field"] is None
    assert all(row["detail"] is None for row in comparison["rows"])
    assert not re.findall(r"href='#result-\d+-\d+'", text)
    assert not re.findall(r"id='result-\d+-\d+'", text)
    assert "id='scope-select'" not in text
    assert "id='result-view'" not in text
    assert "<h2>選擇結果</h2>" not in text
    main = text.split("<div id='overview-view'>", 1)[1].split("<footer>", 1)[0]
    expected = (["overview"] if "overview" in sections else []) + [
        key for key in sections if key != "overview"
    ]
    assert re.findall(r"<section data-section='([^']+)'>", main) == expected
    if "overview" in sections:
        assert "本報告未選取單組詳情章節" in main
        assert "class='table-navigation'" not in main
        assert "查看明細 →</a>" not in main
        encoded = re.search(r"data-table-json='comparison' data-json='([^']+)'", main)
        assert encoded is not None
        assert json.loads(base64.b64decode(encoded[1])) == comparison


@pytest.mark.parametrize(
    "section", ["settings", "metrics", "equity", "ledger", "trades"]
)
@pytest.mark.parametrize("include_overview", [True, False])
def test_selected_detail_sections_have_unique_destinations(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    section: str,
    include_overview: bool,
) -> None:
    _, _, service = context
    _save(context, scopes={"full": _scope(), "other": _scope()})
    selected = ["provenance", section, "limitations"]
    if include_overview:
        selected.append("overview")
    response = service.generate_report(["run"], sections=selected)
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    comparison = _payload(path)["comparison_table"]
    assert comparison["navigate_field"] == "detail"
    targets = [row["detail"] for row in comparison["rows"]]
    assert targets == ["#result-0-0", "#result-0-1"]
    assert re.findall(r"id='(result-\d+-\d+)'", text) == [
        target[1:] for target in targets
    ]
    assert set(re.findall(r"href='(#result-\d+-\d+)'", text)) == set(targets)
    if include_overview:
        overview = text.split("<div id='overview-view'>", 1)[1].split(
            "<div id='result-view' hidden>", 1
        )[0]
        assert "<strong>如何查看詳細結果：</strong>" in overview
        assert "交易紀錄" not in overview
        assert re.findall(
            r"<a href='(#result-\d+-\d+)' aria-label='查看結果 (\d+) 的明細'>"
            r"\d+ · 查看明細 →</a>",
            overview,
        ) == [(target, str(index)) for index, target in enumerate(targets, 1)]
        first_column = comparison["columns"][0]
        assert first_column["field"] == "sequence" and first_column["visible"]
        assert "明細" in first_column["title"]
    details = text.split("<div id='result-view' hidden>", 1)[1].split("<footer>", 1)[0]
    assert re.findall(r"<section data-section='([^']+)'>", details) == [
        "provenance",
        section,
        "limitations",
    ]
    assert ("<h2>選擇結果</h2>" in text) is not include_overview
    assert ("← 返回結果清單" in details) is not include_overview


def test_every_report_table_json_download_contains_the_full_table_specification(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    response = service.generate_report(
        ["run"], sections=["overview", "ledger", "trades"]
    )
    assert response["success"], response
    path = Path(response["path"])
    payload = _payload(path)
    tables = payload["runs"][0]["scopes"][0]["tables"]
    expected = {
        "comparison": payload["comparison_table"],
        "0:0:ledger": tables["ledger"],
        "0:0:trades": tables["trades"],
    }
    exports = re.findall(
        r"data-table-json='([^']+)' data-json='([^']+)'", path.read_text()
    )
    assert {key for key, _ in exports} == set(expected)
    for key, encoded in exports:
        specification = json.loads(base64.b64decode(encoded))
        assert specification == expected[key]
        assert {"columns", "rows", "full_precision_values"} <= specification.keys()


@pytest.mark.parametrize("saved_context", [None, "unknown", ["BTC/USDT"]])
@pytest.mark.parametrize("sections", [[], ["overview", "settings"]])
def test_legacy_nonobject_context_keeps_report_available(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    saved_context: Any,
    sections: list[str],
) -> None:
    _, _, service = context
    _save(
        context, scopes={"full": _scope(metadata={"execution_context": saved_context})}
    )
    response = service.generate_report(["run"], sections=sections)
    assert response["success"], response
    payload = _payload(Path(response["path"]))
    row = payload["comparison_table"]["rows"][0]
    assert row["symbol"] is None and row["timeframe"] is None
    assert (
        payload["runs"][0]["scopes"][0]["metadata"]["execution_context"]
        == saved_context
    )


def test_table_exports_preserve_nanosecond_times_after_utc_conversion(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    timestamps = (
        "2024-01-01T08:00:00.123456789+08:00",
        "2024-01-01T08:00:00.123456790+08:00",
        "2024-01-01T08:00:00.123456791+08:00",
    )
    _save(context, timestamps=timestamps)
    response = service.generate_report(["run"], sections=["overview", "trades"])
    assert response["success"], response
    path = Path(response["path"])
    payload = _payload(path)
    row = payload["comparison_table"]["rows"][0]
    assert row["start"] == "2024-01-01T00:00:00.123456789+00:00"
    assert row["end"] == "2024-01-01T00:00:00.123456791+00:00"
    scope = payload["runs"][0]["scopes"][0]
    assert scope["observations"]["timestamps"] == list(timestamps)
    trade = scope["tables"]["trades"]["rows"][0]
    assert trade["entry_time"] == row["start"] and trade["mark_time"] == row["end"]
    csv_trade = _csv_rows(path.read_text(), "trades-0-0")[0]
    assert csv_trade["進場時間（UTC）"] == row["start"]
    assert csv_trade["期末估值時間（UTC）"] == row["end"]


@pytest.mark.parametrize(
    "scope_id",
    [
        "fold/²/train",
        "fold/٣/test",
        "trial/" + "9" * 5000 + "/train",
        "trial/" + "9" * 4300 + "/train",
    ],
    ids=["superscript", "unicode_decimal", "too_long_to_parse", "too_long_to_display"],
)
def test_unusual_numeric_scope_identifiers_keep_custom_fallback(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], scope_id: str
) -> None:
    _, _, service = context
    _save(context, scopes={scope_id: _scope()}, default_scope=scope_id)
    response = service.generate_report(["run"])
    assert response["success"], response
    path = Path(response["path"])
    assert f"自訂結果範圍（{scope_id}）" in path.read_text()
    assert _payload(path)["runs"][0]["scopes"][0]["scope_id"] == scope_id


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_mode_specific_settings_are_not_presented_as_active_in_signal_mode(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], mode: str
) -> None:
    _, _, service = context
    settings = {"signal_as_position": False, "re_entry_after_sl": True}
    _save(
        context,
        volume=mode == "volume",
        scopes={"full": _scope(mode=mode, metadata={"execution_context": settings})},
    )
    response = service.generate_report(["run"], ["settings"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    for label, active_value in (
        ("訊號代表目標持倉", "否"),
        ("停損／停利後重新進場", "是"),
    ):
        expected = (
            active_value if mode == "volume" else "不適用（此設定僅用於名目金額回測）"
        )
        assert f"<th>{label}</th><td>{expected}</td>" in text
    assert (
        _payload(path)["runs"][0]["scopes"][0]["metadata"]["execution_context"]
        == settings
    )


def test_optional_sections_order_and_commentary_are_saved_and_escaped(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    notes = [{"title": "風險", "text": "<script>alert(1)</script>\n不是計算結果"}]
    response = service.generate_report(["run"], sections=[], commentary=notes)
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    assert "LLM 評語（非計算結果）" in text
    assert "data-section='metrics'" not in text and "<svg" not in text
    assert "<script>alert(1)</script>" not in text
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert manifest["sections"] == [] and manifest["commentary"] == notes
    assert manifest["available_scopes"] == {"run": ["full"]}
    assert service.generate_report(["run"], [], notes) == response
    reordered = service.generate_report(["run"], ["trades", "overview", "metrics"])
    rendered = Path(reordered["path"]).read_text()
    assert rendered.index("data-section='overview'") < rendered.index(
        "data-section='trades'"
    )
    assert rendered.index("data-section='trades'") < rendered.index(
        "data-section='metrics'"
    )
    assert reordered["report_id"] != response["report_id"]
    catalogue = service.get_report_sections()
    catalogue["templates"]["standard"].clear()
    assert service.get_report_sections()["templates"]["standard"]


def test_catalog_titles_and_recipes_match_the_rendered_document(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    catalog = service.get_report_sections()
    section_ids = [section["id"] for section in catalog["sections"]]
    assert catalog["templates"]["standard"] == [
        key for key in section_ids if key != "ml"
    ]
    for selected in catalog["templates"].values():
        report = service.generate_report(["run"], sections=selected)
        if "ml" in selected:
            assert report["code"] == "invalid_report_ml_selection"
            continue
        assert report["success"], report
        content = Path(report["path"]).read_text()
        for section in catalog["sections"]:
            heading = (
                f"<section data-section='{section['id']}'><h2>{section['title']}</h2>"
            )
            assert (heading in content) == (section["id"] in selected)
    catalog["sections"][0]["title"] = "呼叫端修改"
    assert service.get_report_sections()["sections"][0]["title"] == "參數與結果比較總覽"


@pytest.mark.parametrize("sections", [["unknown"], ["metrics", "metrics"]])
def test_invalid_section_selection_does_not_publish(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    sections: list[str],
) -> None:
    workspace, _, service = context
    assert (
        service.generate_report(["run"], sections)["code"] == "invalid_report_sections"
    )
    assert not (workspace.root / "reports").exists()


@pytest.mark.parametrize(
    "notes",
    [
        [{"html": "<p>raw</p>"}],
        [{"title": "", "text": "x"}],
        [{"title": "x", "text": "y"}] * 21,
        [{"title": "x", "text": "y" * 20000}],
    ],
)
def test_invalid_commentary_is_rejected(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    notes: list[dict[str, str]],
) -> None:
    assert (
        context[2].generate_report(["run"], commentary=notes)["code"]
        == "invalid_report_commentary"
    )


def test_all_walk_forward_and_optimization_scopes_preserved_with_summary_semantics(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    summary = PerformanceScope(
        kind="fold_summary",
        mode="signal",
        split="test",
        values={
            "total_return": FoldStats(
                mean=0.25, std=0.0, min=0.25, max=0.25, valid_count=2
            )
        },
        metadata={"aggregation": "fold_descriptive"},
    )
    _save(
        context,
        "wf",
        default_scope="test_summary",
        scopes={
            "fold/0/train": _scope(split="train", fold_index=0),
            "fold/0/test": _scope(split="test", fold_index=0),
            "test_summary": summary,
        },
    )
    _save(
        context,
        "opt",
        default_scope="test",
        selected_train="trial/1/train",
        scopes={
            "trial/0/train": _scope(
                split="train", trial_index=0, parameters={"slow": 23}
            ),
            "trial/1/train": _scope(
                split="train", trial_index=1, parameters={"slow": 26}
            ),
            "test": _scope(split="test", parameters={"slow": 26}),
        },
    )
    response = service.generate_report(["wf", "opt"])
    assert response["success"], response
    assert response["scope_count"] == 6
    payload = _payload(Path(response["path"]))
    assert payload["runs"][0]["scopes"][-1]["observations"] is None
    assert payload["runs"][1]["selected_train_scope"] == "trial/1/train"
    assert payload["default_scope_comparison"]["comparable"] is False
    text = Path(response["path"]).read_text()
    assert "fold_descriptive" in text and "不是串接資金" in text
    assert "<li>平均值：25.00%</li>" in text
    assert "<li>標準差：0.00%</li>" in text
    assert "<li>有效折數：2</li>" in text
    assert "<th>指標是否可比較</th><td>否</td>" in text
    assert "的觀測日期重疊" in text
    assert "scope-1-2" in text and "trial/0/train" in text
    comparison = payload["comparison_table"]
    assert len(comparison["rows"]) == 6
    assert [row["sequence"] for row in comparison["rows"]] == list(range(1, 7))
    assert {row["detail"] for row in comparison["rows"]} == {
        f"#result-{run_index}-{scope_index}"
        for run_index in range(2)
        for scope_index in range(3)
    }
    assert re.findall(r"id='(result-\d+-\d+)'", text) == [
        row["detail"][1:] for row in comparison["rows"]
    ]
    assert "<div id='overview-view'>" in text
    assert "<div id='result-view' hidden>" in text
    assert "href='#overview'" in text
    for row in comparison["rows"]:
        assert f"href='{row['detail']}'" in text
    optimization = [row for row in comparison["rows"] if row["run_id"] == "opt"]
    assert [row["role"] for row in optimization].count("訓練區間") == 2
    assert [row["role"] for row in optimization].count("測試區間") == 1
    selected = next(row for row in optimization if row["selected"] == "選定訓練結果")
    assert selected["parameter:slow"] == 26 and selected["role"] == "訓練區間"
    assert selected["start"] == "2024-01-01T00:00:00+00:00"
    assert selected["end"] == "2024-01-03T00:00:00+00:00"
    assert selected["period"] == "2024-01-01 ～ 2024-01-03"
    assert selected["metric:total_return:fraction"] == 0.25
    visible = {c["field"] for c in comparison["columns"] if c["visible"]}
    assert {
        "sequence",
        "period",
        "parameter:slow",
        "metric:total_return:fraction",
    } <= visible
    assert not {"start", "end", "run_id", "strategy", "symbol", "timeframe"} & visible
    summary_row = next(row for row in comparison["rows"] if row["role"] == "各折摘要")
    assert summary_row["metric:total_return:fraction"] is None
    assert summary_row["start"] is None and summary_row["end"] is None


@pytest.mark.parametrize(
    "artifact", ["performance.json", "observations.json", "manifest.json"]
)
def test_corrupt_saved_source_fails_without_writing_reports(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    artifact: str,
) -> None:
    workspace, _, service = context
    _save(context, modern=True)
    path = workspace.runs / "run" / artifact
    path.write_bytes(b"corrupt")
    response = service.generate_report(["run"])
    assert not response["success"]
    assert response["code"] in {
        "performance_artifact_invalid",
        "execution_manifest_invalid",
    }
    assert not (workspace.root / "reports").exists()
    assert path.read_bytes() == b"corrupt"


def test_existing_report_tampering_is_not_overwritten(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    path = Path(first["path"])
    path.write_text("tampered")
    response = service.generate_report(["run"])
    assert response["code"] == "report_artifact_invalid"
    assert path.read_text() == "tampered"
    assert not list((workspace.root / "reports").glob("*.publishing"))


def test_report_path_symlink_is_rejected(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    tmp_path: Path,
) -> None:
    workspace, _, service = context
    _save(context)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.root / "reports").symlink_to(outside, target_is_directory=True)
    response = service.generate_report(["run"])
    assert response["code"] == "report_path_invalid"
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError("write failed"), TimeoutError("deadline")])
def test_publication_failure_cleans_new_files_and_registrations(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    workspace, store, service = context
    _save(context)
    before = store.list_artifacts()
    original = os.link
    calls = 0

    def interrupted(source: str | Path, target: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        original(source, target)

    monkeypatch.setattr("tradingdev.adapters.reporting.storage.os.link", interrupted)
    response = service.generate_report(["run"])
    assert not response["success"] and response["code"] == "report_generation_failed"
    assert store.list_artifacts() == before
    assert list((workspace.root / "reports").iterdir()) == []


def test_volume_without_timestamps_uses_index_and_amount_drawdown(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context, volume=True, timestamps=None, scopes={"full": _scope(mode="volume")})
    response = service.generate_report(["run"], ["equity"])
    assert response["success"], response
    text = Path(response["path"]).read_text()
    assert "報價貨幣金額" in text and "K 線索引 0" in text and "K 線索引 2" in text


def test_registered_missing_report_is_not_silently_rebuilt(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    path = Path(first["path"])
    path.unlink()
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_artifact_invalid"
    assert not path.exists()


def test_report_download_returns_the_verified_snapshot_without_a_second_read(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    expected = path.read_bytes()
    original_read = Path.read_bytes
    reads = []

    def changed_after_read(target: Path) -> bytes:
        content = original_read(target)
        if target == path:
            reads.append(target)
            target.write_bytes(b"changed after the verified snapshot was read")
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_after_read)
    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert download == {
        "success": True,
        "report_id": report["report_id"],
        "sha256": hashlib.sha256(expected).hexdigest(),
        "content": expected,
    }
    assert reads == [path]
    assert original_read(path) != download["content"]


@pytest.mark.parametrize("change", ["modified", "deleted", "unreadable"])
def test_report_download_rejects_changed_or_unreadable_html(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    if change == "modified":
        path.write_bytes(b"modified")
    elif change == "deleted":
        path.unlink()
    else:

        def unreadable(target: Path) -> bytes:
            raise PermissionError("Report is unreadable")

        monkeypatch.setattr(Path, "read_bytes", unreadable)

    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert not download["success"]
    assert download["code"] == (
        "report_read_failed" if change == "unreadable" else "report_artifact_invalid"
    )
    assert "content" not in download
    if change == "modified":
        assert path.read_bytes() == b"modified"
    elif change == "deleted":
        assert not path.exists()


@pytest.mark.parametrize("field", ["artifact_type", "path", "sha256", "deleted"])
def test_report_download_checks_registration_and_original_expected_hash(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], field: str
) -> None:
    _, store, service = context
    _save(context)
    report = service.generate_report(["run"])
    with store.connect() as connection:
        if field == "deleted":
            connection.execute(
                "delete from artifacts where artifact_id = ?", (report["artifact_id"],)
            )
        else:
            value = "different"
            if field == "sha256":
                # Even replacing both bytes and the database hash cannot change
                # the report snapshot requested by this caller.
                Path(report["path"]).write_bytes(b"changed")
                value = hashlib.sha256(b"changed").hexdigest()
            connection.execute(
                f"update artifacts set {field} = ? where artifact_id = ?",
                (value, report["artifact_id"]),
            )
    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert not download["success"]
    assert download["code"] == (
        "report_not_found" if field == "deleted" else "report_artifact_invalid"
    )
    assert "content" not in download


def test_report_download_rejects_replaced_symlink_even_for_identical_bytes(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService], tmp_path: Path
) -> None:
    _, _, service = context
    _save(context)
    report = service.generate_report(["run"])
    path = Path(report["path"])
    outside = tmp_path / "outside.html"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    download = service.get_report_download(
        report["report_id"], expected_sha256=report["sha256"]
    )

    assert download["code"] == "report_path_invalid"
    assert "content" not in download


@pytest.mark.parametrize(
    ("report_id", "expected_sha256", "code"),
    [
        ("../outside", "a" * 64, "report_path_invalid"),
        ("a" * 64, "not-a-hash", "report_artifact_invalid"),
        ("a" * 64, "b" * 64, "report_not_found"),
    ],
)
def test_invalid_report_download_identity_returns_error_without_content(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    report_id: str,
    expected_sha256: str,
    code: str,
) -> None:
    download = context[2].get_report_download(
        report_id, expected_sha256=expected_sha256
    )

    assert not download["success"] and download["code"] == code
    assert "content" not in download


def test_busy_report_does_not_remove_another_publication_marker(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _save(context)
    first = service.generate_report(["run"])
    marker = workspace.root / "reports" / f".{first['report_id']}.publishing"
    marker.mkdir()
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_busy" and marker.is_dir()
    marker.rmdir()


def test_database_failure_rolls_back_pair_and_cleans_files(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, store, service = context
    _save(context)
    before = store.list_artifacts()
    with store.connect() as connection:
        connection.execute(
            "create trigger reject_report before insert on artifacts "
            "when NEW.artifact_type = 'research_report_manifest' "
            "begin select raise(ABORT, 'database failure'); end"
        )
    failed = service.generate_report(["run"])
    assert failed["code"] == "report_generation_failed"
    assert store.list_artifacts() == before
    assert list((workspace.root / "reports").iterdir()) == []


def test_summary_only_still_checks_observation_bundle_integrity(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    summary = PerformanceScope(
        kind="fold_summary",
        mode="signal",
        split="test",
        values={
            "total_return": FoldStats(
                mean=None, std=None, min=None, max=None, valid_count=0
            )
        },
        metadata={"aggregation": "fold_descriptive"},
    )
    _save(context, scopes={"summary": summary}, default_scope="summary")
    (workspace.runs / "run" / "observations.json").write_text("corrupt")
    failed = service.generate_report(
        ["run"], sections=[], commentary=[{"title": "t", "text": "x"}]
    )
    assert failed["code"] == "performance_artifact_invalid"
    assert not (workspace.root / "reports").exists()


def test_utc_labels_convert_offsets_without_changing_original_timestamps(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    stamps = (
        "2024-01-01T08:00:00+08:00",
        "2024-01-02T08:00:00+08:00",
        "2024-01-03T08:00:00+08:00",
    )
    _save(context, timestamps=stamps)
    result = service.generate_report(["run"])
    path = Path(result["path"])
    text = path.read_text()
    assert ">2024-01-01 00:00 UTC</text>" in text
    assert ">2024-01-03 00:00 UTC</text>" in text
    assert ">2024-01-03T08:00:00</text>" not in text
    assert "<td>2024-01-01T00:00:00+00:00</td>" in text
    assert _payload(path)["runs"][0]["scopes"][0]["observations"]["timestamps"] == list(
        stamps
    )


@pytest.mark.parametrize("scope_count", [1, 35])
def test_report_artifact_read_count_does_not_grow_with_scope_count(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    monkeypatch: pytest.MonkeyPatch,
    scope_count: int,
) -> None:
    workspace, _, service = context
    _save(
        context,
        modern=True,
        default_scope="part/0",
        scopes={f"part/{index}": _scope() for index in range(scope_count)},
    )
    original = Path.read_bytes
    reads: dict[str, int] = {}

    def counted(path: Path) -> bytes:
        if path.parent == workspace.runs / "run":
            reads[path.name] = reads.get(path.name, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    response = service.generate_report(["run"])
    assert response["success"], response
    assert response["scope_count"] == scope_count
    # The observation loader also validates its paired performance file.
    assert reads == {"performance.json": 2, "observations.json": 1, "manifest.json": 1}


@pytest.mark.parametrize(
    "corruption", ["sha256", "path", "artifact_type", "run_id", "bytes"]
)
def test_registered_manifest_identity_and_byte_hash_are_checked_before_reporting(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    corruption: str,
) -> None:
    workspace, store, service = context
    _save(context, modern=True)
    path = workspace.runs / "run" / "manifest.json"
    if corruption == "bytes":
        # Same manifest/canonical execution digest, different registered file bytes.
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
        run = store.get_run("run")
        assert run is not None
        ExecutionManifestStore(workspace).load(
            "run", expected_hash=run["manifest_hash"]
        )
    else:
        bad_values = {
            "sha256": "0" * 64,
            "path": str(workspace.root / "other.json"),
            "artifact_type": "unrelated",
            "run_id": "other",
        }
        with store.connect() as connection:
            connection.execute(
                f"update artifacts set {corruption} = ? where artifact_id = ?",
                (bad_values[corruption], "run:execution_manifest"),
            )
    response = service.generate_report(["run"])
    assert response["code"] == "execution_manifest_invalid", response
    assert not (workspace.root / "reports").exists()
    from tradingdev.app.trade_history_service import TradeHistoryService

    history = TradeHistoryService(workspace=workspace, store=store).get_run_trades(
        "run"
    )
    assert history["code"] == "execution_manifest_invalid"


@pytest.mark.parametrize(
    ("sections", "details"),
    [
        (["overview", "trades"], True),
        (["settings"], True),
        (["research"], False),
        ([], False),
    ],
)
def test_report_delivery_guide_matches_html_and_selected_details(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
    sections: list[str],
    details: bool,
) -> None:
    _, _, service = context
    _save(context)
    response = service.generate_report(["run"], sections)
    assert response["success"], response
    path = Path(response["path"])
    guide = response["navigation_instructions"]
    assert guide in path.read_text()
    assert _payload(path)["navigation_instructions"] == guide
    assert (
        json.loads(path.with_name("manifest.json").read_text())[
            "navigation_instructions"
        ]
        == guide
    )
    assert ("點選總覽表格或結果清單中的結果列" in guide) is details
    assert ("配對交易損益" in guide) is ("trades" in sections)
    assert ("設定與資料覆蓋" in guide) is ("settings" in sections)
    assert "委託、成交與帳戶核對" not in guide


def _benchmark(**changes: Any) -> BenchmarkScope:
    return BenchmarkScope.model_validate(
        {
            "status": "available",
            "asset_class": "crypto",
            "definition": {
                "id": "crypto_btc_buy_and_hold",
                "asset_class": "crypto",
                "symbol": "BTC/USDT",
                "label": "BTC/USDT 買進持有",
                "weighting": "buy_and_hold",
                "rationale": "加密貨幣策略的機會成本基準是持有比特幣。",
            },
            "source": "binance_vision",
            "market_type": "futures/um",
            "dataset_id": "BTC/USDT:1d:2024-01-01:2024-01-31",
            "reused_run_frame": True,
            "init_cash": 100.0,
            "capital_basis": "run_init_cash",
            "fees": 0.0004,
            "slippage": 0.0,
            "coverage": {
                "scope_bars": 3,
                "aligned_bars": 3,
                "missing_bars": 0,
                "complete": True,
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-03T00:00:00Z",
            },
            "timestamps": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "equity_curve": [100.0, 101.0, 110.0],
            "returns": [0.0, 0.01, 0.0891089108910891],
            "values": {"total_return": 0.1, "max_drawdown": 0.02},
            "metadata": {"unavailable": {}, "settings": {"initial_cash": 100.0}},
            **changes,
        }
    )


def test_benchmark_section_compares_the_saved_baseline_with_the_strategy(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    workspace, _, service = context
    _save(context, benchmarks={"full": _benchmark()})
    response = service.generate_report(["run"], sections=["overview", "benchmark"])
    assert response["success"], response
    path = Path(response["path"])
    payload = _payload(path)
    scope = payload["runs"][0]["scopes"][0]
    assert scope["benchmark"]["status"] == "available"
    assert scope["benchmark"]["definition"]["symbol"] == "BTC/USDT"
    text = path.read_text()
    assert "基準：BTC/USDT 買進持有" in text
    assert "策略與基準對照" in text
    # Overview columns carry the baseline return and the strategy's excess.
    assert "基準報酬（買進持有）" in text
    assert "超額報酬（策略−基準）" in text
    assert "15.00%" in text
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert manifest["benchmarks"]["run"]["full"]["symbol"] == "BTC/USDT"
    assert manifest["benchmarks"]["run"]["full"]["status"] == "available"
    assert manifest["sections"] == ["overview", "benchmark"]


def test_report_states_when_a_result_has_no_saved_benchmark(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(context)
    response = service.generate_report(["run"], sections=["benchmark"])
    assert response["success"], response
    text = Path(response["path"]).read_text()
    assert "此結果沒有保存 benchmark 證據" in text
    assert "策略與基準對照" not in text


def test_undefined_asset_class_is_reported_instead_of_a_zero_baseline(
    context: tuple[WorkspacePaths, SQLiteStore, ReportService],
) -> None:
    _, _, service = context
    _save(
        context,
        benchmarks={
            "full": BenchmarkScope.model_validate(
                {
                    "status": "undefined",
                    "asset_class": "us_equity",
                    "reason": "undefined_asset_class",
                }
            )
        },
    )
    response = service.generate_report(["run"], sections=["benchmark"])
    assert response["success"], response
    text = Path(response["path"]).read_text()
    assert "沒有定義 benchmark" in text
    assert "不以其他市場的指數代替" in text
