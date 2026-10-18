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
from pathlib import Path
from typing import Any

import pytest

from tradingdev.adapters.cli.reports import main as report_cli_main
from tradingdev.adapters.reporting.labels import metric_label, unit_label
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
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
    assert "https://" not in text and "<svg" in text
    assert not list((workspace.root / "reports").rglob(".report-*"))
    assert not list((workspace.root / "reports").glob("*.publishing"))


@pytest.mark.parametrize("via_cli", [False, True])
@pytest.mark.parametrize(
    ("status", "direction", "status_label", "direction_label"),
    [
        ("closed", 1, "已平倉", "做多"),
        ("open", -1, "未平倉", "做空"),
        (None, None, "未知", "未知"),
        ("unrecognized", 0, "未知", "未知"),
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
    table = text.split("<table id='trades-0-0' data-sortable>", 1)[1].split(
        "</table>", 1
    )[0]
    headers = re.findall(r"<th><button data-column='(\d+)'>(.*?)</button></th>", table)
    assert headers == list(
        zip(
            map(str, range(13)),
            [
                "序號",
                "進場時間（UTC）",
                "平倉時間（UTC）",
                "期末估值時間（UTC）",
                "交易方向",
                "持倉狀態",
                "進場價格",
                "平倉價格",
                "期末估值價格",
                "數量（標的單位）",
                "手續費（報價貨幣）",
                "淨損益（報價貨幣）",
                "原始資料",
            ],
            strict=True,
        )
    )
    cells = re.findall(r"<td(?: [^>]*)?>(.*?)</td>", table, flags=re.DOTALL)
    final_time = "2024-01-03T00:00:00+00:00"
    missing = "未知／未記錄"
    assert cells[:12] == [
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
        "0.10",
        "9.90",
    ]
    assert "&lt;script&gt;unsafe()&lt;/script&gt;" in cells[12]
    assert "placeholder='狀態 / 方向 / 日期 / 文字'" in text
    assert "每單位標的的報價貨幣金額" in text
    assert _payload(path)["runs"][0]["scopes"][0]["observations"]["trades"] == [trade]
    encoded = re.findall(r"data-csv='([^']*)'", text)
    assert len(encoded) == 1
    csv_rows = list(
        csv.DictReader(io.StringIO(base64.b64decode(encoded[0]).decode("utf-8-sig")))
    )
    assert len(csv_rows) == 1
    assert list(csv_rows[0]) == [
        "進場 K 線索引（從 0 起算）",
        "平倉／期末估值 K 線索引（從 0 起算）",
        "交易方向",
        "持倉狀態",
        "進場價格（報價貨幣／標的單位）",
        "平倉／期末估值價格（報價貨幣／標的單位）",
        "數量（標的單位）",
        "手續費合計（報價貨幣）",
        "淨損益（報價貨幣）",
        '未識別欄位（["note"]）',
    ]
    assert csv_rows[0]["淨損益（報價貨幣）"] == "9.9"
    assert csv_rows[0]["手續費合計（報價貨幣）"] == "0.1"
    assert "<script>unsafe()</script>" in csv_rows[0].values()
    assert "下載全部交易 CSV（中文欄名）" in text


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
    table, after = text.split("<table id='trades-0-0' data-sortable>", 1)[1].split(
        "</table>", 1
    )
    numeric_cells = re.findall(r"<td data-sort-value='([^']*)'>(.*?)</td>", table)
    assert numeric_cells == [(str(value), display)] * 10
    assert "<td>1</td>" in table and "<td>2</td>" in table
    assert "四捨五入至小數點後兩位" in after
    assert "排序、原始資料與 CSV 保留完整精度" in after
    assert _payload(path)["runs"][0]["scopes"][0]["observations"]["trades"] == trades
    assert f"&quot;net_pnl&quot;: {value}" in table


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
    table = text.split("<table id='trades-0-0' data-sortable>", 1)[1].split(
        "</table>", 1
    )[0]
    cells = re.findall(r"<td>(.*?)</td>", table, flags=re.DOTALL)
    assert cells[:4] == ["1"] + ["未知／未記錄"] * 3
    assert cells[4:6] == ["做空", "未平倉"]
    assert cells[6:12] == ["未知／未記錄"] * 6

    _save(context, "empty", trades=[])
    empty = service.generate_report(["empty"], sections=["trades"])
    assert empty["success"], empty
    text = Path(empty["path"]).read_text()
    assert "本範圍沒有保存交易紀錄。" in text
    assert "<table id='trades-" not in text


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
        r"data-download='([^']+)' data-dataset='trades' data-csv='([^']*)'",
        path.read_text(),
    )
    payload = _payload(path)
    assert {key for key, _ in downloads} == {"0:0", "1:0", "1:1"}
    for key, encoded in downloads:
        run_index, scope_index = map(int, key.split(":"))
        rows = list(
            csv.DictReader(io.StringIO(base64.b64decode(encoded).decode("utf-8-sig")))
        )
        source = payload["runs"][run_index]["scopes"][scope_index]["observations"][
            "trades"
        ]
        assert [row["淨損益（報價貨幣）"] for row in rows] == [
            str(trade["net_pnl"]) for trade in source
        ]


@pytest.mark.parametrize("previous_version", ["6", "7", "8", "9", "10"])
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
    reordered = service.generate_report(["run"], ["trades", "metrics"])
    rendered = Path(reordered["path"]).read_text()
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
    assert catalog["templates"]["standard"] == section_ids
    for selected in catalog["templates"].values():
        report = service.generate_report(["run"], sections=selected)
        assert report["success"], report
        content = Path(report["path"]).read_text()
        for section in catalog["sections"]:
            heading = (
                f"<section data-section='{section['id']}'><h2>{section['title']}</h2>"
            )
            assert (heading in content) == (section["id"] in selected)
    catalog["sections"][0]["title"] = "呼叫端修改"
    assert service.get_report_sections()["sections"][0]["title"] == "總覽"


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
