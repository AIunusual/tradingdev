"""Reports expose saved native execution/account streams without reconstructing them."""

from __future__ import annotations

import base64
import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.performance.artifacts import build_artifacts, scope_from_backtest

if TYPE_CHECKING:
    from tradingdev.domain.backtest.result import BacktestResult


class ReportMarkup(HTMLParser):
    """Inspect rendered tables and bind every CSV control to its data stream."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[str] = []
        self.rows: dict[str, int] = {}
        self.downloads: list[dict[str, str | None]] = []
        self._table = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "section" and attributes.get("data-section"):
            self.sections.append(str(attributes["data-section"]))
        if tag == "table":
            self._table = attributes.get("id") or ""
            if self._table:
                self.rows[self._table] = 0
        if tag == "tr" and self._table:
            self.rows[self._table] += 1
        if tag == "button" and "data-download" in attributes:
            self.downloads.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table = ""


def _market(signals: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "open": [100.0, 110.0, 90.0, 95.0, 105.0],
            "high": [110.0, 120.0, 100.0, 105.0, 115.0],
            "low": [90.0, 100.0, 80.0, 85.0, 95.0],
            "close": [105.0, 115.0, 95.0, 100.0, 110.0],
            "signal": signals or [1, -1, 0, 1, 1],
        }
    )


def _save(tmp_path: Path, result: BacktestResult) -> tuple[ReportService, Path]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    scope, observations = scope_from_backtest(result)
    store.create_run(
        run_id="execution",
        job_id="execution",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "execution",
        metrics=result.metrics,
    )
    PerformanceStore(workspace, store).publish(
        build_artifacts(
            "execution", None, "full", {"full": scope}, {"full": observations}
        )
    )
    return ReportService(workspace=workspace, store=store), workspace.runs / "execution"


def _payload(path: Path) -> dict[str, Any]:
    encoded = path.read_text().split(
        "<script id='report-data' type='application/json'>", 1
    )[1]
    result: dict[str, Any] = json.loads(encoded.split("</script>", 1)[0])
    return result


def test_saved_real_engine_records_render_all_rows_and_separate_csv_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SignalBacktestEngine(init_cash=1000.0, fees=0.001, slippage=0.01).run(
        _market()
    )
    service, directory = _save(tmp_path, result)
    saved = {p: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(
        SignalBacktestEngine, "run", lambda *_: pytest.fail("Cannot rerun")
    )
    monkeypatch.setattr("pickle.load", lambda *_: pytest.fail("Cannot unpickle"))
    response = service.generate_report(["execution"], ["executions", "account_history"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    document = ReportMarkup()
    document.feed(text)
    assert document.sections == ["executions", "account_history"]
    assert document.rows == {"execution_records-0-0": 5, "account_history-0-0": 6}
    assert [
        (button["data-dataset"], button["data-download"])
        for button in document.downloads
    ] == [("execution_records", "0:0"), ("account_history", "0:0")]
    exports = {}
    for button in document.downloads:
        encoded = button["data-csv"]
        assert isinstance(encoded, str)
        content = base64.b64decode(encoded, validate=True)
        assert content.startswith(b"\xef\xbb\xbf")
        exports[button["data-dataset"]] = list(
            csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        )
    scope = _payload(path)["runs"][0]["scopes"][0]
    assert result.execution_records is not None and result.account_history is not None
    assert len(exports["execution_records"]) == len(result.execution_records)
    for exported, record in zip(
        exports["execution_records"], result.execution_records, strict=True
    ):
        assert exported["執行紀錄編號"] == str(record.execution_id)
        assert exported["執行前現金（報價貨幣）"] == str(record.before.cash)
        assert exported["執行後現金（報價貨幣）"] == str(record.after.cash)
        assert exported["執行後總持倉（標的單位）"] == str(record.after.position)
        assert exported["成交數量（標的單位）"] == str(record.filled_size)
    assert len(exports["account_history"]) == len(result.account_history)
    for exported, state in zip(
        exports["account_history"], result.account_history, strict=True
    ):
        assert exported["K 線索引（從 0 起算）"] == str(state.bar_index)
        assert exported["現金（報價貨幣）"] == str(state.cash)
        assert exported["總持倉數量（標的單位）"] == str(state.position)
        assert exported["帳戶權益（報價貨幣）"] == str(state.equity)
    assert scope["observations"]["execution_records"] == [
        record.model_dump(mode="json") for record in result.execution_records
    ]
    assert scope["observations"]["account_history"] == [
        state.model_dump(mode="json") for state in result.account_history
    ]
    assert scope["observations"]["execution_records"][1]["after"]["position"] < 0
    for column in (
        "市場開盤價",
        "市場最高價",
        "市場最低價",
        "市場收盤價",
        "委託價格",
        "成交價格",
        "成交數量（標的單位）",
        "成交手續費（報價貨幣）",
        "執行前現金（報價貨幣）",
        "執行後現金（報價貨幣）",
        "執行前總持倉（標的單位）",
        "執行後總持倉（標的單位）",
        "收盤估值價格",
    ):
        assert f">{column}</button>" in text
    assert "一般資產帳本" in text and "不是 Binance" in text
    assert "精確盤中成交時間" in text and "滑價前" in text
    assert "<td>已成交</td>" in text and "<td>買入</td>" in text
    assert "<td data-sort-value='0.001'>0.10%</td>" in text
    assert "<td data-sort-value='1000.0'>1000.00</td>" in text
    assert "排序、原始資料與 CSV 保留完整精度" in text
    manifest = json.loads((path.parent / "manifest.json").read_text())
    availability = manifest["available_data"]["execution"]["full"]
    assert availability["execution_record_count"] == 4
    assert availability["account_history_count"] == 5
    assert {p: p.read_bytes() for p in directory.iterdir()} == saved


@pytest.mark.parametrize(
    ("status", "label"), [("ignored", "已忽略"), ("rejected", "已拒絕")]
)
def test_execution_table_preserves_missing_fills_and_unknown_enum_codes(
    tmp_path: Path,
    status: str,
    label: str,
) -> None:
    result = SignalBacktestEngine(init_cash=1e-12, fees=0.0, slippage=0.0).run(
        _market([1, 0, 0, 0, 0])
    )
    assert result.execution_records is not None
    index = next(
        i for i, row in enumerate(result.execution_records) if row.status == "rejected"
    )
    record = result.execution_records[index].model_copy(
        update={
            "status": status,
            "status_info": "future_<reason>",
            "requested_direction": "future_<direction>",
        }
    )
    result.execution_records[index] = record
    service, _ = _save(tmp_path, result)
    response = service.generate_report(["execution"], ["executions"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    table = text.split("<table id='execution_records-0-0' data-sortable>", 1)[1].split(
        "</table>", 1
    )[0]
    assert "未識別選項：future_&lt;reason&gt;" in table
    assert "未識別選項：future_&lt;direction&gt;" in table
    assert f"<td>{label}</td>" in table
    assert "future_<reason>" not in table
    assert "<td>未知／未記錄</td>" in table
    saved = _payload(path)["runs"][0]["scopes"][0]["observations"]["execution_records"][
        index
    ]
    assert saved == record.model_dump(mode="json")
    assert saved["filled_size"] is None and saved["fees"] is None


def test_standard_template_includes_new_sections_and_custom_order_can_omit_them(
    tmp_path: Path,
) -> None:
    result = SignalBacktestEngine(init_cash=1000.0).run(_market())
    service, _ = _save(tmp_path, result)
    catalogue = service.get_report_sections()
    assert {"executions", "account_history"} <= set(catalogue["templates"]["standard"])
    default = service.generate_report(["execution"])
    document = ReportMarkup()
    document.feed(Path(default["path"]).read_text())
    assert document.sections == catalogue["templates"]["standard"]
    reordered = service.generate_report(
        ["execution"], ["account_history", "executions"]
    )
    document = ReportMarkup()
    document.feed(Path(reordered["path"]).read_text())
    assert document.sections == ["account_history", "executions"]
    empty = service.generate_report(["execution"], [])
    document = ReportMarkup()
    document.feed(Path(empty["path"]).read_text())
    assert document.sections == [] and document.downloads == []


@pytest.mark.parametrize("mode", ["legacy", "volume", "no_orders", "empty"])
def test_missing_unsupported_and_recorded_empty_streams_are_distinct(
    tmp_path: Path,
    mode: str,
) -> None:
    engine = SignalBacktestEngine(init_cash=1000.0)
    if mode == "volume":
        result = VolumeBacktestEngine(position_size=200.0).run(_market())
    elif mode == "empty":
        result = engine.run(_market().iloc[:0])
    else:
        result = engine.run(_market([0, 0, 0, 0, 0]))
        if mode == "legacy":
            result.execution_records = result.account_history = None
    service, directory = _save(tmp_path, result)
    if mode == "legacy":
        # Exercise historical JSON whose schema did not have the new fields.
        path = directory / "observations.json"
        raw = json.loads(path.read_text())
        raw["scopes"]["full"].pop("execution_records")
        raw["scopes"]["full"].pop("account_history")
        path.write_text(json.dumps(raw))
        import hashlib

        with SQLiteStore(WorkspacePaths(directory.parents[1])).connect() as connection:
            connection.execute(
                "update artifacts set sha256 = ? where artifact_id = ?",
                (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    "execution:observations_json",
                ),
            )
    response = service.generate_report(["execution"], ["executions", "account_history"])
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    document = ReportMarkup()
    document.feed(text)
    if mode in {"legacy", "volume"}:
        expected = (
            "not_recorded" if mode == "legacy" else "unsupported_volume_accounting"
        )
        assert expected in text
        assert text.count("data-availability='unavailable'") == 2
        assert document.downloads == []
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert (
            manifest["available_data"]["execution"]["full"]["execution_record_count"]
            is None
        )
    else:
        assert "已記錄：0 筆委託嘗試" in text
        assert "data-availability='unavailable'" not in text
        if mode == "empty":
            assert "已記錄：0 根 K 線" in text and document.downloads == []
        else:
            assert document.rows == {"account_history-0-0": 6}
