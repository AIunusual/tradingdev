"""One report ledger binds human, interactive and CSV views to saved evidence."""

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
        self.headers: dict[str, list[str]] = {}
        self.sort_columns: dict[str, list[str | None]] = {}
        self.cells: dict[str, list[list[str]]] = {}
        self.sort_values: dict[str, list[list[str | None]]] = {}
        self._table = ""
        self._header: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_sort: str | None = None
        self._row: list[str] = []
        self._row_sort: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "section" and attributes.get("data-section"):
            self.sections.append(str(attributes["data-section"]))
        if tag == "table":
            self._table = attributes.get("id") or ""
            if self._table:
                self.rows[self._table] = 0
                self.headers[self._table] = []
                self.sort_columns[self._table] = []
                self.cells[self._table] = []
                self.sort_values[self._table] = []
        if tag == "tr" and self._table:
            self.rows[self._table] += 1
            self._row = []
            self._row_sort = []
        if tag == "th" and self._table:
            self._header = []
        if tag == "td" and self._table:
            self._cell = []
            self._cell_sort = attributes.get("data-sort-value")
        if tag == "button" and self._table and "data-column" in attributes:
            self.sort_columns[self._table].append(attributes["data-column"])
        if tag == "button" and "data-table-csv" in attributes:
            self.downloads.append(attributes)

    def handle_data(self, data: str) -> None:
        if self._header is not None:
            self._header.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "th" and self._header is not None:
            self.headers[self._table].append("".join(self._header))
            self._header = None
        if tag == "td" and self._cell is not None:
            self._row.append("".join(self._cell))
            self._row_sort.append(self._cell_sort)
            self._cell = None
        if tag == "tr" and self._table and self._row:
            self.cells[self._table].append(self._row)
            self.sort_values[self._table].append(self._row_sort)
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


def _report(
    service: ReportService, sections: list[str] | None = None
) -> tuple[Path, str, ReportMarkup, dict[str, Any]]:
    response = service.generate_report(
        ["execution"], ["ledger"] if sections is None else sections
    )
    assert response["success"], response
    path = Path(response["path"])
    text = path.read_text()
    document = ReportMarkup()
    document.feed(text)
    return path, text, document, _payload(path)["runs"][0]["scopes"][0]


def _csv_rows(
    document: ReportMarkup, identifier: str
) -> tuple[list[str], list[dict[str, str]]]:
    button = next(
        button
        for button in document.downloads
        if button["data-table-csv"] == identifier
    )
    encoded = button["data-csv"]
    assert isinstance(encoded, str)
    content = base64.b64decode(encoded, validate=True)
    assert content.startswith(b"\xef\xbb\xbf")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    assert reader.fieldnames is not None
    return list(reader.fieldnames), list(reader)


def test_saved_real_engine_records_share_one_ordered_static_interactive_and_csv_ledger(
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
    path, text, document, scope = _report(service)
    spec = scope["tables"]["ledger"]
    columns, rows = spec["columns"], spec["rows"]
    assert document.sections == ["ledger"]
    assert document.rows == {"ledger-0-0-static": 10}
    assert document.headers["ledger-0-0-static"] == [item["title"] for item in columns]
    assert [item["field"] for item in columns[:2]] == ["timestamp", "sequence"]
    assert [item["title"] for item in columns[:2]] == [
        "K 線時間（UTC）",
        "事件順序（從 1 起算）",
    ]
    assert all(item["title"] != item["field"] for item in columns)
    assert {item["field"] for item in columns if item["visible"]} >= {
        "timestamp",
        "sequence",
        "event_type",
        "status",
        "requested_size",
        "requested_size_unit",
        "requested_size_kind",
        "requested_size_type",
        "requested_price",
        "filled_size",
        "filled_price",
        "fees",
        "before.cash",
        "delta.cash",
        "after.cash",
        "before.position",
        "delta.position",
        "after.position",
        "reconciliation",
    }
    assert any(not item["visible"] for item in columns)
    assert spec["filter_field"] == "filter_status"
    assert {item["value"] for item in spec["event_options"]} == {
        "filled",
        "ignored",
        "rejected",
        "bar_close",
    }
    headers, exported = _csv_rows(document, "ledger-0-0")
    assert headers == [
        item["title"] + ("（比例，1 = 100%）" if item["kind"] == "percent" else "")
        for item in columns
    ]
    assert len(exported) == len(rows) == 9
    assert [row["sequence"] for row in rows] == list(range(1, 10))
    assert [row["filter_status"] for row in rows] == [
        "bar_close",
        "filled",
        "bar_close",
        "filled",
        "bar_close",
        "filled",
        "bar_close",
        "filled",
        "bar_close",
    ]
    assert all(row["reconciliation"] == "可核對項目一致" for row in rows)
    for cells, csv_row, row in zip(
        document.cells["ledger-0-0-static"], exported, rows, strict=True
    ):
        for column, header, cell in zip(columns, headers, cells, strict=True):
            value = row[column["field"]]
            if value is None:
                assert cell == "—" and csv_row[header] == ""
            elif isinstance(value, float):
                expected = (
                    f"{value:z.2%}" if column["kind"] == "percent" else f"{value:z.2f}"
                )
                assert cell == expected
                assert csv_row[header] == str(value)
            else:
                assert cell == str(value)
                assert csv_row[header] == str(value)
    assert result.execution_records is not None and result.account_history is not None
    execution_rows = [row for row in rows if row["filter_status"] == "filled"]
    for row, record in zip(execution_rows, result.execution_records, strict=True):
        assert row["requested_price"] == record.requested_price
        assert row["filled_price"] == record.filled_price
        assert row["filled_size"] == record.filled_size
        assert row["fees"] == record.fees
        assert row["before.cash"] == record.before.cash
        assert row["after.cash"] == record.after.cash
        assert row["delta.cash"] == record.after.cash - record.before.cash
        assert row["before.position"] == record.before.position
        assert row["after.position"] == record.after.position
        assert row["delta.position"] == record.after.position - record.before.position
        assert row["fees_residual"] == pytest.approx(0.0, abs=1e-10)
        assert row["cash_residual"] == pytest.approx(0.0, abs=1e-10)
        assert row["position_residual"] == pytest.approx(0.0, abs=1e-10)
    close_rows = [row for row in rows if row["filter_status"] == "bar_close"]
    for row, state in zip(close_rows, result.account_history, strict=True):
        assert row["after.cash"] == state.cash
        assert row["after.position"] == state.position
        assert row["after.equity"] == state.equity
        assert row["after.valuation_price"] == state.mark_price
        assert row["after.debt"] is row["filled_size"] is row["fees"] is None
    assert rows[1]["after.valuation_price"] != rows[2]["after.valuation_price"]
    assert scope["observations"]["execution_records"] == [
        record.model_dump(mode="json") for record in result.execution_records
    ]
    assert scope["observations"]["account_history"] == [
        state.model_dump(mode="json") for state in result.account_history
    ]
    assert scope["observations"]["execution_records"][1]["after"]["position"] < 0
    assert "Bokeh" in text and "TradingDevTables" in text
    assert "不受篩選、排序或欄位勾選影響" in text
    assert "文字版完整表格（不需 JavaScript）" in text
    assert "不是逐筆報價或交易所精確成交時間" in text
    manifest = json.loads((path.parent / "manifest.json").read_text())
    availability = manifest["available_data"]["execution"]["full"]
    assert availability["execution_record_count"] == 4
    assert availability["account_history_count"] == 5
    assert {p: p.read_bytes() for p in directory.iterdir()} == saved


@pytest.mark.parametrize(
    ("status", "label"), [("ignored", "已忽略"), ("rejected", "已拒絕")]
)
def test_unfilled_rows_preserve_missing_fills_and_escape_unknown_enum_codes(
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
    _, text, document, scope = _report(service)
    table = text.split("<table id='ledger-0-0-static'>", 1)[1].split("</table>", 1)[0]
    assert "未識別選項：future_&lt;reason&gt;" in table
    assert "未識別選項：future_&lt;direction&gt;" in table
    assert f"<td>{label}</td>" in table
    assert "future_<reason>" not in table
    row = next(
        row
        for row in scope["tables"]["ledger"]["rows"]
        if row["filter_status"] == status
    )
    assert (
        row["filled_size"] is row["filled_price"] is row["fees"] is row["side"] is None
    )
    assert row["delta.cash"] == -1e-12
    assert row["reconciliation"] == "可核對項目一致"
    _, exported = _csv_rows(document, "ledger-0-0")
    csv_row = next(row for row in exported if row["委託處理結果"] == label)
    assert csv_row["實際成交數量（標的單位）"] == ""
    assert csv_row["實際手續費（報價貨幣）"] == ""
    saved = scope["observations"]["execution_records"][index]
    assert saved == record.model_dump(mode="json")


def test_standard_template_has_one_ledger_and_custom_section_selection_remains_explicit(
    tmp_path: Path,
) -> None:
    result = SignalBacktestEngine(init_cash=1000.0).run(_market())
    service, _ = _save(tmp_path, result)
    catalogue = service.get_report_sections()
    assert "ledger" in catalogue["templates"]["standard"]
    assert not {"executions", "account_history"} & set(
        catalogue["templates"]["standard"]
    )
    default = service.generate_report(["execution"])
    assert default["success"], default
    document = ReportMarkup()
    document.feed(Path(default["path"]).read_text())
    assert document.sections == catalogue["templates"]["standard"]
    _, _, document, _ = _report(service, ["ledger", "trades"])
    assert document.sections == ["ledger", "trades"]
    _, _, document, scope = _report(service, [])
    assert document.sections == [] and document.downloads == []
    assert scope["tables"]["ledger"]["rows"]  # Selection does not prune embedded data.
    obsolete = service.generate_report(["execution"], ["executions"])
    assert not obsolete["success"]


@pytest.mark.parametrize("mode", ["legacy", "volume", "no_orders", "empty"])
def test_missing_unsupported_and_recorded_empty_ledgers_remain_distinct(
    tmp_path: Path, mode: str
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
    path, text, document, scope = _report(service)
    spec = scope["tables"]["ledger"]
    if mode in {"legacy", "volume"}:
        expected = (
            "not_recorded" if mode == "legacy" else "unsupported_volume_accounting"
        )
        assert (
            spec["execution_availability"] == spec["account_availability"] == expected
        )
        assert expected in text
        assert spec["rows"] == []
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert (
            manifest["available_data"]["execution"]["full"]["execution_record_count"]
            is None
        )
    else:
        assert (
            spec["execution_availability"]
            == spec["account_availability"]
            == "available"
        )
        if mode == "empty":
            assert spec["rows"] == []
        else:
            assert document.rows == {"ledger-0-0-static": 6}
            assert all(row["filter_status"] == "bar_close" for row in spec["rows"])
            assert all(row["filled_size"] is None for row in spec["rows"])


def test_reconciliation_difference_smaller_than_display_precision_survives_csv(
    tmp_path: Path,
) -> None:
    result = SignalBacktestEngine(init_cash=1000.0, fees=0.001, slippage=0.01).run(
        _market()
    )
    assert result.execution_records is not None
    first = result.execution_records[0]
    # Vary the saved requested rate, preserving the actual execution and balances.
    result.execution_records[0] = first.model_copy(
        update={"requested_fee_rate": first.requested_fee_rate + 1e-7}
    )
    service, _ = _save(tmp_path, result)
    _, _, document, scope = _report(service)
    rows = scope["tables"]["ledger"]["rows"]
    index = next(
        index for index, row in enumerate(rows) if row["filter_status"] == "filled"
    )
    row = rows[index]
    assert row["reconciliation"] == "有核對差異"
    assert abs(row["fees_residual"]) < 0.005
    columns = scope["tables"]["ledger"]["columns"]
    column_index = next(
        index
        for index, column in enumerate(columns)
        if column["field"] == "fees_residual"
    )
    assert document.cells["ledger-0-0-static"][index][column_index] == "0.00"
    _, exported = _csv_rows(document, "ledger-0-0")
    assert (
        float(exported[index][columns[column_index]["title"]]) == row["fees_residual"]
    )
