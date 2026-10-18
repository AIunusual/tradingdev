"""One chronological, full-precision view of saved attempts and bar valuations.

This is a projection of observations, never a replay or a broker ledger. A row's
before/after values retain their own valuation prices. In particular, a close
valuation after a fill is not another order and may change equity without cash
or position changing. Missing source fields remain missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.backtest.execution_records import (
        AccountSnapshot,
        AccountState,
        ExecutionRecord,
        MarketReference,
    )

type LedgerValue = str | int | float | None
type Availability = Literal[
    "available", "not_recorded", "unsupported_volume_accounting"
]


@dataclass(frozen=True)
class LedgerColumn:
    """Shared human column order and visibility for every ledger renderer."""

    key: str
    label: str
    group: str
    kind: Literal["number", "integer", "text"] = "number"
    default_visible: bool = False


LEDGER_COLUMNS = (
    LedgerColumn("timestamp", "K 線時間（UTC）", "時間與結果", "text", True),
    LedgerColumn("sequence", "事件順序（從 1 起算）", "時間與結果", "integer", True),
    LedgerColumn("bar_index", "K 線索引（從 0 起算）", "時間與結果", "integer"),
    LedgerColumn("event_type", "事件種類", "時間與結果", "text", True),
    LedgerColumn("status", "委託處理結果", "時間與結果", "text", True),
    LedgerColumn("status_info", "處理原因", "時間與結果", "text"),
    LedgerColumn("execution_id", "委託執行編號", "時間與結果", "integer"),
    LedgerColumn("order_id", "成交單編號", "時間與結果", "integer"),
    LedgerColumn("side", "成交買賣方向", "時間與結果", "text", True),
    LedgerColumn("comparison_basis", "前值比較基準", "時間與結果", "text"),
    *(
        LedgerColumn(f"market.{key}", f"市場{label}價", "市場價格")
        for key, label in (
            ("open", "開盤"),
            ("high", "最高"),
            ("low", "最低"),
            ("close", "收盤"),
        )
    ),
    LedgerColumn("requested_size", "委託規模", "委託內容", default_visible=True),
    LedgerColumn("requested_size_kind", "委託規模數值類別", "委託內容", "text", True),
    LedgerColumn("requested_size_unit", "委託規模單位", "委託內容", "text", True),
    LedgerColumn("requested_size_type", "委託規模類型", "委託內容", "text", True),
    LedgerColumn("requested_direction", "允許的持倉方向", "委託內容", "text"),
    LedgerColumn(
        "requested_price", "委託基準價格（滑價前）", "委託內容", default_visible=True
    ),
    LedgerColumn("requested_fee_rate", "委託手續費率（比例）", "委託內容"),
    LedgerColumn("requested_fixed_fees", "委託固定手續費（報價貨幣）", "委託內容"),
    LedgerColumn("requested_slippage", "委託滑價率（比例）", "委託內容"),
    LedgerColumn(
        "filled_size", "實際成交數量（標的單位）", "成交結果", default_visible=True
    ),
    LedgerColumn("signed_filled_size", "帶正負號成交數量（標的單位）", "成交結果"),
    LedgerColumn("filled_price", "實際成交價格", "成交結果", default_visible=True),
    LedgerColumn(
        "notional", "成交名目金額（報價貨幣）", "成交結果", default_visible=True
    ),
    LedgerColumn("fees", "實際手續費（報價貨幣）", "成交結果", default_visible=True),
    *(
        LedgerColumn(
            f"{phase}.{field}",
            (
                "持倉數量變動（標的單位，負數為減少）"
                if field == "position" and phase == "delta"
                else f"{phase_label}{label}"
            ),
            group,
            default_visible=field in ("cash", "position"),
        )
        for field, label, group in (
            ("cash", "現金（報價貨幣）", "現金與持倉"),
            ("position", "持倉數量（標的單位，負數為空倉）", "現金與持倉"),
            ("free_cash", "可用現金（報價貨幣，VectorBT）", "其他帳戶欄位"),
            ("debt", "帳面負債（報價貨幣，VectorBT）", "其他帳戶欄位"),
            ("equity", "帳戶權益（報價貨幣）", "權益估值"),
            ("asset_value", "持倉估值（報價貨幣）", "權益估值"),
        )
        for phase, phase_label in (
            ("before", "事件前"),
            ("delta", "變動"),
            ("after", "事件後"),
        )
    ),
    LedgerColumn("before.valuation_price", "事件前權益估值價格", "權益估值"),
    LedgerColumn("after.valuation_price", "事件後權益估值價格", "權益估值"),
    LedgerColumn("expected_cash_delta", "依成交計算的現金變動", "核對算式"),
    LedgerColumn("expected_fees", "依費率計算的手續費", "核對算式"),
    LedgerColumn("expected_fill_price", "依滑價率計算的成交價格", "核對算式"),
    LedgerColumn("cash_residual", "現金核對差額（實際減計算）", "核對算式"),
    LedgerColumn("position_residual", "持倉核對差額（實際減計算）", "核對算式"),
    LedgerColumn("fees_residual", "手續費核對差額（實際減計算）", "核對算式"),
    LedgerColumn("fill_price_residual", "成交價格核對差額（實際減計算）", "核對算式"),
    LedgerColumn("equity_residual", "權益核對差額（實際減計算）", "核對算式"),
    LedgerColumn("reconciliation", "可得欄位算術核對", "核對算式", "text", True),
    LedgerColumn("reconciliation_note", "核對範圍與資料缺口", "核對算式", "text", True),
)

_VALUE_LABELS = {
    "event_type": {"execution": "委託處理", "bar_close": "K 線收盤估值"},
    "comparison_basis": {
        "execution_before": "本次委託處理前",
        "last_execution": "本根 K 線最後一次委託處理後",
        "previous_bar_close": "前一根已保存的 K 線收盤",
        "unavailable": "沒有已保存的前值",
    },
    "requested_size_unit": {
        "base_asset": "標的單位",
        "quote_currency": "報價貨幣",
        "ratio": "比例（1 表示 100%）",
    },
    "reconciliation": {
        "matched": "可核對項目一致",
        "mismatch": "有核對差異",
        "unavailable": "沒有足夠資料核對",
    },
}


def ledger_value_label(key: str, value: LedgerValue) -> str:
    """Translate projection enums; unrecognized source values remain visible."""
    if value is None:
        return "未記錄／不適用"
    return _VALUE_LABELS.get(key, {}).get(str(value), str(value))


@dataclass(frozen=True)
class LedgerProjection:
    """Complete event rows and explicit availability, without display rounding."""

    rows: list[dict[str, LedgerValue]]
    execution_availability: Availability
    account_availability: Availability
    notes: tuple[str, ...]


_NOTES = (
    "本表依 K 線索引及原生執行編號排序；收盤估值另列於同根 K 線委託之後。"
    "時間是 K 線時間，不是逐筆報價或交易所精確成交時間。",
    "市場開高低收是該根完整 K 線的保存價格，不表示委託時已知最高、最低或收盤價。"
    "未保存的資料維持空值，不由配對交易回推委託。",
    "帳戶採 VectorBT 通用現金與帶符號持倉計帳，不代表永續合約錢包、保證金或資金費。"
    "收盤帳本沒有保存負債，不能將空值解讀為零。",
    "算術核對使用完整精度及保存契約的相對／絕對容差，畫面小數位不影響結果。"
    "一致只表示所列算式符合保存數值，不表示市場報價、策略或交易所帳務已經驗證。",
    "買入費用另允許成交與費用運算尺度的 4 個浮點間距（ULP），"
    "涵蓋引擎以資金上限扣減成交金額時的捨入誤差；核對差額仍保留原值。",
    "委託規模須同時閱讀數值類別、類型及單位；無限規模表示引擎按可用資源決定，"
    "不是已送出或已成交無限數量。",
)

_BALANCES = ("cash", "position", "free_cash", "debt", "equity", "asset_value")


def _blank_row() -> dict[str, LedgerValue]:
    return dict.fromkeys(column.key for column in LEDGER_COLUMNS)


def _snapshot_values(
    state: AccountSnapshot | AccountState, price: float
) -> dict[str, float | None]:
    return {
        "cash": state.cash,
        "position": state.position,
        "free_cash": state.free_cash,
        "debt": getattr(state, "debt", None),
        "equity": state.equity,
        "asset_value": getattr(state, "asset_value", state.position * price),
        "valuation_price": price,
    }


def _balances(
    row: dict[str, LedgerValue],
    before: dict[str, float | None] | None,
    after: dict[str, float | None],
) -> None:
    for key, value in after.items():
        row[f"after.{key}"] = value
    if before is None:
        return
    for key, value in before.items():
        row[f"before.{key}"] = value
    for key in _BALANCES:
        first, last = before[key], after[key]
        row[f"delta.{key}"] = (
            last - first if last is not None and first is not None else None
        )


def _market(row: dict[str, LedgerValue], market: MarketReference) -> None:
    for key in ("open", "high", "low", "close"):
        row[f"market.{key}"] = getattr(market, key)


def _check(
    row: dict[str, LedgerValue],
    name: str,
    actual: float,
    expected: float,
    *,
    scale: float = 0.0,
    roundoff: float = 0.0,
) -> bool:
    residual = actual - expected
    row[f"{name}_residual"] = residual if math.isfinite(residual) else None
    return math.isclose(
        actual,
        expected,
        rel_tol=1e-10,
        abs_tol=max(1e-8, scale * 1e-10, roundoff),
    )


def _execution_row(record: ExecutionRecord) -> dict[str, LedgerValue]:
    row = _blank_row()
    row.update(
        {
            "event_type": "execution",
            "comparison_basis": "execution_before",
            **{
                key: getattr(record, key)
                for key in (
                    "timestamp",
                    "bar_index",
                    "execution_id",
                    "order_id",
                    "status",
                    "status_info",
                    "side",
                    "requested_size",
                    "requested_size_kind",
                    "requested_size_type",
                    "requested_direction",
                    "requested_price",
                    "requested_fee_rate",
                    "requested_fixed_fees",
                    "requested_slippage",
                    "filled_size",
                    "filled_price",
                    "fees",
                )
            },
        }
    )
    row["requested_size_unit"] = {
        "amount": "base_asset",
        "targetamount": "base_asset",
        "value": "quote_currency",
        "targetvalue": "quote_currency",
        "percent": "ratio",
        "targetpercent": "ratio",
    }.get(record.requested_size_type)
    _market(row, record.market)
    _balances(
        row,
        _snapshot_values(record.before, record.valuation_price),
        _snapshot_values(record.after, record.valuation_price),
    )
    checks = [
        _check(
            row,
            "equity",
            record.after.equity,
            record.after.cash + record.after.position * record.valuation_price,
        )
    ]
    if record.status == "filled":
        assert record.filled_size is not None
        assert record.filled_price is not None
        assert record.fees is not None
        direction = 1 if record.side == "buy" else -1
        signed = direction * record.filled_size
        notional = record.filled_size * record.filled_price
        expected_cash_delta = -direction * notional - record.fees
        expected_fees = (
            notional * record.requested_fee_rate + record.requested_fixed_fees
        )
        expected_price = record.requested_price * (
            1 + direction * record.requested_slippage
        )
        # In a cash-limited buy, VectorBT computes fees as
        # cash_limit - (cash_limit - fixed_fees) / (1 + fee_rate).
        # Reconstructing them from filled_size * filled_price * fee_rate uses
        # different rounded division/multiplication steps. Bound this by four
        # ULPs at the actual transaction scale, not a relative notional tolerance
        # (which could conceal a substantial fee error when the rate is small).
        # Unused account cash must not enlarge a small trade's fee tolerance.
        fee_roundoff = (
            4
            * max(
                math.ulp(notional),
                math.ulp(notional + record.fees),
                math.ulp(record.requested_fixed_fees),
            )
            if record.side == "buy"
            else 0.0
        )
        row.update(
            {
                "signed_filled_size": signed,
                "notional": notional,
                "expected_cash_delta": expected_cash_delta,
                "expected_fees": expected_fees,
                "expected_fill_price": expected_price,
            }
        )
        checks.extend(
            (
                _check(
                    row,
                    "cash",
                    record.after.cash,
                    record.before.cash + expected_cash_delta,
                    scale=max(abs(record.before.cash), notional),
                ),
                _check(
                    row,
                    "position",
                    record.after.position,
                    record.before.position + signed,
                    scale=max(abs(record.before.position), abs(signed)),
                ),
                _check(row, "fees", record.fees, expected_fees, roundoff=fee_roundoff),
                _check(row, "fill_price", record.filled_price, expected_price),
            )
        )
        row["reconciliation_note"] = (
            "核對成交帶符號數量與持倉變動、成交金額加手續費與現金變動、"
            "費率加固定費、方向滑價及事件後權益。未核對交易所保證金。"
        )
    else:
        checks.extend(
            (
                _check(row, "cash", record.after.cash, record.before.cash),
                _check(row, "position", record.after.position, record.before.position),
            )
        )
        row["reconciliation_note"] = (
            "未成交；核對現金與持倉無變動及事件後權益。"
            "引擎可能將極小餘額歸零；成交數量、成交價格與手續費不適用。"
        )
    if row["requested_size_unit"] is None:
        row["reconciliation_note"] = (
            str(row["reconciliation_note"]) + " 未識別委託規模類型，單位未推測。"
        )
    row["reconciliation"] = "matched" if all(checks) else "mismatch"
    return row


def _close_row(
    state: AccountState,
    previous: AccountState | None,
    last_execution: ExecutionRecord | None,
) -> dict[str, LedgerValue]:
    row = _blank_row()
    row.update(
        {
            "timestamp": state.timestamp,
            "bar_index": state.bar_index,
            "event_type": "bar_close",
            "comparison_basis": "unavailable",
            "market.close": state.mark_price,
        }
    )
    before = None
    if last_execution is not None:
        _market(row, last_execution.market)
        before = _snapshot_values(last_execution.after, last_execution.valuation_price)
        row["comparison_basis"] = "last_execution"
    elif previous is not None:
        before = _snapshot_values(previous, previous.mark_price)
        row["comparison_basis"] = "previous_bar_close"
    _balances(row, before, _snapshot_values(state, state.mark_price))
    # Retain the saved value rather than substituting an equivalent calculation.
    row["after.asset_value"] = state.asset_value
    if before is not None and before["asset_value"] is not None:
        row["delta.asset_value"] = state.asset_value - before["asset_value"]
    checks = [
        _check(
            row, "equity", state.equity, state.cash + state.position * state.mark_price
        )
    ]
    if last_execution is not None:
        checks.extend(
            (
                _check(row, "cash", state.cash, last_execution.after.cash),
                _check(row, "position", state.position, last_execution.after.position),
            )
        )
    row["reconciliation"] = "matched" if all(checks) else "mismatch"
    row["reconciliation_note"] = (
        "收盤估值不是成交；核對收盤權益＝現金＋帶符號持倉×收盤估值價。"
        + ("另比對本根最後委託後的現金與持倉。" if last_execution is not None else "")
        + "收盤負債未保存；空白成交欄位不代表零值成交。"
        + (
            "沒有已保存的前值，變動欄位留空。"
            if before is None
            else "前後估值價格可能不同。"
        )
    )
    return row


def build_ledger(
    execution_records: Sequence[ExecutionRecord] | None,
    account_history: Sequence[AccountState] | None,
    *,
    mode: Literal["signal", "volume"] = "signal",
) -> LedgerProjection:
    """Merge saved streams without changing source records or filling their gaps.

    Callers pass the already validated scope observations. Sorting additionally
    guarantees deterministic row order for independently supplied sequences.
    ``None`` means not recorded; an empty sequence means recorded with no events.
    """
    if mode == "volume":
        return LedgerProjection(
            [], "unsupported_volume_accounting", "unsupported_volume_accounting", _NOTES
        )
    records = sorted(
        execution_records or (),
        key=lambda record: (record.bar_index, record.execution_id),
    )
    accounts = sorted(account_history or (), key=lambda state: state.bar_index)
    events: dict[int, list[ExecutionRecord]] = {}
    for record in records:
        events.setdefault(record.bar_index, []).append(record)
    closes = {state.bar_index: state for state in accounts}
    rows: list[dict[str, LedgerValue]] = []
    previous = None
    for bar_index in sorted(events.keys() | closes.keys()):
        attempts = events.get(bar_index, [])
        rows.extend(_execution_row(record) for record in attempts)
        if (state := closes.get(bar_index)) is not None:
            rows.append(_close_row(state, previous, attempts[-1] if attempts else None))
            previous = state
    for index, row in enumerate(rows, start=1):
        row["sequence"] = index
    return LedgerProjection(
        rows,
        "not_recorded" if execution_records is None else "available",
        "not_recorded" if account_history is None else "available",
        _NOTES,
    )
