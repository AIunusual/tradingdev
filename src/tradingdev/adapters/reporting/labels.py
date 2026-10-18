"""Traditional Chinese display labels, independent of saved evidence definitions.

These helpers return plain text. Renderers must escape their output and preserve
the original identifiers, definitions and numeric values in raw evidence.
"""

from __future__ import annotations

_METRIC_LABELS = {
    "total_pnl": "總淨損益",
    "total_return": "累積報酬率",
    "annual_return": "年化複合報酬率",
    "sharpe_ratio": "夏普比率",
    "sortino_ratio": "索提諾比率",
    "calmar_ratio": "卡瑪比率",
    "annual_volatility": "年化波動率",
    "max_drawdown": "最大回撤率",
    "daily_max_drawdown": "每日收盤最大回撤率",
    "max_drawdown_amount": "最大回撤金額",
    "total_trades": "已平倉交易數",
    "open_trades": "期末未平倉交易數",
    "win_rate": "已平倉交易勝率",
    "profit_factor": "獲利因子",
    "trade_expectancy": "每筆已平倉交易平均淨損益",
    "avg_holding_bars": "已平倉交易平均持有根數",
    "total_volume": "累計成交金額",
    "total_fees": "累計手續費",
    "total_slippage": "另計滑價成本",
    "n_days": "實際觀測日數",
    "n_months": "實際觀測月數",
    "monthly_trades_mean": "每月平均已平倉交易數",
    "monthly_volume_mean": "每月平均成交金額",
    **{
        f"{period}_pnl_{stat}": f"{period_label}淨損益{stat_label}"
        for period, period_label in (("daily", "每日"), ("monthly", "每月"))
        for stat, stat_label in (
            ("mean", "平均值"),
            ("std", "標準差"),
            ("min", "最小值"),
            ("max", "最大值"),
            ("median", "中位數"),
        )
    },
}

_UNIT_LABELS = {
    "amount": "報價貨幣金額",
    "fraction": "百分比",
    "ratio": "比率（無單位）",
    "count": "計數",
    "bars": "根 K 棒",
}

_REASON_LABELS = {
    "not_applicable": "此回測模式不適用",
    "embedded_in_fill_prices": "滑價已反映於成交價格，未另計金額",
    "no_trades": "沒有已平倉交易",
    "no_closed_trades": "沒有已平倉交易",
    "insufficient_data": "觀測資料不足",
    "nonpositive_capital": "資金值不符合報酬率計算條件",
    "unbounded": "結果為無限值",
    "zero_denominator": "分母為零或結果未定義",
    "missing_annualization": "未記錄年化期數設定",
    "missing_timestamps": "未記錄時間戳記",
    "unsupported_daily_sampling": "K 棒週期大於一天，無法建立每日觀測",
    "unknown_bar_frequency": "無法識別 K 棒週期",
    "unsupported_volume_accounting": "名目金額回測模式未提供此帳戶資料流",
    "not_recorded": "此保存結果未記錄資料",
}

_ENUM_LABELS = {
    "status": {
        "filled": "已成交",
        "ignored": "已忽略",
        "rejected": "已拒絕",
    },
    "side": {"buy": "買入", "sell": "賣出"},
    "requested_size_kind": {
        "finite": "有限數值",
        "positive_infinity": "正無限數值",
        "negative_infinity": "負無限數值",
        "not_a_number": "無有效數值",
    },
    "requested_size_type": {
        "amount": "交易數量（標的單位）",
        "value": "交易金額（報價貨幣）",
        "percent": "可用資源比例",
        "targetamount": "目標持倉數量（標的單位）",
        "targetvalue": "目標持倉金額（報價貨幣）",
        "targetpercent": "目標持倉占總權益比例",
    },
    "requested_direction": {
        "longonly": "僅做多",
        "shortonly": "僅做空",
        "both": "允許做多與做空",
    },
    "status_info": {
        "sizenan": "委託數量不是有效數值",
        "pricenan": "委託價格不是有效數值",
        "valpricenan": "估值價格不是有效數值",
        "valuenan": "帳戶權益不是有效數值",
        "valuezeroneg": "帳戶權益為零或負數",
        "sizezero": "委託數量為零",
        "nocashshort": "可用現金不足以做空",
        "nocashlong": "現金不足以做多",
        "noopenposition": "沒有可減少或平倉的持倉",
        "maxsizeexceeded": "委託數量超過允許上限",
        "randomevent": "觸發隨機拒單事件",
        "cantcoverfees": "現金不足以支付手續費",
        "minsizenotreached": "成交數量未達允許下限",
        "partialfill": "成交數量小於委託數量",
    },
    "kind": {
        "backtest": "單一回測範圍",
        "fold_summary": "各折測試結果摘要",
    },
    "mode": {
        "signal": "訊號回測模式",
        "volume": "名目金額回測模式",
    },
    "split": {"full": "完整區間", "train": "訓練區間", "test": "測試區間"},
    "parameter_provenance": {
        "scope_snapshot": "保存範圍的參數快照",
        "execution_manifest": "固定執行清單的建構參數",
        "execution_manifest_and_trial": "固定執行清單參數加上本次試驗參數",
        "aggregate_scope_snapshot": "摘要範圍的參數快照",
    },
}


RECORD_LABELS = {
    "execution_id": "執行紀錄編號",
    "order_id": "成交單編號",
    "bar_index": "K 線索引（從 0 起算）",
    "status": "執行狀態",
    "status_info": "狀態原因",
    "side": "買賣方向",
    "timestamp": "K 線時間（UTC）",
    "market.open": "市場開盤價",
    "market.high": "市場最高價",
    "market.low": "市場最低價",
    "market.close": "市場收盤價",
    "requested_price": "委託價格",
    "filled_price": "成交價格",
    "filled_size": "成交數量（標的單位）",
    "fees": "成交手續費（報價貨幣）",
    "valuation_price": "執行時權益估值價格",
    "requested_size": "委託規模（依規模類型）",
    "requested_size_kind": "委託規模數值類別",
    "requested_size_type": "委託規模類型",
    "requested_direction": "允許的持倉方向",
    "requested_fee_rate": "單邊手續費率",
    "requested_fixed_fees": "每次固定費用（報價貨幣）",
    "requested_slippage": "滑價率",
    "mark_price": "收盤估值價格",
    "cash": "現金（報價貨幣）",
    "free_cash": "可用現金（引擎帳本，報價貨幣）",
    "position": "總持倉數量（標的單位）",
    "asset_value": "持倉市值（報價貨幣）",
    "equity": "帳戶權益（報價貨幣）",
    "raw": "原始資料",
    **{
        f"{phase}.{key}": f"{phase_label}{label}（{unit}）"
        for phase, phase_label in (("before", "執行前"), ("after", "執行後"))
        for key, label, unit in (
            ("cash", "現金", "報價貨幣"),
            ("position", "總持倉", "標的單位"),
            ("equity", "帳戶權益", "報價貨幣"),
            ("free_cash", "可用現金", "引擎帳本，報價貨幣"),
            ("debt", "空單負債", "引擎帳本，報價貨幣"),
        )
    },
}


def metric_label(metric_id: str) -> str:
    """Name a known metric without substituting its saved definition."""
    return _METRIC_LABELS.get(metric_id, f"未識別指標：{metric_id}")


def unit_label(unit: str) -> str:
    """Name the displayed unit; fractional values are shown as percentages."""
    return _UNIT_LABELS.get(unit, f"未識別單位：{unit}")


def reason_label(reason: str) -> str:
    """Explain a saved unavailable reason without treating missing data as zero."""
    return _REASON_LABELS.get(reason, f"未識別原因：{reason}")


def enum_label(field: str, value: str) -> str:
    """Translate a known field's value, preserving unknown codes visibly."""
    return _ENUM_LABELS.get(field, {}).get(value, f"未識別選項：{value}")


def scope_name(scope_id: str | None) -> str | None:
    if scope_id is None:
        return None
    names = {
        "full": "完整區間",
        "train": "訓練區間",
        "test": "測試區間",
        "test_summary": "各折測試摘要",
    }
    if scope_id in names:
        return names[scope_id]
    parts = scope_id.split("/")
    if (
        len(parts) == 3
        and parts[0] in ("fold", "trial")
        and parts[1].isascii()
        and parts[1].isdigit()
        and parts[2] in names
    ):
        try:
            index = str(int(parts[1]) + 1)
        except ValueError:
            pass  # Excessively long numeric identifiers remain custom scopes.
        else:
            group = "折" if parts[0] == "fold" else "次參數試驗"
            return f"第 {index} {group} · {names[parts[2]]}（{scope_id}）"
    return f"自訂結果範圍（{scope_id}）"
