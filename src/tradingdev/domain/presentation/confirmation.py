"""Complete, human-readable confirmation documents for captured executions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from tradingdev.domain.backtest.execution_policy import DEFAULT_VOLUME_POSITION_SIZE
from tradingdev.domain.data.crawlers.registry import describe_sources
from tradingdev.domain.execution_assumptions import (
    resolve_execution_assumptions,
    summarize_execution_assumptions,
)
from tradingdev.domain.parameter_values import effective_candidates, parameter_leaves
from tradingdev.domain.performance.sampling import daily_observation_unavailable_reason
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.presentation.confirmation_contract import (
    SECTION_TITLES,
    ConfirmationDocument,
    ConfirmationField,
    ConfirmationIdentity,
    ConfirmationPresentation,
    ConfirmationSection,
    SectionId,
    validate_presentation,
)
from tradingdev.domain.research import ResearchMetadata
from tradingdev.domain.validation.config import parse_walk_forward_config
from tradingdev.domain.validation.date_windows import inclusive_day_window

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Confirmation requires a complete {name} object")
    return dict(value)


def _known_fields(
    values: dict[str, object], known: set[str], name: str, *, complete: bool = True
) -> None:
    unknown = values.keys() - known
    missing = known - values.keys() if complete else set()
    if unknown or missing:
        raise ValueError(
            f"Unrendered or missing {name} settings: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _text(value: object, *, missing: str = "未設定") -> str:
    if value is None:
        return missing
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, dict):
        if value:
            raise ValueError("Structured settings require labels for every leaf")
        return "空的設定集合"
    if isinstance(value, list):
        if value:
            raise ValueError("Structured settings require labels for every leaf")
        return "空的清單"
    return str(value)


def _percentage(value: object) -> str:
    if value is None:
        return "未啟用"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("A percentage setting must be numeric")
    number = format(Decimal(str(value)) * 100, "f")
    return (number.rstrip("0").rstrip(".") if "." in number else number) + "%"


def _field(
    label: str,
    value: object,
    description: str = "",
    *,
    missing: str = "未設定",
    detail: bool = False,
    unit: str | None = None,
) -> ConfirmationField:
    return ConfirmationField(
        label=label,
        value_text=_text(value, missing=missing),
        description=description,
        detail=detail,
        unit=unit,
    )


def _section(
    key: SectionId,
    fields: list[ConfirmationField] | None = None,
    paragraphs: tuple[str, ...] = (),
) -> ConfirmationSection:
    return ConfirmationSection(
        id=key,
        title=SECTION_TITLES[key],
        fields=tuple(fields or ()),
        paragraphs=paragraphs,
    )


_SOURCE_NAMES = {
    "binance_vision": "Binance 歷史資料庫",
    "binance_api": "Binance 行情介面",
    "yahoo_finance": "Yahoo Finance",
    "deribit": "Deribit 波動率資料",
    "custom": "自訂資料",
}
_METRIC_NAMES = {
    "total_pnl": "總淨損益",
    "total_return": "總報酬率",
    "annual_return": "年化報酬率",
    "sharpe_ratio": "夏普比率（每單位波動的超額報酬）",
    "sortino_ratio": "索提諾比率（每單位下行風險的報酬）",
    "calmar_ratio": "卡瑪比率（年化報酬與最大每日回撤之比）",
    "annual_volatility": "年化波動率",
    "max_drawdown": "最大回撤率",
    "daily_max_drawdown": "每日收盤權益最大回撤率",
    "max_drawdown_amount": "最大回撤金額",
    "total_trades": "已平倉交易筆數",
    "open_trades": "未平倉交易筆數",
    "win_rate": "已平倉交易勝率",
    "profit_factor": "獲利因子（總獲利與總虧損之比）",
    "trade_expectancy": "每筆已平倉交易平均淨損益",
    "avg_holding_bars": "平均持倉 K 線數",
    "total_volume": "總成交金額",
    "total_fees": "總手續費",
    "total_slippage": "總滑價成本",
    "n_days": "有資料的日數",
    "n_months": "有資料的月數",
    "monthly_trades_mean": "每月平均已平倉筆數",
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


def _metric(value: object) -> str:
    if not isinstance(value, str) or value not in _METRIC_NAMES:
        raise ValueError(f"Metric lacks a built-in confirmation description: {value}")
    return _METRIC_NAMES[value]


def _source(value: object) -> str:
    # Provider IDs can be names of custom providers, rather than setting keys.
    return _SOURCE_NAMES.get(str(value), f"資料提供者：{value}")


def _timeframe(value: object) -> str:
    text = str(value)
    units = {"m": "分鐘", "h": "小時", "d": "天", "w": "週", "M": "月"}
    if len(text) > 1 and text[:-1].isdigit() and text[-1] in units:
        return f"每根 K 線 {text[:-1]} {units[text[-1]]}"
    return f"K 線週期：{text}"


def _market_fields(
    backtest: dict[str, object], data: dict[str, object]
) -> tuple[list[ConfirmationField], list[ConfirmationField]]:
    _known_fields(
        data,
        {"source", "market_type", "raw_dir", "processed_dir", "requirements"},
        "data",
    )
    requirements = _object(data["requirements"], "data requirements")
    _known_fields(requirements, {"market", "features"}, "data requirements")
    market = _object(requirements["market"], "market data")
    _known_fields(market, {"source", "symbol", "timeframe"}, "market data")
    source = market["source"]
    source_details = next(
        (item for item in describe_sources() if item["source"] == source), None
    )
    if source_details is None:
        raise ValueError(f"Unknown market data source: {source}")
    uses_market_type = bool(source_details["market_types"])
    market_type = {
        "spot": "現貨",
        "futures/um": "以穩定幣保證金計價的合約",
        "futures/cm": "以幣本位保證金計價的合約",
    }.get(str(data["market_type"]), str(data["market_type"]))
    fields = [
        _field("回測商品", backtest["symbol"]),
        _field("回測 K 線週期", _timeframe(backtest["timeframe"])),
        _field("回測開始", backtest["start_date"]),
        _field("回測結束", backtest["end_date"]),
        _field("時間邊界", "依 K 棒時間戳記篩選，包含起訖兩端（有相符資料時）"),
        _field(
            "時間基準",
            "UTC（世界協調時間）",
            "未附時區的時間依 UTC 解讀；純日期代表當日 00:00:00，"
            "結束日期不自動延伸至當日最後一刻。",
        ),
        _field("本次行情來源", _source(market["source"])),
        _field("行情商品", market["symbol"], detail=True),
        _field("行情 K 線週期", _timeframe(market["timeframe"]), detail=True),
        _field("預設資料來源", _source(data["source"]), detail=True),
        _field(
            "市場類型",
            market_type if uses_market_type else f"不適用（{_source(source)}）",
            "本次行情來源使用此設定選擇市場。"
            if uses_market_type
            else "此行情來源不使用市場類型設定；商品種類依來源與商品代碼定義。",
        ),
    ]
    technical = [
        _field("原始行情保存位置", data["raw_dir"]),
        _field("處理後行情保存位置", data["processed_dir"]),
    ]
    if not uses_market_type:
        technical.append(
            _field(
                "未適用的市場類型設定",
                data["market_type"],
                "保留的共用設定值，不參與本次行情選擇。",
            )
        )
    feature_values = requirements["features"]
    if not isinstance(feature_values, list):
        raise ValueError("Confirmation feature requirements must be a list")
    if not feature_values:
        fields.append(_field("額外資料", "無"))
    for index, feature_value in enumerate(feature_values, 1):
        feature = _object(feature_value, "feature")
        _known_fields(
            feature, {"type", "source", "column", "path", "raw_path"}, "feature"
        )
        prefix = f"額外資料 {index}"
        feature_type = {
            "dvol": "隱含波動率指數",
            "funding_rate": "資金費率",
            "custom": "自訂資料",
        }.get(str(feature["type"]))
        if feature_type is None:
            raise ValueError("Feature type lacks a confirmation description")
        fields.extend(
            [
                _field(prefix + "類型", feature_type),
                _field(prefix + "來源", _source(feature["source"])),
            ]
        )
        technical.extend(
            _field(prefix + label, feature[key])
            for key, label in (
                ("column", "策略資料欄位"),
                ("path", "檔案位置"),
                ("raw_path", "原始檔案位置"),
            )
        )
    return fields, technical


def _execution_fields(
    backtest: dict[str, object], parallel: dict[str, object], seed: object
) -> list[ConfirmationField]:
    _known_fields(
        backtest,
        {
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "init_cash",
            "fees",
            "slippage",
            "position_size",
            "stop_loss",
            "take_profit",
            "signal_as_position",
            "re_entry_after_sl",
            "mode",
            "monthly_max_loss",
            "periods_per_year",
            "risk_free_rate",
            "required_return",
        },
        "backtest",
    )
    _known_fields(
        parallel, {"reserve_cores", "safety_factor", "overhead_multiplier"}, "parallel"
    )
    volume = backtest["mode"] == "volume"
    fields = [
        _field(
            "計算方式", "固定名目金額、逐筆累計損益" if volume else "依帳戶資金計算報酬"
        ),
        _field(
            "初始資金",
            backtest["init_cash"],
            "此計算方式不使用初始資金。" if volume else "用來計算帳戶資金與報酬。",
            unit="報價貨幣",
            missing="此計算方式不使用",
        ),
        _field(
            "單次投入金額",
            backtest["position_size"],
            unit="報價貨幣",
            missing=f"{DEFAULT_VOLUME_POSITION_SIZE:g}（執行政策預設）"
            if volume
            else "未指定固定金額",
        ),
        _field("手續費率", _percentage(backtest["fees"])),
        _field("滑價率", _percentage(backtest["slippage"])),
        _field("停損幅度", _percentage(backtest["stop_loss"])),
        _field("停利幅度", _percentage(backtest["take_profit"])),
    ]
    for key, label in (
        ("signal_as_position", "沒有持倉訊號時平倉"),
        ("re_entry_after_sl", "停損停利後允許重新進場"),
        ("monthly_max_loss", "每月最大虧損門檻"),
    ):
        fields.append(
            _field(
                label,
                backtest[key],
                "" if volume else "此計算方式不使用這項設定。",
                detail=True,
                unit="報價貨幣" if key == "monthly_max_loss" else None,
            )
        )
    fields.extend(
        [
            _field(
                "年化計算的每年日數",
                backtest["periods_per_year"],
                detail=True,
                unit="天",
            ),
            _field(
                "年化無風險利率",
                _percentage(backtest["risk_free_rate"]),
                "用於夏普比率等風險指標。",
                detail=True,
            ),
            _field(
                "年化最低目標報酬率",
                _percentage(backtest["required_return"]),
                "用於索提諾比率的下行風險計算。",
                detail=True,
            ),
            _field(
                "隨機種子",
                seed,
                "固定專案提供的隨機產生器；不代表所有第三方模型都會固定。",
                missing="不固定，每次使用獨立亂數",
                detail=True,
            ),
            _field(
                "保留的處理器核心", parallel["reserve_cores"], unit="核心", detail=True
            ),
            _field(
                "平行執行可用資源比例",
                _percentage(parallel["safety_factor"]),
                detail=True,
            ),
            _field(
                "平行執行額外資源估計倍數",
                parallel["overhead_multiplier"],
                unit="倍",
                detail=True,
            ),
        ]
    )
    return fields


def _validation_section(value: object) -> ConfirmationSection:
    if value is None:
        return _section("validation", paragraphs=("本次不採用分段訓練與驗證。",))
    values = _object(value, "validation")
    config = parse_walk_forward_config(values)
    common = {"mode", "target_metric", "label_horizon"}
    dates = ("train_start", "train_end", "test_start", "test_end")
    mode_fields = {
        "automatic": {"n_splits", "train_ratio", "expanding"},
        "explicit": set(dates),
        "purged": {"n_splits", "embargo_bars"},
    }
    _known_fields(values, common | mode_fields[config.mode], "validation")
    fields = []
    if config.mode == "explicit":
        fields.append(_field("分段方式", "使用指定日期的一組訓練／測試區間"))
        date_values = {}
        for start_key, end_key in (
            ("train_start", "train_end"),
            ("test_start", "test_end"),
        ):
            lower, upper = inclusive_day_window(
                datetime.fromisoformat(str(values[start_key])),
                datetime.fromisoformat(str(values[end_key])),
            )
            date_values[start_key] = f"{lower:%Y-%m-%d %H:%M:%S} UTC（含）"
            date_values[end_key] = f"{upper:%Y-%m-%d %H:%M:%S} UTC（不含）"
        fields.extend(
            _field(
                label,
                date_values[key],
                "日期先轉為 UTC（無時區視為 UTC），再忽略時分秒；包含起訖 UTC 整日。"
                f" 原始設定：{values[key]}。",
            )
            for key, label in zip(
                dates, ("訓練開始", "訓練結束", "測試開始", "測試結束"), strict=True
            )
        )
        fields.append(_field("預定分段組數", 1))
    else:
        fields.extend(
            [
                _field(
                    "分段方式",
                    "依資料筆數自動分段"
                    if config.mode == "automatic"
                    else "淨化分段：固定起點、逐次擴大訓練區間",
                ),
                _field(
                    "預定分段組數",
                    config.n_splits,
                    "實際資料須能形成完整分段；資料不足時會拒絕執行。",
                ),
            ]
        )
        if config.mode == "automatic":
            fields.extend(
                [
                    _field(
                        "自動分段的訓練比例",
                        _percentage(config.train_ratio),
                        "每組資料中用於訓練的比例。",
                    ),
                    _field(
                        "訓練區間逐次擴大",
                        config.expanding,
                        "啟用時保留更早的訓練資料。",
                    ),
                ]
            )
        else:
            fields.extend(
                [
                    _field(
                        "訓練區間逐次擴大",
                        True,
                        "額外保留第一段作為訓練；每組訓練區間先扣除標籤前視與隔離根數。",
                    ),
                    _field(
                        "額外隔離根數",
                        config.embargo_bars,
                        "淨化之外再多移除的 K 線。",
                        unit="根 K 線",
                        detail=True,
                    ),
                ]
            )
    horizon = config.label_horizon
    fields.extend(
        [
            _field(
                "驗證紀錄重點指標",
                _metric(config.target_metric),
                "用於驗證紀錄；策略如何訓練另依其策略設定。",
            ),
            _field(
                "標籤前視根數",
                horizon if horizon > 0 else "未啟用檢查（設定值：0）",
                "大於 0 時正式執行會先檢查標籤區間重疊；未啟用不代表標籤安全，"
                "也不代表策略沒有使用未來資料。",
                unit="根 K 線" if horizon > 0 else None,
                detail=True,
            ),
        ]
    )
    return _section("validation", fields)


def _parameter_field(
    pointer: str,
    value: object,
    presentation: ConfirmationPresentation,
    *,
    prefix: str = "",
) -> ConfirmationField:
    entry = presentation.parameter_descriptions[pointer]
    return ConfirmationField(
        label=prefix + entry.label,
        value_text=_text(value),
        description=entry.description,
        unit=entry.unit,
        explanation_source="model",
    )


def _optimization_section(
    manifest: ExecutionManifest, presentation: ConfirmationPresentation
) -> ConfirmationSection:
    search = manifest.optimization
    if search is None:
        return _section(
            "optimization", paragraphs=("本次只執行已列出的設定，不進行參數搜尋。",)
        )
    _known_fields(
        search.model_dump(),
        {
            "param_ranges",
            "optimization_metric",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "direction",
            "trial_timeout_seconds",
            "timeout_scope",
        },
        "optimization",
        complete=search.timeout_scope is not None,
    )
    fields = [
        _field("搜尋目標", _metric(search.optimization_metric)),
        _field(
            "選取方向", "越高越好" if search.direction == "maximize" else "越低越好"
        ),
        _field("訓練開始日", search.train_start),
        _field("訓練結束日", search.train_end, "包含當日。"),
        _field("保留測試開始日", search.test_start),
        _field("保留測試結束日", search.test_end, "包含當日；不拿測試結果選參數。"),
        _field("參數組合總數", search.total_combinations, unit="組"),
        _field(
            "單次試驗與保留測試時限" if search.timeout_scope else "第一組估時計算時限",
            search.trial_timeout_seconds,
            "每組參數及保留測試各自使用此時限；首組未完成時不提供估時。"
            if search.timeout_scope
            else "只限制正式搜尋中第一組參數的估時計算。",
            unit="秒",
            detail=True,
        ),
    ]
    for group_index, candidate_index, pointer, candidate in effective_candidates(
        manifest
    ):
        fields.extend(
            _parameter_field(
                leaf,
                value,
                presentation,
                prefix=f"搜尋組 {group_index}／候選 {candidate_index}：",
            )
            for leaf, value in parameter_leaves(candidate, pointer).items()
        )
    return _section(
        "optimization",
        fields,
        (
            "每個候選列出該搜尋組合併後的完整有效設定：未指定的巢狀欄位保留原值。"
            "不同搜尋組的候選會互相組合；未搜尋的策略設定保持原值。",
        ),
    )


_CHECK_NAMES = {
    "configuration": "完整設定解析",
    "execution_capabilities": "執行模式與訓練需求檢查",
    "parameter_binding": "策略參數綁定",
    "manifest_serialization": "固定執行設定序列化",
    "signal_contract": "合成資料的訊號契約檢查",
    "candidate_binding": "搜尋候選參數綁定",
}
_NOT_CHECKED_TEXT = {
    "historical_data": "尚未取得或檢查真實行情；實際可用日期、筆數與資料缺口尚未驗證。",
    "backtest_engine": "尚未執行回測引擎，成交、成本與停損停利路徑尚未驗證。",
    "training": "尚未訓練模型或執行驗證分段；訓練與訓練後訊號可能在正式執行時失敗。",
    "performance": "尚未計算績效；程式檢查通過不代表策略會獲利。",
}


def _preflight_sections(
    manifest: ExecutionManifest,
    preflight: PreflightReceipt,
    *,
    comparison_candidate: bool = False,
) -> tuple[ConfirmationSection, ConfirmationSection]:
    preflight.verify_manifest(manifest, comparison_candidate=comparison_candidate)
    fields = [
        _field("程式檢查狀態", "已完成必要檢查，可供使用者確認"),
        _field("程式檢查耗時", preflight.elapsed_seconds, unit="秒"),
        _field(
            "訊號檢查範圍",
            {
                "synthetic_contract_passed": "已通過固定合成資料的訊號契約檢查",
                "deferred_until_training": "此策略須先訓練；本次只檢查設定與執行模式",
                "bundled_not_executed": "內建策略只檢查設定與執行模式，未執行訊號",
            }[preflight.signal_check],
            "合成測資不是歷史行情，不能據此宣稱已驗證策略的所有交易規則。",
        ),
    ]
    if preflight.synthetic_fixture_rows:
        fields.append(
            _field(
                "合成訊號檢查筆數",
                "、".join(str(rows) for rows in preflight.synthetic_fixture_rows),
                "分別檢查兩段固定合成測資，不代表歷史暖機或行情覆蓋量。",
                unit="根 K 線",
            )
        )
    fields.extend(
        _field(_CHECK_NAMES[check], "通過") for check in preflight.checked_paths
    )
    if preflight.parameter_combinations_checked is not None:
        fields.append(
            _field(
                "已檢查參數綁定的組合數",
                preflight.parameter_combinations_checked,
                "只檢查參數合法性，不代表已執行候選的歷史回測或比較績效。",
                unit="組",
            )
        )
    limitations = tuple(_NOT_CHECKED_TEXT[check] for check in preflight.not_checked) + (
        "程式檢查不保證完整回測成功，也不是程式安全隔離。",
        "策略名稱、摘要與參數解釋由模型提供；參數值、執行設定與程式檢查結果由後端固定。",
        "確認後只執行這個版本。若程式或有效設定需要修改，必須重新檢查並再次確認。",
        "請選擇開始正式回測、要求調整設定，或取消。確認前不取得真實行情、不訓練模型、不執行回測。",
    )
    return _section("preflight", fields), _section(
        "limitations", paragraphs=limitations
    )


def _research_presentation(
    value: object,
    market_fields: list[ConfirmationField],
    execution_fields: list[ConfirmationField],
    backtest: dict[str, object],
) -> tuple[list[ConfirmationField], list[ConfirmationField]]:
    """Annotate origins without claiming to verify human input."""
    if value is None:
        return [], [_field("研究設定來源", "未保存研究需求來源")]
    metadata = ResearchMetadata.model_validate(value)
    labels = {
        "request": "本次輸入",
        "project_default": f"專案模擬預設（{metadata.policy_version}）",
        "selected_revision": "沿用所選策略版本",
    }
    paths = {
        "回測商品": "/backtest/symbol",
        "回測 K 線週期": "/backtest/timeframe",
        "回測開始": "/backtest/start_date",
        "回測結束": "/backtest/end_date",
        "初始資金": "/backtest/init_cash",
        "手續費率": "/backtest/fees",
        "滑價率": "/backtest/slippage",
        "年化計算的每年日數": "/backtest/periods_per_year",
    }
    for fields in (market_fields, execution_fields):
        for index, field in enumerate(fields):
            if field.label in paths:
                source = labels[metadata.field_sources[paths[field.label]]]
                description = f"{field.description} 設定來源：{source}。".strip()
                fields[index] = field.model_copy(update={"description": description})
    cost_names = {"init_cash": "本金", "fees": "手續費", "slippage": "滑價"}
    if backtest["mode"] == "volume":
        cost_names.pop("init_cash")
    execution_fields.append(
        _field(
            "本金與成本來源",
            "；".join(
                f"{label}：{labels[metadata.field_sources[f'/backtest/{name}']]}"
                for name, label in cost_names.items()
            ),
            "這些是模擬設定，不代表交易所即時費率。",
        )
    )
    sampling_reason = daily_observation_unavailable_reason(str(backtest["timeframe"]))
    execution_fields.append(
        _field(
            "年化統計取樣",
            "此計算方式不提供本金報酬年化"
            if backtest["mode"] == "volume"
            else "此 K 棒頻率不支援每日統計"
            if sampling_reason is not None
            else "有觀測值的 UTC 每日帳戶報酬；缺日不補零",
            "每年日數與 K 棒頻率分開；設定日數不代表已驗證歷史資料覆蓋。",
        )
    )
    strategy_fields = [
        _field(
            "研究構想",
            metadata.strategy_outline,
            "保存本次輸入或所選版本的構想，不代表後端已證明程式符合全部敘述。",
        )
    ]
    technical = [
        _field("研究政策版本", metadata.policy_version),
        _field("研究來源格式版本", metadata.schema_version),
        _field("年化政策判定", metadata.annualization_reason),
        _field(
            "研究構想來源",
            labels[metadata.field_sources["/research/strategy_outline"]],
            "來源標籤記錄解析路徑，不證明輸入來自真人或已獲批准。",
        ),
    ]
    if metadata.selected_revision is not None:
        identity = metadata.selected_revision
        technical.extend(
            (
                _field("沿用設定的策略識別碼", identity.strategy_id),
                _field("沿用設定的修訂版本", identity.revision_id),
                _field("沿用設定的 YAML 摘要", identity.config_hash, missing="未保存"),
            )
        )
    return strategy_fields, technical


def build_confirmation_document(
    manifest: ExecutionManifest,
    presentation: ConfirmationPresentation,
    preflight: PreflightReceipt,
    *,
    plan_id: str | None = None,
    comparison_candidate: bool = False,
) -> ConfirmationDocument:
    """Project effective settings and program evidence into the fixed template."""
    preflight = PreflightReceipt.model_validate(preflight.model_dump(mode="python"))
    validate_presentation(manifest, presentation)
    config = _object(manifest.config_for_execution(), "execution")
    config.setdefault("research", None)
    _known_fields(
        config,
        {
            "strategy",
            "backtest",
            "data",
            "validation",
            "parallel",
            "random_seed",
            "research",
        },
        "execution",
    )
    strategy = _object(config["strategy"], "strategy")
    _known_fields(
        strategy,
        {
            "id",
            "revision_id",
            "version",
            "class_name",
            "description",
            "display_name",
            "source_path",
            "source_hash",
            "parameters",
            "fit",
        },
        "strategy",
        complete=False,
    )
    backtest = _object(config["backtest"], "backtest")
    data = _object(config["data"], "data")
    market_fields, technical = _market_fields(backtest, data)
    execution_fields = _execution_fields(
        backtest, _object(config["parallel"], "parallel"), config["random_seed"]
    )
    assumptions = resolve_execution_assumptions(manifest)
    research_fields, research_identity = _research_presentation(
        config["research"], market_fields, execution_fields, backtest
    )
    technical.extend(research_identity)
    for key, label in (
        ("display_name", "策略名稱"),
        ("id", "策略識別碼"),
        ("revision_id", "策略修訂版本"),
        ("version", "策略標示版本"),
        ("class_name", "策略程式類別"),
        ("source_path", "策略程式位置"),
        ("source_hash", "策略程式內容摘要"),
    ):
        if key in strategy:
            technical.append(_field(label, strategy[key]))
    technical.append(
        _field(
            "策略來源",
            "專案內建" if manifest.strategy_execution.kind == "bundled" else "本次生成",
        )
    )
    parameters = [
        _parameter_field(pointer, value, presentation)
        for pointer, value in parameter_leaves(
            manifest.strategy_execution.constructor_kwargs
        ).items()
    ]
    preflight_section, limitations_section = _preflight_sections(
        manifest, preflight, comparison_candidate=comparison_candidate
    )
    return ConfirmationDocument(
        title=presentation.title,
        summary=presentation.summary,
        sections=(
            _section(
                "strategy",
                [
                    _field(
                        "執行方式",
                        {
                            "backtest": "單次歷史回測",
                            "walk_forward": "分段訓練與驗證",
                            "optimization": "參數搜尋與保留區間測試",
                        }[manifest.kind],
                    ),
                    *research_fields,
                ],
            ),
            _section("market", market_fields),
            _section(
                "parameters",
                parameters,
                ("名稱、說明與單位由模型提供；下列數值取自實際執行設定。",)
                if parameters
                else ("此策略沒有可調整的建構參數。",),
            ),
            _section(
                "execution",
                execution_fields,
                tuple(f"{item.label}：{item.value}" for item in assumptions.items),
            ).model_copy(
                update={
                    "summary_paragraphs": summarize_execution_assumptions(assumptions)
                }
            ),
            _validation_section(config["validation"]),
            _optimization_section(manifest, presentation),
            preflight_section,
            limitations_section,
        ),
        identity=ConfirmationIdentity(
            plan_id=plan_id,
            manifest_hash=manifest.manifest_hash,
            details=tuple(technical),
        ),
    )
