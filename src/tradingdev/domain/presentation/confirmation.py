"""Complete, human-readable confirmation documents for captured executions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.strategies.execution import merge_strategy_parameters
from tradingdev.domain.validation.date_windows import inclusive_day_window

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest

type NonemptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
type SectionId = Literal[
    "strategy",
    "market",
    "parameters",
    "execution",
    "validation",
    "optimization",
    "preflight",
    "limitations",
]

SECTION_TITLES: dict[SectionId, str] = {
    "strategy": "這次要測試什麼",
    "market": "市場與資料",
    "parameters": "策略設定",
    "execution": "資金、交易成本與執行設定",
    "validation": "分段驗證方式",
    "optimization": "參數搜尋方式",
    "preflight": "確認前的試跑結果",
    "limitations": "限制與待確認事項",
}


class _ConfirmationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ParameterDescription(_ConfirmationModel):
    """Model-authored wording for a parameter, never its executable value."""

    label: NonemptyText
    description: NonemptyText
    unit: NonemptyText | None = None


class ConfirmationPresentation(_ConfirmationModel):
    """Plain-language metadata bound to the exact constructor in an execution plan."""

    title: NonemptyText
    summary: NonemptyText
    parameter_descriptions: dict[str, ParameterDescription] = Field(
        description=(
            "Human labels, meanings and units for every effective constructor "
            "parameter, including defaults; never executable values. Keys are JSON "
            "pointers into constructor arguments: /fast_period for generated "
            "strategies, /config/k_period or /fit_config/... for bundled strategies. "
            "Preparation returns required_parameter_paths for missing descriptions."
        )
    )


class ConfirmationField(_ConfirmationModel):
    """One authoritative displayed value with an optional explanation."""

    label: NonemptyText
    value_text: str
    description: str = ""
    unit: NonemptyText | None = None
    detail: bool = False
    explanation_source: Literal["backend", "model"] = "backend"


class ConfirmationSection(_ConfirmationModel):
    """A mandatory block that is shared by text and HTML output."""

    id: SectionId
    title: NonemptyText
    fields: tuple[ConfirmationField, ...] = ()
    paragraphs: tuple[str, ...] = ()


class ConfirmationIdentity(_ConfirmationModel):
    """Machine identity and storage details kept outside the main explanation."""

    plan_id: str | None = None
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    details: tuple[ConfirmationField, ...] = ()


class ConfirmationDocument(_ConfirmationModel):
    """The complete built-in template, without caller-selected omissions."""

    schema_version: Literal[1] = 1
    title: NonemptyText
    summary: NonemptyText
    sections: tuple[ConfirmationSection, ...]
    identity: ConfirmationIdentity

    @model_validator(mode="after")
    def complete_template(self) -> Self:
        if tuple(section.id for section in self.sections) != tuple(SECTION_TITLES):
            raise ValueError("Confirmation requires every built-in section in order")
        for section in self.sections:
            if section.title != SECTION_TITLES[section.id]:
                raise ValueError("Confirmation section titles are fixed by the backend")
            if not section.fields and not section.paragraphs:
                raise ValueError("Confirmation sections cannot be empty")
        return self


class ConfirmationPresentationError(ValueError):
    """Missing or ambiguous metadata can be repaired without editing a revision."""

    def __init__(self, message: str, *, required_parameter_paths: list[str]) -> None:
        super().__init__(message)
        self.required_parameter_paths = required_parameter_paths


def parameter_paths(manifest: ExecutionManifest) -> list[str]:
    """Return exact JSON pointers relative to the captured constructor arguments."""
    manifest.verify()
    paths = set(_parameter_leaves(manifest.strategy_execution.constructor_kwargs))
    for _group, _candidate, pointer, value in _effective_candidates(manifest):
        paths.update(_parameter_leaves(value, pointer))
    return sorted(paths)


def _effective_candidates(
    manifest: ExecutionManifest,
) -> list[tuple[int, int, str, object]]:
    """Resolve each search-group candidate exactly as execution merges its values."""
    if manifest.optimization is None:
        return []
    kwargs = manifest.strategy_execution.constructor_kwargs
    bundled = manifest.strategy_execution.kind == "bundled"
    captured = kwargs.get("config") if bundled else kwargs
    if not isinstance(captured, dict):
        raise ValueError("Strategy does not expose searchable parameters")
    result: list[tuple[int, int, str, object]] = []
    for group, (name, candidates) in enumerate(
        manifest.optimization.param_ranges.items(), 1
    ):
        token = name.replace("~", "~0").replace("/", "~1")
        pointer = ("/config" if bundled else "") + "/" + token
        for candidate_index, candidate in enumerate(candidates, 1):
            effective = merge_strategy_parameters(captured, {name: candidate})
            result.append((group, candidate_index, pointer, effective[name]))
    return result


def _parameter_leaves(value: object, pointer: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        result: dict[str, object] = {}
        for key in sorted(value):
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.update(_parameter_leaves(value[key], pointer + "/" + token))
        return result
    if isinstance(value, list) and value:
        result = {}
        for index, item in enumerate(value):
            result.update(_parameter_leaves(item, f"{pointer}/{index}"))
        return result
    return {pointer: value} if pointer else {}


def validate_presentation(
    manifest: ExecutionManifest, presentation: ConfirmationPresentation
) -> None:
    """Require unambiguous human labels for every captured parameter leaf."""
    paths = parameter_paths(manifest)
    actual = set(presentation.parameter_descriptions)
    required = set(paths)
    missing, extra = required - actual, actual - required
    if missing or extra:
        raise ConfirmationPresentationError(
            "Parameter descriptions must exactly cover the captured constructor; "
            f"missing={sorted(missing)}, extra={sorted(extra)}",
            required_parameter_paths=paths,
        )
    labels = [entry.label for entry in presentation.parameter_descriptions.values()]
    if len(labels) != len(set(labels)):
        raise ConfirmationPresentationError(
            "Parameter labels must distinguish every captured value",
            required_parameter_paths=paths,
        )


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
        _field("時間基準", "UTC（世界協調時間）", "未附時區的時間依 UTC 解讀。"),
        _field("本次行情來源", _source(market["source"])),
        _field("行情商品", market["symbol"], detail=True),
        _field("行情 K 線週期", _timeframe(market["timeframe"]), detail=True),
        _field("預設資料來源", _source(data["source"]), detail=True),
        _field(
            "市場類型",
            market_type,
            "此選項用於 Binance 歷史資料庫；其他資料來源依其商品定義。",
        ),
    ]
    technical = [
        _field("原始行情保存位置", data["raw_dir"]),
        _field("處理後行情保存位置", data["processed_dir"]),
    ]
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
            "策略若提供部位權重，會依每根 K 線調整投入金額。"
            if volume
            else "未指定固定金額時，依回測引擎預設資金配置。",
            unit="報價貨幣",
            missing="200（引擎預設）" if volume else "未指定固定金額",
        ),
        _field("手續費率", _percentage(backtest["fees"]), "每次進場、出場分別計入。"),
        _field(
            "滑價率",
            _percentage(backtest["slippage"]),
            "逐筆另計滑價成本。" if volume else "滑價反映於成交價格。",
        ),
        _field("停損幅度", _percentage(backtest["stop_loss"])),
        _field("停利幅度", _percentage(backtest["take_profit"])),
    ]
    for key, label, description in (
        (
            "signal_as_position",
            "沒有持倉訊號時平倉",
            "啟用時，零訊號會平倉；未啟用時等待反向訊號或停損停利。",
        ),
        (
            "re_entry_after_sl",
            "停損停利後允許重新進場",
            "依仍有效的持倉訊號，允許再次進場。",
        ),
        (
            "monthly_max_loss",
            "每月最大虧損門檻",
            "達門檻後暫停當月新倉；須有時間資料才能按月計算。",
        ),
    ):
        fields.append(
            _field(
                label,
                backtest[key],
                description if volume else "此計算方式不使用這項設定。",
                detail=True,
                unit="報價貨幣" if key == "monthly_max_loss" else None,
            )
        )
    fields.extend(
        [
            _field(
                "年化計算的每年日數",
                backtest["periods_per_year"],
                "未設定時，年化績效不可用。",
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
    if volume and backtest["position_size"] == 0:
        fields[2] = fields[2].model_copy(
            update={"value_text": "200（設定為零時使用引擎預設）"}
        )
    return fields


def _validation_section(value: object) -> ConfirmationSection:
    if value is None:
        return _section("validation", paragraphs=("本次不採用分段訓練與驗證。",))
    values = _object(value, "validation")
    _known_fields(
        values,
        {
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "n_splits",
            "train_ratio",
            "expanding",
            "target_metric",
            "label_horizon",
            "embargo_bars",
            "purged_split",
        },
        "validation",
    )
    dates = ("train_start", "train_end", "test_start", "test_end")
    purged = values["purged_split"] is True
    explicit = not purged and all(values[key] is not None for key in dates)
    horizon = values["label_horizon"]
    if not isinstance(horizon, int) or (purged and horizon < 1):
        raise ValueError("Purged validation requires a positive label horizon")
    if purged:
        mode = "淨化分段：固定起點、逐次擴大訓練區間"
        date_description = "淨化分段不使用指定日期；區間由資料筆數與分段組數決定。"
        ratio_description = (
            "淨化分段不使用此比例；訓練使用測試段之前的資料並扣除隔離區間。"
        )
        expanding_description = (
            f"淨化分段固定逐次擴大；此欄設定值（{_text(values['expanding'])}）不使用。"
        )
    else:
        mode = "使用指定日期的一組訓練／測試區間" if explicit else "依資料筆數自動分段"
        date_description = (
            "日期先轉為 UTC（無時區視為 UTC），再忽略時分秒；包含起訖 UTC 整日。"
            if explicit
            else "未指定日期，由自動分段決定。"
        )
        ratio_description = (
            "指定日期模式不使用此設定。" if explicit else "每組資料中用於訓練的比例。"
        )
        expanding_description = (
            "指定日期模式不使用此設定。" if explicit else "啟用時保留更早的訓練資料。"
        )
    date_values = {key: values[key] for key in dates}
    if explicit:
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
    fields = [
        _field("分段方式", mode),
        *[
            _field(
                label,
                date_values[key],
                f"{date_description} 原始設定：{values[key]}。"
                if explicit
                else date_description,
                missing="不使用" if purged else "由自動分段決定",
            )
            for key, label in zip(
                dates, ("訓練開始", "訓練結束", "測試開始", "測試結束"), strict=True
            )
        ],
        _field(
            "預定分段組數",
            1 if explicit else values["n_splits"],
            f"指定日期模式固定為一組，此設定不使用（設定值：{values['n_splits']}）。"
            if explicit
            else (
                "額外保留第一段作為訓練；後續每段依序測試，資料不足時會拒絕執行。"
                if purged
                else "實際可完成組數取決於資料量。"
            ),
        ),
        _field(
            "自動分段的訓練比例",
            _percentage(values["train_ratio"]),
            ratio_description,
        ),
        _field(
            "訓練區間逐次擴大",
            True if purged else values["expanding"],
            expanding_description,
        ),
        _field(
            "驗證紀錄重點指標",
            _metric(values["target_metric"]),
            "用於驗證紀錄；策略如何訓練另依其策略設定。",
        ),
        _field(
            "標籤前視根數",
            horizon if horizon > 0 else f"未啟用檢查（設定值：{horizon}）",
            "大於 0 時正式執行會先檢查標籤區間重疊；未啟用不代表標籤安全，"
            "也不代表策略沒有使用未來資料。",
            unit="根 K 線" if horizon > 0 else None,
            detail=True,
        ),
        _field(
            "淨化分段",
            values["purged_split"],
            "啟用時每組訓練區間會先移除與測試區間重疊的資料；需要前視根數大於 0。",
            detail=True,
        ),
        _field(
            "額外隔離根數",
            values["embargo_bars"],
            "淨化之外再多移除的 K 線；只有啟用淨化分段時才會生效。",
            unit="根 K 線",
            detail=True,
        ),
    ]
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
        },
        "optimization",
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
            "第一組估時計算時限",
            search.trial_timeout_seconds,
            "只限制正式搜尋中第一組參數的估時計算；其他候選與保留區間測試不受此時限限制。",
            unit="秒",
            detail=True,
        ),
    ]
    for group_index, candidate_index, pointer, candidate in _effective_candidates(
        manifest
    ):
        fields.extend(
            _parameter_field(
                leaf,
                value,
                presentation,
                prefix=f"搜尋組 {group_index}／候選 {candidate_index}：",
            )
            for leaf, value in _parameter_leaves(candidate, pointer).items()
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
    "signal_contract": "策略訊號契約檢查",
    "signals": "以本次設定產生訊號",
    "fit": "策略訓練流程",
    "engine": "實際回測引擎執行",
    "serialization": "結果資料序列化",
    "candidate_binding": "搜尋候選與執行設定綁定",
}
_WARNING_TEXT = {
    "no_trades_observed": (
        "試跑沒有觀察到交易；已完成執行流程，"
        "但尚未驗證此策略在樣本中的成交與交易成本路徑。"
    ),
    "partial_historical_sample": (
        "試跑只涵蓋受限的歷史資料樣本，完整期間仍可能遇到樣本未涵蓋的資料或執行情況。"
    ),
    "single_fold_only": "試跑未涵蓋所有驗證分段；其餘分段會在正式回測執行。",
    "single_candidate_only": "試跑未涵蓋全部參數組合；其餘候選會在正式參數搜尋執行。",
}


def _preflight_sections(
    manifest: ExecutionManifest, preflight: PreflightReceipt
) -> tuple[ConfirmationSection, ConfirmationSection]:
    manifest.verify(expected_hash=preflight.manifest_hash)
    checked = set(preflight.checked_paths)
    required = {
        "configuration",
        "signals",
        "engine",
        "serialization",
    }
    if manifest.strategy_execution.kind == "generated":
        required.add("signal_contract")
    if manifest.kind == "walk_forward":
        required.add("fit")
    if manifest.kind == "optimization":
        required.add("candidate_binding")
    if checked != required or len(checked) != len(preflight.checked_paths):
        raise ValueError("Preflight evidence differs from the required execution paths")
    roles = tuple(window.role for window in preflight.windows)
    expected_roles = ("full",) if manifest.kind == "backtest" else ("train", "test")
    if roles != expected_roles:
        raise ValueError("Preflight windows do not cover the selected execution path")
    config = manifest.config_copy()
    if preflight.data_source != config["data"]["requirements"]["market"]["source"]:
        raise ValueError("Preflight data source differs from the manifest")
    if manifest.optimization is not None:
        if preflight.total_candidates != manifest.optimization.total_combinations:
            raise ValueError("Preflight candidate coverage differs from the manifest")
    elif preflight.total_candidates is not None:
        raise ValueError("Non-optimization preflight cannot claim candidate coverage")
    if manifest.kind == "walk_forward":
        validation = config["validation"]
        explicit = not validation["purged_split"] and all(
            validation[key] is not None
            for key in ("train_start", "train_end", "test_start", "test_end")
        )
        expected_folds = 1 if explicit else validation["n_splits"]
        if preflight.total_fold_count != expected_folds:
            raise ValueError("Preflight fold coverage differs from the manifest")
    if manifest.kind != "walk_forward" and preflight.total_fold_count is not None:
        raise ValueError("Non-validation preflight cannot claim fold coverage")
    fields = [
        _field("試跑狀態", "已完成必要檢查，可供使用者確認"),
        _field("試跑耗時", preflight.elapsed_seconds, unit="秒"),
        _field(
            "試跑資料管道",
            "本次設定的行情資料管道",
            "此項不代表已向外部資料供應商驗證完整行情。",
        ),
        _field("試跑資料來源", _source(preflight.data_source)),
        _field("試跑資料上限", preflight.sample_bars_requested, unit="根 K 線"),
        _field("實際使用資料", preflight.sample_bars_used, unit="根 K 線"),
        _field(
            "宣告的最低歷史資料需求",
            preflight.minimum_history_bars,
            "每個試跑區間均達此需求；此需求由策略準備階段宣告。",
            unit="根 K 線",
        ),
        _field("觀察到的交易", preflight.trade_count, unit="筆"),
        _field(
            "保存的執行紀錄",
            preflight.execution_record_count,
            missing="此引擎未提供",
            unit="筆",
        ),
        _field("樣本觸發交易路徑", preflight.trading_path_exercised),
    ]
    fields.extend(
        _field(_CHECK_NAMES[check], "通過") for check in preflight.checked_paths
    )
    for index, window in enumerate(preflight.windows, 1):
        role = {"full": "策略樣本", "train": "訓練樣本", "test": "測試樣本"}[
            window.role
        ]
        fields.extend(
            [
                _field(f"區間 {index}（{role}）開始", window.start),
                _field(f"區間 {index}（{role}）結束", window.end),
                _field(f"區間 {index}（{role}）筆數", window.rows, unit="根 K 線"),
            ]
        )
    warnings = set(preflight.warnings)
    warnings.add("partial_historical_sample")
    if preflight.trade_count == 0:
        warnings.add("no_trades_observed")
    for tested, total, label, warning in (
        (
            preflight.tested_fold_count,
            preflight.total_fold_count,
            "已試跑的驗證分段",
            "single_fold_only",
        ),
        (
            preflight.tested_candidates,
            preflight.total_candidates,
            "已試跑的參數組合",
            "single_candidate_only",
        ),
    ):
        if tested is not None and total is not None:
            fields.append(_field(label, f"{tested} / {total}"))
            if tested < total:
                warnings.add(warning)
            else:
                warnings.discard(warning)
    if warnings - _WARNING_TEXT.keys():
        raise ValueError("Preflight warning lacks a built-in confirmation description")
    limitations = tuple(
        _WARNING_TEXT[warning] for warning in _WARNING_TEXT if warning in warnings
    ) + (
        "試跑通過不代表策略會獲利，也不能保證完整回測不會出錯。",
        "策略名稱、摘要與參數解釋由模型提供；參數值、執行設定與試跑結果由後端固定。",
        "確認後只執行這個版本。若程式或有效設定需要修改，必須重新試跑並再次確認；暫時性錯誤可在相同設定下重試。",
        "請選擇開始正式回測、要求調整設定，或取消。確認前不會啟動完整回測。",
    )
    return _section("preflight", fields), _section(
        "limitations", paragraphs=limitations
    )


def build_confirmation_document(
    manifest: ExecutionManifest,
    presentation: ConfirmationPresentation,
    preflight: PreflightReceipt,
    *,
    plan_id: str | None = None,
) -> ConfirmationDocument:
    """Project every effective setting and verified sample into the fixed template."""
    preflight = PreflightReceipt.model_validate(preflight.model_dump(mode="python"))
    validate_presentation(manifest, presentation)
    config = _object(manifest.config_for_execution(), "execution")
    _known_fields(
        config,
        {"strategy", "backtest", "data", "validation", "parallel", "random_seed"},
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
    for key, label in (
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
        for pointer, value in _parameter_leaves(
            manifest.strategy_execution.constructor_kwargs
        ).items()
    ]
    preflight_section, limitations_section = _preflight_sections(manifest, preflight)
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
                    )
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
            _section("execution", execution_fields),
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
