"""Backend-owned explanations of the exact simulator policy for an execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingdev.domain.backtest.execution_policy import (
    DEFAULT_VOLUME_POSITION_SIZE,
    EXECUTION_POLICY_VERSION,
)
from tradingdev.domain.backtest.schemas import BacktestConfig

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest

type AssumptionKey = Literal[
    "timing",
    "signals",
    "reversal",
    "sizing",
    "weights",
    "accumulation",
    "leverage",
    "stops",
    "stop_reentry",
    "costs",
    "annualization",
    "monthly_loss",
    "end_of_data",
    "limitations",
]

ASSUMPTION_KEYS: tuple[AssumptionKey, ...] = (
    "timing",
    "signals",
    "reversal",
    "sizing",
    "weights",
    "accumulation",
    "leverage",
    "stops",
    "stop_reentry",
    "costs",
    "annualization",
    "monthly_loss",
    "end_of_data",
    "limitations",
)


class ExecutionAssumption(BaseModel):
    """One fixed backend explanation, never model-authored trading intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: AssumptionKey
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    summary: str | None = Field(default=None, min_length=1)


class ExecutionAssumptions(BaseModel):
    """Persisted policy projection; historical reads do not resolve it again."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_version: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["signal", "volume"]
    items: tuple[ExecutionAssumption, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: object) -> object:
        """Do not interpret boolean or floating-point versions as this schema."""
        if type(value) is not int:
            raise ValueError("Execution assumptions schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def complete_policy(self) -> Self:
        """Require all disclosed rules, in their stable presentation order."""
        if tuple(item.key for item in self.items) != ASSUMPTION_KEYS:
            raise ValueError("Execution assumptions require every policy item")
        return self


def _rate(value: float | None) -> str:
    return "未設定" if value is None else f"{value * 100:.12g}%"


def resolve_execution_assumptions(manifest: ExecutionManifest) -> ExecutionAssumptions:
    """Resolve a verified manifest without market data, strategy execution, or vbt.

    This describes the current execution policy, not completed preflight evidence.
    Call when preparing a new execution or checking its saved policy at submission;
    use the persisted value directly when reading historical results.
    """
    manifest.verify()
    raw = manifest.config.get("backtest")
    if not isinstance(raw, dict) or set(raw) != set(BacktestConfig.model_fields):
        raise ValueError("Execution assumptions require resolved backtest settings")
    config = BacktestConfig.model_validate(raw)
    volume = config.mode == "volume"
    if config.position_size == 0:
        sizing = "單次投入為零：不開新倉，也不產生零數量交易。"
    elif volume:
        amount = (
            DEFAULT_VOLUME_POSITION_SIZE
            if config.position_size is None
            else config.position_size
        )
        sizing = f"每次新倉基礎名目金額 {amount:g} 報價貨幣，不使用初始資金。"
        if config.position_size is None:
            sizing += "未指定金額，使用執行政策預設。"
    elif config.position_size is None:
        sizing = (
            "未指定固定金額，以當時可用資金及成交成本決定最大可成交數量；"
            "不是固定數量或固定初始本金比例。"
        )
    else:
        sizing = (
            f"每次新倉目標名目金額 {config.position_size:g} 報價貨幣，"
            "以訊號成交價換算數量；滑價會改變實際名目金額，資金不足可部分成交。"
        )

    stop_settings = (
        f"停損 {_rate(config.stop_loss)}；停利 {_rate(config.take_profit)}。"
    )
    stops = stop_settings
    stops += (
        "以進場價為基準，既有倉先檢查當棒 high/low，缺欄才用 close；"
        "兩者同棒觸發時停損優先，按門檻價出場，零值停用。"
        "新倉不在進場當棒檢查停損停利。"
        if volume
        else "以進場棒 close 為基準，後續只用 open/close 的高低範圍判斷，"
        "不使用行情 high/low；停損優先於停利與一般訊號。"
        "跳空跨過門檻時以 open 出場，否則以門檻價出場；零值是零距離停損停利。"
        "新倉不在進場當棒檢查停損停利。"
    )
    if volume:
        reentry = (
            "停損停利後，當棒延後訊號非零且月虧損限制未觸發時，"
            "依該訊號於當棒 close 重新進場。"
            if config.re_entry_after_sl
            else "停損停利後本棒不再進場；保持空手，直到後續延後訊號不同於"
            "停損棒的訊號（改為零也解除等待）。跨月不自行解除此等待。"
        )
    else:
        reentry = (
            "停損停利平倉後不在同棒重新進場，須有後續持倉訊號狀態變化才可能進場；"
            "重新進場開關不影響此模式。"
        )
    active_stop = any(
        value is not None and (not volume or value != 0)
        for value in (config.stop_loss, config.take_profit)
    )
    cost_settings = f"手續費率 {_rate(config.fees)}；滑價率 {_rate(config.slippage)}。"
    cost_method = (
        "進場、出場各按名目金額扣除手續費及獨立滑價成本。"
        if volume
        else "各成交收取手續費，一般訊號的滑價反映於成交價。"
    )
    stop_costs = "" if volume else "停損停利採 stoplimit 模擬，出場滑價為零。"
    costs = cost_settings + cost_method + stop_costs
    annualization_basis = (
        "此損益模式沒有初始本金，帳戶報酬及其年化指標不適用。"
        if volume
        else "使用觀測到的 UTC 日報酬，缺日不補值；須有有效時間戳及不粗於日線的資料。"
    )
    annualization_factor = (
        "未設定每年日數，年化績效不可用。"
        if config.periods_per_year is None
        else f"保存的每年日數為 {config.periods_per_year:g}。"
    )
    annualization_rates = (
        f"年化無風險利率 {_rate(config.risk_free_rate)}；"
        f"年化最低目標報酬 {_rate(config.required_return)}。"
    )
    annualization = annualization_basis + annualization_factor + annualization_rates
    leverage_limits = (
        "不模擬完整保證金、清算、借券或資金費率規則，也沒有固定槓桿倍數設定。"
    )
    rows: tuple[tuple[AssumptionKey, str, str], ...] = (
        (
            "timing",
            "訊號與成交時點",
            "訊號延後一根 K 棒，一般訊號進出場使用下一棒 open；"
            "缺少 open 欄位才使用 close。尚未平倉部位以當棒 close 評價。",
        ),
        (
            "signals",
            "多空與零訊號",
            "1 表示做多，-1 表示做空（不是只賣出平多倉）。"
            + (
                "0 保留既有倉位並禁止新倉。"
                if volume and not config.signal_as_position
                else "0 表示平倉。"
            ),
        ),
        (
            "reversal",
            "反向訊號",
            "相反的非零訊號平掉舊方向並嘗試建立新方向；停損停利先於訊號處理。",
        ),
        ("sizing", "投入金額", sizing),
        (
            "weights",
            "部位權重",
            "若策略提供部位權重，延後一棒並於缺值填 1，須有限且大於零；"
            "只在新倉時乘上基礎金額，持有期間不重平衡。"
            if volume
            else "此模式不使用部位權重。",
        ),
        ("accumulation", "同方向持倉", "持續相同方向的訊號不加碼。"),
        (
            "leverage",
            "資金與槓桿限制",
            leverage_limits
            + (
                "此模式不限制名目金額相對於本金的比例。"
                if volume
                else "多倉受可用現金限制；固定金額空倉不鎖定現金，"
                "不能據此聲稱沒有槓桿或保證百分之百成交。"
            ),
        ),
        ("stops", "停損停利成交", stops),
        ("stop_reentry", "停損停利後進場", reentry),
        ("costs", "成交成本", costs),
        ("annualization", "年化解讀", annualization),
        (
            "monthly_loss",
            "每月虧損限制",
            f"門檻 {config.monthly_max_loss:g} 報價貨幣；須有時間戳，"
            "按 UTC 月份計算相對上個有行情月份最後收盤權益的變動（首月基準為零），"
            "計入本月交易成本及未實現損益變動，不重計前月損益。"
            "虧損達門檻時平倉並暫停本月，下個有行情月份解除月暫停。"
            if volume
            else "此模式不使用每月虧損限制。",
        ),
        (
            "end_of_data",
            "資料結束時",
            "剩餘持倉以最後一棒 close 強制平倉並計入出場成本。"
            if volume
            else "剩餘持倉保留為未平倉，以最後 close 評價，不虛構出場成交。",
        ),
        (
            "limitations",
            "規則與檢查的限制",
            "以上是模擬引擎規則，不是策略 Python 或 ML 決策的形式證明。"
            "本快照不證明合成檢查已通過、策略符合構想、沒有未來資料或歷史績效有效；"
            "確認前的合成檢查證據須另行查看，正式資料與績效仍須於授權後執行。",
        ),
    )
    summaries: dict[AssumptionKey, str | None] = {
        "leverage": leverage_limits,
        "stops": stops
        if active_stop
        else stop_settings
        + (
            "零值停用。"
            if volume and 0 in (config.stop_loss, config.take_profit)
            else ""
        ),
        "stop_reentry": reentry if active_stop else None,
        "costs": cost_settings + cost_method + (stop_costs if active_stop else ""),
        "annualization": annualization_basis
        + ("" if volume else annualization_factor)
        + (
            annualization_rates
            if config.risk_free_rate or config.required_return
            else ""
        ),
        "limitations": None,
    }
    if not volume:
        summaries["weights"] = None
        summaries["monthly_loss"] = None
    return ExecutionAssumptions(
        policy_version=EXECUTION_POLICY_VERSION,
        manifest_hash=manifest.manifest_hash,
        mode=config.mode,
        items=tuple(
            ExecutionAssumption(
                key=key, label=label, value=value, summary=summaries.get(key, value)
            )
            for key, label, value in rows
        ),
    )


def summarize_execution_assumptions(
    assumptions: ExecutionAssumptions,
) -> tuple[str, ...]:
    """Render the saved, mode-aware short projection without reinterpreting rules."""
    return tuple(
        f"{item.label}：{item.summary}"
        for item in assumptions.items
        if item.summary is not None
    )
