"""Provider-neutral requirements and independent signal expectations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    parameters: dict[str, int]
    overrides: dict[str, int]
    requirement: str
    repair: bool = False
    legacy: bool = False
    experiment: bool = False
    history: bool = False

    @property
    def strategy_id(self) -> str:
        return f"llm_{self.name}"

    @property
    def prompt(self) -> str:
        if self.legacy:
            preparation = (
                "後端已有這個策略的舊格式檔案。先 list_strategies 找到它，"
                "再 get_strategy 讀取現有原始碼與 YAML，依工具提供的恢復指引處理。"
                "必須先完成這兩次讀取才可 save_strategy，不可跳過舊內容直接建立。"
                "即使舊 metadata 標示 runnable，也必須重新儲存、驗證及 dry-run。"
            )
        elif self.repair:
            preparation = (
                "後端已有這個策略的錯誤草稿。先 get_strategy 並 validate_strategy，"
                "必須先取得失敗診斷，再修改儲存，不能直接覆寫而跳過診斷。"
            )
        else:
            preparation = "請先查詢策略清單。"
        experiment = (
            f"本次為參數實驗：prepare_backtest 必須額外傳 parameters={self.overrides}。"
            "沿用剛通過 dry-run 的 revision_id；不得為這次參數修改另存 revision，"
            "原 YAML 基礎參數保持不變。"
            if self.experiment
            else "本次不是參數實驗：prepare_backtest 必須沿用剛通過驗證與 dry-run 的 "
            "YAML 基礎設定；parameters 與 backtest_overrides 請省略或傳 null，"
            "不要重複傳入基礎值。"
        )
        history = (
            "接著以 find_runs 依策略與本次參數查回同一 run_id/full scope。"
            "呼叫 get_run_trades(run_id,scope=full,limit=1)，若有 next_offset，"
            "再用它讀下一頁。呼叫 get_run_equity 查前兩根觀測。"
            "呼叫 get_run_executions(limit=1) 讀第一筆原生成交，"
            "以及 get_run_account_history(limit=2) 讀前兩根帳戶狀態。"
            "最後先呼叫 get_report_sections，然後 generate_report，"
            "run_ids=[這個run_id]、sections=[overview,metrics,ledger,trades]"
            "（字串陣列），"
            "commentary=[{title:研究評語,text:本次結果僅為歷史模擬。}]。"
            "由後端產生指定章節及你的評語，"
            "確認success及回傳path後才結束。不重跑策略、不另寫報告程式。"
            if self.history
            else "不需要額外取得 artifact 內容或撰寫回測報告。"
        )
        completion = (
            "最後回覆須提供 run_id、[閱讀完整回測報告](<REPORT_PATH>)，"
            "REPORT_PATH 必須替換為 generate_report 回傳的實際完整 path，"
            "不可放在 code block；簡短說明可從總覽依參數選擇結果，"
            "查看委託與帳戶核對表及配對交易明細。"
            "報告連結必須出現在最終回覆，不可只出現在中途說明或工具輸出。"
            if self.history
            else "確認結果後就結束，只回答「完成」與 run_id。"
        )
        return f"""請透過 TradingDev MCP 開發策略 {self.strategy_id}。
{preparation}
讀取 get_strategy_contract，依照契約完成 Python 與 YAML。{self.requirement}
所有參數放在 strategy.parameters 且可覆寫：{self.parameters}。
YAML backtest 設定 BTC/USDT、1h、2024-01-01 至 2024-01-08、init_cash=10000、
mode=signal、fees=0、slippage=0、periods_per_year=365。
random_seed=42 放在 YAML 頂層，與 strategy、backtest 同層。
data.requirements.features 為空。
完成 save_strategy、validate_strategy、dry_run_strategy；若有錯誤請讀取診斷修正，
每次 save 取得的新 revision_id 必須傳給 validate、dry-run 與 prepare_backtest，
確認各工具回覆的 revision_id 一致，直到 runnable。不 promote。
啟動前核對 validate／dry-run 回覆的 effective_config 與上述所有設定一致；
若不一致，修改 YAML 後重新 save、validate、dry-run，不可直接接受預設值。
接著透過 MCP prepare_backtest 準備這個策略的執行計畫，
symbol=BTC/USDT、timeframe=1h、start_date=2024-01-01、end_date=2024-01-08。
minimum_history_bars=20、sample_bars=128；presentation 必須提供中文 title、summary，
及 parameter_descriptions；以每個建構子參數的 JSON pointer 為 key
（例如 /fast_period、/slow_period 或 /lookback），各自提供 label、description、unit。
說明每個參數的白話意義與單位，不要把程式識別字當作唯一說明。
{experiment}
本次行情已預先放入後端快取，不下載行情、不使用外部資料。
取得 success=true、status=ready 的 plan_id 後，將 confirmation_summary 逐行原文引用，
建議每行加 > 作為 Markdown 引用；只可調整排版，
不可改寫、刪除欄名、值、單位或訓練／測試角色。
摘要已由後端完成，不要再摘要。引用外另行用 [閱讀確認單](<FULL_HTML_PATH>) 提供連結，
把 FULL_HTML_PATH 替換為工具回覆 html_path 的實際完整值；角括號可保留空白路徑。
連結放在一般 Markdown 文字行，不放 code block，不加 file:// 或自行創造其他 URI 前綴。
然後結束本回合等待我回覆。一般解釋及回答問題可用自己的表達方式。
現在尚未同意正式執行，不可在本回合呼叫 submit_execution_decision。
只有同一段對話的後續回合收到我的明確同意，才用 plan_id、decision=approve 提交。
拒絕用 decision=decline；要求修改用 decision=revise，再重新準備並呈現新計畫，
結束回合重新等待確認。模糊回覆須澄清，不可當作同意；不可重用已取消的計畫。
收到成功且非空 job_id 後才查詢正式工作。
持續 get_job_status 查詢直到 done，再以回傳的 run_id 呼叫 get_run，
以及 list_artifacts。一般回覆只有摘要；接著呼叫 get_metric_catalog(mode=signal)，
再以 get_run_metrics 查同一 run_id 的 daily_pnl_mean、total_volume、n_days，
確認這些未列於摘要的指標仍可查詢。{history}
{completion}不得只啟動工作就結束。
只能透過 MCP 工具撰寫與執行，不使用 shell 或直接編輯檔案。
直接呼叫工具，不重貼程式碼或完整確認內容；程式、YAML 與 request_summary 保持精簡，
但不得省略需求或驗證步驟。
"""


_SMA_REQUIREMENT = (
    "用收盤價 5 根與 20 根簡單均線判斷：短均線>長均線時每根 signal=1，"
    "小於時每根=-1，相等或資料不足20根時=0。不得修改輸入資料或使用未來資料。"
)
SCENARIOS = {
    "history": Scenario(
        "history",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        history=True,
    ),
    "experiment": Scenario(
        "experiment",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        experiment=True,
    ),
    "sma": Scenario(
        "sma",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
    ),
    "momentum": Scenario(
        "momentum",
        {"lookback": 6},
        {"lookback": 3},
        "比較每根 close 與 lookback 根前的 close；較高每根 signal=1，較低=-1，"
        "相等或沒有足夠歷史=0。不得修改輸入資料或使用未來資料。",
    ),
    "repair": Scenario(
        "repair",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        repair=True,
    ),
    "legacy": Scenario(
        "legacy",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        legacy=True,
    ),
}


def market_frame() -> pd.DataFrame:
    close = pd.Series(
        [
            100.0 + value
            for value in ([*range(1, 41), *range(40, 0, -1), *([1] * 16)] * 2)
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(close), freq="h", tz="UTC"
            ),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )


def expected_signals(
    frame: pd.DataFrame, scenario: Scenario, parameters: dict[str, int]
) -> pd.Series:
    close = frame["close"]
    if scenario.name == "momentum":
        left, right = close, close.shift(parameters["lookback"])
    else:
        left = close.rolling(parameters["fast_period"]).mean()
        right = close.rolling(parameters["slow_period"]).mean()
    expected = pd.Series(0, index=frame.index, name="signal")
    expected.loc[left > right] = 1
    expected.loc[left < right] = -1
    return expected
