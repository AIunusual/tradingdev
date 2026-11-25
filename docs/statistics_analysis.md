# 單一 run/scope 統計分析契約

本契約描述已保存結果的**獨立統計分析**。它讀取一次完整性已核驗的
`PerformanceBundle` / `ObservationsBundle` snapshot，不使用分頁展示回傳重建統計輸入，
不重新執行策略、不下載行情、不改寫原始績效指標或舊報告，也不把新推論登錄為引擎指標、
排名或最佳化目標。

共用入口是 [`StatisticsService`](../src/tradingdev/app/statistics_service.py)。
MCP 的 `analyze_run_statistics` 與 CLI `tradingdev-statistics` 都只轉送
`run_id`、`scope` 與 `save_artifact`，不會各自計算。回傳契約見
[`app/contracts/statistics.py`](../src/tradingdev/app/contracts/statistics.py)。

## 輸入與來源綁定

- 來源必須由 [`TradeHistoryService`](../src/tradingdev/app/trade_history_service.py)
  的 `load_run` / `select_scope` 取得；aggregated `test_summary` 沒有原始 observations，
  回報 `fold_summary_unsupported` 並列出可選 scope。
- `source_hash` 是同一 validated snapshot 的 run 身分、bundle 定義、**所有 scalar scope
  與其 observations**（包含 DSR 使用的全部 `trial/<index>/train`）、所選 scope 參數與
  執行設定的正規化 JSON SHA-256。所有摘要、推論、artifact 與 DSR 試驗母體都綁定此值；
  讀取途中混版、損毀或雜湊不符時回報錯誤，不猜測重建。
- 本金取自 `scope.observations.init_cash`，mode 取自 `scope.performance.mode`；
  `frequency`、`periods_per_year`、`risk_free_rate` 與 `return_std_ddof` 取自該 scope
  的 `metadata.settings`，不使用目前政策預設。缺少 frequency 或 risk-free rate 時，
  需要這些值的摘要與推論回報 unavailable；缺少 annualization 時，描述摘要仍可計算，
  但 Sharpe 類結果回報 `missing_annualization`。
- signal 模式要求 observations 有正且有限的 `init_cash`；volume 模式的累積 P&L
  沒有本金基準，所有報酬型結果回報 `not_applicable`。trial scope 必須以相同 mode、
  本金、frequency、annualization 與 risk-free rate 才可進入 DSR 母體。

## 計算語意

- 整段淨損益從原始本金起算（volume 為 0）；「首筆觀察後變化」是另一個欄位，
  兩者不可互換。初始成本或第一筆觀察前的虧損不得被丟棄。
- 交易分為 `closed_trades` 與 `marked_trades`：保存為 `open` 的部位只用最後 mark，
  不補造出場成交、手續費或滑價。交易成本與週轉成本只來自保存的 `fee` 欄位。
- 日／月 UTC 分桶使用 `domain/statistics/periods.py` 的 observed-period 規則：
  缺日期不填零，period P&L 使用 `ddof=0`，Sharpe 使用 `ddof=1` 與執行時固定的
  annualization / risk-free rate。通用 `period_returns` 以首筆權益為基準，
  不能當作引擎的完整日報酬；服務改用
  [`daily_excess_returns`](../src/tradingdev/domain/statistics/performance.py)，
  首日 opening equity 是原始本金。
- 描述摘要與 breakeven 可獨立於推論計算。勝率、損益平衡與週轉成本只使用交易紀錄；
  volume 模式沒有 capital-relative turnover / cost drag。

## 推論與不可用語意

演算法版本固定為 `statistics-analysis-v1`。每次推論回傳自己的方法、seed、reps、
block size、有效觀察數、頻率、設定、source hash 與 scope。bootstrap 門檻、PSR 與
DSR 公式及限制由 [`domain/statistics/sharpe.py`](../src/tradingdev/domain/statistics/sharpe.py)
定義；服務只固定呼叫參數與回傳語意，不重寫統計演算法：

| 分析 | 方法 | 固定參數與限制 |
| --- | --- | --- |
| Sharpe interval | stationary bootstrap percentile | 95% 區間、1000 reps、seed `20261125`、auto block；少於 20 個日報酬回報 `insufficient_observations`。block size 自動估計不保證名目覆蓋率；平方根年化不移除序列相依或複利造成的 Hotelling 問題。 |
| PSR | normal iid PSR | benchmark 為每期 0；明確標示 iid 假設，序列相關會使機率失效；平方根年化不移除序列相依或複利影響。 |
| DSR | deflated Sharpe | 只對 `selected_train_scope` 計算；試驗 Sharpe 由同一 snapshot 的所有 trial observations 以同口徑重算，`n_trials` 使用 `optimization_selection.raw_attempted`。test scope 回報 `not_selected_training_scope`，不套用 train hurdle。缺少完整試驗、搜尋母體或同口徑 trial Sharpe 時回報 `search_evidence_incomplete`；不從排名頁數、test 分數或未知值臆造搜尋履歷。DSR 假設試驗獨立；相關試驗可能讓 hurdle 低估或高估，修正方向未知。 |

觀測日曆政策固定為 **exact declared bar cadence**：有 timestamp 時，相鄰 bar 必須
等於 `bar_duration_ns(frequency)`。時間缺口回報 `observation_gap_unsupported`，
不把缺值壓縮成相鄰的 bootstrap 樣本；unknown frequency 回報 `bar_cadence_unknown`。
各分析各自 unavailable，不讓一個缺值掩蓋其他可用摘要。

每個 **unavailable 分析 outcome**（`summary`、`breakeven`、`interval`、`psr`、`dsr`）
至少有 `reason`、`next_step`、`source`，可附 `detail`。摘要內的逐指標 unavailable
仍保留 domain 函式的 `dict[str, str]` 原因；MCP/CLI 回傳時一併呈現，不另造下一步與來源。
結果不得為 NaN、inf 或以零假裝成功；ContractModel 在 MCP 邊界再次拒絕非有限值。
回傳不包含原始 equity、trades 或分頁 observable。

## 衍生 artifact

`save_artifact=true` 時，同一分析額外出版一份 UTF-8 Markdown：內容包含演算法版本、
固定設定、限制、完整結果、unavailable 原因與 `source_hash`。檔名與 artifact ID 以
內容 SHA-256 的前 16 位定址；相同分析重複呼叫重用同一檔，內容不同則產生新 ID，
已註冊身分不符時失敗。檔案置於該 run 的 `statistics_analysis/` 下，只新增 artifact
metadata，不修改 `performance.json`、`observations.json` 或既有報告。
服務不自動刪除舊分析；保留期限由使用者依工作區政策處理。

## 驗證入口

- [`tests/app/test_statistics_analysis.py`](../tests/app/test_statistics_analysis.py)：
  本金基準、零交易、歷史未平倉、volume、缺年化、非零利率、短樣本、時間缺口、
  fold summary、損壞 artifact、DSR scope、artifact 重現。
- [`tests/mcp/test_history_contracts.py`](../tests/mcp/test_history_contracts.py)：
  MCP schema、共用 service、unavailable 契約與 artifact 輸出。
- [`tests/adapters/test_statistics_cli.py`](../tests/adapters/test_statistics_cli.py)：
  CLI 只轉送選擇到共用 service。
- [`tests/domain/statistics/test_statistics_performance.py`](../tests/domain/statistics/test_statistics_performance.py)：
  首日 opening equity 與獨立計算的 excess return 不變量。
- [`tests/e2e/test_llm_statistics.py`](../tests/e2e/test_llm_statistics.py)：
  自然語言要求從保存結果取得分析，且不得呼叫策略執行或行情下載工具（需 `live_llm` 環境）。
