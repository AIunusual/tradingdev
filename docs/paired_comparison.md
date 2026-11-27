# 保存序列的配對比較

`compare_saved_series` 與 `compare_signal_series` 共用 `PairedComparisonService`。
比較只讀取兩側已保存、完整性核驗通過的 snapshot，然後把共同有效的觀察值交給
`domain.statistics.comparison.difference_interval_outcome`。比較請求不會回測、
訓練、下載行情、重新 predict，也不會重建原始序列或改寫舊報告。

## 來源與資格

| `kind` | 需要的欄位 | 序列與成本 |
| ---- | ---- | ---- |
| `run` | `run_id`、`scope`（省略用預設） | 從保存的 run/scope observations 重建每期報酬；signal 模式用帳戶權益，成本已在成交中；volume 模式沒有本金相對報酬，回報 `not_applicable` |
| `benchmark`，`benchmark_mode="product"` | `run_id`、`scope`（`benchmark_mode` 必須明示 `product`） | 只讀保存的 `BenchmarkScope` 淨值／報酬與其成本欄位；不逆推原始 closes |
| `benchmark`，`benchmark_mode="zero_cost"` | `run_id`、`price_artifact_id`（`scope` 被忽略） | 服務先確認該 run 的固定 snapshot 存在，再讀註冊且 SHA-256 核對通過的 `raw_price_series` artifact，用 `buy_and_hold_returns` 產生零成本報酬；比較整份已保存的原始價格序列，不依 run 的期間裁切。若無此 artifact 則回報 `zero_cost_prices_unavailable`，不下載也不新增保存流程 |
| `signal` | `artifact_id`、`column` | 已保存的 pointwise evaluation series；`pearson_ic`／`spearman_ic` 是資訊係數，`realized_return` 是無成本訊號模擬，二者不得互換或與成交報酬混檢 |

兩側必須使用**相容定義、同頻率與相容設定**。成本相對的成交報酬可與已保存的
產品基準相比，也可與核驗過的原始價格零成本報酬相比，但兩者的 `cost_basis`
不同，不能把零成本訊號模擬冒充成交績效；IC 只能與同類 IC 相比。run 之間比較
固定本金、年化及無風險利率須一致，這個檢查不套用含有 `zero_cost_price_return`
的配對；signal 比較須有相同的 evaluation settings。entry ID、model ID 或
dataset ID 不同本身不證明 holdout，也不證明未調參。

回覆的 `claim_scope` 是服務依保存證據給出的資格標記：run scope 的 split 為
`test` 且有保存的研究證據時為 `verified_out_of_sample`，其他情況（含所有
signal 序列）為 `in_sample_or_unknown`。這只是保存關係的呈現，不保證沒有
先前選擇或多重比較偏誤。結果保留 training/evaluation 關係、producer source、
signal fingerprint 與可用的 fold 範圍，由讀者判斷資格。

## 對齊與缺口政策

推論模式只接受**精確 UTC 日頻與小時頻**的共同時間戳；其他相同頻率的序列仍可做
描述性比較，但 cadence 標為 `unsupported`，不能進入推論。訊號序列的頻率由保存
時間戳推斷，不規則或不足兩個時間點標為 `unsupported`，不能冒充日頻。run 與產品
基準則使用其保存設定宣告的頻率。交集之前不做 dropna、補零或
名稱對齊；warm-up 的 `NaN` 留在原座標，先依每側自己的時間順序判斷是前段
warm-up 還是中途缺值，再以雙方同位置有效的觀察配對。回報原始長度、共同
日期數、共同有效樣本數、各類排除原因與被排除座標，以及共同有效遮罩。

共同期間缺少預期日曆點時，日曆視為不完整：描述性比較仍可計算並標明缺口，
`mode="inferential"` 回報 `cadence_gap_unsupported`。第一版沒有新的缺口
bootstrap；不完整日曆不壓縮成連續序列。完全沒有共同日期回報
`no_common_timestamps`，共同有效觀察不足
`MIN_BOOTSTRAP_OBSERVATIONS` 回報 `insufficient_observations`；兩點不會補成三對。

## 統計量與輸出

比較統計量固定為 `mean`，差額方向固定為 `a_minus_b`。每次回報
`estimate`、`estimate_a`、`estimate_b`、`lower`、`upper`、`p_value`、
`confidence`、`reps`、`valid_reps`、`seed`、`block_size`、`correlation`、
共同期間、排除原因、兩側來源身分與方法限制。`block_size=None` 時交由既有
[`comparison.difference_interval_outcome`](../src/tradingdev/domain/statistics/comparison.py)
的 difference jackknife 自動估計；指定值必須是正整數。重抽樣次數限制在
`100..10000`。

`mode="descriptive"` 是預設：p 值標為 `descriptive_only`，不承載顯著性結論。
`mode="inferential"` 另要求完整日曆與可識別的 producer；p 值是 percentile
尾端機率，只對選定的這一對有意義。stationary bootstrap 不保證任意非平穩報酬
過程下的有效推論，區間也不保證名目涵蓋率；完整日曆與可識別來源只是必要檢查，
不是充分條件。兩者都不做多重比較校正，第一版只處理單一 pair，不自動宣稱已修正
先前搜尋造成的偏誤。

無法比較（來源不存在、損壞、無序列、定義不相容、資料不足）回傳
`status="unavailable"`，附 `reason`、`next_step`、`source`、`detail`；
請求本身無效（例如 signal 來源缺少 `column`）回傳
`success=false` 與 `code=invalid_comparison_request`；若selector 在 MCP
輸入 schema 驗證時就不完整（例如 signal 缺少 `column`），則是工具輸入錯誤
（`ToolError`），不會進入 service。明細回傳的排除樣本與
缺日樣本各有上限；**回覆本身只帶摘要**。完整有效遮罩、全部排除座標與缺日座標
只寫入出版的 artifact，不在 MCP 回覆中展開。

## 可離線的成果

比較結果可用且 `save_artifact=true` 時（工具預設為 `false`，需明確要求），才會從同一次分析出版兩份內容定址、
不可變的成果；不可用結果的兩個 artifact 欄位都是 `null`，不會產生檔案。
`paired_comparisons/<前16碼>.md`（Markdown）與
`paired_comparisons/<前16碼>.html`（自含 HTML，樣式內嵌、無外部資源）。
兩份各自以本身的內容計算 digest、檔名與 artifact ID，分開註冊到 artifact
registry，分別回傳在 `artifact` 與 `html_artifact`。
內容包含方法版本、模式與資格、限制、分析 JSON、兩側來源，以及完整對齊
Audit（有效遮罩、全部排除座標與缺日）；HTML 另有可離線閱讀的對齊證據章節。重複相同比較會重用同一份內容；不同設定產生不同 ID，舊檔不被改寫。
使用 `get_artifact(include_content=true)` 搭配回傳的 SHA-256 分頁閱讀，或用
`list_artifacts` 查找已註冊的成果；HTML 可直接在離線瀏覽器開啟。出版前會先對
兩份目標做路徑與註冊衝突檢查；若檢查通過但寫入途中發生 OS 錯誤，仍可能只留下
先完成的 Markdown，這時可從 `list_artifacts` 找到它。出版失敗是操作錯誤，
回傳 `success=false`、`code=artifact_publication_failed`，不冒充資料不可用。

## 驗證入口

- [純對齊測試](../tests/domain/statistics/test_statistics_alignment.py)
- [服務測試](../tests/app/test_paired_comparison.py)
- [MCP 契約](../tests/mcp/test_paired_comparison_contracts.py)
- [自然需求測試](../tests/e2e/test_llm_paired_comparison.py)（`live_llm`）
