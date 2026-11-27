# 保存序列的配對比較

`compare_saved_series` 與 `compare_signal_series` 共用 `PairedComparisonService`。
比較只讀取兩側已保存、完整性核驗通過的 snapshot，然後把共同有效的觀察值交給
`domain.statistics.comparison.difference_interval_outcome`。比較請求不會回測、
訓練、下載行情、重新 predict，也不會重建原始序列或改寫舊報告。

## 來源與資格

| `kind` | 需要的欄位 | 序列與成本 |
| ---- | ---- | ---- |
| `run` | `run_id`、`scope`（省略用預設） | 從保存的 run/scope observations 重建每期報酬；signal 模式用帳戶權益，成本已在成交中；volume 模式沒有本金相對報酬，回報 `not_applicable` |
| `benchmark`，`benchmark_mode="product"` | `run_id`、`scope` | 只讀保存的 `BenchmarkScope` 淨值／報酬與其成本欄位；不逆推原始 closes |
| `benchmark`，`benchmark_mode="zero_cost"` | `price_artifact_id` | 只有註冊且 SHA-256 核對通過的 `raw_price_series` artifact 才能用 `buy_and_hold_returns` 產生零成本報酬；否則回報 `zero_cost_prices_unavailable`，不下載也不新增保存流程 |
| `signal` | `artifact_id`、`column` | 已保存的 pointwise evaluation series；`pearson_ic`／`spearman_ic` 是資訊係數，`realized_return` 是無成本訊號模擬，二者不得互換或與成交報酬混檢 |

兩側必須同定義、同頻率與相容設定。run 比較固定本金、年化及無風險利率須一致；
signal 比較須有相同的 evaluation settings。entry ID、model ID 或 dataset ID 不同
本身不證明 holdout，也不證明未調參；結果保留 training/evaluation 關係、
producer source、signal fingerprint 與 fold 範圍，由讀者判斷資格。

## 對齊與缺口政策

第一版支援**精確 UTC 日頻與小時頻**的共同時間戳。交集之前不做 dropna、補零或
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
共同期間、排除原因、兩側來源身分與方法限制。`block_size=None` 時由既有
difference jackknife 自動估計；指定值必須是正整數。重抽樣次數限制在
`100..10000`。

`mode="descriptive"` 是預設：p 值標為 `descriptive_only`，不承載顯著性結論。
`mode="inferential"` 另要求完整日曆與可識別的 producer；p 值是 percentile
尾端機率，只對選定的這一對有意義。兩者都不做多重比較校正，第一版只處理
單一 pair，不自動宣稱已修正先前搜尋造成的偏誤。

無法比較（來源不存在、損壞、無序列、定義不相容、資料不足）回傳
`status="unavailable"`，附 `reason`、`next_step`、`source`、`detail`；
請求本身無效（例如 signal 來源缺少 `column`）回傳
`success=false` 與 `code=invalid_comparison_request`。明細回傳的排除樣本與
缺日樣本各有上限，完整遮罩與座標保存在 artifact 中。

## 可離線的成果

`save_artifact=true` 會從同一次分析出版兩份內容定址、不可變的成果：
`paired_comparisons/<前16碼>.md`（Markdown）與
`paired_comparisons/<前16碼>.html`（自含 HTML，樣式內嵌、無外部資源）。
兩份都註冊到 artifact registry，分別回傳在 `artifact` 與 `html_artifact`。
其內容包含方法版本、模式與資格、限制、完整分析 JSON、兩側來源及排除
Audit。重複相同比較會重用同一份內容；不同設定產生不同 ID，舊檔不被改寫。
使用 `get_artifact(include_content=true)` 搭配回傳的 SHA-256 分頁閱讀；
HTML 可直接在離線瀏覽器開啟。

## 驗證入口

- [純對齊測試](../tests/domain/statistics/test_statistics_alignment.py)
- [服務測試](../tests/app/test_paired_comparison.py)
- [MCP 契約](../tests/mcp/test_paired_comparison_contracts.py)
- [自然需求測試](../tests/e2e/test_llm_paired_comparison.py)（`live_llm`）
