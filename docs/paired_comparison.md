# 保存序列的配對比較

`compare_saved_series` 與 `compare_signal_series` 共用 `PairedComparisonService`。
比較只讀取兩側已保存的 snapshot；訊號序列每次載入都交由既有
`EvaluationSeriesStore` 驗證，不在比較請求中重新 predict。比較把共同有效的觀察值交給
`domain.statistics.comparison.difference_interval_outcome`。run 來源在保存的 `returns` 不存在時，才會由已保存的權益與固定本金推導每期報酬；
此路徑遇保存的 `null` 權益時，該期與下一期都保持不可用，不會跨缺口計算。若保存的
`returns` 存在，則完全依照保存值使用。這是對保存觀察值的衍生，
不是重跑策略或重建原始行情／訊號。比較請求不會回測、訓練、下載行情、
重新 predict，也不會改寫舊報告。

## 來源與資格

| `kind` | 需要的欄位 | 序列與成本 |
| ---- | ---- | ---- |
| `run` | `run_id`、`scope`（省略用預設） | 只接受有 observations 的 signal 模式 scalar backtest scope：優先用保存的 `returns`，但須與 `equity_curve` 等長且與時間戳對齊，否則回報不可用；缺席時才由帳戶權益與固定本金推導每期報酬，成本已在成交中；volume 模式回報 `not_applicable`，彙總 fold summary 沒有觀察值而回報不可用 |
| `benchmark`，`benchmark_mode="product"` | `run_id`、`scope`（`benchmark_mode` 必須明示 `product`） | 只讀保存的 `BenchmarkScope` 淨值／報酬與其成本欄位；即使保存了 `returns`，也必須同時有 `timestamps` 與 `equity_curve`，否則回報 `benchmark_unavailable`；`returns` 缺席時只有在 `init_cash` 為正數下才由淨值推導，否則同樣不可用。不逆推原始 closes |
| `benchmark`，`benchmark_mode="zero_cost"` | `run_id`、`price_artifact_id`（`scope` 被忽略） | 服務先確認該 run 的固定 snapshot 存在，再讀註冊且 SHA-256 核對通過的 `raw_price_series` artifact，用 `buy_and_hold_returns` 產生零成本報酬；比較整份已保存的原始價格序列，不依 run 的期間裁切。selector 仍要求 `run_id` 以確認保存 snapshot 存在，但回傳的來源身分只標明價格
artifact，不含 `run_id` 與 `scope`。此來源固定以日頻
（`frequency="daily"`、`cadence="1d"`）處理，與其他 cadence 的序列配對會被拒絕。若無此 artifact 則回報 `zero_cost_prices_unavailable`，不下載也不新增保存流程。此變更不包含建立此 artifact 的流程；使用前須已有已註冊並核驗過的 artifact，檔案 JSON 需有 `timestamps`、`closes` 与非空 `source`，本文只定義讀取契約 |
| `signal` | `artifact_id`、`column` | 先用 `get_evaluation_series_page` 探索已保存序列；發現與驗證限制見[逐點評估序列](ml_research.md#saved-pointwise-evaluation-series)。此來源是已保存的 pointwise evaluation series；`pearson_ic`／`spearman_ic` 是資訊係數，`realized_return` 是無成本訊號模擬，二者不得互換或與成交報酬混檢 |

兩側必須使用**相容定義、同頻率與相容設定**。成本相對的成交報酬可與已保存的
產品基準相比，也可與核驗過的原始價格零成本報酬相比；零成本來源的身分會標明
`cost_basis`，產品基準則以 `benchmark_costs` 記錄保存的手續費與滑價，兩者不能
混淆，也不能把零成本訊號模擬冒充成交績效。IC 只能與同類 IC 相比。

服務實際檢查的是宣告／推斷的 cadence 標籤相同而非原始間隔完全相等（run 與產品
基準用保存設定宣告的頻率，訊號序列由時間戳推斷；非日／小時都標為 `unsupported`，
因此不同間隔的訊號仍可能在描述模式下通過）；Pearson／Spearman IC 配對的整個 `manifest.settings` 相同（包含候選身分與參數；
即使 horizon／window／quantile 相同，候選設定不同仍回報不相容），`realized_return`
配對不檢查這些設定；run 之間、run 與產品基準之間、產品基準彼此之間的 `init_cash`、
`periods_per_year`、`risk_free_rate` 相同（含零成本價格基準的配對除外）。它**不**核對標的、日期範圍是否完全一致，也不核對 run 之間或 run 與產品基準的
fee/slippage 是否相同，因此可用的比較不一定是受控實驗。entry ID、model ID 或
dataset ID 不同本身不證明 holdout，也不證明未調參。

回覆的 `claim_scope` 是服務依保存證據給出的**這對來源整體**資格標記：只有當
兩側都是 split 為 `test` 且具保存研究證據的 run scope 時才是
`verified_out_of_sample`；訊號序列與基準來源一律保持 `in_sample_or_unknown`，
所以與它們配對也不會取得該標記。這只是保存關係的呈現，不保證沒有
先前選擇或多重比較偏誤。訊號序列來源另保留 training/evaluation 關係、producer source、
signal fingerprint 與可用的 fold 範圍（只有首列範圍，不是完整 fold 清單）；run 與產品基準來源保存 run/scope 身分，
零成本價格基準則只保存價格 artifact 身分。
`claim_scope` 只出現在整份回覆的一層，不是各別來源的欄位。

## 對齊與缺口政策

推論模式只接受**精確 UTC 日頻與小時頻**的共同時間戳；其他相同頻率的序列仍可做
描述性比較，但 cadence 標為 `unsupported`，不能進入推論。訊號序列的頻率由保存
時間戳推斷，不規則或不足兩個時間點標為 `unsupported`，不能冒充日頻。run 與產品
基準則使用其保存設定宣告的頻率。純 `align_masked_pair` 要求兩側都是 timezone-aware
的 `DatetimeIndex`，任一侧 naive 都會被拒絕；公開服務會先把保存來源的時間戳正規化為
UTC 再對齊。交集之前不做 dropna、補零或
名稱對齊；warm-up 的 `NaN` 留在原座標，先依每側自己的 UTC 時間順序（不是傳入的 row 順序）
判斷是前段
warm-up 還是中途缺值，再以雙方同位置有效的觀察配對。回報原始長度、共同
時間戳數、共同有效樣本數、各類排除原因與排除計數；回覆只有
`valid_mask_count`，完整有效遮罩與全部座標需明確出版成果才會提供。

共同期間缺少預期日曆點時，`cadence_complete` 為 false；它只描述時間戳方格。
共同時間戳上存在任一侧中途無效值時另計為 `interior_gap`，推論會獨立拒絕。描述性比較仍可計算；回覆只保留有上限的缺口摘要，完整稽核只在成果出版時提供。bootstrap 會按有效觀察的
序列位置重抽樣，因此描述結果實際上會把有效觀察之間的位置視為相鄰。推論模式
不接受這種相鄰化：通過共同有效樣本數檢查後，`mode="inferential"` 對日曆缺口、間隔與宣告 cadence
不符或中途無效值回報 `cadence_gap_unsupported`；樣本數不足時以
`insufficient_observations` 優先。前段 warm-up 不視為中途缺口。第一版沒有新的缺口
bootstrap。完全沒有共同日期回報
`no_common_timestamps`；保存座標重複等malformed input 會轉為 `comparison_error` 的不可用結果。完全沒有共同有效的觀察回報 `no_valid_observations`；有共同有效觀察但不足
`MIN_BOOTSTRAP_OBSERVATIONS`（定義於
[`domain/statistics/sharpe.py`](../src/tradingdev/domain/statistics/sharpe.py)）
回報 `insufficient_observations`；兩點不會補成三對。

## 統計量與輸出

比較統計量固定為 `mean`，差額方向固定為 `a_minus_b`。**可用結果**回報
`estimate`、`estimate_a`、`estimate_b`、`lower`、`upper`、`p_value`、
`confidence`、`reps`、`valid_reps`、`seed`、`block_size`、`correlation`
（兩條對齊後輸入序列的 Pearson 相關；任一侧為常數時為 `null`，不是重抽樣統計量的相關）、
共同期間、排除摘要、兩側來源身分與方法限制。**不可用結果**的 `result` 為
`null`，不帶這些統計欄位；若失敗發生在對齊之前，連共同期間也只是預設值。`block_size=None` 時交由既有
[`comparison.difference_interval_outcome`](../src/tradingdev/domain/statistics/comparison.py)
的 difference jackknife 自動估計；指定值必須是正整數。重抽樣次數限制在
`100..10000`。

`mode="descriptive"` 是預設：p 值標為 `descriptive_only`，不承載顯著性結論。
`mode="inferential"` 另要求完整日曆與可識別的 producer；p 值是 percentile
由線性插值的 percentile 區邊界反解而得，只對選定的這一對有意義，不是直接
以重抽樣差值的正負比例計算。stationary bootstrap 不保證任意非平穩報酬
過程下的有效推論，區間也不保證名目涵蓋率；完整日曆與可識別來源只是必要檢查，
不是充分條件。兩者都不做多重比較校正，第一版只處理單一 pair，不自動宣稱已修正
先前搜尋造成的偏誤。

無法比較（來源不存在、損壞、無序列、定義不相容、資料不足）回傳
`status="unavailable"`，附 `reason`、`next_step`、`source`、`detail`；
比較選項無效（`reps` 需 100..10000 的整數、`seed` 需非負整數、`block_size` 需正整數、
`confidence` 需嚴格介於 0 與 1 的 float；`mode` 在 MCP schema 中是 Literal，非列舉值
會先被工具輸入驗證拒絕）或出版失敗時，Python service 會拋出 `PairedComparisonError`（`invalid_comparison_request` 或
`artifact_publication_failed`）；MCP 工具捕捉後回傳 `success=false` 與對應的
`code`。若 selector 在 MCP 輸入 schema 驗證時就
不完整（例如 signal 缺少 `column`），則是工具輸入錯誤（`ToolError`），不會
進入 service。明細回傳的排除樣本與
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
Audit（有效遮罩、全部排除座標與缺日）；HTML 另有可離線閱讀的對齊證據章節；其捷徑標籤使用該次要求的 confidence，
完整 JSON 仍保留精確設定值。重複相同比較會重用同一份內容；輸出內容不同才會產生不同 ID，內容相同則重用，
舊檔不被改寫。
用戶端可用 `get_artifact(artifact_id=回傳的 ID, include_content=true,
expected_sha256=回傳的 SHA-256)` 分頁閱讀，並依 `next_offset` 讀到 `null` 後自行存檔；
也可用 `list_artifacts` 查找已註冊的成果。HTML 可直接在離線瀏覽器開啟。出版前會先對
兩份目標做路徑與註冊衝突檢查；若檢查通過但寫入途中發生 OS 錯誤，仍可能只留下
先完成的 Markdown，這時可從 `list_artifacts` 找到它。出版失敗是操作錯誤，
回傳 `success=false`、`code=artifact_publication_failed`，不冒充資料不可用。

## 驗證入口

- [純對齊測試](../tests/domain/statistics/test_statistics_alignment.py)
- [服務測試](../tests/app/test_paired_comparison.py)
- [MCP 契約](../tests/mcp/test_paired_comparison_contracts.py)
- [自然需求測試](../tests/e2e/test_llm_paired_comparison.py)（`live_llm`）
