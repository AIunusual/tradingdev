# 成果需求

本冊對應 UC-01 至 UC-05 的保存與交付；整體範圍見 [SRS](../requirements.md)。

## FR-RES-001 歷史探索與執行設定

- **需求**：以穩定身分查詢保存的策略、工作、run 與 artifact，提供摘要、篩選、分頁及明細入口；查詢依當次保存的設定與來源。
- **驗收**：目前 YAML 或預設值變更不改寫歷史解讀；缺少或損壞證據回報原因，不猜測或重跑策略補造。分頁說明續讀方式及總量，不承諾跨寫入的一致快照；ML 身分與 run 不混用。
- **契約**：[探索分頁](../run_artifacts.md#discovery-pages)、[歷史觀察值](../run_artifacts.md#historical-observation-queries)、[保存設定與來源](../run_artifacts.md#research-settings-and-provenance)、[提交研究說明](../run_artifacts.md#research-evidence-records)、[執行身分與固定設定](../run_artifacts.md#execution-manifest)。
- **驗證**：[MCP 歷史契約](../../tests/mcp/test_history_contracts.py)、[真實 MCP 分頁](../../tests/integration/test_list_pagination.py) 核對呼叫端與保存資料。

## FR-RES-002 績效與基準解讀

- **需求**：回測 run 的績效提供保存的定義、單位、適用性及不可用原因；依模式區分報酬、金額損益、交易統計與分折結果。可提供基準的結果依資產類別的正式政策計算及保存基準。
- **驗收**：缺值不當作零，未平倉不冒充已平倉；粗於日頻資料不假造年化與日／月統計。回測與 walk-forward 的個別執行 scope 可保存基準，`test_summary` 因無單一路徑而沒有基準，最佳化目前不保存基準。不支援或資料覆蓋不足的基準明示原因，不借用其他市場基準；基準取得失敗不使已完成的回測失敗。回測 run 保存的歷史指標定義不被目前 catalog 替換。
- **契約**：[績效解讀與基準](../performance_artifacts.md)。
- **驗證**：[歷史績效查詢](../../tests/app/test_run_metrics.py)、[績效數值](../../tests/domain/backtest/test_performance_metrics.py)、[基準 service](../../tests/app/test_benchmark_service.py) 核對定義、缺值與對齊。

## FR-RES-003 紀錄查詢與可離線報告

- **需求**：以已保存證據產生可離線閱讀的 HTML，提供設定、績效、交易及適用的委託／帳戶明細；計算結果、模型解讀與診斷限制分開標示。
- **驗收**：生成報告不重跑或下載行情；明細保存原值，回測表格可切換完整精度檢視，匯出保留完整紀錄，不受畫面篩選限制。固定比較保留失敗候選；無相應證據的章節明示缺少。報告生成前須驗證引用的 ML series artifact；若引用 artifact 缺失、損毀或與引用 scope 不符，報告生成失敗且不出版，即使 leaderboard 證據仍可用。ML 附加章節不宣稱其模型曾用於旁邊的回測；隱藏章節不移除內嵌資料。
- **契約**：[報告組裝與呈現](../reports.md#composable-offline-reports)。
- **驗證**：[報告 service](../../tests/app/test_report_service.py)、[精度與匯出](../../tests/adapters/test_report_precision.py)、[ML 報告 HTML](../../tests/adapters/test_ml_report_html.py) 核對內容；[ML 報告 service](../../tests/app/test_ml_report_service.py) 是報告保護與引用 series scope 不符時拒絕出版的檢查入口，列為入口不代表測試已通過；瀏覽器互動另需人工操作，離線測試不代表所有瀏覽器相容。

## FR-RES-004 研究解讀與最終交付

- **需求**：要求完成研究並解讀績效時，助手須提供結果、證據限制及報告連結，除非使用者明確拒絕報告或只要摘要；建議不自動授權下一次實驗。
- **驗收**：最終回覆識別所選結果及明細查看方式，確認交付檔案存在可讀。只回報工作啟動或中途貼連結不算完成；報告失敗可沿用已保存結果重試。HTTP 可提供綁定已保存確認單、報告或研究封存的短效連結；檔案變動或連結失效須明示，重新取得不重跑研究。無 HTTP 交付時由客戶端適當交付，遠端主機路徑不冒充下載網址。
- **契約**：[研究解讀與診斷](../reports.md#research-interpretation-and-diagnostic-providers)、[報告交付](../reports.md#composable-offline-reports)、[產物下載與交付限制](../run_artifacts.md#reading-artifact-delivery)。
- **驗證**：[交付證據檢查](../../tests/e2e/test_report_delivery.py) 為離線驗證器測試；[真實模型成果交付](../../tests/e2e/test_llm_research_delivery.py) 及 [HTTP 模型交付](../../tests/e2e/test_llm_http_delivery.py) 驗證受測流程，須另啟模型測試。

## FR-RES-005 研究筆記與封存

- **需求**：可保存研究摘要、更新目前筆記，並將指定版本與報告匯出為可獨立閱讀的封存；筆記不替代原始研究證據或執行授權。
- **驗收**：更新需核對預期版本，衝突不覆寫；匯出核對附件身分、內容與容量，缺少或變更時拒絕。後續筆記更新不改變舊封存；HTTP 下載期限及 stdio 本機路徑差異明示。封存不等同整個工作區、全部行情或重現環境。
- **契約**：[研究筆記與封存](../reports.md#research-notebooks-and-archives)、[封存下載與交付限制](../run_artifacts.md#reading-artifact-delivery)。
- **驗證**：[筆記 service](../../tests/app/test_research_notes.py)、[MCP 筆記](../../tests/mcp/test_mcp_research_notes.py) 檢查版本、附件完整性、匯出及交付入口。

## FR-RES-006 可見對話保存與閱讀

- **需求**：客戶端整合可保存實際觀察的使用者輸入、模型可見回覆與工具紀錄，依原順序匯出可閱讀文字；與研究摘要及執行授權分開。
- **驗收**：紀錄綁定對話與回合身分；並行更新須核對版本，已完成歷史不可改寫。部分輸出與缺漏明示，不用模型摘要補造逐字紀錄；匯出固定所選版本，後續對話不改變舊檔案。私有推理與其他客戶端未觀察的訊息不屬於保存範圍。
- **契約**：[對話紀錄](../reports.md#recorded-conversations)。
- **驗證**：[對話保存與匯出](../../tests/app/test_conversations.py) 檢查文字順序、部分狀態、衝突、容量與固定成果；[Codex 接入](../../tests/adapters/test_codex_recording.py) 以替代協定端檢查事件、續接與實際子程序清理。人工驗收另用真實 Codex CLI／TradingDev MCP 執行多輪對話、關閉後續接，核對原文、工具內容及匯出 HTML 的閱讀與展開操作。

## FR-RES-007 最佳化排名的搜尋證據

- **需求**：最佳化排名候選可追溯到同一執行 manifest、選中訓練 scope 與測試 scope 所屬的原始訓練搜尋。
- **驗收**：排名分頁維持原搜尋的計畫、實際嘗試、完成及 objective 可用數。透過固定引用續讀實際 trial 參數、訓練指標、設定及缺值原因，引用不符時拒絕。未發布完整結果的工作沒有排名候選，無法驗證的歷史搜尋回傳不可用原因。原始嘗試數不代表有效獨立樣本數，訓練搜尋與測試績效分開解讀。
- **契約**：[保存結果排名](../run_artifacts.md#ranking-saved-run-scopes)、[搜尋證據](../performance_artifacts.md#optimization-selection-evidence)。
- **驗證**：[排序與證據續讀](../../tests/app/test_selection_evidence.py) 核對搜尋規模、分頁及歷史證據，[最佳化 MCP 流程](../../tests/integration/test_mcp_optimization.py) 核對 worker 發布及部分結果的排名行為。

## FR-RES-009 保存序列的配對比較

- **需求**：使用者可選兩側已保存、完整性核驗通過的序列，取得同時間觀察的差額、不確定性與可離線閱讀的 Markdown 與自含 HTML 分析成果；不因比較請求啟動回測、訓練、下載或重新 predict。
- **驗收**：共同日期取精確交集，warm-up 缺值保留原座標再按同位置有效觀察配對；回報各自原長度、共同期間、有效樣本數、排除原因與座標，不補零、不按順序硬湊。描述性結果在回覆中保留有上限的缺口摘要，完整稽核只在明確出版成果時提供；推論模式拒絕日曆缺口、間隔與宣告 cadence 不符及中途無效值，不得把缺口壓縮成相鄰觀察。部分／完全無重疊、重複時間、時區處理（公開服務把 naive 與 aware 時間戳正規化為 UTC；純對齊函式本身會拒絕
naive／aware 混用）、頻率不符、來源損壞、舊資料無序列及同 run 不同 scope 都有明確結果。描述與推論資格分開，p 值在描述模式不承載顯著性結論；推論模式只接受精確 UTC 日頻或小時頻的完整日曆、非空且非 `unknown` 的 producer 來源，且不保證 bootstrap 的名目涵蓋率或多比較校正。資訊係數不與報酬混檢，無成本訊號模擬不冒充成交績效；基準來源必須明示 `product` 或 `zero_cost`；零成本價格基準只在有核驗過的原始價格 artifact 時提供。可用結果輸出含方法版本、設定、seed、block size、兩側來源及限制；不可用結果只回報原因與可得的對齊摘要，差異原因與操作錯誤分開，回覆明細有容量界線。兩份成果內容同源、各自內容定址，舊成果不被改寫；出版預設關閉，只有結果可用且明確要求時才產生。回覆中的排除樣本與缺日樣本有容量上限，已出版 artifact 則保存完整對齊證據。
- **契約**：[保存序列配對比較](../paired_comparison.md)（對齊、cadence、來源資格與成果）、[逐點評估序列](../ml_research.md#saved-pointwise-evaluation-series)、[產物註冊](../run_artifacts.md#artifact-references)。
- **驗證**：[純對齊](../../tests/domain/statistics/test_statistics_alignment.py)、[服務](../../tests/app/test_paired_comparison.py)、[MCP 契約](../../tests/mcp/test_paired_comparison_contracts.py) 核對共同索引與來源；[自然需求測試](../../tests/e2e/test_llm_paired_comparison.py) 須另啟模型測試。

## NFR-DATA-001 保存完整性與可追溯性

- **需求**：歷史結果固定各自保存的設定與來源身分；回測 run 另保存當時指標定義。讀取及發佈須核對相關證據，區分完整結果、部分狀態與不可用資料，並區分指標定義的取得時點。
- **驗收**：ML 排行榜紀錄不保存指標定義快照；從相同評估紀錄新產生報告時，採用報告產生時的 signal catalog，不宣稱還原評估當時的定義。既有報告保留其已保存的定義，不隨 catalog 更新改變。缺少或變更來源時不靜默取用其他版本；失敗不把缺值當零或補造交易。Hash 只在預期值可信時檢查一致性，不宣稱抵抗同時控制檔案與資料庫的寫入者；檔案與資料庫不宣稱跨資源交易。
- **契約**：[執行產物](../run_artifacts.md#artifact-references)、[績效保存](../performance_artifacts.md#performance-scopes-and-provenance)、[ML 評估與定義來源](../ml_research.md#recorded-signal-evaluations)、[報告快照](../reports.md#composable-offline-reports)。
- **驗證**：[manifest 儲存](../../tests/app/test_execution_manifest_storage.py)、[績效產物儲存](../../tests/app/test_performance_artifact_storage.py) 核對綁定、損壞及發佈失敗；[ML 報告投影](../../tests/domain/ml/test_ml_report.py)、[ML 報告 service](../../tests/app/test_ml_report_service.py) 核對定義投影與已保存報告的獨立性。

## NFR-UX-001 報告可閱讀與操作

- **需求**：MCP 確認文件與研究成果提供可閱讀的白話說明；報告可離線開啟，研究者能從總覽到明細、檢視完整精度並匯出資料。
- **驗收**：在交付環境開啟實際報告 HTML，操作明細、篩選、排序及匯出，核對所選結果與原數值。MCP 確認文件須可離線閱讀，不能把未執行檢查呈現為已通過；確認文件的驗收不包含互動表格。未實測的瀏覽器與遠端客戶端交付能力須明示。
- **契約**：[報告呈現](../reports.md#composable-offline-reports)、[確認文件](../execution_plans.md#execution-plans-and-confirmation-documents)。
- **驗證**：人工驗收：在目標瀏覽器離線閱讀保存的 MCP 確認 HTML，核對檢查狀態；另以單次及比較報告操作明細、篩選、排序及匯出，核對下載內容。記錄當次瀏覽器與結果，本需求引用不表示已完成操作驗收。
