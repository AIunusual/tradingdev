# 執行需求

本冊對應 UC-01、UC-02；整體範圍見 [SRS](../requirements.md)。

## FR-EXEC-001 準備與明確確認

- **需求**：MCP 正式回測、固定比較、walk-forward 及最佳化須先固定設定、研究說明與離線檢查證據，交付白話摘要及完整確認文件，再等待後續明確同意。
- **驗收**：準備不取得真實行情、不訓練、不執行回測，也不建立正式 job；適用的生成程式檢查使用合成資料；固定比較的每個候選（包含 bundled）均須通過兩組合成 fixture 檢查。最初請求、提問與沉默不是同意；拒絕不啟動，修改須重新準備及確認。直接回測 CLI 命令不採用對話確認，Codex 對話入口仍遵循 MCP 確認流程；獨立 ML 研究不適用此計畫流程。確認單交付：確認單以固定產物身分與雜湊供客戶端取得；HTTP 提供綁定該身分與雜湊的短效閱讀連結；無 HTTP 交付時由客戶端交付完整且經核對的副本，只呈現使用者可讀的閱讀連結，不附上不可存取的伺服器路徑。客戶端無法交付時須明示限制，不宣稱使用者可讀或請其核准未交付文件；讀取與傳輸不啟動工作，也不延長計畫期限。
- **契約**：[執行計畫與確認](../execution_plans.md#execution-plans-and-confirmation-documents)、[研究說明保存與綁定](../run_artifacts.md#research-evidence-records)。
- **驗證**：[preflight service](../../tests/app/test_preflight_service.py) 檢查離線邊界；[HTTP 確認單傳輸](../../tests/mcp/test_http_execution_confirmation.py) 檢查分頁與完整性；[短效閱讀連結](../../tests/mcp/test_artifact_downloads.py) 檢查實際 HTTP、失效與不啟動工作；[交付證據](../../tests/e2e/test_conversation_user.py) 拒絕不完整副本與失效原始連結；[真實模型確認流程](../../tests/e2e/test_llm_execution_plans.py) 及[遠端交付情境](../../tests/e2e/test_llm_research_delivery.py) 檢查跨回合互動與交付選擇，須另啟模型測試。

## FR-EXEC-002 提交完整性與重複請求

- **需求**：同意只能提交原計畫的固定內容。後端須核對期限、來源與執行政策，以唯一提交身分防止重複啟動。
- **驗收**：過期或內容變動時要求重新準備；重複成功提交回傳原工作。啟動成功但狀態寫入失敗時保留可查詢身分，核對相同 manifest 與完整 worker 身分後恢復，不另啟工作。此機制不證明人已閱讀或同意。
- **契約**：[執行計畫提交與恢復](../execution_plans.md#execution-plans-and-confirmation-documents)、[提交快照與歷史完整性](../run_artifacts.md#research-evidence-records)。
- **驗證**：[計畫 service](../../tests/app/test_execution_plan_service.py)、[提交 service](../../tests/app/test_execution_submission.py) 驗證並發、變更、逾期及啟動後寫入失敗。

## FR-EXEC-003 執行模式與回測

- **需求**：以所選策略版本、行情、有效參數與成本執行歷史回測，依策略能力拒絕不支援的模式。Signal 與 volume 模式按各自交易及績效語意處理。
- **驗收**：必須訓練的策略不能以一般回測或不支援的最佳化繞過訓練；可離線判定的設定錯誤於正式資料載入前拒絕，資料相依檢查仍可能在正式執行時失敗。每段回測結束須將剩餘部位完整平倉並計入出場成本，無法完成時不得發布仍有部位的成功結果。結果保存實際設定，不把模擬帳戶宣稱為交易所保證金或資金費率帳本。
- **契約**：[訓練前置條件](../strategy_contract.md#training-before-scoring)、[訊號與成交語意](../strategy_contract.md#signal-and-execution-semantics)。
- **驗證**：[策略能力](../../tests/domain/strategies/test_strategy_capabilities.py)、[回測 service](../../tests/app/test_backtest_service.py) 驗證模式與執行邊界；[期末平倉](../../tests/domain/backtest/test_terminal_liquidation.py) 核對多空成交、成本、期末現金與失敗路徑。

## FR-EXEC-004 固定參數比較

- **需求**：明示固定候選的比較使用同一份不可變行情快照及共同研究假設，各候選獨立策略、資金與亂數串流；不得自動擴大搜尋。
- **驗收**：候選變更須重新確認。保存所有候選帳本與成功結果，失敗、取消及未執行候選不省略；沒有父工作的合併資金績效。此流程限普通回測；生成策略實例各自使用隔離程序，內建策略與可信主程式的任意全域狀態不在隔離保證內。
- **契約**：[固定比較證據](../run_artifacts.md#fixed-comparison-evidence)、[比較執行與隔離](../../ARCHITECTURE.md#準備確認與執行回測)。
- **驗證**：[比較生命週期](../../tests/app/test_comparison_lifecycle.py)、[真實 MCP 比較](../../tests/integration/test_comparison_workflow.py) 核對資料共用、獨立執行及失敗帳本。

## FR-EXEC-005 分段訓練與樣本外評估

- **需求**：Walk-forward 須固定有效訓練／測試切分，依策略能力訓練或分段回測，並依宣告的標籤 horizon 與淨化模式檢查時間重疊。
- **驗收**：日期換算、研究期間包含關係及互斥選項依契約檢查；不能為通過切分而擴大使用者期間。Automatic／explicit 模式的零 horizon 停用重疊稽核，不代表標籤安全；purged 模式要求正值。設定檢查通過不代表真實資料可形成分折。逐折結果與聚合摘要分開，不將分折分布當成單一路徑績效。
- **契約**：[Walk-forward 設定](../strategy_contract.md#strategy-walk-forward-settings)。
- **驗證**：[日期包含](../../tests/domain/validation/test_explicit_dates.py)、[標籤洩漏檢查](../../tests/domain/validation/test_leakage_gate.py)、[保存分折](../../tests/app/test_saved_walk_forward.py) 分別檢查設定、時間邊界與讀回。

## FR-EXEC-006 參數最佳化

- **需求**：最佳化固定搜尋候選、目標方向、互不重疊的訓練／測試期間及首個訓練 trial 的估算時限；同意後才搜尋，並以選定參數執行樣本外測試。
- **驗收**：不把固定比較視為搜尋授權；不與 walk-forward 切分混用。估算時限只限制首個訓練 trial，後續候選與樣本外評估不共用此時限，不承諾整次搜尋有界。不適用或不可用目標不能排名，全部候選無效時失敗；訓練排名與樣本外結果須分開解讀。
- **契約**：[最佳化執行規格](../run_artifacts.md#execution-manifest)。
- **驗證**：[最佳化 service](../../tests/app/test_optimization_service.py)、[真實 MCP 最佳化](../../tests/integration/test_mcp_optimization.py) 核對候選、期間、目標與結果。

## FR-EXEC-007 工作狀態與取消

- **需求**：正式背景工作提供穩定身分、可查狀態、失敗原因與取消能力；程序控制須核對啟動身分，不只相信數字 PID。
- **驗收**：結果保存成功後才回報完成；執行中工作確認所屬程序清理後才能標為已取消，尚未啟動工作可直接取消。異常與取消不得悄悄變成成功或刪除已保存成果；無法證實停止時明示限制。
- **契約**：[工作狀態與程序身分](../run_artifacts.md#job-state-and-worker-identity)、[程序控制與清理限制](../../ARCHITECTURE.md#程序控制與清理)。
- **驗證**：[工作 service](../../tests/app/test_job_service.py)、[真實受監督 worker](../../tests/integration/test_managed_workers.py) 檢查狀態、控制與清理。
