# 策略需求

本冊對應 UC-01、UC-02；整體範圍見 [SRS](../requirements.md)。

## FR-STR-001 研究選擇與模擬假設

- **需求**：建立研究前，助手須取得策略構想、標的、期間與 K 棒頻率；缺項先追問。只有明確選定不可變 generated revision 時才繼承固定選擇。本金、成本與年化使用適用政策，保留有效值及來源。
- **驗收**：缺少必要選擇時不提早建立策略或準備執行；不從 bundled 範例猜日期；bundled 與隱含 current 不提供繼承，舊 revision 未保存構想時須明確補入。明示零成本保持為零，不適用的政策要求釐清；後續政策更新不改變已固定的執行。後端完整性檢查不宣稱輸入來自真人。
- **契約**：[研究輸入與模擬政策](../strategy_contract.md#research-inputs-and-simulation-policy)、[有效設定與來源保存](../run_artifacts.md#research-settings-and-provenance)。
- **驗證**：[需求解析 service](../../tests/app/test_research_resolution.py) 核對繼承、零值與來源；[真實模型需求釐清](../../tests/e2e/test_llm_novice_intake.py) 檢查追問與所有提早操作嘗試，須另啟模型測試。

## FR-STR-002 建立與修訂的身分

- **需求**：使用者不必先命名；系統須區分人類名稱與技術身分、新建與修訂，並保存不可變策略 revision。Bundled 與 generated 策略分開管理。
- **驗收**：新建碰撞不得覆寫既有策略；修訂綁定預期版本，版本衝突拒絕保存。新 revision 不繼承舊版驗證資格；舊版本與其歷史結果仍可指定查詢，不存在的明示版本不退回 current。
- **契約**：[版本身分與資格](../strategy_contract.md#revision-identity-and-eligibility)。
- **驗證**：[策略 service](../../tests/app/test_strategy_service.py) 驗證命名、碰撞及修訂衝突；[revision 儲存](../../tests/adapters/test_strategy_revisions.py) 核對版本與保存。

## FR-STR-003 驗證與訊號契約

- **需求**：生成策略須通過程式政策、品質及合成訊號檢查，取得所選版本的執行資格。訊號須符合契約並保持輸入資料；策略作者不得使用未來資料。
- **驗收**：無效訊號、輸入變更、程式檢查失敗、逾時或清理未確認均不得授予資格；實際執行仍檢查當次有效參數的訊號。通過合成檢查不表示已證明無前視、符合自然語言構想或能在真實資料獲利。
- **契約**：[Python 訊號契約](../strategy_contract.md#python-contract)、[版本驗證與資格](../strategy_contract.md#revision-identity-and-eligibility)、[執行安全邊界](../strategy_contract.md#security-model)。
- **驗證**：[訊號框架](../../tests/domain/strategies/test_signal_frames.py) 驗證輸入與輸出；[真實策略檢查 worker](../../tests/integration/test_strategy_check_workers.py) 驗證正常、失敗與逾時清理。

## FR-STR-004 參數實驗與版本隔離

- **需求**：參數實驗可重用合格 revision，但每次固定有效設定，不改寫原程式、基礎 YAML、驗證證據或 current 指向。
- **驗收**：兩次不同參數執行各有自己的固定設定及結果；revision 綁定宣告的變更須另存新版本。來源被更動或所選資格失效時拒絕執行，不以舊證據放行。
- **契約**：[執行與參數實驗](../strategy_contract.md#execution-and-parameter-experiments)。
- **驗證**：[參數實驗 service](../../tests/app/test_strategy_experiments.py)、[MCP／worker 參數實驗](../../tests/integration/test_parameter_experiments.py) 核對執行設定與版本不變性。

## FR-STR-005 草稿清理

- **需求**：清理生成策略預設提供預覽；實際刪除須明確指定版本並取得授權，只處理符合條件且無引用的非 current 草稿。
- **驗收**：目前版本、非草稿、被工作或結果引用，以及引用損壞或不明的版本均保留；失敗回報原因，不宣稱跨檔案交易。關閉 server 不等於授權清除研究成果。
- **契約**：[草稿清理](../strategy_contract.md#draft-cleanup)。
- **驗證**：[清理 service](../../tests/app/test_strategy_cleanup_service.py) 驗證保護條件；[真實模型清理情境](../../tests/e2e/test_llm_cleanup.py) 檢查預覽、授權與結果，須另啟模型測試。
