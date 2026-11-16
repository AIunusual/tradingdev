# ML 訊號研究需求

本冊對應 UC-04；整體範圍見 [SRS](../requirements.md)。
此為獨立同步研究介面，不建立 execution plan 或背景回測工作，不自動把模型轉為策略。

## FR-ML-001 能力探索與固定資料集

- **需求**：依探索所得的已登錄特徵、模型及驗證方式建置研究；資料集只讀可重用的既有行情，保存不可變內容版本與來源。
- **驗收**：來源、市場、代號、頻率與區間一致；快取缺少、尚不可重用或選取期間為空時失敗，不隱式下載。特徵不完整（含暖機）或無法形成標籤而移除的列提供合計數量，未來標籤不可冒充當時可得特徵；成功不保證來源 K 棒完整。
- **契約**：[資料集與模型輸入](../ml_research.md#dataset-and-model-inputs)、[不可變資料集](../ml_research.md#immutable-datasets)。
- **驗證**：[資料集完整性](../../tests/app/test_ml_dataset_integrity.py)、[時間完整性](../../tests/app/test_ml_temporal_integrity.py) 檢查來源、內容及時間邊界。

## FR-ML-002 訓練、驗證與預測

- **需求**：訓練固定模型參數及種子，保存模型與特徵身分；分折驗證與最後使用全部訓練列擬合的模型分開解讀。預測核對特徵順序與模型內容。
- **驗收**：不把分折分數當作最終模型的樣本外證明。預測核對資料集與模型產物的完整性及特徵順序，不符時拒絕；保存的訓練資料集身分用於來源追溯，不要求預測資料集等同訓練資料集。分折以 panel 共用的相異時間戳切分，排除實際標籤區間與測試區間重疊的訓練時間點，包含 K-fold 右側重疊；含前向填補觀察的 panel 拒絕分折驗證。完整預測與回覆預覽分開，預覽分頁不裁減實際評估的訊號。模型檔案需可信，身分檢查不使 pickle 成為安全格式。
- **契約**：[分折與訊號解讀](../ml_research.md#fold-validation-and-signal-interpretation)、[模型保存](../ml_research.md#trained-models)。
- **驗證**：[訓練驗證](../../tests/app/test_ml_training_validation.py)、[時間完整性](../../tests/app/test_ml_temporal_integrity.py)、[預測 service](../../tests/app/test_ml_predict.py) 核對分折、標籤區間與預測邊界。

## FR-ML-003 訊號與手設規則評估

- **需求**：模型訊號及手設規則可在明示資料與持有期間下評估，核對完整資料對齊；綁定資料集的評估只支援單一標的，panel 須拒絕。評估回覆須區分實際產生來源與呼叫端宣告的方法。選擇記錄評估時，保存來源、方法家族、可取得的參數、訊號指紋與評估條件。
- **驗收**：來源由實際執行入口決定，公開 caller 不可自稱後端模型或規則。未選擇記錄時只回傳評估結果，不寫入排名資料。未宣告設定時空參數表示未知，不是擬合預設值，不能事後推測補齊；caller 宣告的模型產物不存在或載入驗證失敗時可記錄未知參數，後端模型評估仍須有效產物；其他讀取錯誤則使記錄失敗且不寫入排名資料。呼叫端自備訊號的模型訓練欄位只描述被命名的模型，來源確認不代表樣本外。評估保留未啟動交易的列，並說明暖機及不重疊持有的限制；常數訊號無法形成多空組合，須拒絕評估。簡化無成本統計不宣稱等同交易回測或具有成交／權益證據。看過資料後選門檻保留樣本內選擇偏誤；未保存的序列不事後猜測補算。
- **契約**：[分折與訊號解讀](../ml_research.md#fold-validation-and-signal-interpretation)、[已記錄訊號評估](../ml_research.md#recorded-signal-evaluations)。
- **驗證**：[規則評估](../../tests/app/test_ml_rule_evaluation.py)、[訊號收益](../../tests/domain/ml/test_simulated_returns.py) 核對持有區間、評估與無成本口徑；[評估來源與訓練綁定](../../tests/app/test_ml_training_binding.py) 核對入口來源、模型宣告及來源保護。

## FR-ML-004 排名與比較條件

- **需求**：依指標定義的方向排名，保留同分名次、不可用原因與比較條件；只有明確選擇時才跨參數軸比較。
- **驗收**：不可用候選不被轉為零或默默省略。同模型／資料集／條件的不同來源各自保存，同來源重評更新原候選；呼叫端訊號不能靜默覆寫後端結果，舊未知來源不得被升級或推定。排名及固定報告保留來源，並讓同名多列可辨識；不新增來源或樣本外准入限制。Signal 排名預設拒絕不一致的量測條件；`allow_parameter_axes` 只放寬契約列明的參數軸，仍保持研究身分一致與每個 horizon 唯一資料版本；`allow_mixed` 才允許不相容條件並揭露限制，兩者均不繞過明示資料集的驗證。跨 horizon 的來源行情一致性未受保證。Run 與 signal 的共用查詢入口保持各自的資料與評估語意。
- **契約**：[訊號排名](../ml_research.md#recorded-signal-evaluations)、[保存結果排名](../run_artifacts.md#ranking-saved-run-scopes)。
- **驗證**：[ML 排名 service](../../tests/app/test_ml_leaderboard.py)、[MCP 共用排名](../../tests/mcp/test_ml_leaderboard_runs.py) 核對排序、不可用值及資料來源分派；[候選保存](../../tests/domain/ml/test_leaderboard_store.py) 核對同來源替換與跨來源保護，[固定報告](../../tests/app/test_ml_report_service.py) 核對保存來源及未知歷史的保留。
