# 系統架構

TradingDev 讓 LLM 透過標準 MCP 協助使用者建立、驗證與研究交易策略，
提供歷史回測、參數最佳化及結果追蹤，不提供真實下單。
MCP 是主要產品介面；CLI 與 dashboard 共用 application services 的執行與查詢能力。
模型供應商的呼叫與設定留在客戶端整合層，後端不依賴特定 LLM。

本文件說明目前的責任分工、核心概念、執行流程及保證範圍。
操作與資料格式各有以下文件，避免在架構總覽重複維護欄位與版本清單。

| 文件 | 負責內容 |
| ---- | -------- |
| [軟體需求規格](docs/requirements.md) | 產品需求、驗收條件與契約／驗證追溯 |
| [README](README.md) | 安裝、設定、工具使用與使用者升級說明 |
| [策略契約](docs/strategy_contract.md) | 策略 Python／YAML、驗證規則、revision 與執行資格 |
| [執行產物契約](docs/run_artifacts.md) | Workspace、Manifest、SQLite 與歷史查詢入口 |
| [ML 研究](docs/ml_research.md) | 資料、模型、驗證解讀與獨立保存契約 |
| [執行計畫](docs/execution_plans.md) | 確認文件、使用者決策與提交狀態 |
| [績效產物](docs/performance_artifacts.md)、[報告](docs/reports.md) | 指標與觀察值、離線呈現及匯出契約 |
| [開發指南](docs/development.md) | 開發環境、品質檢查、測試及 Git／PR 流程 |
| [代理開發政策](AGENTS.md) | 代理修改 repository 時遵循的政策 |

## 系統邊界與模組責任

以下是主要呼叫關係。Application services 協調策略、資料與執行；
MCP tools 負責協定轉換，CLI 負責同步執行入口，dashboard 負責已保存結果的呈現。

```mermaid
flowchart LR
    client[LLM 客戶端] --> mcp[MCP tools]
    mcp --> app[Application services]
    cli[CLI] --> app
    dashboard[Dashboard] --> app
    app --> domain[策略、資料、回測與績效模型]
    app --> storage[儲存 adapters]
    app --> process[程序 adapters]
    storage --> sqlite[(SQLite)]
    storage --> files[Workspace 檔案]
    process --> supervisor[Supervisor]
    supervisor --> worker[背景 worker]
    worker --> app
    worker --> domain
```

| 位置 | 目前責任 |
| ---- | -------- |
| `mcp/server.py`、`mcp/tools/` | 組裝 services、註冊工具、驗證協定輸入與輸出 |
| `mcp/workers/` | 背景工作入口、狀態更新及流程協調；最佳化 worker 直接協調搜尋與樣本外測試 |
| `app/` | 策略生命週期、研究需求解析、設定固定、限時程式檢查、確認計畫、工作提交、ML 研究、結果保存與查詢；`contracts/` 定義輸入與回覆 DTO |
| `domain/research.py`、`domain/research_policy.py` | 研究選擇與來源模型、共用模擬本金／成本及年化政策 |
| `domain/strategies/` | 策略介面、來源探索與載入、參數解析、訊號契約及 bundled strategies |
| `domain/execution.py`、`domain/randomness.py` | 執行規格與完整性檢查、每次執行的亂數 context |
| `domain/execution_plan.py`、`domain/preflight.py` | 確認內容與執行身分綁定、限時離線檢查請求及驗證證據 |
| `domain/presentation/`、`adapters/presentation/` | 固定文件內容與章節、白話欄位投影，以及共用文字／HTML 呈現 |
| `domain/backtest/`、`domain/validation/`、`domain/optimization/` | 回測引擎、walk-forward、參數搜尋與績效計算整合 |
| `domain/performance/` | 指標定義、適用條件、最佳化方向與具型別的績效產物模型 |
| `domain/statistics/` | 由呼叫端提供觀察值或試驗資料的獨立統計計算與推論；入口見「績效計算與對話摘要」 |
| `domain/ranking.py` | 候選無關的排序核心：依指標宣告方向排序、同分名次、缺值原因與比較群組。兩個呼叫端共用同一套規則而不共用資料模型：`domain/ml/leaderboard.py` 的訊號排名，以及 `app/ranking_service.py` 的 `RankingService.rank_candidates`（依 metric 前綴分派：`signal.` 讀已記錄評估、`run.` 讀已保存 scope 的 run 記錄與 manifest）。MCP 的 `leaderboard` 工具經由 `RankingService` 提供這個共用入口，兩種來源共用同一頁面形狀；CLI 仍未接上 |
| `domain/benchmark.py`、`app/benchmark_service.py` | 依資產類別固定的基準定義、買進持有計算與基準行情取得 |
| `domain/data/`、`domain/indicators/`、`domain/ml/` | 行情與特徵資料處理、技術指標、模型及訓練邏輯 |
| `adapters/storage/`、`adapters/execution/` | SQLite、revision／manifest／績效檔案保存，以及程序啟動與監督 |
| `adapters/cli/`、`adapters/dashboard/` | 終端機與視覺化入口 |
| `shared/` | 路徑解析、JSON 正規化、記錄與平行執行等共用工具 |

這是目前實作的責任分組，尚未形成全面的依賴反轉。
Application services 直接使用具體的儲存與程序 adapters；`domain/data` 內的
crawlers 與 loader 仍執行網路及檔案 I/O。策略檔案讀取與實例生命週期由
`adapters/execution/strategy_factory.py` 負責，domain 的訊號檢查依賴 factory port。
`domain/ml/registry` 的模型、特徵快取與排行榜 stores
也執行檔案 I/O；ML dataset store 位於 `adapters/storage`。
最佳化的完整流程仍由 worker 協調，尚未全部收斂到 app。
修改這些模組時，不能假設 domain 全部都是無副作用的純計算。

`mcp.server.create_server()` 是 MCP 的組裝入口，讓 services 共用 workspace
與 SQLite store。匯入 server 模組不建立 runtime 檔案；建立 server 時才組裝執行資源。
背景 worker 使用相同 Python interpreter，並明確繼承解析後的 workspace 與資料根目錄。
Bundled 策略隨 Git／套件發佈；生成策略、行情與執行結果存於使用者 workspace。

### Repository 開發工具與研究原型

產品程式位於 `src/tradingdev/`；repository 的開發工具、測試與研究原型分開管理：

| 位置 | 目前責任 |
| ---- | -------- |
| `scripts/`、`.githooks/` | Shell 操作與 Git hook 入口，選定 repository 與 interpreter 後呼叫 Python 模組 |
| `tooling/quality.py` | 共用品質檢查與格式化，涵蓋 `src/`、`tests/`、`tooling/`、`research/` |
| `tooling/environment.py` | 按開發操作組合套件、工具鏈、策略 runtime 與模型服務的前置確認，供各入口共用 |
| `tooling/llm_check.py` | 選取並啟動真實模型測試情境，在實際執行前確認環境，保留 pytest 資訊與選測模式 |
| `tooling/requirements.py` | SRS 編號、欄位、導航與引用檢查；`DOCUMENT_SCOPES` 共用於需求文件驗證與審查責任選取 |
| `tooling/git/` | 候選 snapshot／合併 tree 的建立與驗證，以及 hook、本機 full、PR 檢查與分支清理流程 |
| `tooling/review_evidence.py` | 擷取固定 Git 版本的完整來源、diff 與內容身分 |
| `tooling/review_scope.py`、`tooling/review_plan.py` | 固定責任範圍、文件原文章節、完整變更涵蓋與有容量上限的審查規劃 |
| `tooling/review_docs.py`、`tooling/review_options.py` | 共用操作參數，協調模型批次、共同期限、清理與判定 |
| `tooling/process_guard.py`、`tooling/codex_binary.py` | 開發工具與測試共用的程序觀測／清理及 Codex 執行檔定位 |
| `tests/` | 測試案例、fixtures、MCP 與模型測試情境 |
| `research/` | 獨立研究原型；MAE 原型直接讀取指定行情檔案，與正式 Dashboard 分開 |

`tooling` 不直接匯入 `tests`；測試可直接使用共用工具。Git gate 與 PR 檢查共用候選
驗證服務，文件審查使用執行檢查的 checkout 內的工具讀取候選 Git 內容，
品質檢查與離線測試則使用候選 snapshot 內的入口與程式。
`tooling/`、`research/` 不隨產品 wheel 發佈，也不屬於產品 application services。
操作入口與檢查流程見[開發指南](docs/development.md)，研究原型用法見
[開發指南的 MAE 研究原型](docs/development.md#mae-研究原型)。

`.agents/skills/research-feedback/SKILL.md` 是外部開發助手的工程回饋流程入口。
助手讀取明示範圍內的 workspace 策略與證據，評估並整合可重用能力至適當模組，
同步更新使用端、探索入口、測試及契約。來源 revision 與歷史結果維持唯讀；共用程式
的改動仍可能影響匯入它的策略，revision 並不凍結全部相依程式。
此 skill 不屬於產品 MCP 常駐指引或 application services，不自行監看 workspace、
跨研究檢索或發布更新；多人權限與隔離亦不由 skill 提供。
來源選擇及操作方式見[研究成果工程回饋](docs/development.md#研究成果工程回饋)。

## 核心概念與身分

| 概念 | 意義與責任 |
| ---- | ---------- |
| Strategy revision | StrategyService 管理的固定程式、基礎設定、狀態與驗證證據 |
| Execution manifest | 一次執行的策略身分、有效設定、建構子參數與可選搜尋規格 |
| Execution plan | 固定 manifest、離線檢查證據及確認文件，另有期限與提交狀態 |
| Job | JobStore 保存的背景進度、程序身分與控制紀錄 |
| Run | RunService 查詢的一次已保存執行結果 |
| Artifact | ArtifactService 探索與讀取的檔案及索引 |
| ML dataset／model | 獨立 stores 管理的資料版本與訓練模型，不能當作 run ID |

同一 revision 可用不同參數執行多次，每次固定自己的 manifest，保留獨立結果。
實驗不改寫 revision 的程式、YAML、驗證證據或 current pointer；worker 使用已捕捉的
建構子值，不因後續預設值變更重新解釋請求。設定覆寫與完整性規則見
[Execution Manifest](docs/run_artifacts.md#execution-manifest)。

單次背景回測與完整最佳化以 job ID 作為 run ID；部分最佳化以 job ID 查詢試驗帳本。
固定比較由父 job 管理各候選 run，
沒有父工作的合併資金績效。直接回測 CLI 同步執行保存 run，但不建立背景 job。
策略、工作與結果的狀態各自獨立。

## 從策略建立到結果查詢

### 研究需求與有效設定

模型取得使用者的策略構想、標的、期間與 K 棒頻率，缺項須追問。
`app/research.py` 共用解析邏輯，供策略保存與計畫準備使用；`ResearchSpec` 與
共用政策負責研究選擇、模擬預設及來源，作者 YAML 負責策略、資料來源、特徵與執行設定。
後端檢查完整性與版本綁定，不能證明輸入來自真人。

明示的有效值優先，零成本保留；僅可繼承明確選定的 generated revision，不能從
內建範例補日期。政策不適用時須釐清。解析結果隨有效設定保存，worker 不重新套用
更新後的政策。直接回測 CLI 使用完整 YAML；直接 Python override helpers 移除失效來源標籤。

Walk-forward 設定的替換、模式與 UTC 日期包含關係由 schema／準備服務檢查；
正式資料能否形成有效分折，留待授權後執行。輸入與保存契約分別見
[研究輸入](docs/strategy_contract.md#research-inputs-and-simulation-policy)、
[Walk-forward 設定](docs/strategy_contract.md#strategy-walk-forward-settings)及
[研究設定來源](docs/run_artifacts.md#research-settings-and-provenance)。

### 建立與驗證策略

StrategyService 將新建與修訂分開，修訂在 lifecycle lock 內核對預期版本。
成功保存建立不可變 revision；metadata 記錄該版本證據，current 指向最新保存版本。
命名可自動產生，不能覆寫既有 namespace；新版本不繼承舊版本的執行資格。
StrategyService 保留 application 入口與狀態轉移，將清單、來源與版本讀取委派給
StrategyQueries；兩者共用 catalog、revision store 與 SQLite store。
Ruff／Mypy 的程序執行由 strategy_quality adapter 負責，使用 application 傳入的
共用品質設定；找不到執行檔、逾時與非零退出會轉成驗證診斷，不授予執行資格。

```mermaid
stateDiagram-v2
    [*] --> draft: save_strategy
    draft --> validated: validate_strategy 通過
    validated --> draft: 重新驗證失敗
    validated --> runnable: dry_run_strategy 通過
    runnable --> promoted: promote_strategy
```

驗證先做靜態政策與共用 Ruff／Mypy 檢查，再由 StrategyContractService 透過具期限的
worker 執行 SignalContractChecker。StrategyFactory 從首次匯入起把生成程式放入獨立
Linux 執行單元，在合成資料上檢查建構子、設定與訊號；dry-run 使用較長 fixture。
回覆綁定原請求，確認 host worker 與隔離單元清理後才保存證據；
異常退出、逾時及清理失敗皆不授予執行資格。實際執行也套用共用訊號框架檢查，
防止缺列、索引錯位、輸入修改或無效訊號進入引擎。檢查不證明沒有前視或符合自然語言構想。

BacktestService 在提交及執行時重新核對資格與來源完整性。明選 revision 不存在時
不退回 current；省略時於操作開始選定一次。Bundled 策略由 Git 管理，能力由註冊宣告
提供；generated／legacy 能力維持未知，不由 `fit()` 或狀態猜測。

StrategyCleanupService 共用 lifecycle lock，僅在明確授權後清理無引用的非 current
草稿；損壞或不明引用阻止清理。刪除不是跨檔案交易。狀態、參數實驗、舊格式復原與
清理回覆見[策略生命週期](docs/strategy_contract.md#lifecycle)。

### 準備、確認與執行回測

ResearchPreparationService 解析研究需求與已保存的研究說明，固定 manifest。
ExecutionPlanService 經 PreflightService 在受監督的子程序檢查程式與設定，確認
程序清理後才發布摘要與 HTML。準備不建立正式 job、不載入真實行情、不訓練或回測；
合成訊號檢查的適用範圍與未驗證項目隨收據保存。

模型提供白話規則與參數說明，後端提供有效值、執行假設及檢查證據；兩者共同固定於
確認文件。助手交付摘要與 HTML 連結，結束回合等待後續明確決策。
`domain.presentation.confirmation_contract` 定義確認文件型別、固定章節及參數說明的
完整性驗證；`confirmation.build_confirmation_document` 將固定設定與檢查證據轉成
該文件。文字與 HTML adapters 共用同一份確認文件。
`submit_execution_decision` 只在同意時提交原計畫；拒絕或修改要求取消計畫。
修改需重新準備、檢查與確認。最初請求、提問或沉默不構成同意。

伺服器信任助手如實轉交決策，不能證明人類已閱讀或同意。提交仍檢查期限、來源、
設定與執行政策，並以 reservation 防止重複啟動。確認不依賴 MCP elicitation；
完整狀態、重試與啟動後寫入失敗的恢復規則見[執行計畫契約](docs/execution_plans.md)。

```mermaid
sequenceDiagram
    participant User as 使用者
    participant MCP as MCP／助手
    participant Plan as 計畫與離線檢查
    participant Submit as ExecutionSubmissionService
    participant Worker as Supervisor／Worker
    MCP->>Plan: prepare_* 固定需求與證據
    Plan-->>MCP: 簡短摘要與 HTML 連結
    MCP->>User: 交付確認單，結束回合
    User-->>MCP: 後續明確同意
    MCP->>Plan: submit_execution_decision
    Plan->>Submit: 已確認 manifest
    Submit->>Worker: 建立 job、啟動受監督程序
    Worker->>Worker: 核對 manifest，執行並保存結果
    Worker-->>MCP: 可查詢的狀態與結果
```

JobService 與 OptimizationService 共用 ExecutionSubmissionService；提交使用已準備
內容，不重新讀 YAML 或選 current。DataService 供應行情與特徵，BacktestService 協調
訊號與引擎：signal 使用 VectorBT，volume 使用逐筆模擬帳本，walk-forward 逐折訓練與評估。
工作狀態名稱不證明下載已完成；done 表示結果保存成功，程序清理由 supervisor 另行確認。

固定比較由 ComparisonService 使用一份不可變行情快照，逐組重新解碼並建立獨立策略、
引擎、資金與亂數串流。ComparisonStore 保存全部候選帳本，RunPublisher 核對父子身分
後發布成功結果；失敗與未執行組別不被省略。此流程限普通回測。生成策略實例各自有獨立
程序；內建策略與可信主程式的任意 Python 全域狀態不在隔離保證內。
範圍與恢復規則見[固定比較證據](docs/run_artifacts.md#fixed-comparison-evidence)。

直接回測 CLI 同步呼叫 BacktestService，再透過 ArtifactService 保存；命令本身是明確執行
請求，不使用 MCP 對話確認或背景 supervisor，但共用策略資格、manifest 與績效模型。
`tradingdev-codex` 對話入口透過 MCP 操作，仍使用上述執行計畫、確認與背景工作流程。

### 最佳化執行

OptimizationService 固定搜尋候選、訓練／測試期間、目標方向與逐次試驗時限。
使用者同意後，worker 取得行情並以受監督的程序執行試驗。每個執行程序提交自己的
嘗試狀態及結果，SQLite 試驗帳本提供搜尋覆蓋、排序及測試期的共同來源。
明確可歸因的候選失敗可繼續搜尋；程序、資料或儲存故障停止工作，保留已提交的試驗。
完整網格有有限目標才執行保留測試，測試期完成後才進入 RunPublisher 的完整 run 驗證。

最佳化使用自己的日期切分，不能同時套用 walk-forward；固定參數比較不授權擴大搜尋。
保存與查詢見[最佳化試驗產物](docs/run_artifacts.md#optimization-trial-results)。

### 保存與查詢結果

JobStore 委派 RunPublisher 發布結果後更新工作狀態；固定比較另由父工作保存候選帳本。
CLI 透過 ArtifactService 保存，均共用儲存 adapters。SQLite 保存索引、摘要與最佳化試驗帳本，
檔案系統保存 revision、快取與完整產物；發佈與狀態更新是不同操作。

RunService 查詢 run 與指標，ArtifactService 探索與讀取檔案。Dashboard 使用已保存的
PipelineResult；最佳化沒有此 pickle。讀取依保存路徑，不用目前 YAML 重建查找鍵，
也不自動重用以前的回測。格式與查詢入口見[執行產物契約](docs/run_artifacts.md)。

`ArtifactService.read_download` 透過 storage adapter 核對已登錄確認單、報告與
研究 ZIP 的固定路徑、雜湊及實際回傳 bytes；研究封存附件也共用此讀取驗證。
MCP 的 `ArtifactDownloads` 負責短效網址、公開路徑設定與 HTTP 回應，簽發與下載
皆核對檔案，不把簽章或 HTTP 細節放進 domain。完整交付與失敗語意見
[產物交付契約](docs/run_artifacts.md#reading-artifact-delivery)。

### 行情快取

`app/data_manager.py` 協調年度查詢及更新，`domain/data/cache.py` 定義共用身分、
成功查詢證據與重用政策，`adapters/storage/data_cache.py` 負責鎖、不可變版本與
原子發布。DataService 的正式載入、只讀載入、可用性、檢查與取樣使用相同身分；
不同來源、市場及原始代號不共用價格。保留舊版本讓讀取端不受更新影響，
失敗只清除尚未發布的檔案。格式及限制見
[Market Data Cache](docs/run_artifacts.md#market-data-cache)。

`app/dvol_data.py` 管理未指定本機路徑的 DVOL，以精確 UTC 區間建立獨立查詢身分，
共用上述儲存與重用政策。準備執行時固定資料目錄，仍保留自動查詢模式；
取樣取得的少量特徵不發布為完整研究期間的查詢證據。
DataService 委派 FeatureData 統一特徵載入、受限合併及來源檢查，委派
DatasetInspection 組裝行情快取內容與指紋報告；資料查詢身分仍由 DataService
使用共用 DataIdentity 產生，檢查不下載或發布資料。

### 獨立行情取樣

DataService 的受限取樣與確認前檢查分開。Application 層協調日曆窗口，storage
adapter 分批讀 Parquet；crawler 的 `fetch_sample` 在請求與解析階段限制資料量，
不能先完整載入再截斷。特徵只投影所需欄位，合併前檢查筆數膨脹，不改寫來源快取。

一般載入與取樣均精確對齊時間戳，只沿用較早的有效特徵；開頭缺值保持未知，
不能用較晚觀測回填。

行數限制不等於網路流量等比例縮減；Binance Vision 仍需下載選中的壓縮檔。
下載、解壓與列數各有獨立上限，超量失敗。完整取樣及來源限制見
[Bounded Historical Sampling](docs/run_artifacts.md#bounded-historical-sampling)。

## ML 訊號研究

XGBoost 與 SafetyVolume 的逐棒重訓與評分共用 `domain/ml/rolling_gen` 核心；
特徵元件宣告上下文需求，解析後的視窗供規劃、執行與紀錄共用。核心保存類別機率與
不可用狀態，策略負責方向決策或 Risk Gate／持倉管理；規則見 [ML 研究契約](docs/ml_research.md#rolling-retraining-schemes)。

`mcp/tools/ml.py` 透過 MLService 與 MLTrainingService 協調已登錄特徵、資料集、
模型與評估，`app/contracts/ml.py` 驗證回覆。這是同步研究介面，沒有 execution plan、
背景 backtest run，亦未接入回測 CLI／Dashboard；資料集、模型與排行榜 stores 不使用
SQLite 的 run／artifact 索引。

Dataset 建置只讀既有行情，feature cache 重用特徵計算，MLDatasetStore 發布不可變
內容版本。訓練保存全部列擬合的最終模型；分折驗證另建模型，不能把其分數當成最終模型的
樣本外證明。Predict 核對特徵順序與模型 bytes 後產生完整訊號，MCP 只傳預覽頁；
evaluate 在完整單商品資料上驗證對齊與持有區間，簡化無成本統計不等同完整交易回測。
手設門檻規則直接讀取資料集特徵，與模型訊號共用評估器；方法家族區分候選身分，
不改變評估條件。

Leaderboard 保存候選參數、訊號指紋與評估條件，依 domain 指標目錄排序；不可用值保留
`null` 與座標。跨參數軸比較是明確 opt-in。候選跨 horizon 或明確要求驗證原始行情
基線時，排名及固定報告共用 `app/ml_surface.py`，以資料集保存的證據分別核對原始
行情基線與評估列。
特徵快取可淘汰，資料版本與模型有自己的保存規則。輸入、驗證解讀、身分、鎖與保存限制
集中於[ML 研究契約](docs/ml_research.md)，避免在此重列欄位與演算法。

`app/report_ml.py` 依明示候選 ID 讀取當前排行榜，交 `domain/ml/report.py` 投影並計算
完整排名；保留缺值、評估座標及警語，不重新預測或訓練。`ReportService` 將選取內容固定
於獨立 ML 報告，或附加到 run／固定比較報告；MCP 與 report CLI 共用此路徑。
HTML 的 ML 章節與回測證據分開呈現，不推定模型與 run 的來源關聯。排行榜後續替換不改變
已發布報告；未選取 ML 時不加入 ML 區塊。新產生報告採用包含研究解讀與診斷的
模板 24，報告身分會隨內容與模板更新；已保存報告不改寫。
這些報告另行登錄為 SQLite artifacts；選取、呈現與保存契約見[報告文件](docs/reports.md)。

## 績效計算與對話摘要

`DiagnosticService` 透過 domain 的 `DiagnosticProvider` 介面讀取已驗證的 scope 與
原始觀察值。初期 `MaverickDiagnostics` 僅移植適用的規則式建議，不引入上游 server、
行情與資料庫服務；來源版本與 MIT 授權隨 adapter 保存。後端提供指標證據與限制，
外部 LLM 提供結論、觀察、假說與下一步，報告核對結果引用並保存兩者。這個介面
保留後續分析完整交易與行情的空間，不以初期彙總指標限制自研診斷輸入。

| 元件 | 責任 |
| ---- | ---- |
| `domain/backtest/metrics.py` | 對齊引擎觀察值、整合統計套件、記錄不可用原因 |
| `domain/statistics/breakeven.py` | `summarize_breakeven` 從呼叫端提供的交易、評估起訖及可選本金計算損益兩平勝率與周轉成本；欄位、未平倉處理及不可用條件見模組與函式 docstring。獨立計算，不讀取 runs，尚未接入績效發布或 MCP 查詢 |
| `domain/performance/catalog.py`、`sampling.py` | 指標定義、單位、最佳化方向與取樣適用性 |
| `domain/performance/artifacts.py`、`adapters/storage/performance.py` | 具型別的結果範圍、定義快照與觀察值保存／核驗 |
| `domain/benchmark.py`、`app/benchmark_service.py` | 資產類別對應的固定基準、買進持有計算與基準行情取得；報告只讀已保存基準 |
| `domain/statistics/plateau.py` | `trials_from_metrics` 將呼叫端提供的全部參數試驗列轉為網格與有效試驗，`plateau_summary` 比較最佳試驗與鄰域平台；輸入、失敗列及缺失網格語意見模組與函式 docstring。獨立計算，不讀取 jobs／runs，尚未接入績效發布或 MCP 查詢 |
| `domain/statistics/comparison.py` | 對齊呼叫端提供的報酬序列，以 stationary bootstrap 提供配對統計量差值推論與各序列統計量的個別區間；公開入口為 `align_returns`、`difference_interval`、`difference_interval_outcome`、`interval_table` 與 `interval_table_outcome`。各入口的回傳契約、不可用結果、區塊長度、輸入邊界與解讀限制見 `domain/statistics/comparison.py` 的模組與函式 docstring。獨立計算，不讀取 runs，尚未接入績效發布或 MCP 查詢 |
| `domain/statistics/benchmark.py` | 由呼叫端提供的價格序列產生買進持有報酬並對其做策略迴歸比較；公開入口為 `buy_and_hold_returns` 與 `compare_to_benchmark`，結果欄位型別為 `BenchmarkComparison`。屬研究用序列分析，與 `domain/benchmark.py` 的產品基準分屬不同職責。欄位定義、與產品基準的數值差異及解讀限制見 `domain/statistics/benchmark.py` 的模組與函式 docstring。獨立計算，尚未接入績效發布或 MCP 查詢 |
| RunService／JobService | 摘要、探索與詳細查詢；摘要大小不決定保存哪些指標 |

QuantStats 提供報酬／風險統計，VectorBT 提供已平倉交易統計，專案負責成本記帳與資料轉換。
不可用值保留原因，不能以零代替。解讀使用當次保存的定義、單位與設定；詳細查詢不重跑策略
或載入 pickle。Walk-forward 摘要是分折分布，最佳化區分 trial 與樣本外結果。
完整取樣、年化、數值及比較限制見[績效產物契約](docs/performance_artifacts.md)。

需要從原始觀察值獨立計算時，`domain/statistics/periods.py` 提供 UTC 期間權益、
損益及報酬序列，基準是首筆觀察值；`performance.summarize_performance` 接受權益、
時間戳、交易紀錄與明示的原始本金、頻率、年化及無風險利率設定，依績效契約獨立驗算。
`sharpe.py` 的 `sharpe_ratio` 計算比率，`sharpe_interval_outcome` 提供 bootstrap
區間或不可用原因；`probabilistic_sharpe_ratio` 與 `deflated_sharpe_ratio` 分別計算
相對指定門檻及搜尋調整門檻的機率。各函式的 docstring 維護報酬輸入、年化設定、
不可用條件與推論假設。`comparison.py` 提供配對差值推論與各序列的個別區間，
`benchmark.py` 提供研究用的買進持有報酬與策略迴歸比較，後者與前述產品基準的
保存值不是同一種測量；兩者的公開入口、回傳契約與解讀限制見各模組與函式 docstring
（`domain/statistics/comparison.py`、`domain/statistics/benchmark.py`）。
摘要保留金額與比例回撤；帳戶權益的整段淨損益以原始本金為基準，累積 P&L
（`init_cash=None`，例如 Volume 模式）以零為基準，首筆觀測後的權益變化另行命名。
`summarize_performance` 依當次無風險利率計算摘要的 `sharpe_ratio` 欄位，
`return_information_ratio` 欄位另保留未扣無風險報酬的比率；底層
`sharpe.sharpe_ratio` 只計算傳入的報酬序列，不自行扣除無風險報酬。
已平倉與包含未平倉市值的交易統計分組提供，不將未發生成交的出場費假設為已扣除。
這些計算不讀取 run／artifact，也未接入績效發佈或 MCP 結果查詢；不能自行證明呼叫端
提供的原始資料完整或來自同次執行。保存結果仍由上述績效產物流程負責。

### 診斷擴充與替換

`domain/diagnostics.py` 定義 `DiagnosticProvider.diagnose(DiagnosticContext)`。
`DiagnosticContext` 保留原生 `PerformanceScope` 與 `ScopeObservations`（可缺少），
包含保存的交易、權益與帳戶觀察值；不把資料壓成 Maverick 專用輸入。
`DiagnosticService` 的預設 providers 是共同組裝點，MCP 查詢與 ReportService
都經此服務。要替換初期 adapter，可實作相同 Protocol 並在此更換預設組合；
測試也可直接注入 provider tuple。不是把 Maverick 套件當成 domain 基類。

深度診斷應保留 provider／版本、證據、不可用原因與限制，先加入可重現的離線測試。
目前 evidence 是具單位的數值指標；若需時間區段、交易索引或情境分類，應擴充正式
證據模型並同步更新 HTML、工具 schema 與測試，而非把新事實藏在 LLM 文字中。
行情型態／波動診斷若需要 OHLCV，須另以固定資料身分載入並驗證對齊後加入 context；
目前 context 沒有承諾原始行情。資料不足回報 unavailable，不由 provider 隱式抓取行情或回測。
規則或契約改變須更新 provider／報告版本；既有報告及封存保持原樣。

### 研究筆記與可攜封存

對話端 LLM 負責 `ResearchNoteDraft`：目前需求、階段摘要、報告引用與後續研究。
後端 `ResearchNoteService` 供 MCP 與 CLI 共用，`ResearchNoteStore` 在 SQLite
保存目前版本，以 expected_version 防止覆寫其他編輯。筆記不替代不可變的
ResearchEvidence、execution manifest 或執行確認；追加需求不會改動已執行設定。

`ResearchArchiveStore` 核對所選版本與指定 HTML 附件的登錄、路徑及 SHA-256，產生
固定 HTML／JSON／報告 ZIP。附件限所引用的研究報告與執行確認 HTML，
不遍歷打包整個工作區。
`adapters/reporting/notebook.py` 負責閱讀頁，儲存層負責原子發佈與失敗清理。
HTTP 的自訂下載 route 使用每個 server instance 的短期簽章能力網址，
驗證後才交付同一份 ZIP bytes；重新取得舊封存的網址只讀保存 ZIP，不依賴目前筆記或來源報告。
stdio 回傳本機路徑，不假設客戶端能下載。
未來可替換 storage／delivery adapter，但應保留版本衝突、來源完整性、容量限制及
獨立閱讀的契約；聊天供應商的傳輸協定留在客戶端整合層。
欄位與限制見[報告與筆記契約](docs/reports.md#research-notebooks-and-archives)。

### 可見對話紀錄

`ConversationService` 與 `domain/conversations.py` 提供不綁定供應商的對話、回合與
可見項目模型；客戶端 adapter 負責將實際事件轉換並持續保存，MCP server 不推測
客戶端外的訊息。SQLite 以版本核對避免並行覆寫，已完成回合與項目不可改寫。
執行結果與紀錄完整性分開保存，中斷或缺漏不冒充完整紀錄。私有推理不屬於保存內容。

`adapters/reporting/conversation.py` 產生可選取文字及展開工具內容的離線 HTML；
`ConversationExports` 發佈固定 HTML／JSON，登錄到共用 artifact 查詢入口。
筆記與對話匯出共用 `WorkspaceFiles` 的受限讀取，個別封存格式及身分保持獨立。
精確生命週期與限制見[對話紀錄契約](docs/reports.md#recorded-conversations)。

Codex CLI 專用入口由 `adapters/clients/` 驅動 app-server，轉譯可見事件並呼叫
共用保存服務；與模型私有推理分離，不要求 MCP server 觀察客戶端訊息。
續接核對原生歷史、固定設定與生命週期鎖；串流中斷保留部分內容，來源不一致時拒絕追加。
此入口只啟用 TradingDev MCP，通用 MCP 客戶端仍沿用既有連接方式。

## 一致性、程序管理與保證範圍

### 歷史查詢與報告組裝

TradeHistoryService 從已核驗 JSON 與 manifest 提供參數、交易、委託及帳戶查詢，
保留 scope、來源與分頁身分；不能用配對交易補造缺失委託，或將聚合 fold 變成權益序列。
Signal 的原生 order attempts 與帳戶紀錄由共用執行記錄模型保存，使用 generic accounting，
不宣稱是交易所錢包、保證金或資金費率帳本。

ReportService 組裝已保存資料；domain/presentation 定義章節與範本，
adapters/presentation 提供共用文件外殼，adapters/reporting 負責 HTML／圖表與不可覆寫發佈。
MCP、report CLI 及 Dashboard 共用此服務。模型只供應評語與選擇，不替換計算值。
QuantStats 原生報告是可追溯衍生分析，不能覆寫歷史指標。

報告以總覽連到各結果明細，固定比較保留所有候選與失敗原因。可選章節只控制畫面，
不裁減 HTML 內嵌資料；分享時須考慮完整設定、交易紀錄及選取的 ML 證據。最終回覆須交付報告連結、
所選結果與明細操作方式，失敗時沿用保存結果重試報告。互動、完整精度、匯出、
版本與發佈完整性集中於[報告契約](docs/reports.md)。

### 設定與儲存一致性

Manifest 是 worker 的執行設定來源，`config.yaml` 僅為檢視副本；所選 revision 的
基礎設定仍用來核對完整性。非參數策略宣告必須與 revision 相符，參數實驗各自固定有效值。

不可覆寫檔案與 SQLite 紀錄共同綁定身分，但兩者不是跨資源交易。Hash 檢查內容一致性，
不能防禦同時控制檔案與資料庫的寫入者。忙碌標記、失敗恢復與歷史缺證據的處理見
[執行產物](docs/run_artifacts.md)及[績效發佈契約](docs/performance_artifacts.md)。

### MCP 回覆契約

`mcp/prompts.py` 保留階段流程、契約探索、授權與成果交付；詳細契約由標準 MCP tools
按需提供。Schema、模擬政策、指標及靜態規則取自正式定義，不整份載入 Markdown，
也不假設客戶端可讀 server 檔案或自動載入 resources。量測與機械檢查入口見
[開發指南](docs/development.md#量測模型上下文)。

StrictMCPServer 拒絕未知頂層參數，application DTO 驗證回覆；動態策略參數仍保有
明確 JSON 邊界。客戶端按 `tools/list` 的 `outputSchema` 讀取 `structuredContent`：
單一物件在根層，清單及成功／失敗聯集在 `result`。SDK 序列化須保留 wire 別名。

預期應用失敗有穩定 code／diagnostics；參數錯誤、未預期例外與無效回覆使用 MCP error。
`isError` 不能單獨判斷應用是否成功；未預期內部原因留在 server 日誌。
可成長的探索清單以 Page 回傳摘要，詳細內容另查；分頁不提供跨寫入的一致快照。
欄位與排序見[探索分頁](docs/run_artifacts.md#discovery-pages)。

同步工具由 SDK 在線程執行，SQLite connection 不跨 service 呼叫共用。
工具唯讀／破壞性／冪等標註描述副作用，不能當成授權或隔離保證；例如狀態查詢可能
更新失聯 job，而驗證會執行 Python。模型 store 的並發與可信 pickle 邊界見
[ML 產物契約](docs/ml_research.md#trained-models)。

### 程序控制與清理

ProcessRunner 先核對 supervisor 身分，再建立 worker 程序群組。Supervisor 負責
正常退出、失敗與取消後的清理；控制使用唯一 launch 身分，不直接信任資料庫中的數字 PID。
執行中的 job 取得清理確認後才能標記 cancelled；尚未啟動的 queued job 可直接取消。
JobService 在狀態查詢時先透過 JobLifecycle 恢復失聯 worker 的狀態，再交由
JobQueries 讀取結果明細與組裝回覆；清單查詢只讀取已保存狀態。取消與恢復都保留
清理後的狀態重讀及 store 的終態保護，避免覆寫並行完成的工作。

可信 host worker 的監督依賴 POSIX process groups 與 `waitid(WNOWAIT)`，
以尚未回收的直屬 child 維持身分。生成 Python 不進入這些 host 程序：所有入口共用
StrategyFactory 的 context-managed 生命週期，透過有界資料協定呼叫容器中的實例。
fit 狀態與亂數 context 保留於同一實例；主程式不反序列化 worker 提供的 Python 物件。

Supervisor 停止 host worker 後，依[容器執行 adapter](#容器執行-adapter)的清理契約
恢復該次工作的容器控制紀錄，不以程序群組已空代替容器清理。
跨 MCP session 的取消沿用同一 launch 控制身分。

強殺 host supervisor 仍可能使可信 host worker 的清理證據缺失；不得據此宣稱整個工作
已清理。容器隔離與獨立期限的保證範圍見[策略安全模型](docs/strategy_contract.md#security-model)。
控制檔案與證據見[Workspace Layout](docs/run_artifacts.md#workspace-layout)。

### 容器執行 adapter

[`DockerRuntime`](src/tradingdev/adapters/execution/container_runtime.py) 是產品
`StrategyFactory`、直接整合與測試共用的容器執行入口。隔離保證與可信元件的範圍由
[策略安全模型](docs/strategy_contract.md#security-model)說明；
建置映像、環境設定與測試操作見[開發指南](docs/development.md#容器建置與驗證)。

產品 `StrategyFactory` 透過
[`generated_strategy.inspect_parameters` 與 `open_strategy`](src/tradingdev/adapters/execution/generated_strategy.py)
操作容器中的生成策略；整合者或測試也可直接呼叫這些 helper。前者擷取或驗證建構參數，
後者在 context 內保留訓練狀態與亂數序列，
離開時確認容器清理。兩者接收已固定來源雜湊、類別與 runtime 的
[`GeneratedProgram`](src/tradingdev/domain/strategies/execution.py)，以純資料傳送來源與設定。

跨容器訊息由 [`StrategyRequest`、`StrategyResponse` 與 `Operation`](src/tradingdev/adapters/execution/strategy_protocol.py)
定義請求／回覆 schema、協定版本及可用操作；整合者應使用這些正式定義編解碼，
而非自行維護欄位或操作清單。Proxy 逐次核對 request ID、操作、來源雜湊與類別名稱，
拒絕不屬於該次請求或固定策略身分的回覆。

直接呼叫時，`inspect_parameters(..., capture=True)` 擷取缺少的宣告預設、核對 keyword
binding 並回傳完整設定 dict；它不呼叫建構子，但仍會載入及執行模組層程式。
`capture=False` 與 `open_strategy` 要求每個具名參數都有顯式值，包括已有預設的參數。
呼叫端可先 capture，再把回傳 dict 交給驗證或開啟策略；也可自行提供完整設定。

這些 helper 的設定與 proxy hook 回傳值遵循
[`finite_strategy_value`](src/tradingdev/domain/strategies/execution.py) 的有限 JSON 邊界。
`get_parameters()` 須回傳 JSON object；`rolling_trace()` 須回傳 JSON object 或 `None`。
內含 tuple、NumPy 陣列或其他不受支援物件的 Python dict 也會被拒絕，不自動轉型。

DataFrame 透過 [`FramePayload`、`encode_frame` 與 `decode_frame`](src/tradingdev/adapters/execution/frame_codec.py)
跨越邊界。欄名必須是唯一字串，index／columns 的名稱只能是字串或 `None`。
支援 RangeIndex、DatetimeIndex 及一般數值／字串 index；MultiIndex、categorical／自訂
extension dtype 與任意 Python 物件不受支援。`attrs` 與 object 儲存格須為有限 JSON 值。
數值、nullable、datetime 與字串的支援型別由該 codec 定義；無法表示的型別或值會失敗，
不以字串化或截斷替代。時區限已安裝的 IANA 名稱或明示 UTC offset，不接受 dateutil
路徑或任意 timezone 物件。

[`strategy_worker.serve`](src/tradingdev/adapters/execution/strategy_worker.py) 回報操作錯誤後
會結束該 session；呼叫端須離開 context 完成清理後重新開啟，不能沿用已失敗實例的
訓練狀態繼續操作。底層傳輸或程序失敗也不保證可恢復同一實例。
底層容器呼叫契約如下。

1. 建立 `DockerRuntime(control_root)`，以可保留的獨立目錄保存執行歸屬紀錄。
   Adapter 固定所選本機 Docker 端點，後續 Docker context 變更不會改變清理目標。
   `resolve(image=..., limits=...)` 核對 daemon 能力與預建映像，回傳固定 image ID
   及有效政策的 `RuntimeSpec`，不自動下載映像。政策欄位與限制由
   [`RuntimeLimits`](src/tradingdev/domain/runtime.py)定義。
2. 使用 `with runtime.open(spec, command) as session` 啟動映像內的命令及管理生命週期。
   `command` 是程式與參數序列，所需檔案須已在映像內；此入口不掛載 host 工作區。
   每個 context 擁有獨立執行單元及歸屬紀錄。
3. `session.exchange(message)` 依序交換單行 `bytes`：傳入內容不含換行，回傳內容移除
   結尾換行。命令須每次輸入只產生一行 stdout，診斷使用 stderr；超量與已偵測的
   額外 stdout 會失敗。診斷保留有上限的尾端，發生截斷時標示。
4. Context 正常結束或拋出例外時，先停止 attach client，再移除並核對整個執行單元。
   清理核對 daemon 身分、owner 與不可變 container ID；只有確認清理後才移除歸屬紀錄。
   發生 `IsolatedRuntimeError` 時檢查 `cleanup_verified` 與 `control_path`；清理未確認
   就保留控制目錄，不把操作失敗與清理完成視為同一件事。
5. 原呼叫端停止後，可在原 daemon 上呼叫 `runtime.recover_abandoned()` 重試復原；
   `control_root` 使用錯誤所列 `control_path` JSON 的所在目錄。
   它只接手已釋放 lease 的單元，跳過仍持有 lease 的執行；身分不符、紀錄不明或
   daemon 不可用時回報失敗並保留紀錄，不承諾立即清理。

### 可重現性與策略執行安全

策略 YAML 頂層 `random_seed` 建立執行內的 Python／NumPy 亂數 context；walk-forward
按 fold 順序共用該次串流，trial 各自獨立。策略須使用 domain.randomness，第三方種子須
明確傳入；不控制任意全域 RNG。獨立 ML 訓練另用模型的 `random_state`，不能互相推斷。

Manifest、來源 hash、資料指紋及版本提供追蹤；生成策略另固定容器 image ID 與資源政策。
可信 host 依賴、硬體與任意全域 RNG 並未全部封存，不承諾跨環境完全重現。
舊結果缺少證據時明確回報，不能猜測或自動重算。
生成程式使用無主機掛載、無外部網路及有資源上限的 Linux 容器；workspace、靜態驗證
與操作同意本身不是隔離。信任邊界及限制見[策略安全模型](docs/strategy_contract.md#security-model)。
