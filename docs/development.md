# 開發指南

所有命令均在 repository 根目錄執行。產品操作見 [README](../README.md)，
代理開發政策見 [AGENTS.md](../AGENTS.md)。

## 環境與常用命令

開發環境使用 Python 3.14；版本範圍以 `pyproject.toml` 為準。
首次設置或從較舊的 Python 升級：

```bash
uv sync --locked --all-extras --python 3.14
./scripts/install-git-hooks.sh
```

安裝器會調整本機 Git 設定，摘要見文末的[注意事項](#注意事項)。
文件檢查也使用本 repository 的 `.venv` 與 `dev` 群組工具，不依賴全域安裝。
一般文件與簡報專項不要求 Dashboard extras 或 Codex 登入；簡報另需下述 Node.js
工具鏈。其他品質與完整檢查仍需上述 `--all-extras` 環境。

| 命令 | 用途 |
| --- | --- |
| `./scripts/rebuild-env.sh` | 重建 `.venv`，依 `uv.lock` 安裝鎖定版本。 |
| `./scripts/format.sh` | 自動修正 lint 與格式，會修改檔案。 |
| `./scripts/check-docs.sh` | 唯讀檢查 Markdown 與本機檔案／標題連結；不呼叫模型或執行 Python 品質檢查、pytest。 |
| `./scripts/check-presentations.sh` | 檢查文件、簡報重建一致性、頁碼與離線資源；不呼叫模型或完整 pytest。 |
| `./scripts/check-fast.sh` | 唯讀檢查 Ruff、格式、strict Mypy 與上述文件品質。 |
| `./scripts/check-full.sh` | 一律執行快速品質檢查與完整離線 pytest，不依變更類型縮減；不呼叫模型。 |
| `uv run --all-extras pytest` | 完整離線測試，包含真實 MCP／worker；不呼叫模型。 |

`scripts/` 保留 Shell 操作入口，`.githooks/` 保留 Git hook 入口；Python 實作
集中在 `tooling/`，以同一個 `.venv` interpreter 執行 `python -m tooling...`。
`tooling/quality.py` 共用格式化與唯讀檢查流程，Python 檢查範圍為 `src/`、
`tests/`、`tooling/`、`research/`；Markdown 檢查涵蓋根目錄與 `docs/`。

Git 檢查與分支清理位於 `tooling/git/`，Codex 文件審查位於 `tooling/review_docs.py`。
Commit 與候選檢查依不可變 Git tree 分為一般文件、簡報專項與程式變更。
前兩種只執行對應 snapshot；程式候選由目前 checkout 的 reviewer 讀取候選
Git 內容，再執行候選內的品質與離線測試。分流條件見[更新文件](#更新文件)。

`tests/` 保留測試案例、fixtures 與 MCP／模型測試情境；`research/` 保留獨立
研究原型，用法見 [MAE 研究原型](#mae-研究原型)。`tooling/` 與 `research/`
不隨產品 wheel 發佈，Python 程式仍使用相同的 Ruff／strict Mypy 政策。

### 依賴相容性限制

FLAML 2.7 的安裝 metadata 接受 Python `>=3.10`，但
[上游 README](https://github.com/microsoft/FLAML/blob/v2.7.0/README.md)
宣告的範圍仍為 `>=3.10,<3.14`；
[AutoML extra](https://github.com/microsoft/FLAML/blob/v2.7.0/pyproject.toml)
另限制 XGBoost `<3.0.0`。本專案安裝 FLAML 基礎套件並明列所需的模型依賴，
採 Python 3.14 與 XGBoost 3；可成功解析依賴不代表此組合獲得上游完整支援。

專案離線測試逐一強制使用 LightGBM、XGBoost、Random Forest 與 Extra Trees，
驗證真實訓練、holdout、預測及機率，另驗證 CatBoost 搜尋與檔案清理。
驗證範圍限於本專案使用的分類流程；其他 FLAML 功能、平台或模型須另行驗證。
更新此組合時須保留各 learner 的獨立驗證；實際版本以 `pyproject.toml` 與 `uv.lock` 為準。

## 更新文件

文件分工與審查判準見 [AGENTS.md 的政策與溝通](../AGENTS.md#政策與溝通)。
依變更影響更新既有文件：

- [README](../README.md)：使用者的安裝、設定、操作結果及必要限制。
- [架構](../ARCHITECTURE.md)、[策略契約](strategy_contract.md)與[執行產物](run_artifacts.md)：服務分工、介面、格式及相容性。
- 本指南：開發環境、檢查命令與 Git／PR 操作。

直接描述目前操作與適用範圍；同一事項集中說明，其他位置以連結引導。
提交前依文件讀者審閱差異，確認語意、命令正確性與內容歸屬，保留必要操作與
資料限制。`check-docs.sh` 檢查 Markdown 與本機檔案、Markdown 標題連結，
不執行文件中的命令，也不連線檢測外部 URL；通過不代表操作內容已實測。

分流政策以 [AGENTS.md](../AGENTS.md#git-hooks-與本機-pr-審查) 為準。一般文件接受
`README.md`、`ARCHITECTURE.md`、`docs/**/*.md`；已登錄簡報的入口與來源 Markdown
優先進入簡報專項，連同對應 HTML／CSS、空的 `docs/.nojekyll` 及通過內容驗證的
VS Code 簡報工作，確認來源與產物一致。不能只憑 `.html`、`.css`、task label 或
`.vscode` 路徑將任意內容視為文件。
簡報工作須符合固定命令與結構，其他工作須完全未變；tasks 文件頂層只允許
`version` 與 `tasks`，額外的 `options`、`inputs` 等設定即使未變也走程式分流。

分類比較固定 Git tree 的非空差異，改名核對兩側路徑；兩側仍存在的檔案均須為
mode `100644` 的一般檔案，才可進入上述路徑。政策、建置工具與 lock、未知設定、
產品程式、可執行檔、符號連結、含這些項目的混合變更及無法確認的差異都走程式分流。

一般文件與簡報專項的 commit、PR 及本機候選只跑對應 snapshot，不執行 Ruff、Mypy、
自動 Codex 審查或完整 Python pytest。文件語意仍由差異審閱確認，簡報操作與版面
仍須在瀏覽器檢查。政策、程式、依賴與測試基礎設施變更須依
[執行時機](../AGENTS.md#執行時機) 完成相關測試、完整檢查與獨立階段審查。

### 簡報的編輯與輸出

[專案介紹簡報](talks/tradingdev-introduction/README.md)以 Marp Markdown 編輯，
`theme.css` 保存版型，`index.html` 是提交至 Git 的瀏覽器產物。
編輯時另需 Node.js 18 以上與 npm。Marp 及其傳遞依賴由 `tooling/marp/package-lock.json`
鎖定，初始化工具鏈後，手動建置、VS Code 與 snapshot 使用相同建置入口：

```bash
uv run --no-sync python -m tooling.presentations setup
```

Shell 入口 `./scripts/presentations.sh setup|build|watch|check` 呼叫相同模組，
同樣使用 repository 的 `.venv`。
`setup` 安裝限時 300 秒；`build`、`check` 及每次 `watch` 重建的編譯限時
120 秒。簡報 CLI 未提供期限覆寫選項，逾時會失敗並清理其啟動的程序。

初始化會下載 npm 套件；更新工具鏈時須明確更新 lock 並重新初始化。一般 build／check
不安裝或升級依賴，缺少工具或安裝與 lock 不符會失敗，不會退回全域 Marp。
Snapshot 使用開發 checkout 的已安裝工具鏈，並核對其 lock 與候選一致；需要指定
另一套已初始化工具鏈時，可設定 `TRADINGDEV_PRESENTATION_TOOLCHAIN` 指向其目錄。

#### 在 VS Code 儲存時自動編譯

[`.vscode/tasks.json`](../.vscode/tasks.json) 已提供 `slides: watch` 背景工作，
以專案根目錄為工作目錄，只編譯這份 `slides.md`，並監看它使用的 `theme.css`。
啟動時先輸出一次，之後每次儲存這兩個檔案都會更新同目錄的 `index.html`。

1. 完成工具鏈初始化，用 VS Code 開啟整個 repository 資料夾，確認 `uv` 與 Node.js 在 PATH 中。
2. 在可信任的工作區，首次出現自動工作提示時選擇 **Allow Automatic Tasks**。
   若先前已拒絕，可執行 **Tasks: Manage Automatic Tasks** 改為允許。
3. 重新開啟資料夾後會自動啟動。若要在目前視窗立即啟動，執行
   **Tasks: Run Task → slides: watch**。

這使用 VS Code 內建 Tasks，不需額外的 run-on-save 外掛。未儲存的編輯不會觸發編譯，
其他 Markdown 檔也不會產生 HTML。編譯訊息在 `slides: watch` 的專用終端機；
找不到 `uv` 或 Node.js 時，請先修正 VS Code 的 PATH，再重新執行工作。
自動工作規則見 [VS Code 官方說明](https://code.visualstudio.com/docs/debugtest/tasks#_control-automatic-task-execution)。

監看工作在來源變動時重新執行靜態建置，不把本機重新載入程式寫入 HTML。
儲存後可重新整理瀏覽器查看結果，輸出可直接保存或發布。
需要停止時執行 **Tasks: Terminate Task → slides: watch**；單次輸出可執行
**Tasks: Run Task → slides: build**，也可從 **Tasks: Run Build Task** 選取，
不會覆寫專案的預設建置工作。

#### 手動輸出與預覽

在 repository 根目錄輸出或檢查簡報：

```bash
uv run --no-sync python -m tooling.presentations build
./scripts/check-presentations.sh
```

專項檢查在獨立臨時目錄重建 HTML，逐 byte 核對已提交產物，並檢查 HTML 頁碼與
非預期外部資源。檢查不覆寫來源或產物，也不執行 `.vscode` 中的工作。
HTML 已內含樣式與播放程式，閱讀者不需要安裝 Node.js。
Markdown 開頭的註解保存 Marp directives；每頁末尾的註解是講者稿，
`_footer` 則提供可點擊的來源。修改正文、來源或版型後，都應重新輸出 HTML。
大綱、閱讀路線與後續目標集中維護在 `slides.md`；簡報目錄的 README 提供編輯與發布操作。

本機 HTTP 預覽只服務簡報目錄，結束時按 `Ctrl+C`：

```bash
uv run python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/talks/tradingdev-introduction
```

開啟 `http://127.0.0.1:8765/`，檢查全部投影片的換行、表格、頁尾、
章節跳轉及講者模式。來源與 TODO 狀態須依對應程式及官方文件確認。
已登錄簡報走專項檢查；改動建置工具、依賴或其他程式時仍走程式分流。
GitHub Pages 的讀者操作與發布條件見簡報入口，不在此自動部署。

## MAE 研究原型

Repository 另附獨立的 MAE（最大不利變動）Streamlit 研究原型，
可觀察固定持倉時間內 BTC/USDT 的逆向價格幅度。這個原型不屬於正式 Dashboard 功能。

先自行準備 BTC/USDT 一分鐘行情，放在專案根目錄下固定的
`data/processed/btcusdt_1m_2024_2025.parquet`。檔案至少須包含可轉換為 UTC
時間的 `timestamp`，以及數值欄位 `high`、`low`、`close`；原型直接讀取這份檔案。
在專案根目錄安裝圖表依賴並啟動：

```bash
uv sync --locked --extra dashboard
uv run --locked --extra dashboard streamlit run research/mae_analysis.py
```

開啟終端機顯示的本機網址，選擇日期、持倉分鐘數與停損比例，查看 MAE 分布及統計。
結束時按 `Ctrl+C`。

## 選擇性 LLM 測試

`local` 與 `codex` 入口共用策略生成、修復、參數實驗、歷史查詢與報告、舊格式恢復
及草稿清理情境。兩者使用真實模型與 stdio MCP；回測情境先啟動受監督的離線程式
檢查程序，取得模擬使用者同意後才啟動正式 worker，結束後清理臨時檔案及程序。
Harness 以實際工具回覆與保存的產物驗證結果，
串接 `prepare_backtest` 的 plan ID、確認回覆的 job ID 及工作／結果的 manifest hash。

報告情境另核對模型最終回覆的可點擊連結、執行身分及同一份報告的章節與評語，
並在清理前讀取報告確認可用。只生成檔案、只在中途貼連結或最後只回答 run ID
都不代表交付通過。最終回覆文字會留在當次測試輸出，供審閱明細查看指引的語意；
測試不以固定中文詞判定解說是否正確。
自動化情境使用模擬使用者，驗證完即清理；它不替真人保留報告，也不證明桌面客戶端
能開啟檔案或互動表格可操作。對話中的實際驗收與成果保留遵守
[代理政策](../AGENTS.md#測試與清理)，不要直接套用自動化 fixture 的刪除流程。

多回合情境沿用同一對話：Local 保留 messages，Codex app-server 保留 thread。
第一回合須呈現短摘要與 HTML 連結，計畫 ready 且尚無正式工作或結果；模擬使用者
於回合結束後明確同意，後續回合才可用 `submit_execution_decision` 提交同一計畫。
修改計畫時先取消原計畫、準備新計畫，再等待新的同意。拒絕、無效決策、來源異動
與過期另有回歸測試。使用者由 callback 模擬，未涵蓋真人操作；後端也無法自行證明回覆來源。
初始化與所有回合共用情境期限及工具呼叫預算。工具 approval 設定不等於計畫同意。

Codex 測試在臨時工作區啟動獨立 MCP server，停用其他 server、外掛與 hooks；
沿用原有登入，個人設定維持不變。所選 provider 缺少必要能力時，情境直接失敗。

自然需求情境只提供策略條件、資金、日期與交付需求，不提供工具名稱、順序或報告答案。
它開放實際註冊的完整工具集合，由產品 instructions 與工具描述引導探索、讀取契約、
確認及交付；Codex 整合層不另補產品操作規則。驗證會核對正式執行身分、保存的設定與
訊號、標準報告、非空評語及最終連結，拒絕只含 ID 與連結的回覆。
自動檢查不能證明自由文字已正確涵蓋績效、風險、限制及參數選列／明細導覽；
審閱者須檢視測試保存的最終回覆與報告評語，才能宣稱語意交付通過。
既有逐步情境仍用於定位特定生命週期問題，不能取代此情境。

研究需求情境另分別省略策略構想、標的、期間及 K 棒頻率，檢查模型先追問，
並核對所有工具嘗試（包含失敗呼叫）與檔案，拒絕在缺項時建立策略或執行研究。
完整多回合情境僅補上缺少的期間與頻率，之後核對自動命名、模擬本金／成本預設、
確認與正式執行的身分，以及最後的完整報告。它使用明示的單組 10／30 均線需求，
獨立重算做多／空手訊號與績效；不代表模糊指標參數或多組候選比較已驗證。
固定比較另有自然需求情境：只給內建 KD 的兩個指定週期及研究條件，由模型自行
找到整批準備、確認與報告入口。驗證兩組均使用同一份保存行情、各自的完整執行規格
與獨立結果，並在最終報告保留兩組與明細入口；拒絕提早執行、改走自動搜尋或只交付
部分候選。此情境不測試模型自行查找參數來源，也不代表已驗證機器學習的多組比較。
追問內容是否適切仍須審閱模型原文。所有行情均為預先放入的固定合成測資，
不因執行測試而下載市場資料。

KD 最佳化與自建策略 walk-forward 情境核對巢狀參數、搜尋或折數、確認前狀態及
執行產物。Walk-forward 另核對確認前未執行行情試跑或訓練、正式 purged 隔離區間，以及
原 revision／隨機種子未變。離線版本使用相同 MCP／worker 與證據檢查，不呼叫模型。
Walk-forward 的研究沿用核對 fixture 實際建立的 revision、保存 YAML 的 hash，
以及該版本完整且一致的驗證證據；等值研究欄位與等價 UTC 日期可重述，實值變更仍拒絕。
離線 HTTP 回歸測試以 localhost Streamable HTTP、正式 server 及真實 MCP client
驗證準備、確認單讀取與決策提交；離線準備程序及正式 worker 使用替代實作。
桌面客戶端的檔案呈現與對話按鈕須另行驗證。

一般策略建立、修復及歷史查詢情境須沿用已驗證 YAML 基礎設定：`prepare_backtest`
的 `parameters` 與 `backtest_overrides` 須省略或傳 `null`，覆寫相同值也不符合要求。
準備時須明示剛通過 validate／dry-run 的 `revision_id`，並使用
`research.kind=inherit_selected_revision`。研究欄位可省略，或重傳與該版本兩次成功檢查的
`effective_config` 相同的值；兩份研究設定證據都須完整且一致，日期按等價 UTC 時間比對。
`kind=new` 與 `annualization=market` 會重新選擇研究或政策，不符合這些固定設定沿用情境。
參數實驗則沿用同一 runnable revision，以 `parameters` 提交不同設定，並核對結果
及基礎 YAML 未被改寫。入口即時顯示進度與耗時，第一個失敗即停止；
加 `--maxfail=0` 可跑完全部情境。

以本地 provider 單獨執行 ML 研究流程：

```bash
./scripts/check-llm.sh local --llm-model qwen3.8:27b \
  --llm-reasoning-effort none --llm-temperature 0.7 --llm-timeout 900 \
  -k ml_pipeline
```

ML 情境在獨立臨時工作區預放固定種子的行情，要求模型探索契約、建置資料集、
訓練含 purged walk-forward 驗證的 LightGBM、分頁預覽，再以模型與資料集 ID
評估完整資料、記錄結果並查詢排行榜。Harness 串接 dataset／model／entry ID、參數與驗證摘要，
讀取 Parquet、模型及排行榜檔案，重算完整訊號、標籤與命中率，核對預覽切片及評估筆數，
並確認回覆區分訓練集預測與 fold 驗證。檔案與 MCP 程序使用共用清理流程。

離線 ML domain／service／MCP 測試涵蓋數學、身分、儲存及回覆契約；workflow evidence
回歸測試涵蓋證據判定。真實 LLM 情境另驗證所選 provider 的工具理解與串接，
範圍限於上述流程，不含行情下載、策略回測 worker、所有模型種類或實際交易獲利。

以 Codex 執行完整情境：

```bash
./scripts/check-llm.sh codex --llm-model gpt-5.6-luna \
  --llm-timeout 1800
```

本地測試已驗證支援 Ollama 的 Qwen3.8 27B。模型可用 `ollama pull qwen3.8:27b` 安裝。
Local 使用支援工具呼叫的 Chat Completions 服務，預設
`http://localhost:11434/v1`；其他本機服務以 `--llm-base-url` 指定。
其他支援工具呼叫的本地模型可用 `--llm-model` 指定，須另行執行測試驗證。

完整流程會累積工具 schema、策略程式、確認單及結果，需要足夠的 context。
建議為 Qwen3.8 27B 的長流程配置至少 64K；模型宣告的最大 context 不代表服務目前
實際配置。32K 不足時，Ollama 可能回傳 HTTP 500 `no user query found in messages`；
即使回測已完成，尚未完成結果查詢與報告的情境仍不算通過。先檢查服務的 context
與硬體記憶體容量，調整後重新執行完整情境，不能只重試最後一輪。
Ollama 的相容 API 不接受 context size 參數，可依
[官方設定方式](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size)
用 Modelfile 建立專用別名，保留原模型設定：

```bash
task_modelfile=$(mktemp)
cat > "$task_modelfile" <<'EOF'
FROM qwen3.8:27b
PARAMETER num_ctx 65536
EOF
ollama create tradingdev-qwen-test -f "$task_modelfile"
rm -f "$task_modelfile"
./scripts/check-llm.sh local --llm-model tradingdev-qwen-test \
  --llm-reasoning-effort none --llm-temperature 0.7 --llm-timeout 1800
```

不再使用測試別名時，可執行 `ollama rm tradingdev-qwen-test` 移除；原本的
`qwen3.8:27b` 仍保留。較大的 context 會提高記憶體需求，執行時間也受硬體影響。
未指定 temperature 時採服務 API 的預設值，不保證等於模型設定檔；需要覆寫時
加 `--llm-temperature 1.0`（範圍 0–2）。
推理強度以 `--llm-reasoning-effort` 指定，支援值依模型服務而定；省略時使用服務預設。
Local 每輪固定傳送模型採樣 `seed=42`，與策略執行的 `random_seed` 分開；
這不保證跨模型版本或硬體的結果完全相同。不支援此參數的服務會明確失敗。
僅針對已確認的 Ollama `parameter`／`function` XML 工具解析 HTTP 500，
測試 client 會把錯誤回饋模型，要求重新產生下一個工具呼叫，整段對話最多兩次。
恢復時保留工具歷史，沿用原 provider、期限與呼叫預算；已完成的工具不重派。
其他 HTTP 錯誤及逾時仍會失敗。失敗輸出附保存的設定與執行規格，之後照常清理臨時工作區。
Pytest 將 Matplotlib 快取導向獨立暫存目錄，結束、失敗或可處理的中斷後
還原環境並清理。

兩個入口皆可用 `-k` 篩選情境；省略時執行全部情境：

| 篩選 | 涵蓋情境 |
| --- | --- |
| `-k sma` | 均線策略生成、回測與查詢。 |
| `-k natural_request` | 不透露工具操作的自然需求，使用完整工具集合驗證探索與報告交付。 |
| `-k novice` | 四種必要資訊缺漏追問，以及補充需求、模擬預設、自動命名與完整報告的多回合流程。 |
| `-k fixed_comparison` | 自然需求探索固定兩組參數、整批確認、共用行情的獨立回測與完整比較報告。 |
| `-k ml_pipeline` | 上述完整 ML 研究流程。 |
| `-k strategy_execution_guidance` | 探索 Quantile 的訓練需求，診斷不支援的簡單回測／最佳化入口，再根據拒絕回覆說明下一步；不訓練或核准執行。 |
| `-k history` | 查詢指定參數、交易分頁、權益及章節目錄，產生自選章節與 LLM 評語的 HTML，並確認未重跑回測。 |
| `-k legacy` | 探索及讀取舊格式策略，另存、驗證新 revision 並回測，保留原檔。 |
| `-k cleanup` | 授權後先預覽，再指定可刪草稿並執行刪除；逐檔確認 current、歷史可執行版本與回測產物仍保留。 |

歷史查詢另讀取原生成交與逐根帳戶頁，核對總覽、績效、`ledger` 及 `trades` 報告
與保存的 observations，確認模型正確區分原生成交與配對交易。
`--llm-timeout` 是每個模型情境的秒數上限，不是預期執行時間；簡例可用 900 秒，
包含歷史查詢與報告的長流程建議預留 1800 秒，並依硬體調整。

### 量測模型上下文

提示詞與回覆的設計原則集中在 [AGENTS.md](../AGENTS.md#模型上下文與提示詞)。
以下命令只在獨立臨時工作區檢查實際工具註冊，結束即清理，不呼叫模型、啟動 worker
或讀取一般使用者 workspace：

```bash
uv run --no-sync python -m tooling.context_budget
uv run --no-sync python -m tooling.context_budget --tools list_strategies get_strategy_contract
```

輸出分別列出 server instructions、工具名稱／描述／輸入 schema 的 Chat Completions
投影，以及包含輸出 schema 的 MCP wire metadata；每項提供字元與 UTF-8 bytes。
這些不是 token 數，也不代表所有客戶端都將 wire metadata 放進模型上下文。
指定 `--tools` 時須保留所選集合，不能把縮小集合的量測當成完整產品的成本。

Local 真實模型測試另記錄每輪實際序列化 HTTP request 的字元、bytes、相對首輪增量
與累積峰值，以及工具回覆大小。Provider 有回傳 usage 時保留其實際 token 計數；
沒有 tokenizer 或 usage 時不以固定換算假裝精確。Request 大小包含歷史與請求設定，
不能直接當成模型 token 用量；容量評估另須保留輸出空間。Codex 內部序列化與壓縮
由客戶端處理，本量測不聲稱能觀察其完整上下文。前述 64K 是特定本地測試配置經驗，
不是產品對所有模型的最低需求或成功保證。

執行指引情境使用真實模型、stdio MCP 及受監督的 preflight worker，拒絕發生於
行情取得與策略訓練之前，不替代 worker 或 service。Harness 核對探索與拒絕的策略
身分、能力、下一個工具及設定前提，並將模型最後一回合的 JSON 回覆與實際工具結果
比對；資料庫與檔案另確認沒有計畫、工作、結果、模型或行情產物。工具範圍不包含
訓練或確認提交，callback 不給予同意。離線版本走同一 MCP／worker 路徑與證據判定，
不呼叫模型。此情境不涵蓋成功的 Quantile walk-forward 訓練或真人前端操作。

## 開發階段的程式碼審查

審查責任、時機與 findings 處理以
[AGENTS.md](../AGENTS.md#開發階段的程式碼審查) 為準。代理開發時，由主代理
安排獨立 reviewer 或另啟本機審查工作階段；`check-full.sh`、commit 與 push
不會自動呼叫模型審查。

提供 reviewer 任務目的、基準與受查版本、差異範圍及相關契約位置，審查期間
保持內容穩定。Reviewer 唯讀檢視差異及使用端；主代理核實 findings、修正並複查，
回報處理結果與未涵蓋範圍。

使用 Codex CLI 審查時，依範圍選擇命令：

| 命令 | 審查範圍與前提 |
| --- | --- |
| `codex review --base main` | 目前分支相對本機主線的差異；先確認工作樹乾淨。命令不會更新本機主線。 |
| `codex review --uncommitted` | 已暫存、未暫存及未追蹤的修改；先排除其他任務內容。工作樹乾淨時無差異可查。 |
| `codex review --commit HEAD` | 最新一次提交引入的修改；可將 `HEAD` 換成指定 SHA。 |

以最新遠端主線為基準時，先確認工作樹乾淨，再執行：

```bash
git fetch origin main
codex review --base origin/main
```

請代入實際的 remote 與主線。Fetch 只更新遠端追蹤分支 `origin/main`。
審查前記錄基準與受查版本，結束後核對版本及工作樹內容；結果僅適用於受查內容。
已由獨立 reviewer 涵蓋且無未解疑慮的相同內容，可沿用審查結果。

Codex CLI 審查會呼叫模型，需可用的 CLI、登入與額度；用法可用
`codex review --help` 核對，功能說明見
[OpenAI 官方文件](https://learn.chatgpt.com/docs/codex/cli)。
Code review 尋找程式缺陷；`check-full.sh` 驗證品質與離線行為；`check-pr.sh`
另查合併候選文件及遠端版本。模型策略流程由 `check-llm.sh` 驗證，這些檢查各有範圍。

## 從開發到合併

以下以 remote `origin`、主線 `main`、開發分支 `feature/my-task` 示範，
請使用你實際的名稱。

### 1. 分階段提交並推送

在開發分支完成一個階段，依變更完成相關測試與
[開發階段審查](#開發階段的程式碼審查) 後，先 `git add` 要提交的檔案，
再填寫提交訊息：

```bash
./scripts/git-flow-commit.sh "feat: add strategy validation"
git push -u origin feature/my-task
```

Commit 依暫存 tree 相對 HEAD 的差異，自動選擇文件、簡報專項或快速檢查，並檢查
提交格式；各路徑都在獨立 snapshot 執行並核對暫存內容未變。未暫存的修正不能
掩蓋待提交內容的錯誤。一般 push 不執行完整測試或呼叫模型。

本機 hook 的主線推送保護僅適用於 GitHub（網址主機為 `github.com` 或
`ssh.github.com`），拒絕直接更新或刪除主線，包含 `--force`。
主線變更須透過 GitHub PR 合併；GitHub remote 請使用上述主機名稱。
保護規則依各實際推送網址分別判斷。同一 remote 有多個 `pushurl` 時，推送可能
部分成功，失敗也不會回復已完成的更新；重試前須逐一確認各目的地結果。

### 2. 在 GitHub 建立 PR，交由審閱者檢查

從開發分支向主線建立 PR，再以網址中的編號執行檢查；以下用 `/pull/27` 示範：

```bash
./scripts/check-pr.sh 27
```

審閱者可留在目前開發分支。腳本 fetch 主線與 PR head 至獨立臨時 Git repository，
計算合併候選，再依候選相對主線的 tree 差異選擇檢查：

- 一般文件：執行 `check-docs.sh` 的文件 snapshot，跳過 Codex 與完整 pytest。
- 簡報專項：執行 `check-presentations.sh` 的 snapshot，跳過 Codex 與完整 pytest。
- 程式變更：涉及簡報時先通過簡報 snapshot，再執行 Codex 文件一致性與讀者定位審查，最後執行快速品質及完整離線 pytest。

簡報前置檢查也涵蓋 `tooling/presentations.py`、`scripts/presentations.sh`、
`scripts/check-presentations.sh` 或 `tooling/marp/` 內檔案的變更；失敗即停止，
不會繼續呼叫 Codex 或完整 pytest。

所有流程都核對候選不可變性，結束時重新確認 PR 與實際遠端 base／head，
輸出 PR URL、base／head SHA 與候選 tree，並清理臨時產物及啟動的程序。
文件語意與操作命令仍須審閱差異；文件與簡報不要求真實模型策略測試，其他變更
由審閱者依受影響情境另行執行。每次明確檢查都重新執行，不重用先前結果。

長時間檢查保留完整 stdout／stderr 日誌，記錄指令、期限及 session。由執行工具內
的程式等待原 session，只在完成或監看失敗時通知代理，避免模型反覆讀取執行中
狀態。支援 Codex 編排工具的操作見
[wait-for-check skill](../.agents/skills/wait-for-check/SKILL.md)。
結束後確認退出碼與摘要，按需讀取診斷日誌；確認程序停止後再清理。
代理執行細節見 [AGENTS.md](../AGENTS.md#執行時機)。

以下參數適用於 PR 與本機候選檢查。每次 snapshot 預設限時 900 秒；程式分流
涵蓋快速檢查與離線測試，文件與簡報分流涵蓋各自專項。可依實測耗時指定當次期限，
單位為正整數秒：

```bash
./scripts/check-pr.sh 27 --snapshot-timeout-seconds 3600
```

檢查輸出會列出期限；逾時視為失敗，先停止程序再清理暫存。
此參數只調整 snapshot 外層期限，不延長簡報編譯器的 120 秒期限。

以下 Codex 設定只影響程式分流的文件審查證據，不縮減離線測試範圍，也不改變
文件／簡報分流。預設由 `tooling/review_scope.py` 依變更路徑決定責任範圍，
再由 planner 從固定 Git 版本組裝證據；不依賴個人 Skill 或模型猜測檢查範圍。

可先列出範圍，或只產生規劃清單以確認容量及涵蓋內容：

```bash
uv run --no-sync python -m tooling.review_docs --list-scopes
uv run --no-sync python -m tooling.review_docs \
  --repo /absolute/path/to/repository --base origin/main --tree HEAD --plan-only
```

`--plan-only` 不呼叫模型、不執行測試，也不代表審查通過。輸出包含固定 tree、
證據雜湊、路徑歸屬、文件章節的 blob／行號、各批容量與 prompt 雜湊。
內容取自 Git，不讀工作樹中尚未提交的修改。

通常使用自動範圍即可；明確指定時可重複傳入，但必須涵蓋所有已選取變更的責任範圍。
多指定的範圍會增加文件脈絡，漏列則在呼叫模型前失敗。以下只適用於變更確實
涵蓋 `tooling`、`general` 的候選，其他候選須依規劃輸出選定：

```bash
./scripts/check-pr.sh 27 --review-scope tooling --review-scope general
```

納入文字證據的每個變更路徑，其完整候選檔案與原始 diff 都分配給一個責任批次；刪除檔保留 diff
及原版本身分。相關未變更文件依標題選取原文章節，保留祖先段落、引用定義與
未知章節子樹；未知文件保留全文。變更文件本身仍完整交給其責任批次。
已登錄簡報的完整入口、來源、主題與 HTML 由同一批次負責，不能拆開規避容量。
證據只供閱讀，不執行資產。自動選取支援的來源、設定、Markdown、HTML／CSS
及 scripts／tooling 等路徑；不支援的類型（例如一般 `.txt`、`uv.lock`）、binary
資產與 runtime 目錄會列入排除清單並註明原因，不宣稱它們已受模型審查。

多批次時另加一批跨範圍審查，包含全部已選取路徑的原始 diff 與文件／路徑清單，檢查呼叫端、
契約與流程間的影響；不以其他批次的摘要取代原始證據。路徑涵蓋不等於已找到
所有語意或動態相依。模型需要額外來源才能判斷時須回報確切路徑與理由，整次
審查失敗；下次可透過可重複的 `--review-context` 補入候選 Git 的完整文字檔。
`--review-context` 的補充內容會交給每批。若只有特定責任範圍需要全文，使用可重複的
`--review-context-for SCOPE:PATH`；`SCOPE` 為上列責任範圍或 `cross_scope`，只在第一個
冒號分隔，路徑中的其餘冒號保留。指定範圍必須已被選取；額外範圍可用
`--review-scope` 明確加入。指定 `cross_scope` 時，即使只有一個責任批次仍執行跨範圍
審查。各範圍的補充全文只交給該範圍每批，不取代其他範圍原本需要的文件章節，
僅因特定範圍補充而納入的未變更 Markdown 不會成為其他範圍的共享文件；
原本自動選取的文件仍保留各範圍需要的原文章節。同批已有全文則不重複章節。
規劃輸出保留自動文件來源、映射、每批來源 blob 身分及證據／prompt
雜湊，沒有送達、無效範圍或缺少檔案均在呼叫模型前失敗。
兩種補充方式都允許自動選取類型以外的普通 UTF-8、無 NUL 檔案；若該檔
有變更，也將其 diff 納入責任範圍。仍不接受 runtime 目錄、glob、目錄、工作樹
內容或符號連結：

```bash
./scripts/check-pr.sh 27 --review-context src/tradingdev/domain/strategies/base.py
./scripts/check-pr.sh 27 --review-context-for tooling:tests/e2e/test_llm_comparison.py \
  --review-context-for cross_scope:docs/run_artifacts.md
```

容量採三個獨立的 UTF-8 byte 預算，所有批次均在呼叫模型前完成容量檢查：

| 參數 | 預設 | 限制對象 |
| ---- | ---- | -------- |
| `--max-evidence-bytes` | 600,000 | 組裝前的完整唯一證據，包含候選來源及原始 diff |
| `--max-prompt-bytes` | 240,000 | 每批完整 prompt，包含指示、證據及批次資訊 |
| `--max-total-prompt-bytes` | 3,000,000 | 所有批次 prompt 合計，包含重複脈絡與跨範圍審查 |

此外每批仍限制為 1,024,000 字元，低於已觀察到的 Codex `turn/start`
1,048,576 字元限制。Bytes／字元不是 tokens，這些預設不保證任意模型視窗都
容得下；需按實際模型容量及輸出空間調整。單一完整檔案、簡報組或跨範圍 diff
放不下時直接失敗，不截斷證據。可拆分 PR，或確認容量後明確提高當次預算：

```bash
./scripts/check-pr.sh 27 --max-evidence-bytes 1000000 \
  --max-prompt-bytes 400000 --max-total-prompt-bytes 4000000
```

最多八批並行，各自使用唯讀、無工具的 Codex 與獨立臨時目錄；所有批次共用
預設 180 秒期限，包含排隊時間，程序清理另計。分批可能增加呼叫及總用量；
每批均須完成且通過。Finding、必要脈絡不足、容量不足、執行失敗或逾時皆令
整體失敗。模型仍可能漏判，不能將規劃雜湊或路徑涵蓋視為語意正確的證明。

可依批次數與實測耗時，指定當次文件審查期限（1–86400 秒）：

```bash
./scripts/check-pr.sh 27 --documentation-timeout-seconds 600
```

此期限由所有文件批次共用，與 snapshot 分開計時。外層程序期限為 900 秒與
「文件期限加 60 秒」的較大值，保留準備與清理空間。

單獨執行文件審查時，`--repo` 指定 Git 證據所在 repository，
`--base` 與 `--tree` 在該 repository 內解析：

```bash
uv run --no-sync python -m tooling.review_docs \
  --repo /absolute/path/to/repository --base main --tree HEAD
```

PR 檢查仍須透過 `check-pr.sh` 完成對應分流的 snapshot 檢查及遠端版本核對。

### 3. 在 GitHub 審閱並合併

完成評論與審閱後，核對下列兩個版本是否仍與檢查結果相同，再於網頁合併：

```bash
git ls-remote origin refs/heads/main refs/pull/27/head
```

主線或 PR 有新提交就重新檢查；本機檢查不能攔截 GitHub 網頁合併。

### 4. 合併後更新 main、清理本機分支

```bash
git switch main
git pull --ff-only origin main
./scripts/cleanup-pr.sh 27 --branch feature/my-task
```

腳本核對合併狀態後，只刪除符合條件的本機分支；無法確認時保留並說明原因。
Squash／rebase 合併可能需要人工確認；遠端分支、worktree 與 branch config 會保留。

## 尚未建立 PR 的本機檢查

在乾淨且已提交的工作樹執行：

```bash
uv run --no-sync python -m tooling.git.gate full --base main
```

此命令檢查指定本機主線與目前提交的合併候選，採用與 PR 相同的文件／簡報／程式
分流，並核對本機 refs 與工作樹未變；可使用上述證據上限與期限參數。
結果僅涵蓋本機指定版本；遠端 PR 須另行檢查。此處的 `gate full` 是候選驗證
入口；手動 `scripts/check-full.sh` 則一律執行完整離線檢查，不依差異分流。

## 注意事項

- **測試清理**：若無法確認 worker 已停止，測試會報錯並列出保留的臨時目錄，
  供確認程序後清理。監督程序需具備程序群組觀察權限；若被外部強制殺死，
  不會假定其子程序已一併停止。
- **Git 設定**：安裝器只調整本 repository，設定 `core.hooksPath=.githooks`、
  `branch.<主線>.rebase=false` 與 `pull.ff=only`，讓一般 pull 遇分歧時停止。
  遷移時僅移除唯一值為 `--no-ff` 的 `branch.<主線>.mergeOptions`，
  保留其他自訂 merge options 與 `pull.rebase`。
- **環境**：腳本使用 Bash，目前未保證原生 Windows 相容；Git 需支援
  `merge-tree --write-tree`。更新依賴時執行 `uv lock`，並提交 lockfile 與設定變更。
- **模型與費用**：日常 pytest、文件與簡報專項不需模型服務。Codex 策略測試、
  程式候選的 Codex 文件審查需要已登入的 Codex CLI、網路與額度；
  可用 `TRADINGDEV_CODEX_BIN` 指定 CLI。
  上方 Codex 範例明確指定 Luna，須有該模型的存取權限；可用 `--llm-model`
  指定其他模型。模型不可用時測試會失敗，不會自動改用其他模型。
  本地測試須自行安裝並啟動模型服務，不會自動下載模型或改用付費服務；本地
  模型通過不代表 Codex／Claude 相容性已驗證。PR 檢查與清理另需 `gh` 能存取 repository。
- **PR 與 remote**：PR 檢查要求乾淨且已提交的工作樹，以及開啟且指向主線的 PR。
  兩個 PR 腳本僅支援 `github.com` 的標準 HTTPS／SSH remote，預設為 `origin`；
  例如改用名為 `upstream` 的 remote，就加 `--remote upstream`。
  自訂主線使用 `git config tradingdev.primaryBranch 分支名稱`。
- **舊版遷移**：重新安裝 Git hooks；若客戶端仍註冊指向已刪除的
  `scripts/hooks/finalize-*` 或 `.Codex/hooks/` 的 Stop hooks，請移除這些註冊。
