# 開發指南

本指南提供開發工具的負責範圍、使用方式與測試維護入口。命令均在 repository
根目錄執行；產品操作見 [README](../README.md)，開發政策與檢查時機以
[AGENTS.md](../AGENTS.md) 為準。

## 環境與常用命令

使用 Python 3.14 與 repository 的 `.venv`；版本和依賴以
[pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock) 為準。

```bash
uv sync --locked --all-extras --python 3.14
./scripts/install-git-hooks.sh
```

一般文件檢查不需要模型服務；簡報另需 Node.js 與 npm。安裝器調整的 Git 設定及
執行環境限制見[注意事項](#注意事項)。

| 工具 | 負責範圍 | 修改時應維護的內容 |
| --- | --- | --- |
| `scripts/rebuild-env.sh` | 依 lock 重建本專案 `.venv`。 | 依賴變更同步設定、lock 及受影響套件測試。 |
| `scripts/format.sh` | 修改程式碼的 lint／格式修正。 | 品質政策維護於 `pyproject.toml`，流程實作見 `tooling/quality.py`。 |
| `scripts/check-docs.sh` | Markdown 格式與本機檔案／標題連結。 | 改檢查器時維護 `tests/tooling/test_documentation.py`；文件語意由差異審閱確認。 |
| `scripts/check-presentations.sh` | 文件品質、簡報重建一致性、頁碼及離線資源。 | 改建置／檢查行為時維護 `tests/tooling/test_presentations.py`；視覺與瀏覽器操作另行確認。 |
| `scripts/check-fast.sh` | Ruff、格式、strict Mypy、文件品質與 MCP 契約探索／上下文預算。 | 改工具契約時維護對應 MCP 測試；改機械檢查時維護 `tests/tooling/`。 |
| `scripts/check-full.sh` | 快速品質檢查與完整離線 pytest；不呼叫模型。 | 行為變更補對應模組及使用端回歸，修復問題須能重現原本失敗。 |
| `scripts/check-llm.sh codex\|local` | 模型實際使用 MCP 的探索、互動、確認及成果交付。 | 改模型可見契約或流程時維護受影響的 `tests/e2e/` 情境。 |
| `scripts/check-pr.sh PR編號` | 固定合併候選的文件／程式檢查與遠端版本核對。 | 改 Git 流程、範圍規則或審查證據時維護 `tests/tooling/` 對應測試。 |
| `scripts/cleanup-pr.sh PR編號 --branch 分支` | 核對已合併 PR 後刪除符合條件的本地分支。 | 改清理條件時維護 `tests/tooling/test_pr_cleanup.py`。 |

Shell 入口位於 [scripts/](../scripts/)，Git hooks 位於 [.githooks/](../.githooks/)，
實作集中於 [tooling/](../tooling/)。測試按 domain、app、adapters、MCP、整合流程及
開發工具放在 [tests/](../tests/)；新增能力時依責任擴充相應測試，避免只依檔名選取。
測試案例、測資與斷言以測試程式為準，不在本指南逐項重述。

離線測試可直接使用 `uv run --all-extras pytest`，加檔案路徑或 `-k` 執行相關範圍。
`check-full.sh` 一律跑完整離線套件；提交與 PR 的檢查分流見下方流程。

### 依賴相容性限制

本專案使用 Python 3.14、XGBoost 3 及 FLAML 基礎套件；FLAML 2.7 的
[上游宣告](https://github.com/microsoft/FLAML/blob/v2.7.0/README.md)仍有限制，
可安裝不代表所有上游功能均支援。修改 ML 依賴時須保留各 learner 的實際訓練與
評估驗證，入口見 [tests/domain/ml/](../tests/domain/ml/) 及
[整合測試](../tests/integration/)。安裝問題亦見 [README](../README.md)。

## 更新文件

| 文件 | 維護責任 |
| --- | --- |
| [README](../README.md) | 使用者安裝、操作與結果解讀。 |
| [架構](../ARCHITECTURE.md) | 模組責任與跨層流程。 |
| [策略契約](strategy_contract.md)、[執行產物](run_artifacts.md) | 策略作者及整合者需要的正式契約。 |
| 本指南 | 工具入口、負責範圍、操作方式與測試維護位置。 |
| [AGENTS.md](../AGENTS.md) | 唯一代理開發政策來源。 |

行為改變時同步更新受影響的契約與操作說明。實作或測試案例調整而工具職責未變時，
通常不需要擴寫本指南。刪減內容前確認仍保留必要操作、資料保存與安全限制。

一般文件只走文件檢查；已登錄簡報走簡報專項；政策、程式、依賴或檢查工具變更走
程式檢查。完整分類規則見 [AGENTS.md](../AGENTS.md#git-hooks-與本機-pr-審查)。
文件檢查不執行文件中的命令或驗證外部網址，也不代替語意與操作審閱。

### 簡報的編輯與輸出

[簡報入口](talks/tradingdev-introduction/README.md)提供編輯、預覽與發布操作。
編輯環境須有 Node.js 18 以上及 npm；初始化鎖定的 Marp 工具鏈：

```bash
./scripts/presentations.sh setup
```

`setup` 會下載 npm 套件；一般 build／check 使用已安裝工具鏈，不自動安裝。
依賴更新須同步 `tooling/marp/package-lock.json` 並重新初始化。

#### 在 VS Code 儲存時自動編譯

[VS Code Tasks](../.vscode/tasks.json) 提供 `slides: watch` 與 `slides: build`。
完成初始化後，在可信任的工作區允許自動工作，或以 **Tasks: Run Task** 啟動。
監看時儲存來源會重建 HTML；結束使用 **Tasks: Terminate Task**。
詳細操作見[簡報入口](talks/tradingdev-introduction/README.md#編輯與預覽)。

#### 手動輸出與預覽

```bash
./scripts/presentations.sh build
./scripts/check-presentations.sh
uv run python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/talks/tradingdev-introduction
```

瀏覽 `http://127.0.0.1:8765/` 確認版面、頁碼、導覽與講者模式；結束伺服器按
`Ctrl+C`。修改來源或樣式後同步提交重建的 HTML。
專項檢查以臨時目錄重建並比對，不覆寫工作樹；缺少工具鏈或輸出不同步會失敗。
可用 `TRADINGDEV_PRESENTATION_TOOLCHAIN` 指定另一套已初始化且 lock 相符的工具鏈。

## MAE 研究原型

[research/mae_analysis.py](../research/mae_analysis.py) 是獨立的最大不利變動研究原型，
不屬於正式 Dashboard。先準備 `data/processed/btcusdt_1m_2024_2025.parquet`，
包含可轉換為 UTC 的 `timestamp` 與數值 `high`、`low`、`close`：

```bash
uv sync --locked --extra dashboard
uv run --locked --extra dashboard streamlit run research/mae_analysis.py
```

開啟終端機顯示的網址；結束按 `Ctrl+C`。此入口不下載行情。

## 選擇性 LLM 測試

此工具驗證模型透過真實 MCP 完成策略研究互動及成果交付。修改模型可見的指引、
工具 schema／回覆契約、策略生成修復或模型整合層時，依
[執行時機](../AGENTS.md#執行時機) 選擇受影響情境。
測試入口為 [scripts/check-llm.sh](../scripts/check-llm.sh)，情境與驗證程式位於
[tests/e2e/](../tests/e2e/)，provider 選項定義於 [tests/llm_plugin.py](../tests/llm_plugin.py)。

```bash
./scripts/check-llm.sh codex --llm-model gpt-6.1-sol --llm-timeout 1800
./scripts/check-llm.sh local --llm-model qwen3.8:27b --llm-timeout 1800
```

Codex 需要已登入的 CLI 與所選模型存取權；Local 需要已啟動且支援工具呼叫的本機
服務，預設 `http://localhost:11434/v1`，以 `--llm-base-url` 指定其他 loopback 位址。
工具不下載模型、不自動切換 provider；所選能力不可用會失敗。

以 `-k` 選取相關情境，省略時執行全部。可先列出目前可用情境，不呼叫模型：

```bash
./scripts/check-llm.sh codex --collect-only -q
./scripts/check-llm.sh codex --llm-model gpt-6.1-sol -k novice --llm-timeout 1800
```

依變更責任選擇與維護測試：

| 責任範圍 | 測試入口 |
| --- | --- |
| 策略生成、修復與既有結果查詢 | [test_llm_workflows.py](../tests/e2e/test_llm_workflows.py) |
| 自然需求、必要資訊追問與工具探索 | [test_llm_novice_intake.py](../tests/e2e/test_llm_novice_intake.py)、[test_llm_discovery.py](../tests/e2e/test_llm_discovery.py) |
| 執行計畫、使用者決策與訓練指引 | [test_llm_execution_plans.py](../tests/e2e/test_llm_execution_plans.py)、[test_llm_strategy_execution_guidance.py](../tests/e2e/test_llm_strategy_execution_guidance.py) |
| 固定參數比較與參數提案 | [test_llm_comparison.py](../tests/e2e/test_llm_comparison.py)、[test_llm_research_proposal.py](../tests/e2e/test_llm_research_proposal.py) |
| ML 研究 | [test_llm_ml.py](../tests/e2e/test_llm_ml.py) |
| 授權後的草稿清理 | [test_llm_cleanup.py](../tests/e2e/test_llm_cleanup.py) |

測試使用隔離工作區、固定合成行情與模擬使用者，不下載市場資料；完成驗證後清理。
它不替真人保留報告，也不驗證桌面客戶端的互動。自然語言解讀仍須審閱輸出的最終
回覆與報告評語；機械斷言通過不等於語意正確。使用者實際驗收的成果保存遵守
[AGENTS.md](../AGENTS.md#測試與清理)。

`--llm-timeout` 是每個情境的期限。長流程需要足夠上下文；已驗證的本地 Qwen 配置
為 64K，但不代表所有模型的最低需求。容量不足須調整服務配置後重跑完整情境。
Ollama 設定方式見[官方說明](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size)。
Local 可用 `--llm-temperature`、`--llm-reasoning-effort` 覆寫支援的採樣設定；
這些選項不適用於 Codex。模型版本與設定不同，驗證結果不能直接互相替代。

### 量測模型上下文

```bash
uv run --no-sync python -m tooling.context_budget
uv run --no-sync python -m tooling.context_budget --tools list_strategies get_strategy_contract
uv run --no-sync python -m tooling.prompt_contracts
```

| 工具 | 負責範圍與維護入口 |
| --- | --- |
| [context_budget](../tooling/context_budget.py) | 量測實際註冊的指引、工具投影及 MCP wire metadata；修改量測時維護 `tests/tooling/test_context_budget.py`。 |
| [prompt_contracts](../tooling/prompt_contracts.py) | 機械檢查契約探索、schema、明示章節引用與 UTF-8 預算；已納入 `check-fast.sh`，修改時維護 `tests/tooling/test_prompt_contracts.py`。 |

兩者使用可清理的臨時工作區，不呼叫模型、下載行情或執行回測。Bytes 不是 tokens；
指定工具集合的量測也不能當成完整產品成本。預算集中於 `prompt_contracts.py`，
超量須檢查內容歸屬，不以截斷內容通過檢查。
Local 真實模型測試另記錄 request 增量與峰值；Codex 完整上下文及內部壓縮不在此量測範圍。

## 開發階段的程式碼審查

依 [AGENTS.md](../AGENTS.md#開發階段的程式碼審查) 安排獨立唯讀 reviewer，提供任務目的、
基準／受查版本、差異範圍與相關契約。保持受查內容穩定，核實 findings、修正後複查。

| 命令 | 適用範圍 |
| --- | --- |
| `codex review --base origin/main` | 相對已取得遠端主線的分支差異；先確認工作樹乾淨並 fetch。 |
| `codex review --uncommitted` | 已暫存、未暫存及未追蹤內容；先排除其他任務修改。 |
| `codex review --commit HEAD` | 指定 commit 引入的差異。 |

CLI 參數以 `codex review --help` 核對。已由獨立 reviewer 涵蓋且無未解疑慮的相同內容
可沿用審查結果。程式碼審查、離線測試、文件一致性與真實模型流程驗證各有用途，
不能互相替代；一般 commit／push 不會自動執行模型 code review。

## 從開發到合併

以下以 `origin`、`main`、`feature/my-task` 示範，請代入實際名稱。
分支、提交與授權政策見 [AGENTS.md](../AGENTS.md#工作樹與分階段提交)。

### 1. 分階段提交並推送

更新遠端主線後建立 task branch；每階段完成相關測試與必要審查，再明確提交檔案：

```bash
git fetch origin main
git switch -c feature/my-task origin/main
./scripts/git-flow-commit.sh "feat: add strategy validation" path/to/changed_file.py
git push -u origin feature/my-task
```

已有開發分支時沿用。Commit hook 依暫存內容執行文件、簡報或快速檢查；一般 push
不執行完整 pytest 或呼叫模型。本機 hook 的主線推送保護限於 GitHub remote。

### 2. 在 GitHub 建立 PR，交由審閱者檢查

```bash
./scripts/check-pr.sh 27
```

須有乾淨且已提交的工作樹、可存取 repository 的 `gh`，以及開啟且指向主線的 PR。
工具在隔離 repository 檢查合併候選，不切換開發分支：

| 候選變更 | 執行檢查 |
| --- | --- |
| 一般文件 | 文件格式與連結。 |
| 已登錄簡報 | 文件與簡報專項；視覺操作仍須人工確認。 |
| 程式／政策／工具 | 必要的簡報專項、Codex 文件一致性審查、快速品質與完整離線 pytest。 |

真實模型策略測試另依變更選取。程式候選的文件審查需要 Codex 登入與額度；
這一步不要求使用 Astra，建議使用 Sol 或其他具足夠上下文容量的模型，透過
`TRADINGDEV_CODEX_MODEL` 指定，例如：

```bash
TRADINGDEV_CODEX_MODEL=gpt-6.1-sol ./scripts/check-pr.sh 27
```

文件審查會帶入完整的變更檔案及按責任選取的既有文件原文，輸入量不只取決於
修改行數；先用下方 `--plan-only` 量測證據與 prompt 大小，再依容量與成本選擇模型。
上述命令為這次 PR 文件審查指定模型；開發階段 `codex review` 的模型須另行指定。
審查程式與來源入口見
[review_docs.py](../tooling/review_docs.py)、[review_scope.py](../tooling/review_scope.py)。

先列出責任範圍或規劃固定 Git 版本的證據；以下命令不呼叫模型、不代表檢查通過：

```bash
uv run --no-sync python -m tooling.review_docs --list-scopes
uv run --no-sync python -m tooling.review_docs \
  --repo /absolute/path/to/repository --base origin/main --tree HEAD --plan-only
```

通常使用自動推導的範圍；需要調整時，`check-pr.sh` 支援：

| 選項 | 使用目的 |
| --- | --- |
| `--review-scope SCOPE` | 明確增加審查責任範圍；不可漏掉變更所屬範圍，可重複。 |
| `--review-context PATH` | 為所有批次補入候選 Git 中的完整文字檔，可重複。 |
| `--review-context-for SCOPE:PATH` | 僅為指定責任範圍或 `cross_scope` 補入完整檔案，可重複。 |
| `--max-evidence-bytes`、`--max-prompt-bytes`、`--max-total-prompt-bytes` | 分別限制唯一證據、單批 prompt 與總 prompt；先量測，再依模型容量明確設定。 |
| `--documentation-timeout-seconds` | 文件審查所有批次共用的期限。 |
| `--snapshot-timeout-seconds` | 品質／測試 snapshot 的期限；不改變測試範圍或工具自身期限。 |

例如為單一範圍補上必要來源，並依實測耗時設定期限：

```bash
./scripts/check-pr.sh 27 \
  --review-context-for interfaces:src/tradingdev/mcp/server.py \
  --documentation-timeout-seconds 900 --snapshot-timeout-seconds 3600
```

請以實際規劃的 scope／檔案代換。預設值與其他選項以 `scripts/check-pr.sh --help`
為準。範圍選取不縮減離線測試；變更檔案保留完整內容與 diff，相關文件按責任取原文。
缺少必要證據、容量不足、finding 或逾時均須修正後重查，不能截斷內容或忽略失敗。

長時間檢查保留完整日誌，透過[完成通知 skill](../.agents/skills/wait-for-check/SKILL.md)
等待原執行結束；完成後讀退出碼與摘要，有異常再讀診斷。未確認程序停止前保留臨時目錄。

### 3. 在 GitHub 審閱並合併

完成檢查與評論處理後，核對即時遠端版本與檢查輸出的 base／head SHA：

```bash
git ls-remote origin refs/heads/main refs/pull/27/head
```

任一版本改變都須重查。確認後才合併；本機檢查不會鎖住 GitHub 的遠端變更。

### 4. 合併後更新 main、清理本機分支

```bash
git fetch origin main
git switch main
git merge --ff-only origin/main
./scripts/cleanup-pr.sh 27 --branch feature/my-task
```

清理工具核對 PR、分支身分與祖先關係，只刪除符合條件的本地分支；無法確認時保留。
Squash／rebase 合併可能無法證明祖先關係。遠端分支、worktree 與 branch config
不由此工具刪除；遠端刪除須另核對來源倉庫及使用者授權。

## 尚未建立 PR 的本機檢查

在乾淨且已提交的工作樹執行：

```bash
uv run --no-sync python -m tooling.git.gate full --base main
```

使用與 PR 相同的候選分流及範圍／期限選項，但結果僅涵蓋指定本機版本；遠端 PR
仍須執行 `check-pr.sh`。此入口與一律執行完整離線檢查的 `check-full.sh` 有不同用途。

## 注意事項

- **環境**：使用 repository 的 `.venv`；腳本依賴 Bash，未保證原生 Windows 相容。
  Git 需支援 `merge-tree --write-tree`。更新依賴須執行 `uv lock` 並提交設定及 lock。
- **Git 設定**：hooks 安裝器只調整本 repository；實際變更見安裝輸出與
  [安裝腳本](../scripts/install-git-hooks.sh)。重新安裝時移除客戶端仍引用舊版 Stop hooks 的註冊。
- **模型**：日常 pytest 不呼叫模型。Codex 測試及程式候選文件審查需要 CLI、登入、網路與額度；
  可用 `TRADINGDEV_CODEX_BIN` 指定 CLI。Local 服務須自行啟動，模型與上下文配置須個別驗證。
- **程序清理**：無法確認 worker 停止時，檢查失敗並保留臨時目錄；確認程序後才清理。
  外部強制終止不保證子程序停止。使用者成果不適用測試暫存清理規則。
- **PR 與 remote**：PR 工具僅支援 `github.com` 標準 HTTPS／SSH remote，預設 `origin`，
  可用 `--remote` 指定其他名稱；自訂主線使用 `git config tradingdev.primaryBranch 分支名稱`。
