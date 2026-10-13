# 開發指南

所有命令均在 repository 根目錄執行。產品操作見 [README](../README.md)，
代理開發政策見 [AGENTS.md](../AGENTS.md)。

## 環境與常用命令

首次設置：

```bash
uv sync --locked --all-extras
./scripts/install-git-hooks.sh
```

安裝器會調整本機 Git 設定，摘要見文末的[注意事項](#注意事項)。

| 命令 | 用途 |
| --- | --- |
| `./scripts/rebuild-env.sh` | 重建 `.venv`，依 `uv.lock` 安裝鎖定版本。 |
| `./scripts/format.sh` | 自動修正 lint 與格式，會修改檔案。 |
| `./scripts/check-fast.sh` | 唯讀檢查 Ruff、格式、strict Mypy 與 Markdown。 |
| `./scripts/check-full.sh` | 執行上述品質檢查與完整離線 pytest；不呼叫模型。 |
| `uv run --all-extras pytest` | 完整離線測試，包含真實 MCP／worker；不呼叫模型。 |

`scripts/` 保留 Shell 操作入口，`.githooks/` 保留 Git hook 入口；Python 實作
集中在 `tooling/`，以同一個 `.venv` interpreter 執行 `python -m tooling...`。
`tooling/quality.py` 共用格式化與唯讀檢查流程，Python 檢查範圍為 `src/`、
`tests/`、`tooling/`、`research/`；Markdown 檢查涵蓋根目錄與 `docs/`。

`tooling/git/candidates.py` 管理獨立的 snapshot／合併候選與驗證；`gate.py`、
`check_pr.py`、`cleanup_pr.py` 分別協調 hook／本機 full、PR 檢查與分支清理，
`pr_context.py` 共用 GitHub PR 身分讀取。文件審查位於 `tooling/review_docs.py`，
與測試共用 `tooling/` 的程序管理及 Codex 執行檔定位，不依賴 `tests/`。
候選驗證使用執行檢查的 checkout 內的 reviewer 讀取候選 Git 內容，再執行
候選 snapshot 內的品質檢查與離線測試。

`tests/` 保留測試案例、fixtures 與 MCP／模型測試情境；`research/` 保留獨立
研究原型，用法見 [README](../README.md#mae-研究原型)。`tooling/` 與 `research/`
不隨產品 wheel 發佈，Python 程式仍使用相同的 Ruff／strict Mypy 政策。
原有 `scripts/*.py` 路徑已移除，不保留相容轉接；操作時使用既有 Shell 入口，
直接呼叫 Python 工具則在 repository 根目錄使用新的模組命令。

## 選擇性 LLM 測試

兩個入口共用均線、動量、參數實驗、歷史查詢與報告、錯誤草稿修復與舊格式策略恢復情境，要求模型完成策略生成、
計畫準備、確認請求、回測及結果查詢；腳本獨立核對訊號與結果，並清理臨時檔案及程序。
證據檢查串接 `prepare_backtest` 的 plan ID、確認回覆的 job ID，以及相同 manifest hash
的工作與結果，不能用模型聲稱「完成」代替實際工具回覆。
Local 與 Codex harness 使用真實模型、stdio MCP、受監督的試跑與正式 worker；確認由明確注入的
測試使用者 callback 回應並記錄。這是模擬使用者批准，不是已驗證真人操作的客戶端 UI。
Codex 使用 app-server 雙向 JSON-RPC，把 MCP 確認表單交給同一套 callback，再回傳其決定；
不使用無法接回表單的 `codex exec`。測試會停用其他 MCP server、外掛與 hooks，
在臨時工作區啟動獨立的 MCP server，保留原有 Codex 登入；不修改個人設定。
需要支援 `mcpServer/elicitation/request` 的 Codex CLI；初始化、模型及 callback
共用同一個情境逾時上限，協定不支援或 callback 缺少時直接失敗。
Runner 宣告 `experimentalApi`，使用僅允許 MCP 表單的 granular approval policy；
`never` 會自動拒絕表單，不能用來驗證成功批准。
另有內建 KD 最佳化與自建策略 walk-forward 的執行計畫情境，檢查巢狀參數說明、
搜尋或折數覆蓋、確認及結果身分；兩個 provider 共用相同的確認前狀態觀察，
在 callback 收到完整確認單時核對尚無正式工作與結果，批准後再核對工作及執行產物。
一般測試連線預設不支援 elicitation；成功案例須明確選擇 callback，拒絕、取消、
不支援與無效回覆另有回歸測試。不得把模型輸入的 `approved` 當成確認。
參數實驗情境要求沿用同一 runnable revision，以 `parameters` 提交不同於基礎 YAML
的設定，並核對執行結果與基礎設定未被改寫。
另有明確授權的草稿清理情境：先預覽再指定可刪版本，並逐檔確認 current、
歷史可執行版本及回測產物仍保留；可用 `-k cleanup` 單獨執行。
入口即時顯示進度與耗時，第一個失敗就停止並顯示原因；要跑完全部情境可加
`--maxfail=0`。

Codex 與 local 入口都涵蓋完整情境，不需排除執行計畫測試：

```bash
./scripts/check-llm.sh codex --llm-model gpt-5.6-luna \
  --llm-timeout 1800
```

各情境仍須具備其要求的模型及客戶端能力，能力不足會失敗，不會默默略過。

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
恢復時保留已完成工具的歷史，不重新派發、不切換 provider，也不重設時間或工具次數上限。
其他 HTTP 錯誤及逾時仍會失敗。失敗輸出附保存的設定與執行規格，之後照常清理臨時工作區。
快速檢查可加 `-k sma`，只跑均線的生成、回測與查詢；移除此選項才涵蓋全部情境。
`-k history` 驗證模型查回指定參數、翻頁交易、讀取權益與章節目錄，並請後端產生
自選章節及 LLM 評語的 HTML；對照保存的原始交易，確認沒有再次啟動回測。
同一情境也讀取原生成交與逐根帳戶頁，並選用這兩個報告章節；回覆逐欄對照
保存的 observations，驗證模型沒有把配對交易當作逐次成交。
`-k legacy` 驗證模型先探索及讀取舊格式策略，再另存、驗證新 revision 並執行回測，
同時確認原有檔案未變。
`--llm-timeout` 是每個模型情境的秒數上限，不是預期執行時間；簡例可用 900 秒，
包含歷史查詢與報告的長流程建議預留 1800 秒，並依硬體調整。
Codex 入口使用 CLI 的 MCP 客戶端能力；工具 approval 設定與 MCP 使用者表單是不同機制。
若該客戶端無法完成 elicitation，執行情境會明確失敗，不能跳過確認或改用另一個 provider。

## 開發階段的程式碼審查

審查責任、時機與 findings 處理以
[AGENTS.md](../AGENTS.md#開發階段的程式碼審查) 為準。代理開發時，由主代理
主動安排獨立 reviewer 或另啟本機審查工作階段；使用者不必每次手動輸入命令。
這個流程不會透過 `check-full.sh`、commit 或 push 自動呼叫模型，也沒有額外
的 repository 審查腳本。

代理有獨立 reviewer 可用時，提供任務目的、基準與受查版本、差異範圍，以及
相關契約的位置。Reviewer 直接檢視差異及相關使用端，只回報 findings；主代理
負責核實與修正。開始審查後暫停修改受查內容，修正 findings 後再複查受影響範圍。

改用 Codex CLI，或由開發者手動審查時，在 repository 根目錄選擇符合範圍的
一個入口即可。若修改已提交，要審查目前分支相對本機主線的完整差異，
先確認工作樹乾淨，再執行：

```bash
codex review --base main
```

`main` 指本機主線，此命令不會自動更新它。分階段 commit 後，仍可用這個入口
審查整個分支；不需要取消 commit。

階段修改尚未提交時：

```bash
codex review --uncommitted
```

這只審查已暫存、未暫存及未追蹤的修改，需先確認範圍沒有混入其他任務。
工作樹乾淨時會回報沒有差異，不代表已審查分支上的 commits。

若只要審查最新一個 commit：

```bash
codex review --commit HEAD
```

可將 `HEAD` 換成指定 commit 的 SHA；此入口只涵蓋該次提交引入的修改。
若要以最新遠端主線為基準審查完整分支，先確認工作樹乾淨，再執行：

```bash
git fetch origin main
codex review --base origin/main
```

`main`、`origin/main` 是範例，使用實際的 remote 與主線。Fetch 更新遠端追蹤
分支 `origin/main`，不更新本機 `main`。上述命令不推送或建立 PR；審查前記錄
所選基準與受查版本，結束後確認版本及工作樹內容未變。未提交內容也要在交付時
說明所涵蓋的範圍，不能把舊結果套用到後續修改。已由獨立 reviewer 涵蓋的相同
內容不需要再跑一次 CLI review。

例如 reviewer 指出某個預期的 service 拒絕會在 MCP 邊界變成未處理例外，
先核對呼叫鏈與例外型別，再用能重現該失敗的測試驗證；修正後測試應通過。
若原行為已符合契約，記錄不採納的程式／測試證據。最後回報受查範圍、findings
處理與未涵蓋項目，不只列出命令的 exit code。

Codex CLI 審查會呼叫模型，需可用的 CLI、登入與額度；用法可用
`codex review --help` 核對，功能說明見
[OpenAI 官方文件](https://learn.chatgpt.com/docs/codex/cli)。
此審查用來找程式缺陷；`check-full.sh` 執行品質檢查與離線測試，
`check-pr.sh` 另外檢查合併候選的文件一致性及遠端版本。Code review 不會取代
這些檢查，也不代表真實模型策略流程已驗證。

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

Commit 會自動跑快速檢查；一般 push 不執行完整測試或呼叫模型。

### 2. 在 GitHub 建立 PR，交由審閱者檢查

從開發分支向主線建立 PR。編號取自 PR 標題的 `#編號` 或網址 `/pull/編號`；
例如 `/pull/27` 的編號就是 `27`。以下 `27` 均為範例，請換成實際 PR 編號：

```bash
./scripts/check-pr.sh 27
```

審閱者可留在目前開發分支執行；腳本會自動檢查合併候選、執行 Codex 文件
審查與完整離線 pytest，並清理臨時產物；真實模型策略測試由審閱者視變更另行執行。

文件審查預設接受最多 600,000 bytes 的完整證據，超過時會在呼叫模型前失敗，
不截斷檔案或差異。確認所用模型可容納完整內容後，可針對這次檢查明確提高上限：

```bash
./scripts/check-pr.sh 27 --max-evidence-bytes 1000000
```

參數必須是正整數，只調整原始完整證據的 UTF-8 byte 預算，不會提高 Codex 的
單次輸入或模型 token 上限。腳本另限制每次完整 prompt 最多 1,024,000 字元，
包含審查指示與批次資訊，低於已觀察到的 Codex `turn/start` 1,048,576 字元限制。

放不進單次請求時，腳本按完整檔案分批。每批都包含全部 diff、Markdown 文件及
候選檔案清單，其餘候選原始檔逐批完整涵蓋，不裁切或以摘要取代。所有批次先完成
容量檢查；若共同證據或單一檔案連同共同證據仍放不下，會在呼叫模型前失敗，
此時須拆分變更，提高 `--max-evidence-bytes` 無法解決單次輸入限制。

最多八批並行，各自使用唯讀、無工具的 Codex 與獨立臨時目錄；所有批次共用
原有 180 秒文件審查期限，排隊也計時，程序清理另計。分批會增加模型呼叫與重複
context 的用量。每批必須完成且通過；任何 finding、缺少必要跨檔脈絡、容量不足、
執行失敗或逾時都令整體失敗。全部 diff 提供跨檔變更脈絡，但分批不等於所有
原始檔同時放在單一上下文，也不能保證模型不漏判。
若要單獨執行文件審查，從本 repository 根目錄呼叫模組；`--repo` 指定要讀取
Git 證據的 repository，`--base` 與 `--tree` 在該 repository 內解析：

```bash
uv run --no-sync python -m tooling.review_docs \
  --repo /absolute/path/to/repository --base main --tree HEAD
```

此命令僅執行文件審查；PR 檢查仍須使用上述入口完成離線測試及遠端版本核對。

### 3. 在 GitHub 審閱並合併

完成評論與審閱後，核對下列兩個版本是否仍與檢查結果相同，再於網頁合併：

```bash
git ls-remote origin refs/heads/main refs/pull/27/head
```

### 4. 合併後更新 main、清理本機分支

```bash
git switch main
git pull --ff-only origin main
./scripts/cleanup-pr.sh 27 --branch feature/my-task
```

腳本核對合併狀態後，只刪除符合條件的本機分支；無法確認時會保留並說明原因。

## 尚未建立 PR 的本機檢查

```bash
uv run --no-sync python -m tooling.git.gate full --base main
```

此命令檢查指定本機主線與目前提交的合併候選。
同樣可加 `--max-evidence-bytes 1000000`，只調整這次文件審查的證據上限。

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
- **模型與費用**：日常 pytest 不需模型服務。Codex 策略測試及 PR 文件審查
  需要已登入的 Codex CLI、網路與額度；可用 `TRADINGDEV_CODEX_BIN` 指定 CLI。
  上方 Codex 範例明確指定 Luna，須有該模型的存取權限；可用 `--llm-model`
  指定其他模型。模型不可用時測試會失敗，不會自動改用其他模型。
  本地測試須自行安裝並啟動模型服務，不會自動下載模型或改用付費服務；本地
  模型通過不代表 Codex／Claude 相容性已驗證。PR 檢查與清理另需 `gh` 能存取 repository。
- **PR 與 remote**：PR 檢查要求乾淨且已提交的工作樹，以及開啟且指向主線的 PR。
  兩個 PR 腳本僅支援 `github.com` 的標準 HTTPS／SSH remote，預設為 `origin`；
  例如改用名為 `upstream` 的 remote，就加 `--remote upstream`。
  自訂主線使用 `git config tradingdev.primaryBranch 分支名稱`。
- **合併前版本**：主線或 PR 有新提交就重新檢查。本機檢查不能攔截 GitHub 網頁
  合併；尚未建立 PR 時的本機檢查，也不代表遠端 PR 已驗證。
- **分支清理**：先在 GitHub 合併再清理；squash／rebase 合併可能需要人工確認。
  遠端分支與 worktree 會保留。
- **舊版遷移**：重新安裝 Git hooks；若客戶端仍註冊指向已刪除的
  `scripts/hooks/finalize-*` 或 `.Codex/hooks/` 的 Stop hooks，請移除這些註冊。
