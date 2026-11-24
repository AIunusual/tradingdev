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

一般文件檢查不需要模型服務。本機簡報建置與檢查另需 Node.js 22.17 以上；
含 Mermaid 時另需 Chrome／Chromium。VS Code 的 build／watch 工作使用 uv，
啟動時先同步鎖定的 Python 依賴，再按需初始化本機 Marp 工具鏈；
只有工具鏈需要安裝或修復時才需要 npm。
Python 簡報 build／watch 命令未加 `--ensure-toolchain` 時，以及 check／package 與離線測試，
均要求工具鏈已初始化，執行時不呼叫 npm。Cloudflare 發布入口會先執行 `npm ci`。
完整離線 pytest 與 `tests/tooling` 包含真實圖表渲染測試，需要 Node.js、已初始化的工具鏈與瀏覽器；
離線表示執行時不呼叫模型或下載依賴。
初始化操作與替代瀏覽器設定見[簡報的編輯與輸出](#簡報的編輯與輸出)。
安裝器調整的 Git 設定及
執行環境限制見[注意事項](#注意事項)。

| 工具 | 負責範圍 | 修改時應維護的內容 |
| --- | --- | --- |
| `scripts/rebuild-env.sh` | 依 lock 重建本專案 `.venv`。 | 依賴變更同步設定、lock 及受影響套件測試。 |
| `scripts/format.sh` | 修改程式碼的 lint／格式修正。 | 品質政策維護於 `pyproject.toml`，流程實作見 `tooling/quality.py`。 |
| `scripts/check-docs.sh` | Markdown、連結及 SRS 結構與追溯引用。 | 改檢查器時維護 `tests/tooling/test_documentation.py`、`test_srs_requirements.py`；不執行引用的測試。 |
| `scripts/check-presentations.sh` | 文件品質、簡報來源建置、頁碼及離線資源。 | 改建置／檢查行為時維護 `tests/tooling/test_presentations.py`；打包與發布 staging 維護 `tests/tooling/test_presentation_bundle.py`；視覺與瀏覽器操作另行確認。 |
| `scripts/check-fast.sh` | Ruff、格式、strict Mypy、文件品質與 MCP 契約探索／上下文預算。 | 改工具契約時維護對應 MCP 測試；改機械檢查時維護 `tests/tooling/`。 |
| `scripts/check-full.sh` | 快速品質檢查與完整離線 pytest；不呼叫模型。 | 行為變更補對應模組及使用端回歸，修復問題須能重現原本失敗。 |
| `scripts/check-tooling.sh` | 快速品質檢查與完整 `tests/tooling`；不呼叫模型。 | 修改已登錄工具或其 Markdown 指引時，依[分流表](#2-在-github-建立-pr交由審閱者檢查)確認適用範圍。 |
| `uv run --no-sync python -m tooling.sharpe_calibration` | 離線執行固定 Sharpe 推論校準矩陣，輸出 JSON 至 checkout 外。 | 改程序生成公式、seed、block size 或判讀規則時維護 `tests/domain/statistics/test_statistics_sharpe_calibration.py`。 |
| `scripts/check-llm.sh codex\|local` | 模型實際使用 MCP 的探索、互動、確認及成果交付。 | 改模型可見契約或流程時維護受影響的 `tests/e2e/` 情境。 |
| `scripts/check-pr.sh PR編號` | 固定合併候選的文件／程式檢查與遠端版本核對。 | 改 Git 流程、範圍規則或審查證據時維護 `tests/tooling/` 對應測試。 |
| `scripts/claim-issue.sh 編號 --repo OWNER/REPO --user LOGIN` | 核對 CLI 登入者並將本人加入 issue assignees。 | 改指派流程時維護 `tests/tooling/test_claim_issue.py`。 |
| `scripts/cleanup-pr.sh PR編號 --branch 分支` | 核對已合併 PR 後刪除符合條件的本地分支。 | 改清理條件時維護 `tests/tooling/test_pr_cleanup.py`。 |

Sharpe 校準入口以 `--profile full` 執行完整固定矩陣，`--profile smoke` 做快速抽樣；
900 與 120 秒是重複之間的合作式期限，進行中的計算不會被中斷，實際時間可能超過；
檢查發現逾時會丟出 `TimeoutError` 且不寫出 JSON。兩者都要求絕對路徑的 `--output`，
且路徑必須在 checkout 外。報告欄位與適用限制見
[`build_report`／`main`](../tooling/sharpe_calibration.py)。

檢查與格式化入口在開始工作前，以 [tooling/environment.py](../tooling/environment.py)
確認所需的 Python 工具；缺少時回報套件名稱與 `uv sync --locked --all-extras`。
格式化會在第一次修改檔案前完成工具確認。完整測試另共用產品的
`DockerRuntime.resolve()`，確認本機 Docker 端點、daemon 資源能力、策略映像及
有效資源設定；失敗時不啟動品質檢查或 pytest，也不自動下載或建置映像。
快速、文件與開發工具檢查不要求 Docker 或模型服務。本機 Python 簡報入口在安裝或編譯前
確認 Node.js 版本，檢查入口亦先核對已安裝的鎖定工具鏈。Cloudflare 發布腳本直接執行
npm／Node，使用平台的 `NODE_VERSION` 設定，未經此 Python 前置檢查；設定方式見
[簡報網站發布設定](#簡報網站發布設定)。

可只確認完整測試的前置環境：

```bash
uv run --no-sync python -m tooling.environment full
```

此命令回報解析到的策略映像 ID，不啟動策略容器；它確認的是當下的前置條件，
不保證映像與來源同步，也不取代實際容器、程序清理或策略行為測試。
依下方容器建置流程準備映像後，再執行完整檢查。直接執行 pytest 選測個別案例時，
仍須自行備妥該案例需要的環境。

Shell 入口位於 [scripts/](../scripts/)，Git hooks 位於 [.githooks/](../.githooks/)，
實作集中於 [tooling/](../tooling/)。測試按 domain、app、adapters、MCP、整合流程及
開發工具放在 [tests/](../tests/)；新增能力時依責任擴充相應測試，避免只依檔名選取。
測試案例、測資與斷言以測試程式為準，不在本指南逐項重述。

### 型別檢查與契約保護

Ruff 檢查程式品質與格式，Mypy 檢查靜態型別相容性；`strict` 通過仍可能包含
`Any`，不能據此判定型別契約足夠精確。未填元素型別的 `pd.Series` 也可能使用
stub 的 `Any` 預設值，使不存在的方法或錯誤輸入失去檢查。

已建立精確契約的核心模組在 [pyproject.toml](../pyproject.toml) 以逐模組
`disallow_any_explicit` 限制新增的 `Any`；模組範圍以該設定為準。
新增同類模組時，先確認輸入、輸出及使用端契約，再納入設定並執行快速檢查。
動態 JSON、外部套件等邊界若需要 `Any`，應在邊界說明來源及限制，並在可驗證後
收窄型別；不能用廣泛的忽略、`cast` 或擴大例外取代契約修正。
此規則不禁止所有推導出的 `Any`，第三方回傳值及跨模組傳遞仍需契約案例保護。

靜態型別案例以 `typing_*.py` 放在受測模組所屬的測試目錄，由快速檢查中的 Mypy
驗證，不由 pytest 執行。公開介面新增或變更時，選擇有風險的輸入與輸出，
以 `assert_type` 驗證跨模組型別保留；預期錯誤使用具錯誤代碼的局部忽略。
若型別退化成 `Any` 而不再報錯，strict Mypy 的 unused-ignore 檢查會使案例失敗。
這些案例驗證靜態契約，執行期資料驗證仍由相關行為測試負責。
設定的拒絕能力由 [test_quality.py](../tests/tooling/test_quality.py) 使用實際 Mypy 驗證。

快速檢查的 Python 目錄以 [tooling/quality.py](../tooling/quality.py) 的 `PYTHON_PATHS`
為準。新增目錄時，確認可執行、持續維護的 Python 已納入該入口；歷史案例與封存
證據另依其用途維護，不因副檔名相同就改寫。單獨執行 Mypy 時須指定 repository 的
`pyproject.toml`，不能用另一份較寬鬆設定代替快速檢查。

### 執行離線測試

離線測試可直接使用 `uv run --all-extras pytest`，加檔案路徑或 `-k` 執行相關範圍。
手動執行 `check-full.sh` 一律跑完整離線套件。`check-tooling.sh` 執行開發工具套件；
交付時先[產生整個任務差異的檢查計畫](#1-分階段提交並推送)，依 `delivery_profile` 執行相應命令。

### 平行執行離線測試

`check-full.sh` 與 `check-tooling.sh` 使用 pytest-xdist 的 `worksteal` 排程，
將測試分配給多個程序；worker 異常退出時不自動重啟。完整檢查仍執行全部離線測試。
直接執行 pytest 預設循序執行，加 `-n auto --dist worksteal` 才啟用平行模式。

`-n auto` 的程序數由 [tests/parallel_plugin.py](../tests/parallel_plugin.py) 決定：
Apple Silicon macOS 使用 `hw.perflevel0.physicalcpu` 的最高效能層級核心數，
不計入節能核心；查詢失敗時顯示警告並使用一個 worker。
Linux／Windows 使用目前程序可用的邏輯核心數減二，最少一個 worker；
Intel Mac 與其他作業系統沿用 xdist 的實體核心偵測及其後備方式。
此設定只控制程序數，不綁定 CPU 核心，也不限制測試內啟動的子程序或容器。

可透過標準環境變數覆寫自動程序數，`0` 表示循序執行；負數或無效值會失敗。
此環境變數也覆寫 `-n logical`；未設定時，該選項沿用 xdist 的邏輯核心數。
直接使用 pytest 時，明示的 `-n 數字` 優先於此環境變數：

```bash
PYTEST_XDIST_AUTO_NUM_WORKERS=10 scripts/check-full.sh
PYTEST_XDIST_AUTO_NUM_WORKERS=0 scripts/check-tooling.sh
uv run --no-sync python -m pytest -n 4 --dist worksteal --durations=30
```

直接執行 pytest 時，以 `--durations=30` 檢視最慢的測試階段。完整與 tooling 檢查
腳本不讀取 `PYTEST_ADDOPTS`，以固定命令確保測試範圍。程序數增加不保證等比例加速，
須確認記憶體、Docker 資源與測試內部平行工作量。平行測試應隔離檔案、資料庫與
port，清理只處理自己擁有的程序；session fixture 會在各個使用它的 worker 分別建立。
未指定 `--basetemp` 時，每次平行執行會建立專屬 pytest 暫存根並於結束後清除，
不清掃其他 pytest 執行的目錄；既有自訂測試工作區仍逐案例清理。
偵錯時使用 `-n 0`，或將上述環境變數設為 `0`。

### 依賴相容性限制

本專案使用 Python 3.14、XGBoost 3 及 FLAML 基礎套件；可安裝不代表所有上游
功能均已驗證。修改 ML 依賴時須保留各 learner 的實際訓練與
評估驗證，入口見 [tests/domain/ml/](../tests/domain/ml/) 及
[整合測試](../tests/integration/)。安裝問題亦見 [README](../README.md)。

### 容器建置與驗證

[Dockerfile](../Dockerfile) 使用與專案相同的 Python 次版本，透過
`uv sync --locked --no-dev --no-editable` 安裝執行環境；Ruff、Mypy 與內建策略的
ML 套件屬執行期依賴，不能因為容器部署而移除。品質設定隨 wheel 收錄。
映像採多階段建置，執行層只複製安裝環境；[.dockerignore](../.dockerignore) 限定傳入建置的檔案。

獨立策略 worker 使用 `strategy-runtime` target。完整離線測試包含真實 Linux 容器邊界
檢查，需先準備可透過本機 Unix socket 連線、支援 cgroup v2 資源限制的 Docker Engine
與此映像；所有支援平台都不接受 SSH 或 TCP 遠端端點。建置另需 Docker CLI 的
Buildx 元件，不能只以 Engine 可用判定工具已齊備。先執行 `docker buildx version`，
再以 `docker buildx create --help` 確認包含 `docker-container` driver、
`docker buildx build --help` 確認支援 `--load`；缺少命令或能力時須先安裝或更新
Docker 安裝來源提供的 Buildx 元件，不能繼續下列建置。

在 repository 根目錄為本次檢查建立專用 builder 與標籤，明確選用 builder 而不切換預設 builder：

```bash
tradingdev_check_id="$(uv run --no-sync python -c 'import uuid; print(uuid.uuid4().hex)')"
tradingdev_check_builder="tradingdev-check-${tradingdev_check_id}"
tradingdev_check_image="tradingdev-strategy-runtime:check-${tradingdev_check_id}"
docker buildx create --name "$tradingdev_check_builder" --driver docker-container
docker buildx build --builder "$tradingdev_check_builder" --load \
  --target strategy-runtime -t "$tradingdev_check_image" .
export TRADINGDEV_STRATEGY_IMAGE="$tradingdev_check_image"
```

macOS 可使用 Docker Desktop 提供的 Linux Engine。
測試本身不下載映像；缺少 daemon、
必要能力或預建映像會失敗。`TRADINGDEV_STRATEGY_IMAGE` 可指定專用測試標籤，執行前
會解析為固定 image ID。建立 adapter 前，以 `TRADINGDEV_RUNTIME_LIMITS` 的 JSON 物件設定測試資源政策；
未設定時使用預設政策，直接呼叫明示的 `limits` 優先。可設定欄位、預設值及
欄位間限制以 [`RuntimeLimits`](../src/tradingdev/domain/runtime.py) 為準。

直接整合或診斷 adapter 時，查閱[容器執行 adapter](../ARCHITECTURE.md#容器執行-adapter)，
取得 `DockerRuntime.resolve/open`、`ContainerSession.exchange`、生成策略呼叫與身分綁定、
清理失敗判定及 `recover_abandoned` 的契約。傳輸、程序身分與故障清理的驗證入口為
[替代 daemon 的 adapter 測試](../tests/adapters/test_container_runtime.py)及
[真實容器測試](../tests/integration/test_strategy_isolation.py)。
生成策略資料協定另由 [codec 測試](../tests/adapters/test_frame_codec.py)及
[直接 worker／proxy 測試](../tests/integration/test_generated_protocol.py)驗證。

變更 worker、資料協定或其依賴後，須從待驗證內容重建策略映像，再執行相關測試。
固定 image ID 不會隨標籤重建而改變。保留上述變數值；確認本次 host 工作已停止、
所有關聯執行容器均已停止並移除，且映像沒有其他執行引用後，才在同一個 shell 清理：

```bash
docker image rm "$tradingdev_check_image" &&
  docker buildx rm "$tradingdev_check_builder"
```

移除專用 builder 時不加 `--keep-state`，會一併移除其建置快取。
只清理本次資源，不使用 force 或全域 prune。任一步驟失敗或無法確認清理時，
停止後續清理並保留剩餘資源、控制目錄及紀錄，依上述 adapter 復原契約處理。

MCP runtime target 含 Docker CLI，Compose 只把本機 socket 掛給可信控制端；操作權限與
安裝設定由 [README 的 Docker 部署](../README.md#使用-docker-部署) 說明。

若開發檢查回報未確認清理，保留所列控制目錄；停止相應 host 工作並恢復原 Docker daemon
後，依[容器執行 adapter 的復原契約](../ARCHITECTURE.md#容器執行-adapter)選定控制目錄並重試。

容器修改除完整離線檢查外，須驗證 Compose 解析、實際建置與 HTTP MCP 連線，
並檢查非 root 身分的工作區寫入、生成策略驗證所需工具／套件、容器重建後資料讀取，
以及錯誤 Host／Origin 的拒絕行為。使用獨立 Compose project、臨時工作區及映像標籤，
結束後先停止並移除該 project 容器，再清理測試資料與專用映像／建置快取；
不要清理共用 Docker 資源。未實測的 CPU 架構與 NAS 平台須明列限制。
使用者部署、設定與備份操作統一見 [README](../README.md#使用-docker-部署)。

## 測試設計與精簡

本節是落實 [AGENTS.md 測試政策](../AGENTS.md#測試與清理) 的本機設計基準。
涉及測試設計、修改、刪減或品質審查時讀取本節即可，不需閱讀外部準則或整份開發指南。
授權、必要檢查與清理依 AGENTS.md；下列方法按變更風險選用，不增加固定比例或工具門禁。

1. **撰寫前確認責任**：先辨識要保護的需求／契約、觸發情境與預期錯誤，再查既有
   測試。選擇能可靠捕捉該錯誤、成本合理的層級；高層測試聚焦跨層接線、序列化、
   設定傳遞、狀態與副作用，避免在各層重跑完整業務分支。不同使用端、模型實作及
   程序邊界可能有獨立風險，斷言相似不代表重複。
2. **依風險選案例**：從正常、邊界與失敗路徑選擇代表輸入。同樣處理方式的輸入
   可歸為等價類別，再涵蓋邊界值、狀態轉移及會改變結果的條件組合，避免無理由
   枚舉同類數值。公開預設值、schema 約束與回傳型別若影響契約，仍有驗證價值。
3. **斷言對準行為**：預期值來自契約、可手算案例、獨立參考或明確不變量，
   避免複製被測實作當答案。比較重複結果可驗證確定性，不能取代正確性與輸入
   敏感性驗證。名稱反映實際保證；測平倉先確認進場再確認期限內退出，測設定
   生效需觀察設定的影響，測 JSON 相容性就實際序列化，不能只檢查欄位或長度。
   內部名稱、簽名與呼叫順序，僅在其本身是必要契約或保護副作用時固定。
4. **避免斷言空跑**：檢查空集合、未進入的條件與零次迴圈，確保目標情境確實
   發生。測拒絕時讓其餘條件有效，核對錯誤原因與必要副作用，避免因別的錯誤
   而通過。刻意測空結果或不拋錯時，明確檢查該接受條件。
5. **控制替代物與重用**：mock／fake 適合隔離外部依賴、昂貴流程或製造失敗，
   不替換正在驗證的行為；只檢查 mock 回傳值不代表真實整合成立。共用契約可
   用有可辨識名稱的參數化案例，重複設定可抽成隔離 fixture，同時保留各實作及
   邊界必要的真實路徑。允許有助理解的重複，避免為縮短行數建立複雜框架，
   或讓案例依賴執行順序與前一案例的可變狀態。
6. **完成前驗證偵錯能力**：回歸案例先確認能因原缺陷失敗，再驗證修正後通過。
   高風險或斷言效力有疑慮時，可用反例、隔離環境內可還原的故障注入或局部
   mutation testing 確認測試會拒絕錯誤行為，不需每次跑全套 mutation testing。
   記錄實際驗證範圍；單一測試漏抓不代表整套失效，全部通過也不證明沒有缺陷。
7. **精簡前核對保護能力**：比對輸入、執行路徑、契約與故障種類，在差異審查／
   完成回報指出保留的測試及涵蓋範圍；有獨立邊界風險就保留。弱斷言但需求仍
   必要時優先改寫。資料洩漏、執行身分、保存完整性、授權與程序清理的驗證不能
   因精簡而遺失。重複設定與多餘案例分別評估，不只憑名稱、行數、coverage
   重疊或一次故障注入決定刪除；沒有合理預期效益時不為減少行數而改寫。

## 更新文件

| 文件 | 維護責任 |
| --- | --- |
| [軟體需求規格](requirements.md) | 產品需求、驗收條件與契約／驗證入口；按責任分冊。 |
| [README](../README.md) | 使用者安裝、操作與結果解讀。 |
| [架構](../ARCHITECTURE.md) | 模組責任與跨層流程。 |
| [策略契約](strategy_contract.md) | 策略作者的 Python／YAML、訊號、訓練與 revision 要求。 |
| [執行產物](run_artifacts.md) | Workspace、身分、保存格式及歷史查詢入口。 |
| [ML 研究](ml_research.md)、[執行計畫](execution_plans.md) | 各自的輸入、驗證、保存及執行邊界。 |
| [績效產物](performance_artifacts.md)、[報告](reports.md) | 保存數值及觀察值的語意、報告呈現與交付。 |
| [簡報編輯指南](talks/tradingdev-introduction/README.md) | 編輯者的內容與版型維護、預覽、打包及日常發布。 |
| 本指南 | 工具入口、負責範圍、操作方式與測試維護位置。 |
| [AGENTS.md](../AGENTS.md) | 唯一代理開發政策來源。 |

依 [文件政策](../AGENTS.md#政策與溝通) 判斷本次變更是否改變使用方式、對外語意、
資料保存／讀取、相容性或安全邊界。修正受影響的敘述；既有說明仍正確且足夠時，
不因實作或測試調整再追加段落。契約保留行為與限制，schema、catalog、常數與
正式 model 負責可機械取得的定義，不逐欄複製到 Markdown。

整理契約時先找主要維護來源，合併重複敘述，再提供按需入口。參照應包含「何時讀、
檔案與類別／函式或工具、可取得的資訊」，例如設定欄位查 `BacktestRunConfig`，
MCP 使用端則取得 `get_strategy_contract(section="configuration")`。讀者需要的
範例、重要語意及失敗處理仍保留；新增小文件不是精簡的必要步驟。

新增、改名或拆分契約時，同步檢查 [review_scope.py](../tooling/review_scope.py) 的責任選取
及 [test_review_scope.py](../tests/tooling/test_review_scope.py) 的證據涵蓋。新文件會自動納入
審查，但未知章節保守保留，拆檔本身不代表減少上下文。量測應分開報告文件閱讀量、
審查輸入與產品 MCP；不把 bytes 當成 tokens。

PR 文件 reviewer 無法自行開啟參照。審查需要的正式定義若未包含在原始證據中，
以 `--review-context-for SCOPE:PATH` 補入同一候選 Git 版本的完整檔案；跨範圍問題
使用 `cross_scope`。參照本身不證明內容正確，也不會自動載入來源；必要內容缺失時
依 finding 補齊證據後重查，不為通過審查把 schema 重抄回文件。

交付文件、簡報或工具指引前，依[整體差異的計畫](#1-分階段提交並推送)選擇檢查。
文件檢查不執行文件中的命令或驗證外部網址，也不代替語意與操作審閱。

### SRS 維護與檢核

`scripts/check-docs.sh` 自動檢查 `docs/requirements.md` 與 `docs/requirements/`
中的需求。規則及審查責任唯一來源是 [requirements.py](../tooling/requirements.py)：
`check_requirements` 拒絕重複／格式錯誤的編號、缺少需求／驗收／契約／驗證欄位、
未從入口連到的章節，以及未登錄於 `DOCUMENT_SCOPES` 的需求文件。
同一份責任登錄直接供 reviewer 選取文件，不另維護第二份章節清單。

需求使用二級標題 `FR-EXEC-001 名稱` 或 `NFR-UI-001 名稱`，四個頂層清單欄位以
`**需求**：`、`**驗收**：`、`**契約**：`、`**驗證**：` 開始。
契約欄連到本機 Markdown 詳細契約，驗證欄連到 `tests/` 中的 `test_*.py`；
沒有自動化證據時，可用 `人工驗收：` 說明操作、預期結果與限制。
檢查器核對這些格式與引用存在性，失敗會指出檔案、需求及原因；不匯入或執行測試，
不驗證測試函式、驗收文字的語意或覆蓋充分性。引用是查核入口，不是通過收據。
SRS 入口與章節都不存在時允許尚未引入 SRS 的版本；部分缺失或導航斷裂則失敗。

SRS 的結構與引用由文件檢查驗證；需求與契約的一致性納入既有候選審查，
操作及限制見[PR 檢查](#2-在-github-建立-pr交由審閱者檢查)。

### 簡報的編輯與輸出

[簡報入口](talks/tradingdev-introduction/README.md)提供編輯、預覽與發布操作。
命令列建置前需備妥 Node.js 22.17 以上與鎖定的 Marp 工具鏈。
手動初始化工具鏈時另需 npm，再執行：

```bash
./scripts/presentations.sh setup
```

`setup` 會下載 npm 套件；命令列 build／watch 預設及 check 使用已安裝工具鏈。
執行 `setup` 前必須已安裝 Node.js 與 npm——它只安裝
[package.json](../tooling/marp/package.json) 定義並由 lock 鎖定的 npm 工具鏈，
不會安裝 Node.js 或 npm 本身。若報錯提示
`Node.js is unavailable` 或 `Node.js and npm are required`，
請先安裝 Node.js 與 npm，而不是執行 `setup`。
依賴更新須同步 `tooling/marp/package-lock.json` 並重新初始化。

#### 在 VS Code 儲存時自動編譯

[VS Code Tasks](../.vscode/tasks.json) 提供 `slides: watch` 與 `slides: build`。
備妥 uv 與 Node.js 後，在可信任的工作區允許自動工作，或以 **Tasks: Run Task** 啟動。
兩個工作會先依 `uv.lock` 同步專案 Python 環境及全部 extras，再檢查 Marp 工具鏈；
缺少套件或安裝版本與 npm lock 不符時才執行安裝，已齊備時直接開始建置或監看。
工具鏈需要安裝或修復時另需 npm；首次使用或鎖定依賴更新後可能需要網路，不會自行更新 lock。
Node.js 及 Mermaid 所需的 Chrome／Chromium 仍須先安裝；依賴安裝失敗時不會開始編譯。

兩個工作共用安裝鎖，避免同時初始化 Marp。監看開始後，儲存來源只會重建 HTML；
變更依賴後須重新啟動工作。結束使用 **Tasks: Terminate Task**。
命令列需要相同行為時，執行
`uv run --locked --all-extras python -m tooling.presentations build --ensure-toolchain`，
或將 `build` 改為 `watch`。此選項只適用於 build／watch，
不修改 `TRADINGDEV_PRESENTATION_TOOLCHAIN` 指定的外部工具鏈；外部安裝須由其維護者修復。
品質檢查與打包入口維持不下載依賴。
詳細操作見[簡報入口](talks/tradingdev-introduction/README.md#編輯與預覽)。

#### 手動輸出與預覽

```bash
./scripts/presentations.sh build
./scripts/check-presentations.sh
uv run python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/talks/tradingdev-introduction
```

瀏覽 `http://127.0.0.1:8765/` 確認版面、頁碼、導覽與講者模式；結束伺服器按
`Ctrl+C`。Git 保存來源與資源；本機產生的 `index.html` 由 `.gitignore` 忽略。

簡報採整包離線交付，`assets/` 圖片及 `evidence/` 報告沿用相對路徑。
`scripts/check-presentations.sh` 另核對交付 HTML 的 `href`、`src`、`poster`、`xlink:href`
靜態引用與 HTML 片段；這些本機引用須指向交付範圍內的檔案，整包不接受符號連結。
報告內的 CSS、`srcset` 與 JavaScript 動態資源不在此連結檢查範圍，須另用瀏覽器驗證。
案例證據依 [presentation_bundle.py](../tooling/presentation_bundle.py) 的
`evidence_content_allowed` 登錄相對路徑角色與格式，新增、更新或刪除已登錄的報告、
資料、截圖、下載包及保存的 `strategy.py` 走簡報專項；保存的策略只供閱讀／下載，
檢查不匯入或執行它。分類同時檢查變更前後的 Git 內容與檔案模式，未知角色、
不符格式、可執行檔、符號連結或混合產品修改仍走原有保守分流。
格式辨識不證明報告數值、ZIP 內容或策略正確，案例語意、來源與完整性仍須另行審閱。
`scripts/presentations.sh package --output <新資料夾>` 檢查後建立資料夾及同名 ZIP，
不覆寫既存輸出。打包與檢查都從來源建置，無須先保存預覽 HTML。
`scripts/stage-presentations.sh` 使用已安裝的簡報工具編譯來源，以 Python 標準函式庫
驗證成品後，將 HTML、圖片與案例證據放入 `site-dist/tradingdev/`；成功後才替換舊發布內容。
直接執行時預設使用 `python3`，可用 `TRADINGDEV_CHECK_PYTHON` 指定 interpreter；
腳本不選擇 Python 版本，Cloudflare 的 Python 3.14 由下方建置變數固定。
Cloudflare 使用 `scripts/build-presentations-site.sh` 先安裝鎖定的 npm 工具與 Chromium，
再呼叫上述編譯及發布準備步驟，不安裝 TradingDev 的 Python 產品依賴。
打包、解壓與發布的使用方式見[簡報編輯指南](talks/tradingdev-introduction/README.md)。
專項檢查在臨時目錄產生與驗證完整交付，不依賴本機預覽檔，也不覆寫工作樹；缺少工具鏈會失敗。
可用 `TRADINGDEV_PRESENTATION_TOOLCHAIN` 指定另一套已初始化且 lock 相符的工具鏈。

含 Mermaid 圖表時，build／check 及相關工具測試另需本機 Chrome，或以
`TRADINGDEV_PRESENTATION_BROWSER` 指定 Chromium 執行檔；`setup` 安裝雲端 Chromium
套件，不安裝本機 Chrome。
[render-diagrams.mjs](../tooling/marp/render-diagrams.mjs) 使用鎖定的 Mermaid 與 Noto Sans TC
在獨立 headless 瀏覽器產生 SVG，圖表頁面不載入網路資源；完成或失敗後關閉瀏覽器，
本機 Python 入口的程序監督器負責逾時清理。Cloudflare staging 直接呼叫 Node，
沒有這層 Python 逾時監督。[build.mjs](../tooling/marp/build.mjs) 是本機與雲端共用的
編譯入口。SVG 與所需字型嵌入 HTML，觀看端不需要建置工具。
[presentation-url.mjs](../tooling/marp/presentation-url.mjs) 讀取簡報來源的網址設定，
以鎖定的 QR 套件產生 PNG 並嵌入 HTML，同步填入圖片及頁尾連結。
此步驟不呼叫外部圖片服務，也不需要瀏覽器；設定與編輯方式見
[封面網址與 QR code](talks/tradingdev-introduction/README.md#封面網址與-qr-code)。
替代工具鏈的套件宣告、lock 及渲染腳本都須與受查 snapshot 相符。
瀏覽器版本仍可能影響 SVG 版面；須以瀏覽器確認本機與分支預覽的圖表。本機使用已安裝
的 Chrome 或明示執行檔；渲染器在 `WORKERS_CI=1` 的建置環境選用鎖定的 Linux x64
Chromium 套件與內附函式庫，在獨立臨時目錄解壓並於正常關閉後清理；
若無法確認瀏覽器關閉，會回報錯誤並保留該目錄。雲端使用該套件的
serverless 啟動參數，包含停用瀏覽器 sandbox；僅在隔離的建置環境渲染 repository 來源，
渲染時封鎖外部網路請求，不將此流程宣稱為執行任意來源的安全隔離。
圖表編輯方式與支援範圍見[簡報編輯指南](talks/tradingdev-introduction/README.md#頁面版型與撰寫方式)。

### 簡報網站發布設定

本節供維護簡報發布環境的管理者使用。日常編輯、檢查與發布確認見
[簡報編輯指南](talks/tradingdev-introduction/README.md#檢查與發布)。
變更網站設定前先確認使用的發布服務與網域責任；一般內容修改不需要重新設定網站。

#### Cloudflare 自動發布

Cloudflare Workers Builds 可連接本 repository，在 `main` 更新時從 `slides.md`、`theme.css`
與 Mermaid 建置 `index.html`，再將 HTML、`assets/` 與 `evidence/` 一起發布。
Git 不保存產生的簡報 HTML；雲端只安裝簡報專用的 npm 工具與瀏覽器，
成品驗證使用 Python 3.14 標準函式庫，不安裝 TradingDev 產品依賴或執行產品測試。

在 Cloudflare 的 **Workers & Pages → Create application → GitHub** 選擇本
repository；GitHub 授權範圍只選需要發布的 repository。設定如下：

| 欄位 | 設定 |
| --- | --- |
| Worker name | 與根目錄 [wrangler.jsonc](../wrangler.jsonc) 的 `name` 相同。 |
| Production branch | `main`。 |
| Root directory | repository 根目錄。 |
| Build command | 下方的安裝與編譯指令。 |
| Deploy command | `npx wrangler deploy`。 |
| Preview command（非正式分支） | 啟用分支預覽時使用 `npx wrangler preview`。 |

Build command：

```bash
bash scripts/build-presentations-site.sh
```

在 **Settings → Build → Build Variables and Secrets** 加入
`SKIP_DEPENDENCY_INSTALL=1`、`NODE_VERSION=24.22.1` 與 `PYTHON_VERSION=3.14.8`。
Workers Builds 自動提供 `WORKERS_CI=1`，供渲染器選擇雲端瀏覽器；
此變數由平台注入，建置腳本與 `wrangler.jsonc` 不設定它。
停用自動安裝後，由建置腳本只安裝簡報依賴。部署仍會使用 Wrangler；
Workers Builds 可自動建立部署 token，不需將 token 放進 repository。
`site-dist/tradingdev/` 是忽略的發布產物，每次重新建立，只包含簡報 HTML、圖片與明列的
案例證據；不會將整個 `docs/` 或一般使用者 workspace 發布。
`wrangler.jsonc` 的 `build.command` 呼叫 `scripts/stage-presentations.sh`，
使用已安裝的工具重新編譯及驗證成品；直接執行 Wrangler 前須先準備簡報工具與瀏覽器。
Workers Builds 的 Build command 由 Cloudflare 獨立保存；請將 **Production** 與
**Preview base** 同步設定為上述安裝與編譯腳本。
既有分支預覽保留各自的設定，不會因修改 Preview base 自動同步；亦須進入該預覽的
**Settings** 更新建置命令、建置變數與監看路徑，再重試建置。

分支預覽使用 `wrangler.jsonc` 的 `previews` 區塊；即使沒有獨立設定，也須保留空物件。
靜態資源仍使用頂層 `assets`，正式網域的 `routes` 不放入 `previews`。
在建置記錄確認分支與 commit，成功後開啟產生的預覽網址並加上 `/tradingdev/` 查看簡報。
分支預覽與 `main` 正式部署是兩次獨立建置，須分別核對結果；本機簡報檢查不代表雲端部署成功。

在 **Settings → Builds** 的 **Production** 與 **Preview base** 分頁，分別設定
**Build watch paths**：從 **Include paths** 移除預設的 `*`，加入以下路徑，
並讓 **Exclude paths** 保持空白。

- `docs/talks/tradingdev-introduction/slides.md`
- `docs/talks/tradingdev-introduction/theme.css`
- `tooling/marp/**`
- `tooling/presentation_bundle.py`
- `tooling/presentation_html.py`
- `tooling/presentation_errors.py`
- `scripts/build-presentations-site.sh`
- `docs/talks/tradingdev-introduction/assets/**`
- `docs/talks/tradingdev-introduction/evidence/**`
- `scripts/stage-presentations.sh`
- `wrangler.jsonc`

這項設定由 Cloudflare 保存，用來略過一般程式、測試與文件更新，不寫在 `wrangler.jsonc`。
`slides.md`、`theme.css` 或編譯工具的提交會觸發發布；產生的 HTML 不作為監看輸入。
Cloudflare 對零檔案變更、至少 3,000 個檔案變更或至少 20 個 commits 的推送會略過
路徑篩選，因此這些事件仍可能觸發建置。核對略過行為時，使用一般的小型文件提交，
不要用空提交作為不應建置的測試。

若建置設定顯示 Git 帳戶中斷連線，按 Git repository 旁的 **Manage**，在 GitHub
完成必要的重新驗證，確認 Cloudflare 應用程式仍可存取本 repository，再回到
Cloudflare 重新整理設定頁。確認連線警告消失、Production branch 為 `main`，
且 **Disable builds** 關閉。下一次主線更新後，核對建置記錄的 commit 與部署結果；
手動重試建置成功不代表 GitHub 自動觸發已驗證。

##### 共用網域與專案路徑

`slides.klwang.tw` 是多個專案共用的簡報網域。Custom Domain 綁定獨立的
`klwang-slides` 入口 Worker，由管理者在 Cloudflare 維護；本 repository
只部署 `klwang-slides-tradingdev`，不接管整個網域。

根目錄的 `wrangler.jsonc` 宣告本專案的兩條 Route：

- `slides.klwang.tw/tradingdev`：不帶尾端斜線的入口。
- `slides.klwang.tw/tradingdev/*`：簡報及其子路徑。

Route 優先於共用入口的 Custom Domain；路由保留原始路徑，因此發布檔案仍放在
`site-dist/tradingdev/index.html`。不要改成 `slides.klwang.tw/*` 或將整個
Custom Domain 綁回本專案，也不要使用會涵蓋 `/tradingdev-other` 的
`/tradingdev*`。不帶尾端斜線且附有 query 的網址由共用入口補上斜線後轉址。

首次設定時，管理者先在 **Workers & Pages → klwang-slides → Domains →
Add domain** 選擇 `klwang.tw`，在 Subdomain 填入 `slides`，啟用於 Production。
網域須屬於同一 Cloudflare 帳號的有效 zone，Cloudflare 負責 DNS 與 HTTPS。
TradingDev 的部署權限須包含該 zone 的
Workers Routes 編輯權限；部署後在本專案 Worker 的 **Domains** 核對兩條 Route。
沿用舊設定時，先建立上述 Route，再將 Custom Domain 移到共用入口，以保留簡報入口。

新增其他專案時，使用獨立 Worker 與不同的路徑前綴，各自連接自己的 repository；
不覆寫本專案 Worker，也不移動共用 Custom Domain。共用入口的根路徑 `/` 回應
HTTP 302，轉址目標（`Location`）為 `https://www.klwang.tw/talks/`；該入口由管理者
另外維護，不隨 TradingDev 發布更新。

簡報位址為 [https://slides.klwang.tw/tradingdev/](https://slides.klwang.tw/tradingdev/)。只有部署成功才會更新網站；
repository 中的設定檔不代表已連接 Cloudflare 或發布。

## MAE 研究原型

[research/mae_analysis.py](../research/mae_analysis.py) 是獨立的最大不利變動研究原型，
不屬於正式 Dashboard。先準備 `data/processed/btcusdt_1m_2024_2025.parquet`，
包含可轉換為 UTC 的 `timestamp` 與數值 `high`、`low`、`close`：

```bash
uv sync --locked --extra dashboard
uv run --locked --extra dashboard streamlit run research/mae_analysis.py
```

開啟終端機顯示的網址；結束按 `Ctrl+C`。此入口不下載行情。
日期範圍包含起訖日的 UTC 全日，以結束日次日 00:00 為排他上界。
分析只保留所選期間內具完整未來持倉視窗的進場點；資料不足時顯示警告，
不產生分位數或停損觸發率。可擴大日期範圍或縮短持倉視窗後重新分析。

## 研究成果工程回饋

需要把策略案例中的共通邏輯、檢查或方法整合回專案時，使用
[research-feedback skill](../.agents/skills/research-feedback/SKILL.md)。它提供來源固定、
共用價值判斷、元件整合、使用端探索與驗證的工作流程；Git、審查及授權政策仍由
[AGENTS.md](../AGENTS.md) 管理。單純建立策略或查看回測不需要此 skill。

在能存取 repository 與來源 workspace 的開發助手中呼叫 `$research-feedback`；
若客戶端未發現 repository skills，明確要求讀取上述 `SKILL.md` 再執行。
只有 MCP 工具存取權的客戶端無法藉此修改 server 程式，也不假設客戶端能讀 server
本機路徑。提供實際可讀取的來源，例如：

> 使用 $research-feedback，讀取 `/absolute/workspace-a` 的策略與指定版本，及
> `/absolute/workspace-b` 的另一個策略與指定版本。這些來源允許用於共用元件開發；
> 僅分享抽取後的邏輯與去識別化合成測資，保留私人策略、行情與原始報告。
> 請判斷是否有值得重用的機制，若有則在本專案完成整合、使用入口與驗證。

先提供策略 ID、選定 revision（或明示採目前版本）、實際授權範圍，以及相關 run／
筆記（若有）；不必為了使用 skill 重跑研究。只有評估需求時可以明示「唯讀分析，
不修改」。已有授權不重複確認，
缺少來源或可共用範圍時補齊後才處理相依工作。若沒有合適的抽取候選，回報理由與
證據即可，不為交付而新增元件。

原 workspace 的 revision、設定、驗證證據、結果與筆記保持唯讀。要驗證新的使用方式，
在隔離工作區建立新策略版本；匯入的共用程式不受舊 revision 雜湊完整凍結，須記錄
實作版本並驗證受影響使用端。需要使用者續查的驗收成果須保留，不能當測試暫存清理。

這是由人啟動、開發助手執行的流程，不會在每次策略保存時自動運行，也不會自行
掃描其他 workspace、分享私人研究或發布更新。效果以實際抽取、使用端採用及驗證
證據判斷；合成案例只能驗證流程，不能證明真實多人研究的改善成效。

## 選擇性 LLM 測試

此工具驗證模型透過真實 MCP 完成策略研究互動及成果交付。修改模型可見的指引、
工具 schema／回覆契約、策略生成修復或模型整合層時，依
[執行時機](../AGENTS.md#執行時機) 選擇受影響情境。
測試入口為 [scripts/check-llm.sh](../scripts/check-llm.sh)，情境與驗證程式位於
[tests/e2e/](../tests/e2e/)，provider 選項定義於 [tests/llm_plugin.py](../tests/llm_plugin.py)。

```bash
./scripts/check-llm.sh codex --llm-timeout 1800
./scripts/check-llm.sh local --llm-model qwen3.8:27b --llm-timeout 1800
```

Codex 需要已登入的 CLI 與所選模型存取權；Local 需要已啟動且支援工具呼叫的本機
服務，預設 `http://localhost:11434/v1`，以 `--llm-base-url` 指定其他 loopback 位址。
工具不下載模型、不自動切換 provider；所選能力不可用會失敗。

入口由 [tooling/llm_check.py](../tooling/llm_check.py) 保留情境選擇與 pytest 選項，
維持循序執行及即時進度輸出；拒絕透過命令列或 `PYTEST_ADDOPTS` 啟用 xdist 平行或監看重跑，
避免 worker 略過主程序的環境確認。離線套件的平行設定不套用到此入口。
在選出實際要執行的情境後、建立測試工作區前共用環境確認：檢查策略容器前置條件，
再確認 Codex CLI 登入狀態或 Local 服務連線。help、列出情境／fixtures、
`--setup-plan` 等純資訊模式與沒有選到情境時不檢查服務；`--setup-only` 會執行
fixtures，仍須確認環境。Local 前置確認只建立有期限的連線，不呼叫模型；它不證明 API、
所選模型、工具呼叫或上下文容量可用，這些能力仍由真實情境驗證。

Codex 產品流程的預設參考配置為 `gpt-6.1-sol` 與 `high`。解析順序為 `--llm-model`／
`--llm-reasoning-effort`、專用環境變數 `TRADINGDEV_LLM_MODEL`／
`TRADINGDEV_LLM_REASONING_EFFORT`、固定基準；不沿用宿主或文件審查的模型設定。
啟動模型回合前核對及記錄實際模型與推理強度，所選能力不可用或不符即失敗。
推理強度的可接受值由 [provider 選項](../tests/llm_plugin.py) 的
`CODEX_REASONING_EFFORTS` 定義；不支援會自動委派的 Ultra。
這些情境驗證單一客戶端透過 MCP 操作，不允許子代理或其他原生工具代做，
也不修改使用者的全域 Codex 設定。

此處指定的是受測模型。驗證某個模型的相容性時，明確指定該模型與推理強度；
其他模型的結果不能代替。此測試不執行程式碼審查或 PR 文件一致性審查。

以 `-k` 選取相關情境，省略時執行全部。可先列出目前可用情境，不呼叫模型：

```bash
./scripts/check-llm.sh codex --collect-only -q
./scripts/check-llm.sh codex -k novice --llm-timeout 1800
```

依變更責任選擇與維護測試：

| 責任範圍 | 測試入口 |
| --- | --- |
| 策略生成、修復與既有結果查詢 | [test_llm_workflows.py](../tests/e2e/test_llm_workflows.py) |
| 自然需求、必要資訊追問與工具探索 | [test_llm_novice_intake.py](../tests/e2e/test_llm_novice_intake.py)、[test_llm_discovery.py](../tests/e2e/test_llm_discovery.py) |
| 執行計畫、使用者決策、訓練、已保存結果查詢與生成策略隔離／資源限制／清理復原指引 | [test_llm_execution_plans.py](../tests/e2e/test_llm_execution_plans.py)、[test_llm_strategy_execution_guidance.py](../tests/e2e/test_llm_strategy_execution_guidance.py) |
| 自動 walk-forward 設定、尾端覆蓋與結果交付 | [test_llm_automatic_walk_forward.py](../tests/e2e/test_llm_automatic_walk_forward.py)（`-k automatic_walk_forward`） |
| 固定參數比較、已保存 scope 比較／排名與參數提案 | [test_llm_comparison.py](../tests/e2e/test_llm_comparison.py)、[test_llm_research_proposal.py](../tests/e2e/test_llm_research_proposal.py) |
| ML 研究與已保存序列 | [test_llm_ml.py](../tests/e2e/test_llm_ml.py)、[test_llm_evaluation_series.py](../tests/e2e/test_llm_evaluation_series.py)（序列情境用 `-k evaluation_series`） |
| 已保存結果解讀、研究筆記更新、封存匯出與既有封存續讀 | [test_llm_research_delivery.py](../tests/e2e/test_llm_research_delivery.py) |
| HTTP 確認單可讀／不可讀、使用者拒絕執行、既有報告與封存網址交付 | [test_llm_http_delivery.py](../tests/e2e/test_llm_http_delivery.py)（`-k http_delivery`） |
| 授權後的草稿清理 | [test_llm_cleanup.py](../tests/e2e/test_llm_cleanup.py) |

測試使用隔離工作區與固定合成行情，不下載市場資料；需要使用者決策的情境會在
同一對話注入模擬回覆。完成驗證後清理工作區。
它不替真人保留報告，也不驗證桌面客戶端的互動。自然語言解讀仍須審閱輸出的最終
回覆與報告評語；機械斷言通過不等於語意正確。使用者實際驗收的成果保存遵守
[AGENTS.md](../AGENTS.md#測試與清理)。

`--llm-timeout` 是每個情境的期限。長流程需要足夠上下文；已驗證的本地 Qwen 配置
為 64K，但不代表所有模型的最低需求。容量不足須調整服務配置後重跑完整情境。
Local 可用 `--llm-temperature`、`--llm-reasoning-effort` 覆寫支援的採樣設定；
未指定時使用本地服務預設，不讀取 Codex 測試的專用環境變數。
`--llm-temperature` 不適用於 Codex。模型版本與設定不同，驗證結果不能直接互相替代。

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
使用 CLI 時明確指定 review 模型及推理強度；以下範例使用 `gpt-6.1-sol` 與 `high`：

```bash
codex -c 'review_model="gpt-6.1-sol"' -c 'model_reasoning_effort="high"' \
  review --base origin/main
```

| 命令 | 適用範圍 |
| --- | --- |
| `codex review --base origin/main` | 相對已取得遠端主線的分支差異；先確認工作樹乾淨並 fetch。 |
| `codex review --uncommitted` | 已暫存、未暫存及未追蹤內容；先排除其他任務修改。 |
| `codex review --commit HEAD` | 指定 commit 引入的差異。 |

CLI 參數以 `codex review --help` 核對。已由獨立 reviewer 涵蓋且無未解疑慮的相同內容
可沿用審查結果。程式碼審查、離線測試、文件一致性與真實模型流程驗證各有用途，
不能互相替代。本機 commit／push hooks 不會自動執行模型 code review。

## 撰寫 issue

建立或改寫 GitHub issue 時，依下列步驟整理內容。發佈授權與接手指派仍依
[AGENTS.md](../AGENTS.md)。

1. **確認問題與讀者**：查核相關 issues 與 repository 現況，每個 issue 聚焦一個
   可獨立追蹤的問題。以未參與對話的接手者為讀者，標題點明問題或預期成果，
   開頭交代受影響的功能、現況與影響，不以對話經過或未解釋的事件代替背景。
2. **依類型補齊資訊**：bug 說明環境、重現方式、預期與實際結果；偶發或無法
   重現時保留已知證據並明示限制。改善工作說明能力缺口、範圍與完成條件；
   調查工作說明待解問題、已有證據與結束調查的判準，不把推測寫成已確認故障。
3. **安排資訊歸屬**：背景交代問題，範圍界定工作邊界，驗收描述可觀察的完成結果。
   每項資訊只在適當位置完整說明，不在各節重列同一份功能、交付或操作清單。
   標題與篇幅依任務調整，不強制套滿模板，也不以刪除必要限制換取簡短。
4. **保留必要參照與取捨**：實作入口只列有助接手的檔案／符號及用途；共用政策
   與既有契約引用負責來源，不複製到每個 issue。區分已確認需求、建議方案與
   待決事項，只保留影響範圍或結果的取捨，不將一般實作選擇包裝成需要使用者決策。
5. **發布前自查**：確認讀者不靠聊天紀錄能理解為何要做、要處理什麼及如何判定
   完成；合併重複資訊，移除過時方案與無關排程討論，保留必要證據和相依關係。

## 接手 issue

決定負責某個 issue 時，將自己的 GitHub 帳號加入 **Assignees**，讓協作者知道
由誰接手；僅閱讀或分析不算接手。AI 協作的判斷與授權依
[Issue 接手與指派政策](../AGENTS.md#issue-接手與指派)。

準備好 repository 的 `.venv` 與已登入本人帳號的 GitHub CLI（`gh`），代入實際值：

```bash
./scripts/claim-issue.sh 123 --repo OWNER/REPO --user YOUR_LOGIN
```

腳本僅支援 `github.com` 的開啟 issue，核對 `YOUR_LOGIN` 與登入帳號相符，
只新增本人、保留其他 assignees，並讀回確認；已指派時不重複變更。
若失敗或結果未確認，依輸出的 issue 連結查看 **Assignees**，必要時手動設定；
已授權的本機工作可繼續。參數說明可用 `./scripts/claim-issue.sh --help` 查看。

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

已有開發分支時沿用。Commit hook 依暫存內容執行文件、簡報或快速檢查。
一般 push 的本機 hook 不執行完整 pytest 或呼叫模型。主線推送保護限於 GitHub remote。

交付前對整個任務差異取得機械可讀的檢查計畫；以下 `origin/main` 應替換成實際任務基準：

```bash
uv run --no-sync python -m tooling.git.change_scope --base origin/main --tree HEAD
```

有尚未提交的最後階段時，先暫存相關檔案並以 `--staged` 取代 `--tree HEAD`，base 仍涵蓋
先前所有階段，不能只比較 `HEAD` 與最後的暫存內容。`--repo <path>` 可指定 checkout。
`verification_planned` 回覆固定 tree 身分與 `ChangeScope.payload` 產生的計畫，並非通過證明。
`delivery_profile` 指定任務交付要跑的 `scripts/check-<profile>.sh`；`commit_profile`
供 hook 使用，`candidate_checks` 是 PR／本機合併候選的檢查順序。
每階段仍只要求受影響的相關測試，當次 PR 候選另外核對必要證據；計畫不新增逐階段完整檢查。
暫存模式不包含未暫存或未追蹤內容；這時提交使用 `git-flow-commit.sh "type: message"`，
不再傳檔名。工具失敗時不能假設可走較輕的分流。

### 2. 在 GitHub 建立 PR，交由審閱者檢查

啟用 Actions 並設定 `OPENAI_API_KEY` secret 後，以 GitHub 使用者帳號提出的非草稿 PR 在開啟、重開、轉為可審查或更新時
可自動觸發[網頁初審](pr_review.md)。[工作流程](../.github/workflows/pr-agent.yml)使用目標 repository 的設定與 GitHub API，
也能審查 fork PR，過程中不執行 PR 程式碼。確認作者具協作者資格後才將 PR 內容、差異、commit 訊息及先前發現送交 OpenAI。

作者在 **Files changed → Conversations** 核對未結討論，包含行號已過時的項目。僅見於一般留言或摘要的問題，
推送前逐項列為 PR 描述待辦，附原留言連結、受查 commit 與驗證情境。在原處回覆修正 commit 與驗證結果，再由審閱者結案。

身分查詢失敗或額度用完時，可在 **Actions → PR-Agent 初審** 查看原因。額度用完會標記當次初審未完成，
下次更新仍會嘗試，可用 **Disable workflow** 暫停。準備合併時仍須執行本機候選檢查：

```bash
./scripts/check-pr.sh 27
```

程式與開發工具候選會先確認所選測試分流的前置環境，再啟動耗時檢查及模型審查；
文件審查亦先確認 Codex CLI 登入，缺少時不啟動審查批次。一般文件與簡報候選
仍只要求其分流需要的工具。環境確認不取代候選內容、版本與執行結果驗證。

須有乾淨且已提交的工作樹、可存取 repository 的 `gh`，以及開啟且指向主線的 PR。
工具在隔離 repository 檢查合併候選，不切換開發分支：

| 候選變更 | 執行檢查 |
| --- | --- |
| 一般文件 | 文件格式、連結與 SRS 結構；涉及 SRS 時再執行 Codex 一致性審查。 |
| 已登錄簡報 | 文件與簡報專項；混合 SRS 時再執行 Codex 審查；視覺操作仍須人工確認。 |
| 開發政策／已登錄開發工具及其已登錄的 Markdown 指引／已登錄工具專用測試支援檔案 | 必要的簡報專項、Codex 文件一致性審查、快速品質與完整 `tests/tooling`。 |
| 產品／依賴／產品共用測試設施／未知或混合產品變更 | 必要的簡報專項、Codex 文件一致性審查、快速品質與完整離線 pytest。 |

查詢 PR 與本機候選的分流規則時，參閱 [change_scope.py](../tooling/git/change_scope.py) 的
`classify_changes` 與 `ChangeScope`。

### 測試證據與文件修正

乾淨 checkout 執行 `scripts/check-full.sh` 或 `scripts/check-tooling.sh` 時，會以 Git
snapshot 執行完整檢查；成功後由工具保存紀錄。手動要求完整檢查仍會實際執行，不讀取
舊紀錄略過測試。未提交的修改會在目前工作樹檢查，並使該分流的舊紀錄失效，不產生
可重用紀錄。兩個 scripts 可加 `--timeout-seconds <正整數>` 設定期限，預設 5400 秒；
PR snapshot 仍使用 PR 命令自己的期限。

PR 與本機合併候選先按整體差異決定檢查範圍，文件語意審查照常執行。隨後由
[候選檢查](../tooling/git/candidates.py) 核對所需分流的最新紀錄：

- 測試輸入的路徑、Git blob 與 mode 全部相同，只有已登錄的一般文件改變時，
  沿用測試結果並補做文件格式、連結及 SRS 結構檢查。
- 程式、測試、政策、已登錄的開發工具與指引、模型提示詞、依賴、簡報或未知路徑改變，
  或環境指紋不同時，重新執行該候選所需的完整分流。`AGENTS.md` 按開發政策處理。
- 紀錄缺失、無法解析、執行中、失敗或不完整時不重用；不以最後一個文件 commit
  掩蓋更早的未驗證程式修改，也不退回較舊的成功紀錄。

[紀錄工具](../tooling/git/receipts.py) 將每個分流的最新狀態保存於 Git common directory
下的 `tradingdev-checks/`，不加入版本控制；同分流使用鎖避免同時寫入。每次先保存
執行中狀態，才探查環境或補做檢查，避免後續磁碟寫入失敗時留下舊成功紀錄。
一般文件品質失敗仍阻擋候選；確認清理與環境正常後，可保留原測試 tree 的成功紀錄，
修正文件後只需補驗證文件。逾時、中斷或清理未確認則不恢復該紀錄。紀錄綁定實際
通過的 tree、指令、檢查工具內容及環境指紋。環境包含 Python 與已安裝套件、作業系統、
可用 CPU、Git／uv／Node、已安裝的簡報套件與瀏覽器；完整分流另核對 Docker CLI、
daemon 身分／版本／資源及容器映像。Python 與 Node 套件核對已安裝清單及檔案的
路徑、mode、大小與修改時間；不接受無檔案清單的 Python 套件或連結的 Node 套件目錄。
環境變數只保存雜湊，不把憑證內容寫入紀錄；runner 管理的工作目錄、暫存路徑及
主機的呼叫／工作階段追蹤身分不參與比較。權限、sandbox 與未知設定仍參與比較。
工具無法確認環境時仍執行檢查，但不產生可重用成功紀錄。這是本機執行證據，並非
防竄改或跨機器驗證機制；不匯入手寫摘要或把既有日誌轉成成功紀錄。

重用成功會輸出 `test_evidence_reused` JSON，列出 `tested_tree`、`candidate_tree`、
`profile` 與紀錄位置；這表示先前測試仍適用，不代表當次又執行了 pytest。
若要排查新疑慮，直接執行對應的完整檢查 script 即可強制重新驗證。

### 檢查期間的主線更新

檢查期間若只有主線前進，且更新僅涉及與 PR 不重疊的已登錄簡報檔案，工具可重建
候選並補做簡報檢查；PR 身分與 head 必須不變，PR 本身也不能涉及簡報或其工具。成功時輸出
`pr_base_refreshed` JSON，列出最新 base／head／tree、沿用的 `reused_tree` 及補做的
`checks`。審閱者仍須確認新增差異的語意與必要視覺行為。

其他主線變更、改寫歷史或補驗證期間再次變動均須重新檢查；合併前仍要
[核對即時遠端版本](#3-在-github-審閱並合併)。若需追查自動補驗證為何不適用，
查看 [change_scope.py](../tooling/git/change_scope.py) 的 `presentation_only_advance`，
取得路徑重疊、檔案類型及候選差異的判定條件。跨次測試重用另依
[測試證據規則](#測試證據與文件修正)。

### 文件審查設定

涉及 SRS 的文件／簡報、開發工具與完整程式候選需要 Codex 文件一致性審查，
須有 CLI 登入與額度。預設模型為 `gpt-6-luna`、推理強度為 `high`；以
`TRADINGDEV_DOC_REVIEW_MODEL`、`TRADINGDEV_DOC_REVIEW_REASONING_EFFORT` 覆寫：

```bash
TRADINGDEV_DOC_REVIEW_MODEL=gpt-6-luna ./scripts/check-pr.sh 27
```

檢查會輸出實際模型與推理強度，不讀取使用者的全域 Codex 設定，也不在失敗時
自動切換模型。上述設定只影響文件審查；真實模型策略測試另依
[選擇性 LLM 測試](#選擇性-llm-測試) 執行，開發階段的獨立 code review 也須另行設定。

### 規劃證據與補充來源

審查輸入包含完整變更檔案、原始 diff 及相關文件原文，大小不只取決於修改行數。
選擇模型或排查容量失敗前，先規劃固定 Git 版本的證據；以下命令不呼叫模型，
產生計畫也不代表審查通過：

```bash
uv run --no-sync python -m tooling.review_docs --list-scopes
uv run --no-sync python -m tooling.review_docs \
  --repo /absolute/path/to/repository --base origin/main --tree HEAD --plan-only
```

依規劃 JSON 核對下列資訊：

| 要確認的問題 | 輸出與讀法 |
| --- | --- |
| 變更由哪個範圍審查？ | `path_owners`、`selected_scopes` 列出負責範圍；owner 不代表所有背景都必須載入。 |
| 為何選到這些文件？ | `behavior_contexts`、`requirement_contracts`、`development_contexts` 與各批 `document_sections` 列出責任及來源；`fallback_scope` 表示仍採該 owner 的保守背景。各 owner 批次都會承接同 owner 其他變更路徑的 registered behavior／development obligations，即使那些路徑在不同的 source units。 |
| 必要來源是否在同一批？ | `source_contexts` 列出登錄來源及 Git 版本，再核對各批完整檔案與章節；交付清單或雜湊不等於原文，也不是審查通過證明。 |
| 容量花在哪裡？ | `evidence_utf8_bytes` 是唯一原文；`delivered_original_*` 與 `repeated_original_*` 顯示交付及重複量，`prompt_overhead_*` 是指引與封裝成本。 |
| 模型實際收到多少輸入？ | 查看各 `batches` 的 prompt 大小及 `total_prompt_*` 合計；`source_pool_utf8_bytes` 只是本機資料池，不是模型輸入。 |

大小分別以 UTF-8 bytes 與 Unicode 字元計量，兩者都不是 tokens。若預設容量讓
`--plan-only` 中止，可明示容量選項取得完整計畫；這不代表已核實模型容量。
先檢查內容歸屬與重複，再核對所選模型容量並預留輸出空間，才設定實際審查上限。
容量不足時，依錯誤訊息確認超量來自完整 diff、固定來源或文件脈絡。工具可將
文件脈絡分批，且每批都會重複完整的 assigned files 與 diff；partition 不能拆開
assigned files。各 owner 批次的跨路徑承接會把同 owner 的 registered obligations
放進該批的 selected context，即使那些路徑不在同一 source unit；這不限於保守 fallback，
估容量時要一併計入。
若單批 prompt 仍超過上限，須拆分變更，或核實模型容量後調整上限。

無工具 reviewer 無法自行開啟參照。必要原文須出現在需要它的批次；來源清單或
其他批次的結論不能代替原文。缺少來源時，以 `--review-context-for SCOPE:PATH`
補入完整候選檔案；跨範圍比對使用 `cross_scope`。各批均須獨立通過。

補充及必要背景來源須為一般 UTF-8 文字檔，不接受符號連結。一般變更中的
符號連結只審查連結自身，不讀取其目標內容。

| 選項 | 使用目的 |
| --- | --- |
| `--review-scope SCOPE` | 增加審查責任範圍，不可漏掉變更所屬範圍；可重複。 |
| `--review-context PATH` | 為所有批次補入候選 Git 中的完整文字檔；可重複。 |
| `--review-context-for SCOPE:PATH` | 僅為指定範圍或 `cross_scope` 補入完整檔案；可重複。 |
| `--max-evidence-bytes`、`--max-prompt-bytes`、`--max-total-prompt-bytes` | 分別限制唯一原文、單批 prompt 與總 prompt；先量測再設定。 |
| `--documentation-timeout-seconds` | 所有文件審查批次共用的期限。 |
| `--snapshot-timeout-seconds` | 品質／測試 snapshot 的期限，不改變測試範圍或工具自身期限。 |

例如為單一範圍補上必要來源，並依實測耗時設定期限：

```bash
./scripts/check-pr.sh 27 \
  --review-context-for interfaces:src/tradingdev/mcp/server.py \
  --documentation-timeout-seconds 900 --snapshot-timeout-seconds 7200
```

請以實際規劃的 scope／檔案代換，預設值與其他選項見 `scripts/check-pr.sh --help`。
文件審查範圍不縮減測試範圍；變更全文與 diff 完整保留，背景章節使用原文。
缺少證據、容量不足、finding 或逾時均須處理後重查，不截斷證據或忽略失敗。

維護或排查選取規則時，使用以下入口；修改登錄前須核對相關使用端與契約，
並以證據涵蓋測試確認必要原文仍在：

| 維護任務 | 正式入口與可取得的資訊 |
| --- | --- |
| 核對路徑責任與所需來源 | [review_scope.py](../tooling/review_scope.py) 的 `behavior_context_for_path`、`development_context_for_path`、`source_context_for_path`，提供行為、開發操作及必要來源的登錄。 |
| 維護產品文件的章節責任 | [review_scope.py](../tooling/review_scope.py) 的 `behavior_context_for_section` 提供章節與需求／契約的對應；登錄與證據涵蓋測試見 [test_review_scope.py](../tests/tooling/test_review_scope.py)、[test_review_plan.py](../tests/tooling/test_review_plan.py)。 |
| 追查簡報參照選取 | [review_evidence.py](../tooling/review_evidence.py) 的 `_presentation_context`、`_operation_definitions_only`、`automatic_document_paths`，判定何時需要編輯指南或完整簡報來源。 |
| 追查容量及批次完整性 | [review_plan.py](../tooling/review_plan.py) 的 `_selected_evidence_size`、`build_review_plan`、`ReviewPlan.verify`，提供原文計量、批次規劃與完整性核對。 |

### 檢查失敗與程序清理

Codex 執行失敗時，先看批次身分、退出碼與錯誤摘要；診斷包含 stderr 尾端及可辨識的
`error`／`turn.failed` 事件。過長診斷以 `[truncated]` 標示，這只縮短錯誤輸出，
不裁減審查證據。日誌可能包含服務回傳的敏感內容，分享前須檢查；工具不會自動發布。
執行失敗或無效的審查回覆都不算通過，須排除原因後重查。

退出碼 `86` 表示程序清理未確認，相關工作區會保留；依輸出定位未停止的程序與
保留路徑，確認停止後才能清理。需要解析 JSON 診斷時，查看
[ProcessCleanupError.payload](../tooling/process_guard.py) 的欄位定義；文字 `message`
只供顯示。一般失敗或已確認停止的逾時仍會自動清理。

長時間檢查保存完整日誌，透過[完成通知 skill](../.agents/skills/wait-for-check/SKILL.md)
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
