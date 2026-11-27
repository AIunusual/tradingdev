# TradingDev

用對話把交易想法變成可驗證的策略。TradingDev 讓你的 AI 助手撰寫策略、
檢查程式、執行歷史回測、比較參數，並保存每次研究的設定與結果。
目前提供歷史研究功能，不提供真實下單。

你可以從內建策略開始，也可以直接描述自己的規則。助手透過 MCP
（讓 AI 呼叫外部工具的標準協定）操作 TradingDev；回測在執行 server 的主機上進行。
TradingDev 本身不提供對話模型，需搭配支援 MCP 工具呼叫的 AI 客戶端。

你提供策略構想、標的、回測期間與 K 棒頻率，助手負責撰寫與檢查策略。
構想可以先用白話描述；缺少上述資訊時，助手會先追問，不用先替策略取名。正式執行前，你會收到
一份確認單，核對設定並同意後才開始回測；完成後可在對話中解讀結果或開啟圖表。

- [安裝與連接助手](#安裝與連接助手)
- [使用 Docker 部署](#使用-docker-部署)
- [完成第一次回測](#完成第一次回測)
- [建立與研究自己的策略](#建立與研究自己的策略)
- [研究機器學習訊號](#研究機器學習訊號)
- [查看結果與圖表](#查看結果與圖表)
- [資料與工作區](#資料與工作區)
- [終端機用法](#終端機用法)
- [常見問題](#常見問題)
- [相關文件](#相關文件)

## 安裝與連接助手

若要部署在伺服器或 NAS，可直接使用下方的 [Docker 部署方式](#使用-docker-部署)，
主機不必另裝 Python 或 uv。

### 1. 準備環境並安裝

你需要 macOS（Apple Silicon）或 Linux、Git、
[uv](https://docs.astral.sh/uv/getting-started/installation/)，
以及支援 MCP stdio 或 Streamable HTTP 的 AI 客戶端。專案使用 Python 3.14；
鎖定的 Numba／llvmlite 不提供 Intel Mac 預編譯套件，
原生 Windows 目前不支援生成策略驗證、執行計畫檢查及背景回測與最佳化；
這些程序監督功能需要 POSIX 環境。

執行生成策略另需本機 Docker CLI 與支援 cgroup v2 的 Linux Docker Engine，
並開啟記憶體、CPU 配額與程序數限制。macOS 可使用 Docker Desktop；MCP server
與 CLI 仍在 macOS 原生執行，生成 Python 從匯入開始在 Linux 容器中執行。
目前只接受本機 Unix socket，不支援遠端 SSH／TCP Docker context。

首次下載與安裝：

```bash
git clone https://github.com/Kuan-Lun/tradingdev.git
cd tradingdev
uv sync --locked
docker build --target strategy-runtime -t tradingdev-strategy-runtime:0.1.0 .
```

已有專案時，在現有專案根目錄執行 `uv sync --locked`，並依下方更新流程重建策略映像。
這會建立 `.venv` 並安裝鎖定版本的套件，包含內建策略需要的機器學習依賴；
首次安裝可能需要較長時間。從 Python 3.12／3.13 升級或明確選擇版本時，
使用 `uv sync --locked --python 3.14`；舊版 Python 不再支援。

FLAML 自動選模在此環境由專案測試驗證；Python 3.14 與所用 XGBoost 組合
仍超出 FLAML 上游宣告的支援範圍。使用前請參閱
[依賴相容性限制](docs/development.md#依賴相容性限制)。

以下終端機命令皆在專案根目錄執行。

更新程式後也要重建策略映像。自訂標籤時，在啟動 MCP／CLI 的環境設定
`TRADINGDEV_STRATEGY_IMAGE`；未設定時使用已安裝專案版本的映像標籤。
缺少映像、daemon 或必要隔離能力會明確失敗，不會改在本機執行生成 Python。
已固定的執行綁定原映像內容，仍需重播時請保留該映像。

### 2. 啟動 MCP server 並連接助手

依客戶端支援的連線方式，選擇以下其中一種：stdio 由客戶端自動啟動 server，
HTTP 則由你在終端機手動啟動，再讓客戶端連線。

#### 由客戶端啟動（stdio）

在客戶端的 MCP 設定中新增一個本機 stdio server，名稱填 `tradingdev`，
啟動命令填 `uv`。支援 `mcpServers` JSON 格式的客戶端可使用：

```json
{
  "mcpServers": {
    "tradingdev": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/tradingdev",
        "run",
        "--locked",
        "tradingdev-mcp",
        "--workspace",
        "/absolute/path/to/tradingdev-workspace"
      ]
    }
  }
}
```

請替換兩個絕對路徑：

- `/absolute/path/to/tradingdev`：剛下載的專案目錄，資料夾名稱不同也可以。
- `/absolute/path/to/tradingdev-workspace`：你選擇存放策略、行情與回測結果的資料夾，
  TradingDev 會建立所需目錄。之後查詢結果與開啟圖表都使用同一工作區。

若客戶端以表單設定，將 `command` 與 `args` 分別填入命令與參數欄位；
設定檔位置與格式依客戶端而定。若它找不到 `uv`，在終端機執行
`command -v uv`，將輸出的完整路徑填入 `command`。

儲存後重新載入 MCP 設定或重啟客戶端，並啟用 TradingDev 工具。
客戶端會自行啟動 server，不需要另外開終端機常駐執行。

#### 手動啟動並使用 HTTP 連接

若客戶端支援 Streamable HTTP，在專案根目錄執行：

```bash
uv run --locked tradingdev-mcp \
  --workspace /absolute/path/to/tradingdev-workspace \
  --web --transport streamable-http --port 8000
```

將 `--workspace` 換成執行 server 的主機上用來保存策略、行情與回測結果的資料夾絕對路徑。
將同一台電腦上的客戶端 MCP URL 設為 `http://127.0.0.1:8000/mcp`，
並保持此終端機執行；停止 server 時按 `Ctrl+C`。
這是 MCP 工具端點，不是瀏覽器操作介面；策略的網頁操作請使用 Dashboard。

`127.0.0.1` 指向發起連線的設備本身，另一台設備不能直接用這個網址連到你的主機。
服務預設只監聽 `127.0.0.1`；`--host` 可指定監聽 IP，例如 `--host 0.0.0.0`
監聽所有 IPv4 網路介面，或 `--host ::` 監聽 IPv6。此選項只適用於 `--web`，
不會自動放寬允許的 Host／Origin，也不提供登入驗證。
MCP HTTP 端點預設僅接受本機 `127.0.0.1`、`localhost`、`[::1]` 的 Host，
包括指定連接埠；瀏覽器 Origin 也須為對應的本機 HTTP 來源。沒有 Origin 的原生客戶端
仍可連線，不可信 Host／Origin 分別收到 HTTP 421／403。
跨設備連線須使用具驗證與存取控制的反向代理或網路轉送，並限制後端連接埠的存取。
若代理保留公開 Host，或瀏覽器使用不同來源，明確增加對應的完整值，例如：

```bash
uv run --locked tradingdev-mcp \
  --workspace /absolute/path/to/tradingdev-workspace --web --port 8000 \
  --allow-host research.example.com \
  --allow-origin https://research.example.com \
  --public-base-url https://research.example.com
```

兩個允許選項可重複指定；值須精確符合請求（含非預設連接埠），不接受萬用字元。
允許清單檢查適用於 Streamable HTTP 與 SSE 的 MCP 端點；產物下載端點仍依簽章網址
驗證，不使用此允許清單。允許清單不是登入驗證，也不授予跨來源瀏覽器讀取權限。
代理須另行處理使用者驗證、TLS 及所需的瀏覽器跨來源政策。
`--public-base-url` 指定使用者可達的公開網址；代理有路徑前綴時一併填入，
例如 `https://research.example.com/trading`。代理須將該前綴下的 `/mcp` 與
`/artifact-download/` 轉送到後端相應路徑。未設定時使用請求的 base URL，
需確保代理轉送的協定、Host 與路徑正確；這個設定不會建立代理或放寬允許清單。
stdio 不使用這些 HTTP 設定。
只支援雲端 URL 的客戶端也無法透過這個本機位址連接。

HTTP 客戶端須支援 Streamable HTTP 的 SSE 串流回覆；若經過代理轉送，也須保留串流。
執行前的確認仍在一般對話中完成。
Streamable HTTP 每次請求的本文上限為 4 MiB，超過會收到 HTTP 413；
傳送大型策略程式或設定時，請縮小單次內容。

### 3. 確認助手能使用工具

在對話中輸入：

> 請使用 TradingDev 列出可用策略與行情資料來源，並說明哪個內建策略適合用來確認回測流程。

助手應實際呼叫工具，取得策略清單與資料來源。
清單會區分隨專案提供的內建策略，以及保存在工作區的自建策略。
若助手只提供一般建議，請先確認客戶端已連接 `tradingdev` 並允許工具呼叫。

## 使用 Docker 部署

適合長期運行 MCP Server 的 Linux 伺服器或 NAS。主機需有 Docker Engine 與
Docker Compose v2 以上，以及 64 位元 x86-64 或 ARM64 CPU；不支援 32 位元 NAS。
Engine 須支援前述 cgroup v2 限制，並提供本機 Unix socket。
映像包含 Python 與內建策略所需依賴，首次建置會下載較多套件，請預留磁碟空間。
這套部署只提供 MCP Server；查看結果時，可請助手產生 HTML 報告並匯出研究 ZIP，
下載後離線開啟，操作見[查看結果與圖表](#查看結果與圖表)。
容器保存的成果路徑以 `/workspace` 開頭；主機上的原生 Dashboard 不能只把
`TRADINGDEV_WORKSPACE` 改成掛載資料夾就讀取這些成果，還需要相同的絕對路徑環境。

### 準備檔案與資料夾

下載完整專案，在專案目錄執行首次設定：

```bash
git clone https://github.com/Kuan-Lun/tradingdev.git
cd tradingdev
cp .env.example .env
mkdir -p workspace
id -u
id -g
```

編輯 `.env`，將 `TRADINGDEV_UID`、`TRADINGDEV_GID` 分別填入最後兩個命令的數字，
使用能寫入 `workspace` 的一般帳號。容器以該身分執行，避免以 root 建立研究檔案。
若要另存到 NAS 共享資料夾，將 `TRADINGDEV_DATA_DIR` 改為該資料夾的絕對路徑，
例如 `/volume1/docker/tradingdev-workspace`，先建立它並授予該帳號讀寫權限。
遇到 `Permission denied` 時核對 UID／GID 與 NAS 資料夾權限；部署不會自動變更既有檔案擁有者。

MCP 容器需要連接同一個 Docker daemon 來建立策略執行單元。Linux 上先用
`stat -c '%g' /var/run/docker.sock` 查詢 socket 群組，將數字填入
`TRADINGDEV_DOCKER_GID`；socket 不在此路徑時一併設定 `TRADINGDEV_DOCKER_SOCKET`。
這會授予可信 MCP 控制程式操作 daemon 的權限；socket 的唯讀掛載不會限制 Docker API。
生成策略的容器不會取得 socket、主機工作區或主機環境變數。

首次使用請選擇空的工作區。既有研究可能記錄絕對路徑，不要直接把原生安裝的工作區
搬到容器內當作遷移。容器內的工作區固定為 `/workspace`，往後更新與還原都保持此路徑。
`.env` 是本機部署設定，已排除於 Git 與映像建置內容之外；各項設定範例見
[.env.example](.env.example)。

### 設定連線入口

預設只在伺服器本機開放 `127.0.0.1:8000`，適合同機的反向代理。
例如對外網址為 `https://research.example.com/mcp`，在 `.env` 設定：

```dotenv
MCP_BIND_IP=127.0.0.1
MCP_PORT=8000
MCP_ALLOWED_HOST=research.example.com
MCP_ALLOWED_ORIGIN=https://research.example.com
```

將既有反向代理的目的地設為 `http://127.0.0.1:8000`，保留原始 Host、路徑與
SSE 串流；公開 `/mcp` 及確認單、報告與研究封存所需的 `/artifact-download/` 路徑。
舊版的 `/research-download/` 已由此共用路徑取代，升級後請向助手重新取得連結。
若代理與 Docker 主機不同，或代理本身在另一個容器中，`127.0.0.1` 指向代理自身；
請將 `MCP_BIND_IP` 改為 Docker 主機可達的內網 IP，代理也改連該 IP，並限制只有代理能存取。

使用 HTTPS 代理時，另將 `MCP_TRUSTED_PROXIES` 設為**容器實際看到的代理來源 IP**
（常是 Docker 網路閘道，可由請求後的 `docker compose logs mcp` 存取紀錄確認），
並讓代理傳送 `X-Forwarded-Proto: https`，讓後端辨識公開的協定；
未信任的轉送標頭會被忽略。多個可信 IP 可用逗號分隔，勿將不受信任的來源納入。
若代理改寫 Host 或使用路徑前綴，可在 `docker-compose.yml` 的 `command` 追加
`--public-base-url` 及公開網址，明確指定產物連結的目的地。

若要在受控網路直接以 IP 連線，例如伺服器為 `192.168.1.20`，改為：

```dotenv
MCP_BIND_IP=0.0.0.0
MCP_PORT=8000
MCP_ALLOWED_HOST=192.168.1.20:8000
MCP_ALLOWED_ORIGIN=http://192.168.1.20:8000
```

另一台電腦的 MCP URL 就是 `http://192.168.1.20:8000/mcp`；使用外網 IP 時，
同樣替換成客戶端實際連線的 IP。`0.0.0.0` 是監聽設定，不能當作客戶端網址。
改連接埠時同步調整網址及允許值；容器內連接埠維持 `8000`。
Host 值不含協定或路徑，Origin 包含協定但不含路徑；非預設連接埠須保留。
需要多個公開 Host／Origin 時，可在 `docker-compose.yml` 的 `command` 追加
成對的 `--allow-host`／值或 `--allow-origin`／值。

允許清單不提供登入驗證。服務能保存並執行生成的 Python 策略，請只開放給可信使用者；
公開網路入口須由反向代理提供驗證、TLS 與存取控制。共用此容器的客戶端會使用同一工作區，
不具備每位使用者的資料隔離。生成程式的容器邊界不取代帳號權限與服務部署保護。
瀏覽器客戶端所需的跨來源政策仍由代理處理，詳細限制見
[HTTP 連接說明](#手動啟動並使用-http-連接)。

### 啟動、更新與保存資料

在專案目錄執行：

```bash
docker compose config --quiet
docker build --target strategy-runtime -t tradingdev-strategy-runtime:0.1.0 .
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 -f mcp
```

`ps` 應顯示服務執行中；`healthy` 表示容器內的連接埠已可連線，工具操作仍須依
[確認助手能使用工具](#3-確認助手能使用工具)實際確認。查看日誌時按 `Ctrl+C`
只結束追蹤，Server 仍在背景執行。Docker 服務重新啟動後會自動啟動此容器，
手動停止的容器則需再次執行 `docker compose up -d`。

Synology 使用者可把完整專案放在 NAS 上，以 Container Manager 的專案功能選用
`docker-compose.yml`，並使用同目錄的 `.env` 設定；亦可透過 SSH 執行上述命令。
需要完整原始碼與 `uv.lock` 才能建置，不能只上傳 Compose 檔。
反向代理可使用 NAS 已有功能，目的地依上一節設定。

更新前先等待回測／最佳化結束，並停止服務：

```bash
docker compose stop
```

停止期間備份 `TRADINGDEV_DATA_DIR` 指向的**整個資料夾**，包含 SQLite、策略、行情、
模型與研究結果，並另存 `.env`。還原時先停止服務，還原完整資料夾與權限，再以相同
容器路徑 `/workspace` 啟動。備份完成後更新程式並重新啟動：

```bash
git pull --ff-only
docker build --target strategy-runtime -t tradingdev-strategy-runtime:0.1.0 .
docker compose up -d --build
```

停止或重建會中斷未完成的工作及客戶端連線，
不會自動續跑；重新連線後查詢工作狀態再決定是否重做。

`docker compose down` 可移除容器與 Compose 網路；研究資料保留在主機資料夾，
更新、停止及移除容器都不會刪除它。再次啟動時仍須使用同一個 `TRADINGDEV_DATA_DIR`。
修改 `.env` 後用 `docker compose up -d` 重新套用設定，單純 `restart` 不會載入變更。

## 完成第一次回測

先用內建 KD 交叉策略確認整個流程，不必先撰寫 Python：

> 請用 TradingDev 的內建 KD 交叉策略，以 Binance Vision 的 BTC/USDT USDT-M 期貨行情、
> 1 小時 K 線，回測 UTC 2024-01-01 00:00:00 到 2024-02-01 00:00:00。
> 資金、手續費與滑價使用專案模擬預設，完成後說明交易規則、實際設定、總損益、
> 最大回撤與已平倉交易數。

助手會先檢查程式與執行設定，再提供簡短摘要與完整 HTML 確認單連結。
未另行指定時，本金使用 10,000 單位報價貨幣、單邊手續費 0.06%、滑價 0.05%；
這些是模擬假設，不是交易所即時費率。你明確指定的有效值優先，零手續費或零滑價也會保留。
確認單會列出實際設定、來源與年化依據；未指定的策略參數會由助手提出並說明理由，
無法查證的提案會明確標示。這些說明與程式檢查不保證策略正確或可獲利。
確認前不下載真實行情、不訓練模型，也不執行回測。
接下來依序完成：

1. **閱讀確認單**：核對交易規則、參數、行情來源與市場類型、期間、資金、成本及尚未驗證的事項。
2. **回覆決定**：在對話明確回覆「同意執行」、拒絕，或用自然語言說明修改要求。
   修改後會收到新的確認單，需再次同意；拒絕則不啟動正式工作。
3. **等待正式結果**：助手提交工作、追蹤進度，完成後讀取結果並說明績效。

助手會在提供確認單後等待你的回覆。最初請它回測、對計畫提問或未回覆，都不算同意。
跨電腦以 HTTP 連線時，助手可提供確認單與報告的短效閱讀網址，不必存取伺服器檔案路徑。
連結有效十五分鐘，持有者皆可讀取該份文件；請只分享給預期的讀者。
連結過期或伺服器重啟後，可請助手重新取得，無須重跑研究，也不延長計畫有效期。
stdio 或無法使用 HTTP 交付時，須透過客戶端保存並核對完整的本機檔案或附件，
助手才提供該副本的閱讀連結。無法交付可讀文件時，助手會說明限制，待你能閱讀後才請你確認。
這個流程依賴助手如實轉交你的決定，系統無法獨立驗證回覆是否來自你本人；
詳細限制見[執行確認契約](docs/execution_plans.md#execution-plans-and-confirmation-documents)。
確認單有效期為一小時，過期後需重新準備。通過檢查不代表行情可取得或正式執行一定成功；
失敗時可請助手查詢原因，依[常見問題](#常見問題)處理。

如需指定其他行情來源或使用現貨資料，請在提示詞中說明，並於執行前核對確認單。
可用資料範圍依來源而異，詳見[資料與工作區](#資料與工作區)。
助手會追蹤回測進度，完成後提供結果與解讀；僅回報工作已啟動，尚不代表回測完成。

接著可以追問：

> 請解讀剛才的結果，列出回測設定、成本與資料期間。哪些指標無法計算？
> 請說明原因，並找出值得進一步驗證的地方。

## 建立與研究自己的策略

### 把交易規則交給助手

先提供策略構想、交易標的、期間與 K 線頻率；資金與成本可選填。
助手可以協助把模糊構想寫成具體程式，但不能替你猜測缺少的四項研究選擇。例如：

> 請建立均線策略，使用 Binance Vision 的 BTC/USDT USDT-M 期貨行情、1 小時 K 線。
> 10 期均線向上穿越 30 期均線時做多，向下穿越時平倉，不做空。
> 初始資金 10,000 USDT、單邊手續費 0.06%、滑價 0.05%，
> 回測期間為 UTC 2024-01-01 00:00:00 到 2024-07-01 00:00:00，報酬率以全年 365 日年化。
> 請完成回測，說明績效、風險與結果限制。

自建策略會先經過程式檢查，再進入前述確認流程。請核對確認單上的規則是否符合
你的想法；檢查通過不保證策略已完整實現你的交易想法或具有獲利能力。

策略可由助手命名，也可以使用你提供的名稱。修訂時請指定既有策略，
助手會核對版本；若期間已被更新，須重新確認要修改的版本。
每次成功保存都產生新版本，修正後需要重新檢查；
舊版本與歷史結果仍保留，可請助手指定版本執行或查詢。
自行撰寫 Python 或 YAML 時，請參考[策略契約](docs/strategy_contract.md)。

自建策略在檢查與回測時都可能執行 Python，使用有期限與資源上限的 Linux 容器。
策略可使用傳入的研究資料及映像內的檔案，不能連外下載、直接讀取主機檔案或操作回測引擎。
超限會失敗，需由操作者調整 `TRADINGDEV_RUNTIME_LIMITS` 或修改策略後重新準備執行。
隔離範圍、可傳輸資料及清理失敗的處理見[策略安全模型](docs/strategy_contract.md#security-model)。

升級後，舊的生成策略執行設定可能需要重新準備；原始碼與結果不會自動改寫或刪除。
若舊策略依賴直接操作回測引擎，須請助手修改為使用傳入資料後重新驗證。

### 比較參數

沿用已有策略，即可比較不同參數的效果：

> 請沿用內建 KD 交叉策略，把隨機指標的回顧週期分別設為 9、14、21 根 K 線，執行三次回測。
> 交易對、期間、其他參數與成本都沿用第一次回測，完成後比較損益、回撤與交易數。
> 請說明參數改變後的差異，以及各次結果是否適合直接比較。

這類比較可一次核對一到八組固定參數，在同一份確認單同意後執行。
每組使用相同的行情資料與成本設定，獨立計算資金、保存結果，不會改寫策略的基礎設定或先前的回測。
報告保留所有組別；成功結果可點選查看交易明細，失敗、取消或未執行的組別會列出原因。
目前這種整批比較限於普通回測，需要訓練或 walk-forward 的策略仍使用下方的流程。
要修改已確認的候選或設定，須重新準備並確認，不會自動擴大搜尋。

### 使用樣本外驗證與最佳化

**Walk-forward（分段訓練與測試）** 將資料分成一組或多組訓練與測試區間，
用來觀察策略在不同期間的表現。具備訓練能力的策略會在每組訓練區間訓練，
其他策略只會分段回測，不會自動學習或調整參數。
你可以描述希望的切分方式；指定日期時以 UTC 完整日計算，訓練與測試區間都須落在研究期間內。
執行前請核對確認單上的實際區間，保存的策略版本不會因這次分段而改變。

訓練標籤若使用未來價格，請說明向前看多少根 K 線，並要求助手避免訓練標籤與測試區間重疊。例如：

> 訓練標籤使用未來 5 根 K 線，請做三段淨化（purged）驗證，另留 2 根 K 線的間隔。

一般分段不會自動排除標籤重疊；實際資料不足或無法形成有效分段時，正式執行仍可能失敗。
自行編輯 YAML 時，設定範例與限制見
[策略 walk-forward 設定](docs/strategy_contract.md#strategy-walk-forward-settings)。

**參數最佳化** 在訓練期間逐組回測指定的候選值，依目標指標選出參數，
再以獨立的測試期間回測。這個搜尋流程不會額外訓練機器學習模型，僅適用於支援此模式的策略。例如：

> 請對內建 KD 交叉策略，搜尋回顧週期為 9、14、21 根 K 線、D 值平滑週期為 3、5 根 K 線的組合，
> 使用 Binance Vision 的 BTC/USDT USDT-M 期貨行情、1 小時 K 線，以總損益為目標。
> 訓練期間為 2024-01-01 至 2024-03-31，測試期間為 2024-04-01 至 2024-06-30。
> 請找出訓練期間總損益最高的組合，並說明它在測試期間的表現。

最佳化的確認單會列出候選範圍、組合數與目標指標，你同意後才開始搜尋。
訓練與測試日期按 UTC 日曆日計算，包含終日且不得重疊。
結果應同時查看訓練表現與保留測試表現；部分指標可能因交易或資料不足而無法計算。
如果工作只留下部分結果，可用 job ID 請助手列出各組的完成、失敗、逾時、中斷及未嘗試狀態。
工作失敗或取消後，已開始但尚未提交結果的試驗會標為中斷。你仍可查到已完成試驗的指標，
並確認搜尋是否完整，以及是否實際執行保留測試。
[最佳化試驗產物](docs/run_artifacts.md#optimization-trial-results)說明可查的內容。

**建議預留最終評估資料。** 開始開發前，可以先保留一段資料作為最終評估，等規則、特徵與參數
都選定後才使用。若看過這段資料的結果後又回頭修改策略，它就不再是獨立的最終測試，之後的結果
應視為開發過程的一部分。這是使用上的建議：系統不會檢查你是否這樣做，也不會驗證某段資料
未被使用過；開發時可以使用最近一段期間的資料，保留測試期或樣本外證據也不是結果列入比較的條件。

## 研究機器學習訊號

助手可選用內建特徵與方向分類模型，直接建立資料集、訓練及評估。
也可以不訓練模型，直接研究手設規則，例如「沿用已保存資料集，評估 RSI14 低於 50
時偏多、否則偏空的訊號並保存，與同一資料集的模型比較」。
此流程會依需求取得資料及訓練，不經回測確認單，也不會自動把模型轉成交易策略。
以下以訓練模型為例：

> 請使用 Binance Vision 的 BTC/USDT USDT-M 期貨行情、1 小時 K 線，研究方向訊號。
> 期間涵蓋 UTC 2024 年 1 月 1 日至 1 月 31 日的完整日期。
> 請說明可選的特徵與模型，以固定種子訓練，並做淨化（purged）walk-forward 驗證。
> 保存候選與評估結果，產生可離線閱讀的比較報告。
> 請區分訓練集表現、驗證結果及真正樣本外的證據，說明資料、成本與排名的限制。

研究期間須已結束至少 7 天；行情缺漏及指標暖機等因素也可能減少可用資料，
請依結果確認實際資料範圍與筆數。
產生報告前會檢查所選候選引用的逐點序列；檢查發現序列缺失、損毀或與所選候選不一致時，
報告不會出版，即使已保存的排行榜指標仍可讀。檢查可能沿用先前已驗證的快取，限制見
[ML series evidence](docs/reports.md#composable-offline-reports)。

後續可以請助手「比較剛才保存的候選，列出排名與沒有分數的原因」。
有保存序列的候選，可以請助手分批查看逐筆訊號、報酬與指標缺值原因。
歷史候選若未保存序列，會明確回報證據不可用，不會事後重建。
訓練時採用分折驗證，也可請助手讀取當時保存的預測，說明每折的資料範圍。
分折資料可能不連續，不能僅憑起訖範圍認定每筆資料都有參與。
需要核對原始座標、欄位定義、讀取容量或復原限制時，見
[已保存逐點評估序列](docs/ml_research.md#saved-pointwise-evaluation-series) 與
[Fold prediction series](docs/ml_research.md#fold-prediction-series)。

不同期間、資料或評估條件的結果預設不混合排名；若要比較不同預測長度或視窗，
請在需求中說明，並留意報告列出的比較限制。

解讀結果時請留意：

- 用訓練資料評估的表現不代表樣本外效果。若要驗證樣本外表現，請明確要求使用未參與訓練與選參的資料。
- 訊號績效不含手續費與滑價，也不是完整的交易回測，不能直接當作可實現的交易報酬。
- 候選越多，最高排名越可能受到挑選偏誤影響，排名第一不保證未來表現。

目前支援多商品訓練與預測，但訊號績效只評估單一商品。
需要核對驗證方法、資料處理與指標定義時，見
[ML 研究契約](docs/ml_research.md#ml-research-contract)。

## 查看結果與圖表

### 在對話中查詢

回測解讀會提供重要績效、觀察依據、尚未確認的原因與下一步研究建議，並保存於
HTML 報告，方便交給同仁閱讀。自動診斷依固定規則提供提示；門檻不是
你的風險容忍度，也不代表修改一定有效。目前尚未提供市場情境或波動敏感性診斷。
缺少資料的檢查會標示未評估，未提供 LLM 解讀的報告也會明確標示。
建議不會自動啟動新實驗。資料與格式見[報告契約](docs/reports.md)。

可以直接用研究內容或期間找回結果，不必記住工具名稱或識別碼：

| 想做的事 | 可以這樣說 |
| --- | --- |
| 找回之前的結果 | 「列出最近的回測，顯示策略、期間與績效。」 |
| 追蹤或停止工作 | 「剛才的回測進行到哪裡？」或「取消剛才的回測。」 |
| 查摘要以外的指標 | 「查詢這次回測的其他可用指標，說明定義、單位與無法計算的原因。」 |
| 比較兩次實驗 | 「比較這兩次回測的損益與回撤，說明差異與比較限制。」 |
| 檢查實際設定 | 「這次回測用了哪個策略版本、參數與成本？」 |
| 找指定參數的交易 | 「找出短均線 12 期、長均線 29 期的歷史結果，查看每筆進出場、數量與費用。」 |
| 開啟完整報告 | 「為這兩次回測產生離線 HTML 報告，保留完整交易表並加入你的研究評語。」 |

需要精確指定時，可請助手提供工作代號（job ID）以追蹤或取消工作，
或提供已保存結果的代號（run ID）以供後續查詢、比較及終端機操作。

解讀結果時請留意：

- `N/A` 或 `null` 代表無法計算或不適用，不是零；可請助手查詢原因。
- 每段回測結束時，剩餘部位會按最後收盤價強制平倉並計入出場成本；有本金的
  回測以平倉後現金計算期末損益。既有結果保留當時的規則，不會自動重算。
  成交及失敗條件見[策略契約](docs/strategy_contract.md#signal-and-execution-semantics)。
- 年化報酬、Sharpe 等指標使用保存的每年觀測日數。只提供加密資產的行情來源可採 365；
  Yahoo 等涵蓋不同市場的來源，需另行指定日數或選擇不計算年化。
  它表示每年日數，與 1 小時或 5 分鐘 K 線分開。
  週線、月線等粗於日頻的資料不提供年化及日／月損益統計。
- Volume 模式沒有初始資金基準，應查看金額損益、成交量與回撤金額。
  交易數、勝率等交易統計只計算已平倉交易。
- Walk-forward 摘要是各測試區間的統計分布，不能當成整段期間的單一績效；
  比較時可要求逐區間查看。最佳化則應確認查看的是訓練或樣本外測試結果。

執行時的設定與結果會保留，但行情、套件及環境未完整封存，
日後重跑不保證得到完全相同的數字。詳細定義見[執行結果與產物](docs/run_artifacts.md)。

### 交易紀錄與 HTML 報告

要求助手完成回測並解讀績效、風險時，預設也會產生完整離線 HTML 報告；
若只需要對話摘要，可明確說「只要摘要，不需要報告」。完成回覆應包含報告連結、
對應結果及查看明細的方法，只有績效數字或 run ID 不代表報告已交付。

請助手為已完成或歷史回測產生離線 HTML，可一次比較最多 8 次執行。例如：

> 請為剛才的三次回測產生比較報告，包含參數、績效、權益曲線、交易紀錄與你的評語。

標準報告先顯示總覽，點選各列的「查看明細 →」即可查看該組結果的設定、績效、權益／回撤、委託與帳戶核對表、
配對交易及資料限制。你也可以請助手只顯示需要的章節；助手的評語會與計算結果分開標示。
報告使用已保存的結果，不會重新回測或下載行情。HTML 可離線開啟，不需安裝圖表套件。

「基準比較」以固定的買進持有策略作為對照，不能在對話中自選基準。
支援的 Binance USDT 計價市場使用 BTC/USDT，美股使用 S&P 500 價格指數（不含股利）。
台灣掛牌標的（含上市、上櫃及 ETF）統一使用台股加權價格指數（不含股利）；
這是市場對照，不表示每個標的的實際投資範圍都與台股大盤相同。
基準使用相同成本及共有的 K 線，最後一筆共有收盤也會平倉並計入出場成本；
資料未完整涵蓋策略期間時，不計算超額報酬。
沒有適用基準或取得失敗時，回測結果仍可保存，報告會說明原因，不把缺少資料當作零報酬。
沿用策略行情時，既有快取的實際來源仍未經獨立核驗。

一般回測與 walk-forward 的個別訓練／測試結果可提供基準；跨分段摘要與最佳化不提供。
較早完成、沒有保存基準的受支援結果需重新回測，單純重建報告不會補算。
市場對應與計算口徑見[基準比較契約](docs/performance_artifacts.md#benchmarks)。

ML 訊號報告可獨立產生，也可附加在回測或固定比較報告中。「ML 訊號研究」章節提供候選
排序、未排名原因、評估設定與指標定義；展開候選及其他指標可查看詳細證據，亦可下載完整
ML 證據 JSON。一次可選最多 50 個候選；兩類研究分開呈現，並不表示模型曾用於旁邊的回測。
ML 的模擬績效不含成本，沒有逐筆成交或權益曲線。跨預測長度時，報告會分別說明原始
K 棒是否共用、實際評估列是否一致。舊結果可能缺乏核對證據。若研究要求共用行情基線，
可請助手只接受原始行情基線已驗證的比較，並另外查看報告中的評估列一致性結果。

報告固定保存產生當下選取的 ML 內容；之後重新評估或移除排行榜紀錄不會改變已產生的
報告。新產生報告會讀取目前的紀錄；未保存的預測與 IC 時間序列不會自行補算。

想回答「A 比 B 差多少、證據夠嗎」，可請助手對兩側**已保存**的 scalar、signal
模式且有觀察值的 run/scope（volume 與彙總 fold summary 不適用）、訊號序列或基準做
配對比較；零成本基準需工作區已保存並核驗過的原始價格 artifact，
比較本身不會建立它，也無法由含成本淨值反推價格。只有相容定義能相比（例如同類 IC 相比、訊號模擬不與成交報酬混檢）。比較只使用共同的 UTC 時間戳與雙方有效的觀察，不補值、
不重新回測、不重新 predict，也不會下載行情。預設是描述性結果；推論另須精確 UTC
日頻或小時頻的完整日曆（中途無效值也不接受）與可識別來源，仍不代表已修正先前搜尋造成的偏誤，bootstrap 區間也不保證名目涵蓋率。可要求出版
可離線閱讀的 Markdown／HTML 成果（預設不出版）。服務不保證兩側標的、期間與成本
完全相同；來源選擇、資格、排除證據與成果格式見[配對比較契約](docs/paired_comparison.md)。

標準報告的研究說明保留確認時的買賣規則、公式、參數依據與參考來源，方便核對各組設定。
這些是助手或使用者提供的說明，仍需核實來源與程式是否一致。
舊結果若沒有保存研究說明，報告會標明缺少，不能事後當作原始依據。

查看紀錄時，請區分以下內容：

| 內容 | 用途與限制 |
| --- | --- |
| 績效分析 | 查看報酬、風險、月報酬與回撤。QuantStats 分析的勝率等統計按報酬期間計算，與逐筆交易統計不同；詳細口徑見報告說明。 |
| 委託與帳戶核對表 | 依時間查看委託、成交與每根 K 線收盤時的資金及持倉。這是模擬帳戶，不是交易所的保證金或資金費率帳本。 |
| 配對交易 | 查看進出場、數量、費用與損益。未平倉部位的期末估值會另外標示，不代表已實際平倉。 |

時間是 K 線時間，價格來自保存的行情，並非盤中逐筆成交或買賣報價。
舊回測或不支援帳戶紀錄的模式可能缺少部分內容；缺失會說明原因，不代表沒有交易。

表格可篩選、排序與選擇欄位；核算價格、數量與費用前，請勾選「顯示完整精度」。
總覽、核對表與配對交易可下載 CSV 或 JSON，匯出會保留全部紀錄、欄位與完整精度，
不受畫面篩選影響。CSV 使用中文欄名，費率保留原始比例（1 表示 100%），缺值留空。

**分享前請留意**：隱藏章節只改變畫面，HTML 仍包含所選結果的全部已載入資料，
包括設定、交易紀錄及選取的 ML 證據；接收檔案的人仍可讀取這些內容。
報告保存在工作區的 `reports/`。若助手提供的檔案路徑無法在你的電腦開啟，
請它透過客戶端取得檔案；執行 MCP 的主機上的路徑不是公開網址。
關閉 MCP server 不會刪除報告或回測結果；離線 HTML 仍可閱讀，重新連接同一工作區
可繼續查詢原結果。若報告產生或交付失敗，請沿用已完成的結果重試，不必重新回測。

報告格式、指標口徑與程式查詢介面見
[報告契約](docs/reports.md#composable-offline-reports)。

### 保存研究筆記與打包分享

可直接請助手「把目前需求、各階段結論與報告存成研究筆記，打包給同事閱讀」。
尚未回測也能保存；後續追加需求時，請它更新同一本筆記，將尚未執行的想法標明。
助手整理的是階段摘要，不會自動取得或保存聊天客戶端的完整對話。

匯出的 ZIP 包含 `index.html`、筆記 JSON 與指定的報告。請完整解壓後開啟
`index.html`，由各階段連到報告；報告中的觀察、限制與建議也會保留。
同一本筆記可隨時重新匯出，已匯出的版本不隨後續編輯改變。

HTTP 連線會提供 15 分鐘有效的 ZIP 下載網址；持有網址者可下載，請只交給預期收件者。
網址過期或 server 重啟後，請助手依封存代號重新取得下載網址；筆記更新或來源報告移除也不影響舊封存。stdio 提供 server 本機路徑，
需客戶端支援檔案存取／傳輸；遠端 server 路徑不能直接當成你電腦上的檔案。
筆記最多 16 KiB，單次封存最多 20 份指定 HTML 報告、64 MiB；過大時需精簡摘要或報告選擇。
來源報告遺失或被更改時會拒絕打包，不會偷偷重跑策略。

封存用於閱讀與討論，不包含整個工作區、完整行情或執行環境；若要繼續查詢研究，
仍須保留原工作區。儲存與介面契約見[研究筆記與封存](docs/reports.md#research-notebooks-and-archives)。

若希望把研究中可重用的邏輯整合回專案，另見開發指南的
[研究成果工程回饋](docs/development.md#研究成果工程回饋)；這需要能存取專案的開發助手與明確授權。

### 自動保存 Codex CLI 對話

第一版提供 TradingDev 專用的 Codex CLI 入口。先安裝並登入 Codex CLI，確認
`codex login status` 可用；若執行檔不在 PATH，將 `TRADINGDEV_CODEX_BIN` 設為其
絕對路徑。研究仍須具備前述 TradingDev 執行環境，模型使用既有 Codex 登入。
`tradingdev_recorder` 是此入口保留的 MCP 名稱；若 Codex 設定已有同名項目，
即使已停用也會拒絕啟動。請先將既有項目重新命名，再重試。

```bash
uv run --locked tradingdev-codex --workspace ./research --model gpt-6.1-sol \
  --reasoning-effort medium --message '我想用均線買賣0050，可以幫我看看嗎？'
```

每次命令送出一輪文字，自動保存使用者輸入、AI 代理的可見回覆與 TradingDev 工具
參數／結果，並回傳 `conversation_id`、回覆文字與 `export.html_path`。較長的輸入
可用 `--prompt-file FILE` 取代 `--message`。收到追問後，用同一個工作區與代號續接：

```bash
uv run --locked tradingdev-codex --workspace ./research --conversation-id CONVERSATION_ID \
  --message '先看2024年1月1日到7月1日，用日線。'
```

續接沿用原模型與推理設定，並核對 Codex 歷史。請保留原工作區與 Codex 的原生
工作階段資料，不要從其他入口修改同一個來源對話。中斷時已保存文字仍可匯出；
若未取得來源回合身分或歷史不一致，會拒絕續接，請保留舊紀錄並另開新對話。
關閉對話命令不會自動取消已交給 TradingDev 的背景研究工作，可從同一工作區查詢。

這個入口僅啟用 TradingDev MCP，不是一般程式開發的 Codex 終端介面；不會擷取
其他 `codex`、VS Code、桌面版或 Claude 的聊天。第一版只接受文字，私有推理不保存。
它不提供互動式 MCP 表單，遇到表單會明示由客戶端取消，不會代替使用者同意；
一般文字追問仍由下一輪命令回答。需要 app-server 的啟動、續接及分頁歷史介面，
不支援的 CLI 版本或事件會明確失敗。預設單輪期限為 1200 秒，可用 `--timeout` 指定。

### 閱讀保存的對話

客戶端整合已保存的對話可用 `uv run --locked tradingdev-conversation --workspace PATH list` 找到，
再以 `read CONVERSATION_ID` 讀取原文與版本。匯出指定版本：

```bash
uv run --locked tradingdev-conversation --workspace PATH export CONVERSATION_ID \
  --expected-version VERSION
```

命令回傳可開啟的 `html_path`；頁面提供可選取文字、逐輪導航及可展開的完整工具紀錄。
模型回合狀態與紀錄完整性分開標示，中斷不冒充完整。HTML 可離線閱讀，若也要下載
JSON，請一併保留同目錄的檔案。分享會包含所有已記錄的訊息與工具內容；後續回合
不會改寫先前匯出的版本。這個查詢入口不會自動擷取其他聊天程式或匯入歷史對話。
保存範圍與限制見[對話紀錄契約](docs/reports.md#recorded-conversations)。

### 開啟 Dashboard

Dashboard 可顯示具有可載入圖表產物的一般回測與 walk-forward 結果。先安裝圖表依賴，
並將工作區設為 MCP 設定使用的同一路徑：

```bash
uv sync --locked --extra dashboard
export TRADINGDEV_WORKSPACE="/absolute/path/to/tradingdev-workspace"
uv run --locked --extra dashboard streamlit run src/tradingdev/adapters/dashboard/app.py
```

開啟終端機顯示的本機網址，從側邊欄選擇結果。也可以直接指定 run ID：

```bash
uv run --locked --extra dashboard streamlit run src/tradingdev/adapters/dashboard/app.py \
  -- --run-id "YOUR_RUN_ID"
```

側邊欄也可生成並下載同一份標準 HTML 報告。
請將 `YOUR_RUN_ID` 換成助手查到的實際值。最佳化結果請透過對話查詢或生成 HTML 報告，
目前不產生 Dashboard 所需的圖表產物。結束圖表服務時，在終端機按 `Ctrl+C`。

「Rolling 30d MDD」只計算目前時間往前 30 個日曆日（不含左端點）的已觀測權益，
以報價幣別顯示最大回撤金額；高點必須早於低點。所選期間不足 30 天時顯示缺值，
休市或缺 K 棒不補值，週／月 K 也使用實際時間戳；沒有有效時間戳時顯示不可用。

## 資料與工作區

### 行情來源與下載

專案提供 Binance Vision、Binance API 與 Yahoo Finance 資料來源；
可在提示詞中指定，或請助手列出來源，選擇適合的市場與代號。
外部來源能提供的交易對、頻率與歷史期間各有不同，專案不保證任意區間都有資料。

首次執行可能需要下載行情，請預留等待時間與磁碟空間。正式回測按年度取得資料，
即使只回測一個月，也可能需要下載該年度的行情。
系統會重用符合條件的本機行情；只有在請求區間結束至少 7 天後成功查詢的資料才會重用，
近期區間每次載入都會重新查詢。請勿指定未來日期，延長期間時也可能需要下載新資料。
已有快取或資料檢查通過，不保證每根 K 棒齊全，也不排除來源日後修正資料。

舊版未記錄來源的快取會保留，但新研究需要重新下載行情。
更新失敗不會刪除原資料；更新成功後也會保留舊版本，因此反覆更新會增加磁碟用量，
目前不自動淘汰。快取重用條件與儲存格式見
[行情快取契約](docs/run_artifacts.md#market-data-cache)。

DVOL 未指定檔案路徑時，會依完整研究期間從 Deribit 取得，並使用上述 7 天重用政策；
延長或更換期間會查詢新的範圍，下載失敗會保留既有版本。
若指定本機 DVOL 檔案，已有檔案不會因研究期間延長而自動更新，須自行確認資料範圍。
資金費率與自訂特徵需要設定指定的本機 Parquet 檔案。
額外特徵的開頭若缺資料，會保留缺值，不會用較晚的資料回填。
策略須處理這些缺值，該期間可能無法產生有效訊號。

### 選擇與保存工作區

工作區保存自建策略、行情、模型、研究筆記與回測結果，與隨專案提供的內建策略分開。
請讓助手、Dashboard 與終端機命令使用同一個工作區，才能查到相同研究。
HTML 報告位於 `reports/`，可分享的研究 ZIP 位於 `research_archives/`；
其他目錄與檔案用途見[工作區儲存結構](docs/run_artifacts.md#workspace-layout)。

只使用可信來源的工作區與模型檔案。升級 Python 或 ML 套件後，舊模型可能無法載入，
請在新環境重新訓練；相容性與保存方式見[ML 研究產物](docs/ml_research.md#ml-research-artifacts)。

MCP 與報告 CLI `tradingdev-report` 的 `--workspace` 優先於環境變數
`TRADINGDEV_WORKSPACE`；兩者都未設定時，使用啟動目錄下的 `workspace/`。
回測 CLI 與 Dashboard 使用同一環境變數及預設目錄，但沒有 `--workspace` 參數。
每次開新終端機時，請重新設定環境變數，或在自己的 shell 設定中保存它。

若需要另外存放行情，可設定 `TRADINGDEV_DATA_ROOT`，預設使用其下的 `raw/` 與
`processed/`；CLI 的圖表快取也會跟隨這個環境變數。
策略 YAML 若明確設定 `data.raw_dir`、`data.processed_dir`，則優先使用那些行情路徑。
自訂相對資料路徑以程序啟動目錄解析，建議使用絕對路徑。

保留研究結果時，請保留工作區的資料庫與檔案，不要只留下 run ID 或單一結果檔。
若另外指定外部資料或快取目錄，也需保留那些路徑下的檔案。
現有紀錄包含絕對路徑，還原時應保持原位置；搬移工作區不會自動更新這些紀錄。
清理舊草稿時，工具預設先預覽可刪版本，由你確認清單並授權刪除。
已使用的版本及無法確認可安全刪除的內容會保留。

### 升級舊工作區

升級至先確認再執行的流程時，舊版「等待確認」的最佳化工作不會自動遷移，
可能導致列出工作、查詢或取消失敗；建立新確認單也無法修復舊紀錄。
遇到這種情況，先用建立工作的舊版停止其工作與服務，確認程序已結束，
保留原工作區於原位置，再將客戶端 MCP 設定的 `--workspace` 改成新的空白資料夾絕對路徑，
重新載入客戶端。

新工作區不會自動帶入原有策略與歷史，舊結果仍保留在原工作區。
不要將舊資料庫或程序控制檔複製過去；這是另建工作區的復原方式，不是原地遷移。
相容性與復原限制見[工作狀態契約](docs/run_artifacts.md#job-state-and-worker-identity)。

## 終端機用法

### 直接執行設定檔

不透過 AI 也可以執行回測。先設定與 MCP 相同的工作區，再使用內建 KD 範例：

```bash
export TRADINGDEV_WORKSPACE="/absolute/path/to/tradingdev-workspace"
uv run --locked python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
```

這會依檔案中的期間與成本執行，並在終端機顯示結果。每次執行都保存獨立 run，
之後可由助手查詢，或在 Dashboard 選取。
若要調整內建範例，先複製 YAML 作為自己的執行設定，再修改交易對、日期與參數。
行情的 `backtest` 與 `data.requirements.market` 設定請保持一致；
一般回測的日期欄位是時間邊界，僅填日期會解析為當日 00:00:00，不會自動包含終日。
含 `validation` 的設定必須加上 `--walk-forward`，例如：

```bash
uv run --locked python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml \
  --walk-forward
```

自建策略須使用已通過程式與合成訊號檢查的版本。可複製該版本的 YAML 調整
`strategy.parameters`，但保留其餘策略宣告與指向原版本的 `source_path`；
交易對、日期與成本則在 `strategy` 區段以外調整。
若手動修改研究設定，例如標的、K 棒頻率、日期、本金、成本或年化，請在執行用的
YAML 副本移除整個頂層 `research` 區段，並繼續保留上述策略身分。
CLI 直接讀取 YAML 不會重算或清除來源紀錄；留下原紀錄會讓新設定帶著過時的來源標籤。
只修改執行用副本，保存的策略版本不變；詳見[研究設定與來源紀錄](docs/run_artifacts.md#research-settings-and-provenance)。
修改策略程式或其他策略宣告時，請讓助手另存版本並重新驗證，
完整規則見[策略生命週期](docs/strategy_contract.md#lifecycle)。

### 從已保存的結果產生報告

將 `YOUR_RUN_ID` 換成實際的 run ID，即可產生標準 HTML 報告：

```bash
uv run --locked tradingdev-report --workspace /absolute/path/to/tradingdev-workspace \
  --run-id YOUR_RUN_ID
```

要比較多次已保存的結果，可重複加入 `--run-id`，最多 8 次。

若要查看一整批固定參數比較，請使用該批次的父工作 ID：

```bash
uv run --locked tradingdev-report --workspace /absolute/path/to/tradingdev-workspace \
  --comparison-job-id YOUR_COMPARISON_JOB_ID
```

父工作須已完成、失敗或取消，且每組候選的狀態都已定案。這份報告保留全部候選，
包括失敗、取消與未執行的原因；即使沒有任何成功結果，仍可產生報告，並說明缺少可用績效。
`--comparison-job-id` 與 `--run-id` 不能同時使用。兩種命令都會輸出報告的本機路徑；
自選章節與評語參數可用 `uv run --locked tradingdev-report --help` 查看。

ML 訊號研究也可使用已查到的候選 ID 產生報告：

```bash
uv run --locked tradingdev-report --workspace /absolute/path/to/tradingdev-workspace \
  --ml-entry-id CANDIDATE_ID --ml-entry-id ANOTHER_CANDIDATE_ID
```

同一研究的參數比較可加 `--ml-allow-parameter-axes`。若要展示評估條件不相容的
候選，可加 `--ml-allow-mixed`，報告會標明限制。用 `--ml-require-verified-baseline`
要求原始行情基線通過驗證，混合展示時也適用。評估列一致性可另看報告的核對結果。
詳見[報告選項與比較規則](docs/reports.md#composable-offline-reports)。

### 從終端機操作研究筆記

與 MCP 共用相同工作區。先用 `read` 找到筆記代號與版本，再匯出指定版本：

```bash
uv run --locked python -m tradingdev.adapters.cli.research_notes \
  --workspace /absolute/path/to/tradingdev-workspace read
uv run --locked python -m tradingdev.adapters.cli.research_notes \
  --workspace /absolute/path/to/tradingdev-workspace export NOTE_ID VERSION
```

`save draft.json` 可新建筆記；更新時加上 `--note-id` 與 `--expected-version`。
輸入格式與完整替換語意見[筆記契約](docs/reports.md#research-notebooks-and-archives)。

## 常見問題

| 情況 | 處理方式 |
| --- | --- |
| 客戶端找不到 `uv` 或 server 無法啟動 | 用 `command -v uv` 取得完整路徑；確認 `--directory` 指向專案根目錄，並已成功執行 `uv sync --locked`。 |
| 客戶端只有一般建議，沒有 TradingDev 工具結果 | 確認已連接並啟用工具；若仍無法執行，查看客戶端的 MCP 錯誤紀錄。只產生程式草稿或收到工作啟動訊息，都不代表回測已完成。 |
| 有確認單，但尚未開始正式回測 | 開啟 HTML 閱讀後，在對話明確回覆同意。若已過期、要修改設定或仍使用升級前的舊版確認單，須重新產生並確認；更新程式後請重新載入客戶端的工具清單。 |
| 手動執行 `uv run tradingdev-mcp` 後沒有畫面 | 預設為 stdio，正在等待 MCP 客戶端通訊，不會顯示對話視窗；一般由客戶端自動啟動即可。 |
| 行情下載失敗或資料不足 | 依工作錯誤確認來源、交易對、頻率與期間，並檢查來源能否連線；資金費率／自訂特徵需有本機檔案，DVOL 則也可能在下載時失敗。已有快取不代表已更新到最新日期。 |
| 助手回報策略檢查失敗 | 工具會提供診斷供助手修正；若需改變你的交易規則或補充資料，再確認需求。未通過檢查的自建版本不會取得回測資格。 |
| 回測完成，Dashboard 卻找不到結果 | 確認 `TRADINGDEV_WORKSPACE` 與 MCP 的 `--workspace` 相同。Dashboard 需要該次回測保存的圖表資料；最佳化結果請改在對話中查詢或生成 HTML 報告。 |
| 升級後舊策略或圖表無法使用 | 請助手依[策略契約](docs/strategy_contract.md)檢查舊設定，另存並驗證新版策略；原策略與檔案保留，不會直接覆寫或自動遷移。舊結果可讀範圍取決於當時保存的資料，完整新版圖表可能需要重新回測。 |
| 更新 GLFT 後無法沿用設定，或更換 K 棒頻率後結果不同 | 舊設定不會自動遷移，請助手依[GLFT 參數說明](docs/strategies/glft_strategy.md#multi-timeframe)更新後建立新研究。策略要求固定間距行情，更換頻率也會改變持倉及預測的實際時長，詳見[時間尺度](docs/strategies/glft_strategy.md#時間尺度)。 |
| 升級後無法查詢或取消舊工作 | 若涉及舊版等待確認的最佳化工作，依[升級舊工作區](#升級舊工作區)停止舊服務並另建工作區，保留原研究資料。 |
| macOS 出現 `libomp.dylib` 載入錯誤 | XGBoost／LightGBM 需要 OpenMP runtime。有 Homebrew 的 macOS 可執行 `brew install libomp`；沒有 Homebrew（例如只有 Anaconda）請見下方「沒有 Homebrew 時設定 OpenMP」。修改後請重新啟動客戶端或命令。 |

### 沒有 Homebrew 時設定 OpenMP（macOS）

以下方式適用於 Apple Silicon Mac 的特定 Anaconda 安裝，尚未驗證其他 conda 安裝。
僅在錯誤訊息列出的搜尋路徑包含
`/opt/homebrew/opt/libomp/lib/libomp.dylib`，且本機已有與 Python 相同架構的
`/opt/anaconda3/lib/libomp.dylib` 時使用。其他 conda 安裝位置須先確認實際路徑；
來源不存在時不要建立連結。

先檢查來源與架構，確認輸出包含 Python 所用的架構（Apple Silicon 為 `arm64`）：

```bash
uv run --locked python -c "import platform; print(platform.machine())"
test -f /opt/anaconda3/lib/libomp.dylib
file /opt/anaconda3/lib/libomp.dylib
```

確認來源有效、目的地尚未存在，再建立連結；`ln -s` 不會覆寫既有檔案：

```bash
sudo mkdir -p /opt/homebrew/opt/libomp/lib
sudo ln -s /opt/anaconda3/lib/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib
```

驗證直接啟動與透過 shell 啟動都能載入，再重新啟動客戶端：

```bash
uv run --locked python -c "import xgboost, lightgbm; print('ok')"
/bin/bash -c 'uv run --locked python -c "import xgboost, lightgbm; print(\"ok\")"'
```

注意事項：

- 這是機器本地設定，不由 `uv sync` 管理。移除或升級 conda 後，須重新檢查
  連結是否仍可讀、架構相符，並重新執行上方兩種載入檢查。
- 若日後改用 Homebrew，先確認目的地確實是自己建立、指向上述來源的 symlink，
  再移除該連結並執行 `brew install libomp`；不要刪除其他來源管理的檔案。
- 單靠 `DYLD_FALLBACK_LIBRARY_PATH` 設定可能無法讓客戶端啟動的程序找到函式庫，
  不能只以目前終端機載入成功判定已解決；請以上述兩種啟動方式確認，並重新啟動客戶端。

技術指標套件 TA-Lib 隨安裝命令一併安裝。若平台有相容 wheel，不需另外安裝 C 函式庫；
只有必須從原始碼編譯時，才需依[TA-Lib 安裝說明](https://github.com/TA-Lib/ta-lib-python#installation-)
準備底層依賴。

## 相關文件

- [軟體需求規格](docs/requirements.md)：產品範圍、需求編號、驗收條件與查核入口。
- [專案介紹簡報](https://slides.klwang.tw/tradingdev/)：給非技術聽眾的完整閱讀版，包含瀏覽器播放、工具比較與後續開發方向。
- [內建策略說明](docs/strategies/README.md)：各策略的規則、參數與資料需求。
- [策略契約](docs/strategy_contract.md)：自行撰寫 Python／YAML、版本與驗證規則。
- [執行結果與產物](docs/run_artifacts.md)：完整指標、執行設定與保存格式。
- [開發指南](docs/development.md)：修改 TradingDev 本身時的環境、品質檢查、測試與 Git／PR 流程。
- [專案架構](ARCHITECTURE.md)：服務分工、MCP 回覆契約與執行流程。
- [代理開發政策](AGENTS.md)：協助開發此專案的代理須遵守的規則。
