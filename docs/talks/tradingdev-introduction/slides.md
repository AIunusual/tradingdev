<!--
marp: true
theme: tradingdev
size: 16:9
paginate: true
lang: zh-Hant
title: TradingDev｜讓 AI 助手協助研究交易策略
description: 產品介紹、使用場景、分類比較與後續開發方向
-->

<!-- _class: cover -->

# TradingDev

讓 AI 助手協助研究交易策略

自己的想法，可檢查的實驗，可繼續追問的結果

*完整閱讀版 · 2026 年 10 月。歷史回測與策略研究。*

<!--
開場可以問聽眾是否曾有一個交易想法，卻不知道如何驗證。這份簡報介紹的是把想法變成可檢查實驗的過程，不展示或承諾投資績效。
本專案事實基準：d57da5c20aa7dc578dabef92bbf14d796df8adb6。外部資料查核：2026-10-25 至 2026-10-26。
來源：https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/README.md
-->

---

## 閱讀路線

| 主題 | 內容 |
| --- | --- |
| [策略研究](index.html#3) | 一個交易想法如何接受檢驗 |
| [運算與使用自由度](index.html#15) | 助手、主機與公司的 AI 產品 |
| [相關工具比較](index.html#21) | 函式庫、MCP 工具與完整平台 |
| [經驗如何回饋專案](index.html#32) | 研究紀錄與共用能力 |
| [後續開發方向](index.html#35) | 可獨立指定的開發目標 |

*各段可獨立閱讀。方向鍵換頁，P 開啟講者模式。*

<!--
這是完整閱讀版，不限定演講時間。現場可選擇前半主線與一兩個比較案例，其餘內容保留給網站閱讀。
-->

---

<!-- _class: statement -->

## 這裡的「研究能力」

> 把交易想法寫成明確規則，用歷史資料實驗，
> 比較修改前後的差異，並查明結果的原因。

AI 助手協助溝通與撰寫。TradingDev 提供工具、計算與研究紀錄。

<!--
研究不是泛稱投資分析。此處具體指策略程式的建立、檢查、歷史回測、參數實驗、時序驗證與結果查詢。
來源：專案 README、ARCHITECTURE.md。
-->

---

## 一個交易想法

> 「短期平均價格高於長期平均價格時買進。」

這個想法還需要補上明確規則：

- 平均多久的價格？何時賣出？
- 每次投入多少？是否允許做空？
- 使用哪段行情？手續費與滑價如何計算？

*這是說明流程的假設例子，沒有宣稱已測得績效。*

<!--
讓一般聽眾看到，口頭想法和可執行策略之間還有許多選擇。助手的任務之一是協助釐清，而不是替使用者悄悄決定。
-->

---

## 回測提供的證據

用過去的行情，模擬規則在當時會如何運作。

觀察收益之外，也要看**損失、交易成本與失敗期間**。

> 過去的結果，支持我們檢查假說。未來是否仍有效，需要繼續驗證。

<!--
可以用「歷史實驗」解釋回測。資料、費用、成交假設和選取期間都會影響答案，不把漂亮結果視為未來保證。
-->

---

## 使用者、助手與後端

| 角色 | 負責的事 |
| --- | --- |
| 使用者 | 提出目標、釐清規則、確認實驗 |
| AI 助手 | 理解需求、撰寫與修正程式、解讀結果 |
| TradingDev | 檢查策略、執行計算、保存及查詢證據 |

**MCP** 是讓 AI 助手使用外部工具的共同介面。

<!--
讀作 MCP，不是 MPC。TradingDev 不提供對話模型，搭配支援 MCP 的客戶端使用。研究的判斷和工具的計算各有責任。
來源：https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/ARCHITECTURE.md
-->

---

## 一次研究的完整流程

1. **形成策略**：從內建範例開始，或請助手撰寫自己的規則。
2. **檢查與試跑**：找出程式、參數及訊號問題。
3. **確認實驗**：閱讀設定，決定是否正式執行。
4. **執行與追問**：查看結果，再修正規則或比較設定。

<!--
新策略會保存版本，通過驗證及 dry-run 後才取得執行資格。試跑只涵蓋有限資料，不能冒充完整回測。
來源：README「建立與研究自己的策略」、docs/strategy_contract.md。
-->

---

## 自己的規則也能成為策略

內建策略提供起點。助手也能透過 MCP 保存新的 Python 策略版本。

檢查失敗時，後端回覆診斷，供助手修正後再驗證。

> 研究範圍可以從調整參數，延伸到修改交易規則本身。

*策略仍須符合訊號契約與檢查政策。驗證不等於完整安全隔離。*

<!--
這是與僅選擇固定模板的工具相比可具體展示的差異，但不是所有 MCP 專案都只支援模板。cloudQuant/backtrader-mcp 和 bcosm 也支援自訂策略，後面分別比較。
-->

---

## 執行前的確認單

正式回測前，先確認：

- 交易規則、資料期間與參數。
- 資金或交易規模、手續費與滑價。
- 試跑涵蓋了什麼，以及還有哪些限制。

**使用者明確同意後，才提交正式工作。**

*這份 HTML 確認單是執行前流程，與完成後的分析報告不同。*

<!--
修改設定後要準備新的計畫並重新確認。後端依賴助手如實轉交使用者決定，不能獨立驗證決定的人類來源。這項限制留在講者稿，操作細節以 README 為準。
來源：README「完成第一次回測」、docs/run_artifacts.md 的 execution plans。
-->

---

## 回測完成後，繼續問問題

> 「哪一段時間損失最多？」
> 「交易成本占了多少？」
> 「上次使用什麼參數？和這次差在哪裡？」

助手讀取保存的指標與交易紀錄，提出解釋及下一個實驗。

*查詢已保存的結果，不一定需要重新回測。*

<!--
結果主要供 LLM 查詢和分析。詳細交易、帳戶序列與執行紀錄可分頁查詢；不存在的舊紀錄不能靠助手猜測重建。
來源：src/tradingdev/mcp/tools/history.py、runs.py 與 mcp/prompts.py。
-->

---

## 修正規則與比較參數

| 想研究的問題 | 系統如何保存 |
| --- | --- |
| 改變進出場規則 | 保存新的策略版本，再驗證 |
| 同一規則換一組參數 | 固定該次實驗的有效設定 |
| 比較前後差異 | 找回各次結果與比較條件 |

結果有編號，讓對話可以指回特定實驗。

<!--
strategy/revision、plan、job、run 的用途不同。job 用於進度與取消，run 用於已保存結果；現有 ID 是追蹤基礎，不等於已綁定使用者擁有權。
來源：ARCHITECTURE.md 的 strategy identity 與 run artifacts。
-->

---

## 換一段資料，再檢驗一次

**調整規則的期間**與**檢驗效果的期間**分開。

前一段資料用來選參數，後一段觀察是否仍然成立。
也可以沿時間反覆進行，形成滾動驗證。

> 「這個結果，是普遍有效，還是剛好適合那段歷史？」

*已提供參數最佳化與 walk-forward。驗證仍受資料及實驗設定影響。*

<!--
避免用「自動找到最佳策略」形容參數最佳化。工具在指定候選、指標與訓練期間中選擇結果，再用保留區間檢驗。
來源：README「研究與比較參數」、ARCHITECTURE.md 的 optimization。
-->

---

## 依研究目的選擇模式

| 模式 | 主要回答的問題 |
| --- | --- |
| **signal：資金表現** | 給定一筆起始資金，資產、報酬與回撤如何變化？ |
| **volume：交易量與成本** | 按指定交易規模運作，累計成交量、成本與金額損益如何變化？ |

相同的策略研究流程，可以服務不同的衡量目標。

*這是 TradingDev 明確提供的選擇。其他引擎也能自訂部位與模擬方式。*

<!--
signal 使用 VectorBT，包含初始資金。volume 從零累計 P&L，沒有初始資金，不能當作本金報酬率；名目規模可由 size_weight 調整。volume 也不是市場深度或成交量限制撮合模型。
來源：https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/src/tradingdev/domain/backtest/result.py
其他引擎：https://vectorbt.dev/api/portfolio/base/#simulation-modes 、 https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing
-->

---

## 報告保留給需要細看的使用者

日常研究可以在對話中完成查詢、解讀與修正。

需要細查時，再產生 HTML 報告，觀察指標、圖表、交易與帳戶紀錄。

**計算結果與助手的文字解讀分開呈現。**

<!--
不要把使用者必須閱讀圖表說成產品核心。執行後報告是進階閱讀入口，與執行前確認單分開。保存紀錄不等於保存完整軟體環境，不能保證跨環境完全重現。
來源：README「查看結果與圖表」、docs/run_artifacts.md。
-->

---

<!-- _class: chapter -->

## 對話設備與回測主機

使用者與助手交談。

**計算可以交給資源更充足的主機。**

*遠端連線需安排網路與存取控制。集中排程及多人隔離屬後續擴充。*

<!--
以筆電與公司主機說明。遠端放置後端是架構可行性；目前 CLI 的 HTTP 預設綁定 localhost，跨設備需轉送。不能宣稱已有多節點叢集或大型租戶平台。
來源：README 的 Streamable HTTP 設定與限制。
-->

---

## 整合者可以決定的事

| 選擇 | 具體例子 |
| --- | --- |
| 對話與研究流程 | 先問哪些問題，何時修正或停止實驗 |
| 模型與費用 | 沿用個人助手，或由公司提供模型服務 |
| 資料與運算 | 使用自己的資料、主機與快取政策 |
| 研究規則與輸出 | 修改檢查、策略契約或對客戶的呈現方式 |

<!--
這是定位重點，但不能說只有 TradingDev 能外接 LLM 或自架。差異在於可以直接維護自己的研究服務實作，以及願意承擔哪些整合責任。
來源：ARCHITECTURE.md；QC MCP 與 Private Cloud 文件供比較。
-->

---

## 公司自己的 AI 產品

| 產品層 | 可以安排的責任 |
| --- | --- |
| 公司的 AI 入口 | 與客戶對話，選擇模型，安排用量及收費 |
| TradingDev 研究後端 | 接收 MCP 工具請求，驗證策略及執行實驗 |
| 公司的資料與主機 | 提供行情、計算與結果保存 |

**可行方向：把策略研究接進既有的 AI 網站或服務。**

*此頁說明可開發的整合方式，並非已完成的多人收費產品。*

<!--
對話 LLM 在公司的產品入口，token 使用量也由該層計量。後端可以配合公司的既有產品流程，公開多人產品仍需另行整合。
真正收費上線時另核實依賴及行情授權。這項工作不屬本次簡報實作。
-->

---

## 研究團隊自己的流程

沿用團隊的資料、實驗格式與檢查規則。

研究員提出策略。共同後端讓實驗採用一致的檢查與紀錄方式。

> 有價值的共用規則，可以直接改進到團隊維護的服務中。

*團隊情境為部署方向。使用者權限、策略隔離與集中資源管理列為 TODO。*

<!--
目前可先在可信環境採獨立部署；同一 server 的 workspace 不能當作現成多租戶隔離。QuantConnect Private Cloud 也可保留公司資料與運算，不把私有部署說成獨有。
-->

---

## 個人研究與教學

**個人研究**：沿用自己的 AI 助手，將想法變成實驗，找回過去的結果。

**教學情境**：用相同題目與資料，觀察不同規則、成本或驗證方式如何改變答案。

*教學可先採每人獨立環境。共用伺服器是另一項部署需求。*

<!--
這些是依現有能力提出的使用情境，不是假裝已完成的客戶案例。教學重點是如何檢查假說，不是比較誰的回測收益最高。
-->

---

## 已整合的基礎能力

| 研究工作 | TradingDev 提供 |
| --- | --- |
| 準備與實驗 | 行情來源、技術指標、歷史回測 |
| 比較與檢驗 | 參數最佳化、時序驗證、結果追蹤 |
| ML 訊號研究 | 已登錄模型的訓練、預測與評估 |

*機器學習研究也是 QuantConnect 提供的共同功能類別，各自支援範圍不同。TradingDev 目前不提供真實下單。*

<!--
ML 不作為獨有賣點。TradingDev MCP 模型目前包含 XGBoost、LightGBM、CatBoost 方向模型；QC 支援更多常見框架。模型種類不代表較泛用。
TradingDev 實際使用 VectorBT、TA-Lib、QuantStats 等套件；沒有把 Freqtrade 整合為依賴。
來源：pyproject.toml、src/tradingdev/domain/ml/models/registry.py；https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/popular-libraries/pytorch
-->

---

<!-- _class: chapter -->

## 三類相關工具

**基礎計算工具**提供運算能力。

**AI／MCP 研究工具**讓助手操作研究。

**完整平台**進一步整合資料、團隊與交易流程。

*依產品層次比較，才能理解各自適合的場合。*

<!--
這是為了說明的分類，不是嚴格互斥的產品分類。Freqtrade 是完整交易應用，比單純函式庫更高一層；LEAN 引擎與 QuantConnect 平台也要分開。
-->

---

<!-- _class: compact -->

## 基礎能力已有成熟實作

| 工具 | 主要用途 | 使用時還要安排什麼 |
| --- | --- | --- |
| VectorBT | Python 回測與參數研究函式庫 | 組裝研究流程、服務及保存方式 |
| Backtrader | 可擴充策略與交易模型的框架 | 選擇維護分支，整合資料與操作流程 |
| Freqtrade | 含回測、最佳化與交易功能的應用 | 配合其策略介面與部署方式 |

TradingDev 使用成熟計算套件，再整合成助手可操作的研究服務。

<!-- _footer: "來源：[VectorBT](https://vectorbt.dev/)、[Backtrader](https://www.backtrader.com/docu/)、[Freqtrade](https://www.freqtrade.io/en/stable/)" -->

<!--
重點是基礎功能有被整合，不把底層套件當作功能不足的對手。Freqtrade 本身已提供很多產品功能，不能說使用它一定要從頭自行組裝。
原版 Backtrader 主線更新停在 2023 年，社群分支持續修正。維護情況依分支而異，不用 issue 數量推論品質。
來源：https://github.com/cloudQuant/backtrader/blob/development/CHANGELOG.md 、 https://github.com/backtrader2/backtrader/issues/98
-->

---

## VectorBT PRO MCP

助手可以查文件、讀 API，並在 Python 環境執行與修正回測程式。

**適合的工作方式**：延續個人的程式研究環境，保留變數與資料繼續實驗。

**比較重點**：TradingDev 將策略版本、檢查、背景工作與結果查詢做成明確服務。

*PRO 的程式執行使用持續存在的 kernel，且限 stdio。多人使用需安排環境隔離，未實測衝突。*

<!-- _footer: "來源：[VectorBT PRO MCP 官方說明](https://vectorbt.pro/features/ai/mcp-server/)" -->

<!--
不能說 VectorBT PRO MCP 只能查文件，或不能開發策略。它也有可由助手讀取的研究 skills。其公開設計不能直接當作完整多人後端，但每人獨立環境是可能的。
-->

---

## MaverickMCP

提供策略模板、參數研究、驗證及分析工具，也有 ML 與訊號集成能力。

**適合的工作方式**：沿既有策略類型進行分析與實驗。

**比較重點**：TradingDev 讓助手保存及修正新的 Python 策略版本，並持續查詢完整研究紀錄。

*所查版本的自然語言解析會選擇模板與參數，不會把任意交易描述編譯成新策略。*

<!-- _footer: "來源：[MaverickMCP 回測 API](https://github.com/wshobson/maverick-mcp/blob/da70d074dcdac5fb21e3b2e6383cfebf391f28b5/docs/api/backtesting.md) · 查核版本 da70d07" -->

<!--
MaverickMCP 有自己的優勢，包括模板便利性、Monte Carlo 與 ensemble 工具。不要把自然語言解析成模板說成不能做任何策略研究。
所查 MCP 介面未見等價的新策略原始碼保存、revision 驗證及持久背景 job 生命週期。這是介面範圍比較，不是斷言開發者不能擴充原始碼。
-->

---

## cloudQuant/backtrader-mcp

已提供策略草稿、驗證、修復、工作追蹤及研究引導。

**共同方向**：讓助手參與策略開發，而不只呼叫一次回測。

**比較重點**：它採 Backtrader 的策略寫法；TradingDev 採訊號輸出規則，並提供逐筆交易與帳戶紀錄查詢。

*它已有 MCP prompts。TradingDev 使用 server instructions 與工具回覆提供引導。*

<!-- _footer: "來源：[backtrader-mcp](https://github.com/cloudQuant/backtrader-mcp/tree/ad6312a11c2fe7ff9a4c3fcfcfab5037a951b58c) · 查核版本 ad6312a" -->

<!--
cloudQuant 與 QuantConnect 是不同專案。它有持久工作、取消及日誌，與 TradingDev 的部分功能相近。移入策略時要依各自格式改寫，對話追問能力則取決於結果查詢的範圍。
獨立的 backtrader-skills、backtrader-agent 另有流程與有限 sweep 經驗保存，不能自動視為主 MCP repository 已整合的功能。
-->

---

## bcosm/backtester-mcp

可接收策略程式，提供參數最佳化、穩健性檢驗及結果保存。

**比較重點**：TradingDev 把策略版本、驗證資格、執行計畫與實際結果串在同一個生命週期。

它另有 Python 層的**獨立策略淨值加權分析**。

*TradingDev 尚缺等價組合分析，列入 TODO-04。這項 API 尚未接到對方 MCP／CLI，也不是共用資金帳戶。*

<!-- _footer: "來源：[MCP 工具](https://github.com/bcosm/backtester-mcp/blob/ddc0394d0ad6edb07cf836de4f931a641adf23f5/src/backtester_mcp/mcp_server.py)、[組合 API](https://github.com/bcosm/backtester-mcp/blob/ddc0394d0ad6edb07cf836de4f931a641adf23f5/src/backtester_mcp/portfolio.py) · 查核版本 ddc0394" -->

<!--
bcosm 的驗證工具有 verdict/reasons，流程範例放在 examples/agent_workflow_demo.py。未註冊 prompts 不能簡化成完全沒有引導。
save_run 接收呼叫端提交的 metrics/code/params，與 TradingDev 固定實際執行設定、保存產物的流程不同。
-->

---

## QuantConnect 與 LEAN

| 層次 | 作用 |
| --- | --- |
| LEAN 引擎 | 事件驅動的策略、資料、委託與帳務模型 |
| QuantConnect 平台 | 專案、資料、運算、協作與交易服務 |
| 官方 MCP 與 Agents | 對外提供工具，或由研究代理操作流程 |

三者相關，但使用條件與整合責任不同。

<!-- _footer: "來源：[LEAN](https://github.com/QuantConnect/Lean)、[QC MCP](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server/key-concepts)、[QC Agents](https://www.quantconnect.com/docs/v2/ai-assistance/agents)" -->

<!--
LEAN engine 為 Apache-2.0，可自行建置；官方 CLI/Local Platform 與 Private Cloud 有各自方案條件。不要把官方 remote MCP 的付費 Agent Node 要求套到所有自行使用 LEAN 的情境。
-->

---

<!-- _class: compact -->

## 與 QuantConnect 比較的重點

| 問題 | QuantConnect | TradingDev |
| --- | --- | --- |
| AI 如何研究策略？ | 可用外部 MCP 客戶端或內建 Agents | 外部助手操作研究後端 |
| 程式依循什麼模型？ | LEAN 的策略、事件及交易模型 | 訊號輸出規則與驗證流程 |
| 團隊要維護什麼？ | 選用平台服務或自管方案 | 自己的研究服務與整合 |
| 資料與主機能自管嗎？ | 有 Local／Private Cloud 路線 | 可自行安排，依需求擴充 |

*雙方都有外部 AI、回測與 ML 研究。價格、速度與準確性未作同條件實測。*

<!-- _footer: "來源：[Private Cloud](https://www.quantconnect.com/docs/v2/local-platform/private-cloud)、[官方 MCP](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server/key-concepts)、本專案架構" -->

<!--
QC 已有新策略建立、編譯、回測、最佳化及結果診斷，不能宣稱 TradingDev 的完整研究流程對 QC 也是獨有。
本頁表達工作方式與維護責任。要主張某個情境更省事或成本更低，仍需具體部署評估。
-->

---

## 使用場合決定選擇

| 你的主要需求 | 可優先考慮的方向 |
| --- | --- |
| 已有 AI 網站，要加入策略實驗 | 將 TradingDev 接成可修改的研究元件 |
| 已有 Python 流程，要自訂檢查與紀錄 | 維護自己的 TradingDev 服務 |
| 希望採用完整資料與團隊平台 | 評估 QuantConnect 的現成服務 |
| 要接券商、處理多資產帳務與實盤 | QC／LEAN 已有較完整能力 |

*這是依現有能力提出的情境建議。TradingDev 目前不提供真實下單。*

<!--
舉例：「公司已經有和客戶聊天的 AI，只缺一套能真正跑策略實驗的工具。」這比抽象的自由度更容易理解。
需要私有資料時 QC Private Cloud 也可行，不能用資料是否留在公司作單一判準。
-->

---

<!-- _class: statement -->

## TradingDev 的產品選擇

> 將策略研究做成可由自己的 AI 助手操作、
> 也可由團隊直接修改的後端服務。

使用者保留對話與產品設計的選擇。
研究服務保留明確的檢查、執行與證據。

<!--
這是對前面事實的定位總結，不是宣稱所有能力市場獨有。其價值取決於整合者是否需要掌握研究服務實作。
-->

---

## 目前提供與後續擴充

| 已提供 | 後續方向 |
| --- | --- |
| 策略建立、修復、驗證與試跑 | 更完整的研究引導與記憶 |
| 背景執行、參數研究、歷史結果查詢 | 多策略組合與資金配置 |
| 策略及實驗編號、共用檢查 | 多人權限、策略隔離與工程回饋 |

*後續項目尚未完成，也沒有在此承諾開發時程。*

<!--
現有身分 ID 是未來權限設計可用的基礎，尚未完成使用者隔離。此頁集中區分已提供的產品能力與未來方向。
-->

---

<!-- _class: chapter -->

## 兩種「整合策略」

**整合開發經驗**
把個別問題轉成共用檢查與程式元件。

**組合交易策略**
研究不同策略如何配置資金、互補風險。

<!--
兩種整合共享研究證據，但目的與工程責任不同。工程回饋改善開發工具，策略配置研究整體資金與風險。
-->

---

## 一次問題，成為下一次的檢查

1. 某次策略開發發現時間對齊問題。
2. 保留可重現案例，確認錯誤原因。
3. 加入共用檢查或元件，並完成測試與審查。
4. 下一位開發者使用同一套工具時，得到這項改進。

*這是工程回饋的示例。現有版本與紀錄提供基礎，自動回饋流程仍待開發。*

<!--
可以讓觀眾想像「前一個人踩過的坑，成為下一個人的防護」。這不是自動分享私人策略，也不代表系統已會自行修改並發布核心程式。
-->

---

## 外部開發助手的擴充方式

外部助手讀取**策略案例、執行證據與專案程式**，提出共用能力的修改。

初版可以獨立於回測服務運作，沿用專案的測試與審查流程。

> 可獨立掛接的工程流程，讓後端逐步累積改進。

*開發可行性。實際修改範圍依問題決定，尚未實作自動整合。*

<!--
外部開發流程可以獨立掛接，實際修改是否涉及核心程式，仍取決於所發現的問題。抽象出的共用能力也需要測試與審查。
現有 record_feature_request、revision 與 run 可作輸入，並未完成自動提案與程式整合。
-->

---

<!-- _class: compact -->

## 後續開發目標

| ID | 目標 |
| --- | --- |
| TODO-01 | 研究引導與診斷 |
| TODO-02 | 可引用證據的研究記憶 |
| TODO-03 | 共用元件與工程回饋 |
| TODO-04 | 多策略配置研究 |
| TODO-05 | 成熟引擎評估 |
| TODO-06 | 多人服務身分與權限 |

<!--
以下目標保持簡短。之後可指定「閱讀 TODO-04 再調查與開發」，開始時仍需重新確認需求、程式與外部套件現況，不把候選方案視為已決定的規格。
-->

---

## TODO-01 研究引導與診斷

**目標**：讓助手依研究狀態與失敗原因，選擇下一個檢查或實驗。

現有基礎是流程 instructions、策略診斷與執行限制。

可參考其他 MCP 的工作流程，以及 QC 的結構化回測診斷。

*待開發。以實際研究情境檢驗引導效果，不只增加提示文字。*

<!-- _footer: "參考：[QC Agents 與 diagnostics](https://www.quantconnect.com/docs/v2/ai-assistance/agents)、[backtrader-mcp](https://github.com/cloudQuant/backtrader-mcp/tree/ad6312a11c2fe7ff9a4c3fcfcfab5037a951b58c)" -->

<!--
具體診斷規則、引導內容與評估案例，留待開發任務依當時的研究需求分析。
-->

---

## TODO-02 可引用證據的研究記憶

**目標**：讓下一次對話知道已經試過什麼，以及為何保留或放棄某個假說。

保存研究結論時，連回使用的策略版本與實驗結果。

可參考 QuantConnect 的 project memory。

*待開發。歷史結果是證據，研究記憶再補上結論與適用前提。*

<!-- _footer: "參考：[QC Custom Memories](https://www.quantconnect.com/docs/v2/ai-assistance/agents#Custom-Memories)" -->

<!--
外部 LLM 可以整理結論，後端提供保存與檢索。無須因為新增記憶，就把對話模型供應商放進研究核心。
-->

---

## TODO-03 共用元件與工程回饋

**目標**：把確認過的問題與解法，轉成可重用的檢查和程式元件。

採外部開發流程提出修改，經測試及審查後整合。

可參考 QC Shared Libraries，並保留元件版本與實驗的關聯。

*待開發。公開共用知識與私人策略內容，需依使用情境區分。*

<!-- _footer: "參考：[QC Shared Libraries](https://www.quantconnect.com/docs/v2/local-platform/projects/shared-libraries)" -->

<!--
不預先規定整個自動開發系統。初期可由人選擇案例，再讓開發助手提出改進。元件更新後，仍要能解釋舊實驗引用的版本。
-->

---

## TODO-04 多策略配置研究

**目標**：理解多個策略放在一起，能否改善整體風險與表現。

先做日期、成本與資本基準一致的固定權重分析。
再按需求評估 Riskfolio-Lib 或 PyPortfolioOpt 的配置功能。

*待開發。signal 淨值與 volume 累計損益不能直接混算。共用資金及訂單衝突另需執行模型。*

<!-- _footer: "候選：[Riskfolio-Lib](https://portfoliooptimization.org/riskfoliolib/portfolio.html)、[PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html)、[VectorBT shared cash](https://vectorbt.dev/api/portfolio/base/)" -->

<!--
先區分訊號投票、獨立淨值加權、配置最佳化和共享帳戶。初始配置後讓權重漂移，與每期固定權重再平衡也不同。此頁記錄目標，不要求現在完成這些設計。
-->

---

## TODO-05 成熟引擎評估

**目標**：確認 LEAN 等引擎，是否更適合未來的資產、委託與帳務需求。

| 可能優勢 | 需要付出的成本 |
| --- | --- |
| 成熟的交易時間、資產與帳務模型 | 調整策略契約與資料轉接 |
| 減少自行維護基礎機制 | 核對成交語意、結果與執行管理 |

*待評估。先用可人工核對的案例比較，再決定是否遷移。*

<!-- _footer: "參考：[LEAN Reality Modeling](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts)" -->

<!--
保留研究服務責任，不代表凍結目前的訊號契約。若需要事件、委託與成交回報，應允許原生事件策略設計。其他候選依需求再調查，例如 order book/多 venue 需求才評估 NautilusTrader。
-->

---

## 成熟引擎仍需要正確假設

LEAN 的內建回測成交模型預設完全成交。
預設券商模型使用零滑價模型。

**資料時間、交易成本、流動性與訓練切分，仍需核對。**

> 採用成熟引擎可以減少基礎實作，正確的實驗仍需要驗證。

<!-- _footer: "來源：[LEAN Trade Fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts)、[Slippage](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts)" -->

<!--
具體是 DefaultBrokerageModel 的 NullSlippageModel，不是說 LEAN 無法模擬滑價。部分成交、流動性等可自訂，代價是仍須設定與驗證模型。
-->

---

## TODO-06 多人服務身分與權限

**目標**：將使用者、專案、策略版本與執行紀錄關聯起來。

依個人、團隊或公開產品情境，決定哪些內容可以共用、哪些需要隔離。

現有 ID 與工作紀錄，提供後續設計的基礎。

*待開發。現有 workspace 與靜態檢查不是完整的租戶或執行安全隔離。*

<!-- _footer: "現況依據：[TradingDev 架構與執行限制](https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/ARCHITECTURE.md)" -->

<!--
不把缺口重複鋪陳成警告清單。真正開始多人產品開發時，再確認權限、隔離、配額與工作排程需求。
-->

---

## 依使用場合決定優先序

| 使用場合 | 優先處理 |
| --- | --- |
| 個人反覆研究 | 引導、診斷與研究記憶 |
| 團隊共享開發經驗 | 共用元件與工程回饋 |
| 多個策略一起評估 | 組合分析與配置模型 |
| 對外提供多人服務 | 身分、權限與執行隔離 |

對話介面、模型路由與計費可留在整合層。
真實下單維持目前產品邊界之外。

<!--
不承諾所有 TODO 都一定做。把對話模型與產品計費留在外部，是責任分工；多人服務必需的隔離則不能用自由度為理由省略。
-->

---

<!-- _class: chapter -->

## TradingDev 的方向

今天，讓自己的 AI 助手完成**可追問的策略實驗**。

接下來，讓研究中得到的證據與經驗，
逐步成為下一次開發可以使用的能力。

*後續附有參考來源。演講後可繼續在線上閱讀。*

<!--
結尾回到一個人的想法如何變成有證據的研究，以及工程改進如何幫助下一個人。避免用未驗證的獲利或效能數字作收尾。
-->

---

<!-- _class: references -->

## 參考資料：本專案與基礎工具

- [TradingDev 使用說明](https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/README.md)與[架構](https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/ARCHITECTURE.md)
- [策略契約](https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/docs/strategy_contract.md)與[執行產物](https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/docs/run_artifacts.md)
- [VectorBT](https://vectorbt.dev/)與[VectorBT PRO MCP](https://vectorbt.pro/features/ai/mcp-server/)
- [Backtrader](https://www.backtrader.com/docu/)與[社群分支更新](https://github.com/cloudQuant/backtrader/blob/development/CHANGELOG.md)
- [Freqtrade](https://www.freqtrade.io/en/stable/)

*本專案基準 d57da5c。外部文件查核於 2026-10-25 至 2026-10-26。*

<!--
原始碼來源固定版本，外部官方文件可能持續變動。簡報更新時應重新核對涉及的主張。
-->

---

<!-- _class: references -->

## 參考資料：MCP 與完整平台

- [MaverickMCP](https://github.com/wshobson/maverick-mcp/tree/da70d074dcdac5fb21e3b2e6383cfebf391f28b5)
- [cloudQuant/backtrader-mcp](https://github.com/cloudQuant/backtrader-mcp/tree/ad6312a11c2fe7ff9a4c3fcfcfab5037a951b58c)
- [bcosm/backtester-mcp](https://github.com/bcosm/backtester-mcp/tree/ddc0394d0ad6edb07cf836de4f931a641adf23f5)
- [QuantConnect MCP](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server/key-concepts)、[Agents](https://www.quantconnect.com/docs/v2/ai-assistance/agents)與[Private Cloud](https://www.quantconnect.com/docs/v2/local-platform/private-cloud)
- [LEAN 引擎](https://github.com/QuantConnect/Lean)與[Algorithm Framework](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview)

*比較依公開文件與所列版本。不代表已進行效能、併發或準確性基準測試。*

<!--
需要進一步調查時，從各比較頁引用的具體程式或官方文件開始。候選配置套件與引擎的來源已附於 TODO 頁。
-->
