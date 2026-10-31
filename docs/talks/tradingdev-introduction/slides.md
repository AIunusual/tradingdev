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
封面後先提供閱讀路線，再交代個人的研究背景與共用後端的目標，以必要能力比較現有方案，提出 TradingDev 的核心研發課題與目前基礎。接著用一個交易想法，展示如何形成可檢查的實驗；逐項產品比較保留在後段供深入閱讀。
初稿專案事實基準：d57da5c20aa7dc578dabef92bbf14d796df8adb6；執行確認與報告交付流程依對應頁面所列的新版本更新。外部資料查核：2026-10-25 至 2026-10-26。
開場的比較摘要另於 2026-10-30 查核官方文件；「目前已建立的研究基礎」依 cd95d6091bbc2a46c6a58cdb326aa5df730519c4 更新。
來源：https://github.com/Kuan-Lun/tradingdev/blob/d57da5c20aa7dc578dabef92bbf14d796df8adb6/README.md
-->

---

<!-- _class: compact -->

## 閱讀路線

| 章節 | 主題 | 內容 |
| --- | --- | --- |
| 01 | [開發動機與專案定位](index.html#3) | 研究需求、現有方案與核心研發課題 |
| 02 | [策略研究](index.html#10) | 一個交易想法如何接受檢驗 |
| 03 | [運算與使用自由度](index.html#23) | 助手、主機與公司的 AI 產品 |
| 04 | [產品詳細比較](index.html#30) | 各工具的能力、工作方式與使用場合 |
| 05 | [經驗如何回饋專案](index.html#40) | 研究紀錄與共用能力 |
| 06 | [後續開發方向](index.html#44) | 可獨立指定的開發目標 |

<!-- _footer: "點選章名跳轉。方向鍵換頁，P 開啟講者模式。" -->

<!--
這是分為六章的完整閱讀版，不限定演講時間。各章可獨立閱讀，內容頁頂端會標示目前章節。開場以比較摘要建立研發定位，再看研究流程及使用情境。後段的逐項產品比較可供深入閱讀與問答，接著說明經驗回饋與後續開發目標。
-->

---

<!-- header: "01　開發動機與專案定位" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 01　開發動機與專案定位

研究背景、必要能力與本專案的研發定位。

---

## 開發動機

我是臺大資工計算理論暨金融計算實驗室的一員，

許多實驗室成員的研究與金融交易相關。

> **我希望建立共用的研究後端，**
> **支援成員開發各自的交易策略。**

<!--
我是國立臺灣大學資訊工程學系計算理論暨金融計算實驗室的一員。許多成員的研究與金融交易相關，因此希望建立共用的後端，協助大家開發自己的策略。
共用的對象是研究所需的基礎能力，各位成員仍能研究自己的策略構想。接下來先說明這個目標需要哪些能力，再看現有工具能提供什麼。
-->

---

<!-- _class: compact -->

## 共用研究後端的必要能力

**共用的實驗基礎**
在同一後端執行回測，保存可重現的設定與結果。

**多樣的策略與模型**
支援成員建構、修改與驗證不同的量化方法。

**協助改進策略的回饋**
檢查策略問題，分析實驗結果，協助下一輪修正。

**AI 驅動的自動迭代**
讓 AI 從使用者的策略開發經驗，自動提出並驗證後端改進。

<!-- _footer: "開發目標，尚未全部完成。" -->

<!--
共用實驗基礎需要清楚區分各自的策略、版本及結果，並保存重現實驗所需的設定與依據。多樣的策略與模型包含交易規則、統計方法及機器學習模型的建構，以及適用的訓練與驗證。
策略回饋應引用檢查與實驗證據，協助提出修改，再驗證修改是否有效。
AI 開發助手從多位使用者的策略開發紀錄中辨識共通問題與需求，自動產生後端改進方案及程式修改，執行測試並依結果繼續修正。通過審查後，將適合共用的檢查、元件或流程改善整合回後端，支援下一輪研究。
這些是目標能力。目前已有回測與研究紀錄等基礎，多人服務身分、權限及自動工程回饋仍待開發。以下以這些需求檢視現有工具與整合責任。
-->

---

<!-- _class: compact -->

## 現有工具已經能做什麼

| 選擇 | 已提供的能力 |
| --- | --- |
| VectorBT、Backtrader | 策略與歷史回測的計算基礎 |
| VectorBT PRO MCP 等研究工具 | 讓 AI 查工具、撰寫程式及操作研究 |
| QuantConnect／LEAN 生態 | 資料、策略研究、團隊與交易流程 |

**回測與 AI 研究已有成熟工具。**
本專案以這些能力為基礎，聚焦研究決策、經驗累積與後端改進。

<!-- _footer: "來源：[VectorBT](https://vectorbt.dev/)、[Backtrader](https://www.backtrader.com/docu/)、[VBT PRO MCP](https://vectorbt.pro/features/ai/mcp-server/)、[QC Agents](https://www.quantconnect.com/docs/v2/ai-assistance/agents)" -->

<!--
回測是用歷史行情模擬交易規則。現有工具已能完成許多計算與研究工作，TradingDev 也沿用其中的成熟套件。
Freqtrade 是含回測、最佳化及交易功能的應用。cloudQuant/backtrader-mcp、bcosm 也支援自訂策略，後段逐一比較。QuantConnect 平台已涵蓋完整研究流程，不能把它描述成只有回測引擎。
-->

---

<!-- _class: compact -->

## 現有方案與研究需求的比較

| 必要能力 | 現有方案的相關能力 | TradingDev 的研發目標 |
| --- | --- | --- |
| 共用實驗 | QuantConnect：專案回測與結果管理 | 以一致規則與紀錄支援成員研究 |
| 策略與模型 | VBT PRO MCP 程式研究、QC 機器學習 | 支援不同方法與一致驗證 |
| 策略回饋 | QC Agents：研究流程與診斷 | 依實驗證據提出下一步研究 |
| 後端自動迭代 | QC：專案記憶與共用程式庫 | 從研究案例提出、驗證後端改進 |

*右欄是本專案的研發目標，各項能力仍分階段開發。*

<!-- _footer: "來源：[QC 回測](https://www.quantconnect.com/docs/v2/local-platform/backtesting/getting-started)、[ML](https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/key-concepts)、[Agents](https://www.quantconnect.com/docs/v2/ai-assistance/agents)、[程式庫](https://www.quantconnect.com/docs/v2/local-platform/projects/shared-libraries)、[VBT PRO MCP](https://vectorbt.pro/features/ai/mcp-server/) · 查核 2026-10-30" -->

<!--
QC 是 QuantConnect。這頁依四項研究需求，列出官方文件已說明的能力與本專案的研發方向；逐項工具的操作方式與使用情境留在後段。
QuantConnect 已提供專案回測、結果管理與 ML 研究；Agents 能依工具結果繼續研究，並接收回測診斷。VBT PRO MCP 能讓助手執行與修改 Python 回測程式。
自動迭代列中的專案記憶與共用程式庫，是經驗保存與元件重用的相關能力，不等同於自動修改平台後端。此處不據此判定其他產品能否完成後端自動改進，也不宣稱右欄能力是 TradingDev 獨有。
現有方案能否直接滿足特定研究需求，仍須以案例驗證。右欄列的是本專案希望建立與評估的能力，不是已測得的功能或效能優勢。
來源：本頁頁尾的官方文件，查核於 2026-10-30。
-->

---

## TradingDev 的核心研發課題

**有證據的 AI 研究決策**
讓 AI 根據檢查與實驗結果，提出下一步研究方向。

**可累積的研究經驗**
將不同策略的問題與解法，整理成可重用的知識與檢查。

**可驗證的自動迭代**
讓 AI 提出後端修改，經測試與審查後回饋下一輪研究。

*以可重現案例驗證成效，各項能力仍分階段開發。*

<!--
研發重點是讓研究證據持續支持判斷與改進。成熟回測工具提供計算基礎，接下來要用具體案例驗證 AI 是否提出有依據的實驗、找回適用的研究經驗，以及產生能解決共通問題的後端修改。
這些是待驗證的研發課題。現有實驗與紀錄可作基礎，研究引導、跨研究記憶及自動工程回饋的進度與限制，在後續頁面分別說明。
-->

---

## 目前已建立的研究基礎

> 透過 MCP，讓 AI 運用研究指引建立與修正策略，
> 依據檢查與實驗結果，提出可追問的分析。

- **策略實驗**：建立與修正策略，完成檢查與歷史回測。
- **證據保存**：連結策略版本、有效設定與實驗結果。
- **研究回饋**：診斷提示、報告解讀與可續讀的研究筆記。

<!-- _footer: "現況依據：[TradingDev 使用說明](https://github.com/Kuan-Lun/tradingdev/blob/cd95d6091bbc2a46c6a58cdb326aa5df730519c4/README.md)與[架構](https://github.com/Kuan-Lun/tradingdev/blob/cd95d6091bbc2a46c6a58cdb326aa5df730519c4/ARCHITECTURE.md)" -->

<!--
後端除了接收工具請求、執行計算和保存資料，也提供模型可讀取的研究流程指引、策略契約與診斷。使用者不必自己把一次次實驗的程式、設定和結果拼回來。
診斷先提供規則式提示，外部 LLM 的報告解讀須保留觀察依據與限制，研究建議不會自動啟動新實驗。研究筆記可保存階段摘要及引用報告，目前尚未完成跨研究自動檢索與後端自動改進。
TradingDev 不內建對話模型。這些能力是研究後端的現有基礎，不代表已提供多人權限或完整執行隔離。
研究指引來源：src/tradingdev/mcp/prompts.py 的 SERVER_INSTRUCTIONS、server.py 的 instructions 註冊，以及 tools/strategy.py 的按需契約。
-->

---

<!-- header: "02　策略研究" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 02　策略研究

一個交易想法如何成為可檢查、可追問的實驗。

---

<!-- _class: statement -->

## 這裡的「研究能力」

> 把交易想法寫成明確規則，用歷史資料實驗，
> 比較修改前後的差異，並檢查可能的原因。

TradingDev 透過 MCP 提供**研究流程指引與檢查規則**，
引導 AI 釐清需求、開發策略，並依實驗證據提出下一步。

<!--
研究不是泛稱投資分析。此處具體指策略程式的建立、檢查、歷史回測、參數實驗、時序驗證與結果查詢。
指引由後端提供，推理與互動由使用者選擇的 AI 客戶端承擔。直接在 Codex 開發策略時，Codex 就是這個角色，不需要另建一套助手服務。實際呈現與指引採用方式由客戶端決定，不能把提示詞當成所有客戶端必然遵循的保證。
來源：src/tradingdev/mcp/prompts.py、server.py 與 tools/strategy.py。
-->

---

<!-- _class: compact -->

## MCP 的提示詞引導：均線策略案例

> **使用者**：「我想測試短均線高於長均線時買進。」

| TradingDev 的指引 | 前端 AI 可能接著問 |
| --- | --- |
| **缺項時追問** | 「想測哪個標的、哪段期間？每根 K 棒代表多久？」 |
| **將策略具體化** | 「均線各用幾根？短均線跌破時，是平倉還是做空？」 |
| **執行前確認** | 「本金、手續費與滑價有指定嗎？若沒有，我先列出模擬預設供你確認。」 |

*示意對話，問法依情境調整；整理成確認單後，使用者同意才回測。*

<!--
本頁保留使用者只有均線交易構想的情境，以具體問句展示 TradingDev 的指引如何影響前端互動。表中的問題可分散於多輪對話，不是要求一次問完；補齊必要研究選擇後，再釐清策略細節與提出方案。這是依現有指引編寫的示意，不是真實模型逐字紀錄或已執行的回測。
提示詞引導由 TradingDev 的 server instructions、工具說明與按需策略契約提供；此處不是指另行註冊的 MCP prompts，也不是固定問卷。實際問句與互動由 Codex 等客戶端的 LLM 組織。
建立新研究前，構想、標的、回測期間與 K 棒頻率缺項時須追問，不能從範例猜出選擇。只有明確要求沿用已選定研究時，才可繼承其設定。
策略具體化包含整理進場、持有、出場與交易方向，確認參數並形成可驗證的程式。模型可提出有說明依據的方案，不能把自己的假設冒充使用者要求。
本金與成本未指定時可依共用模擬政策解析，明示有效值優先。前端須呈現後端產生的確認摘要與 HTML，待使用者明確同意該方案才提交執行。這些是歷史回測假設，不是即時費率或真實下單設定；本金與交易規模依執行模式適用。
提示詞要求不保證所有客戶端一定遵循；後端另檢查必要欄位、設定與執行狀態，但不能證明輸入或同意真的來自使用者。
來源：src/tradingdev/mcp/prompts.py 的 SERVER_INSTRUCTIONS、server.py 的 instructions 註冊，以及 src/tradingdev/domain/strategies/templates.py 的 research 契約與 src/tradingdev/domain/research_policy.py。
-->

---

<!-- _class: compact -->

## 回測證據與 AI 解讀：均線策略案例

> **使用者**：「這個均線策略為什麼虧損？該改哪裡？」

| TradingDev 的機制 | 前端 AI 可能如何回應 |
| --- | --- |
| **查詢實驗證據** | 「先核對這次的期間與成本，再查回撤和虧損交易。」 |
| **規則式診斷** | 「若回撤觸發診斷提示，我會先檢查那段期間的交易。」 |
| **引導 AI 分析** | 「先區分觀察與假說，再提議用相同期間與成本，比較一種出場規則。」 |
| **保存報告與引用** | 「把分析與對應的回測結果存入報告，方便你繼續追問。」 |

*以上是現有功能的對話示例，實際分析依回測結果而定。*

<!--
承接上一頁的均線策略，假設使用者已確認並完成一次歷史回測，再詢問虧損與修改方向。對話是依實作編寫的情境示意，不是真實模型逐字紀錄，也沒有宣稱本例已測得虧損、回撤或改善效果。
查詢證據：get_run_metrics 回傳保存的指標、定義、有效設定及不可用原因；get_run_trades、get_run_equity 等工具讀取對應執行範圍的原始紀錄。TradeHistoryService 與 PerformanceStore 核對保存資料的身分與完整性，查詢不會重新執行策略。缺少資料須保留原因，不能當成零或由模型猜測。
規則式診斷：DiagnosticService 讀取保存的結果，交給可替換的 DiagnosticProvider。目前的 MaverickDiagnostics 依六項固定門檻提供數值依據、建議檢查及限制。例如 signal 模式的最大回撤超過 0.25，會提示先檢查回撤期間的交易，再以原策略為基準比較明確的出場或風控變體。門檻不是使用者風險上限，也不證明出場規則就是虧損原因；診斷沒有直接找出虧損期間或完成市場情境、波動敏感度的因果分析。
AI 分析指引：SERVER_INSTRUCTIONS 要求先讀診斷，涉及特定虧損期間時再查交易或權益資料；解讀須包含 summary、observations、hypotheses_and_limits、next_steps 與精確的 run_id/scope 引用。文字推理由前端 LLM 完成，後端沒有自行呼叫另一個模型產生評語。新比較須另提可確認的方案，研究建議不授權自動執行。
報告保存：generate_report 接收模型的結構化分析，ReportService 核對引用是否屬於所選報告結果，將分析與規則式診斷保存在 HTML 及 manifest。計算證據與 LLM 解讀分開呈現，沒有解讀時明確標示；引用檢查不等於證明文字推論正確。歷史結果只支持檢驗假說，不保證未來有效。
來源：src/tradingdev/mcp/prompts.py、mcp/tools/runs.py、mcp/tools/history.py、app/diagnostic_service.py、adapters/reporting/maverick_diagnostics.py、app/contracts/reports.py、app/report_service.py，以及 docs/reports.md「Research interpretation and diagnostic providers」。
-->

---

## 使用者、AI 客戶端與 TradingDev

| 角色 | 負責的事 |
| --- | --- |
| 使用者 | 提出目標、釐清規則、確認實驗 |
| AI 客戶端 | 依研究指引理解需求、撰寫與修正策略、解讀結果 |
| TradingDev | 研究流程指引、策略契約與檢查、實驗計算與證據 |

**MCP** 將研究指引與工具接到 Codex 等 AI 客戶端。

可直接使用既有客戶端，也可整合進自己的應用。

<!--
讀作 MCP，不是 MPC。表中列的是責任，不是要求使用者部署三套系統。直接在 Codex 中開發策略時，Codex 承擔 AI 客戶端角色；TradingDev 不限定對話介面，也不內建對話模型。
TradingDev 以 server instructions、工具描述與按需契約提供研究指引，並由後端強制檢查可機械驗證的輸入與執行條件。客戶端決定互動與推理方式，提示詞不能保證模型遵循，也不能證明使用者真的閱讀或同意。
來源：src/tradingdev/mcp/server.py、prompts.py、tools/strategy.py，以及 ARCHITECTURE.md。
-->

---

## 後端提供的研究流程指引

1. **釐清構想**：確認交易想法、標的、期間與 K 棒頻率。
2. **開發與修復**：依策略契約撰寫，根據檢查結果修正。
3. **確認實驗**：閱讀設定，決定是否正式執行。
4. **解讀證據**：依回測與診斷，區分觀察、假說及限制。
5. **延續研究**：提出下一步，依需要保存研究筆記。

<!--
新策略會保存版本，通過驗證及 dry-run 後才取得執行資格；其中的訊號檢查使用固定合成資料，並非歷史回測。準備確認單時只做離線程式與設定檢查，以及適用的合成訊號檢查。單次準備的內建策略不在此時執行訊號；固定參數比較則對每組候選（包含內建策略）執行 80 與 240 根合成 K 棒的訊號檢查。確認前不取得真實行情、不訓練模型、不執行歷史回測；行情、訓練及績效須同意後才開始驗證。
這些步驟已有 SERVER_INSTRUCTIONS 引導，並搭配工具說明及 get_strategy_contract 的按需內容。缺少必要研究選擇時須追問；保存研究筆記須有保存或分享需求。客戶端可安排自己的互動方式，但工具仍會檢查契約與執行狀態。
來源：src/tradingdev/mcp/prompts.py、tools/strategy.py、README「完成第一次回測」與 docs/strategy_contract.md。
-->

<!-- _footer: "來源：[執行確認流程](https://github.com/Kuan-Lun/tradingdev/blob/35d709d4ac9ed8c9d3a71a3e24cd98996bf3a1e6/README.md#完成第一次回測)" -->

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
- 離線檢查的範圍，以及尚未驗證的行情、訓練與績效。

**使用者明確同意後，才提交正式工作。**

*這份 HTML 確認單是執行前流程，與完成後的分析報告不同。*

<!--
確認前不取得真實行情、不訓練模型、不執行歷史回測；自建策略適用的訊號檢查只用固定合成資料。單次準備的內建策略只檢查設定、執行模式與參數；固定比較的每組候選則都要通過 80 與 240 根合成 K 棒的訊號檢查，包含內建策略。一到八組普通回測的固定候選可共用一份確認單，使用者批准整組後才執行；這不包含需要訓練或 walk-forward 的多組比較，也不會自動擴大搜尋。修改候選或設定後要準備新的計畫並重新確認。後端依賴助手如實轉交使用者決定，不能獨立驗證決定的人類來源。這項限制留在講者稿，操作細節以 README 為準。
來源：README「完成第一次回測」、docs/run_artifacts.md 的 execution plans。
-->

<!-- _footer: "來源：[執行確認流程](https://github.com/Kuan-Lun/tradingdev/blob/35d709d4ac9ed8c9d3a71a3e24cd98996bf3a1e6/README.md#完成第一次回測)" -->

---

## 回測完成後，繼續問問題

> 「哪一段時間損失最多？」
> 「交易成本占了多少？」
> 「上次使用什麼參數？和這次差在哪裡？」

TradingDev 提供診斷與分析指引，讓 AI 依保存的證據說明
**觀察依據、假說與限制，以及下一步研究建議**。

*查詢保存的結果不必重新回測。研究建議仍須驗證，新實驗需確認。*

<!--
結果主要供 LLM 查詢和分析。詳細交易、帳戶序列與執行紀錄可分頁查詢；不存在的舊紀錄不能靠助手猜測重建。
SERVER_INSTRUCTIONS 要求先讀 get_run_diagnostics，再以 summary、observations、hypotheses_and_limits、next_steps 及精確的 run/scope 引用提交分析，保留於 HTML。generate_report 的工具說明與 ReportAnalysis schema 也公開這個結構。後端提供規則式診斷並核對引用，文字推理由客戶端模型產生，未提供時會標示；不是後端自行呼叫另一個 LLM。涉及特定虧損期間時須先查交易或權益資料，不能只憑摘要判定原因。
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
strategy/revision、plan、job、run 的用途不同。job 用於進度與取消，run 用於已保存結果；現有 ID 是追蹤基礎，不等於已綁定使用者擁有權。固定比較以同一份行情快照獨立執行各組普通回測，使用各自資金與策略實例，不合併成一個投資組合。完整比較報告依父工作選取所有候選，包含失敗、取消及未執行的原因；成功結果可點選查看明細。
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

## 對話解讀與報告明細

日常研究可以在對話中完成查詢、解讀與修正。

完成回測與績效解讀時，預設提供 HTML 報告連結。
需要細看時，可開啟指標、圖表、交易與帳戶紀錄。

也可以明確要求：「只要摘要，不需要報告」。

**計算結果與助手的文字解讀分開呈現。**

<!-- _footer: "來源：[報告交付方式](https://github.com/Kuan-Lun/tradingdev/blob/7298a4d8cf0769c0db7f92f0df46e2e739c8e470/README.md#交易紀錄與-html-報告)" -->

<!--
對話仍是查詢、解讀與修正策略的主要入口，使用者不必閱讀圖表。完成回測並解讀績效、風險時，助手預設交付標準報告連結、對應結果及查看明細的方法；使用者可明確選擇只要摘要。執行後報告與執行前確認單分開。保存紀錄不等於保存完整軟體環境，不能保證跨環境完全重現。
來源：https://github.com/Kuan-Lun/tradingdev/blob/7298a4d8cf0769c0db7f92f0df46e2e739c8e470/README.md#交易紀錄與-html-報告 及同版本 docs/run_artifacts.md。
-->

---

<!-- header: "03　運算與使用自由度" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 03　運算與使用自由度

AI 助手、運算主機與研究流程的分工。

---

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
| TradingDev 研究後端 | 提供研究指引與診斷，驗證策略及執行實驗 |
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

<!-- header: "04　產品詳細比較" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 04　產品詳細比較

**基礎計算工具**提供運算能力。

**AI／MCP 研究工具**讓助手操作研究。

**完整平台**進一步整合資料、團隊與交易流程。

*逐一比較能力、工作方式，以及適合採用的情境。*

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

TradingDev 以成熟計算套件支援策略實驗，後續研發聚焦研究決策、記憶與自動改進。

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
| 團隊要維護什麼？ | 選用平台服務或自管方案 | 研究後端與自訂驗證規則 |
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

## TradingDev 的研發定位

> 建立能累積研究經驗，
> 並由 AI 持續改進的量化研究後端。

以可追問的策略實驗為基礎，逐步驗證 AI 研究決策、跨研究記憶與後端自動迭代。

*目前已具備實驗與紀錄基礎，自動改進流程仍待開發。*

<!--
現有工具提供可採用的研究能力，也提供評估設計的參考。本專案以研究決策、經驗累積與後端改進作為持續研發方向，成效須由具體研究案例與修改前後的證據驗證。
既有能力與未來目標分開說明，不把記憶、Agents 或可自架說成市場獨有功能，也不以採用的工具數量代表研究價值。
-->

---

<!-- header: "05　經驗如何回饋專案" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 05　經驗如何回饋專案

讓個別研究中的問題，成為下一次開發可用的能力。

---

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

<!-- header: "06　後續開發方向" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 06　後續開發方向

以現有成果為基礎，安排下一階段的開發與驗證。

---

<!-- _class: compact -->

## 目前提供與後續擴充

| 已提供 | 後續方向 |
| --- | --- |
| 研究流程指引、策略開發與驗證 | 更完整的研究引導 |
| 規則式診斷、AI 解讀與研究建議 | 市場情境、波動敏感度與跨實驗引導 |
| 研究筆記、報告引用、續讀與匯出 | 跨研究自動檢索與假說追蹤 |
| 背景執行、參數研究、歷史結果查詢 | 多策略組合與資金配置 |
| 策略及實驗編號、共用檢查 | 多人權限、策略隔離與自動工程回饋 |

*後續項目尚未完成，也沒有在此承諾開發時程。*

<!--
現有身分 ID 是未來權限設計可用的基礎，尚未完成使用者隔離。此頁集中區分已提供的產品能力與未來方向。
規則式診斷、外部 AI 解讀及研究筆記，已能支持研究判斷與成果續用。右欄進一步擴充分析深度、自動檢索與工程回饋，不把現有紀錄與解讀能力一併列為未來功能。AI 解讀由外部助手提供，未提供時報告會標示，建議不會自動啟動新實驗。
現況來源：README「在對話中查詢」「保存研究筆記與打包分享」，ARCHITECTURE.md「績效計算與對話摘要」「研究筆記與可攜封存」。
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

**目標**：深化研究狀態與失敗原因的分析，協助選擇下一個實驗。

已提供研究流程指引、規則式診斷與 LLM 解讀。

診斷元件可以替換，報告保留依據與未驗證事項。

*市場情境、波動敏感度與跨實驗引導仍待深度開發。*

<!-- _footer: "參考：[QC Agents 與 diagnostics](https://www.quantconnect.com/docs/v2/ai-assistance/agents)、[backtrader-mcp](https://github.com/cloudQuant/backtrader-mcp/tree/ad6312a11c2fe7ff9a4c3fcfcfab5037a951b58c)" -->

<!--
初期 adapter 移植 Maverick 的六項適用門檻規則，保留固定來源版本與 MIT 授權，沒有安裝整個上游 server。
DiagnosticProvider 接收原生績效與觀察值；替換入口及證據模型擴充方式見 ARCHITECTURE.md「診斷擴充與替換」。目前沒有完成市場情境或波動敏感度的因果診斷，研究建議仍待驗證，也不自動授權下一個實驗。
-->

---

## TODO-02 可引用證據的研究記憶

**目標**：讓下一次對話知道已經試過什麼，以及為何保留或放棄某個假說。

PoC 已可保存需求與階段摘要，引用報告並匯出離線閱讀包。

追加想法維護在同一本筆記，已匯出的版本獨立保留。

*跨研究的自動檢索與假說追蹤仍待開發，目前可依筆記代號續讀。*

<!-- _footer: "參考：[QC Custom Memories](https://www.quantconnect.com/docs/v2/ai-assistance/agents#Custom-Memories)" -->

<!--
外部 LLM 整理可編輯的研究摘要，後端提供版本檢查、保存與封存。ZIP 包含 HTML、JSON 與所引用報告，任何階段都可產生，沒有逐字聊天攔截器。
這只完成可保存、續讀與分享的 PoC；跨研究搜尋與自動追蹤假說尚未實作。研究核心不綁定對話模型供應商，參考方向仍可包含 QuantConnect project memory。
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

<!-- _class: closing -->

## TradingDev 的方向

今天，讓自己的 AI 助手完成**可追問的策略實驗**。

接下來，讓 AI 從研究經驗提出與驗證後端改進，
形成持續累積的共用研究能力。

*後續附有參考來源。演講後可繼續在線上閱讀。*

<!--
結尾回到個人的研究構想、可驗證的策略實驗，以及 AI 如何將研究經驗轉成後端改進。後者是持續研發方向，仍須經過測試與審查，沒有在此宣稱已完成自動迭代或測得績效優勢。
-->

---

<!-- header: "附錄　參考資料" -->
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
