# 策略文件索引

本目錄包含 bundled strategies 的詳細說明文件。對應程式碼與 config 位於
`src/tradingdev/domain/strategies/bundled/<strategy>/`。

| 策略 | 檔案 | 說明 |
|------|------|------|
| KD Stochastic 交叉 | [kd_strategy.md](kd_strategy.md) | KD 隨機指標金叉/死叉 + 超買超賣過濾 |
| Safety-First Volume | [safety_volume_strategy.md](safety_volume_strategy.md) | 風險門控 + SMA 方向，最小虧損達成交易量目標 |
| GLFT Market Making | [glft_strategy.md](glft_strategy.md) | GLFT 最優做市模型，解析型 spread-based 進出場（支援 DVOL implied vol） |
| GLFT + ML Direction | [glft_ml_strategy.md](glft_ml_strategy.md) | GLFT 做市 + FLAML AutoML 方向預測，limit order market making |
| XGBoost Direction | [xgboost_strategy.md](xgboost_strategy.md) | XGBoost 方向預測 + rolling retrain，volume mode 交易量策略 |
| Quantile Regime Volume | [quantile_strategy.md](quantile_strategy.md) | XGBoost regime classifier + DVOL/funding rate feature requirements |

Quantile、XGBoost Direction 與 Safety-First Volume 需要先訓練，請使用含有效
`validation` 設定的 walk-forward；一般回測與參數最佳化不支援這三個策略。
KD、GLFT 與 GLFT ML 的訓練則為可選，各模式仍須符合設定與資料需求。
能力說明不代表模型已訓練或績效已驗證；完整限制與宣告契約見
[訓練前置條件](../strategy_contract.md#training-before-scoring)。

Generated strategies 不放在此目錄；它們由 MCP
lifecycle 管理，每次保存都在
`workspace/generated_strategies/<strategy_id>/revisions/<revision_id>/` 建立
`strategy.py`、`config.yaml` 與 `metadata.json`。每個 revision 分別保存驗證證據與
`draft → validated → runnable → promoted` 狀態，新版本不繼承舊版本的執行資格。
`workspace/configs/<id>.yaml` 僅為舊格式策略的恢復讀取位置；舊檔須明確另存並重新驗證。

## 內建策略的參數搜尋

KD、GLFT、GLFT-ML 與 Quantile 的 `fit()` 使用指標目錄定義的優化方向。
`target_metric` 必須是已登錄且可優化的指標；例如 `total_pnl`、`total_volume`
取最大值，`max_drawdown_amount` 取最小值。不可用（`null`）與非有限數值
不當作零，也不參與排名；沒有有效候選值時以 `ValueError` 失敗。
GLFT、GLFT-ML 與 Quantile 設定的月均損益約束仍另外套用。

目標須適用於回測模式：volume mode 沒有本金基準，報酬率與 Sharpe 等指標
不可用，可選 `total_pnl`、`total_volume` 或 `max_drawdown_amount`。
年化指標還需要有效的 `backtest.periods_per_year`、時間戳及日級觀測；
`periods_per_year` 表示每年觀測日數，不是每年 K 棒數。K 棒頻率未知或大於
一天時，年化及日／月損益統計不可用；若目標或啟用的月均損益約束因此
無有效值，搜尋便無法選出參數。
