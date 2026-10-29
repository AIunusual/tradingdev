# GLFT + ML Direction Prediction Strategy

## 策略概述

結合 GLFT（Gueant-Lehalle-Fernandez-Tapia）解析型做市模型與 FLAML AutoML
方向預測器。ML 模型預測短期（5-15 分鐘）方向，用於提前掛 limit order，
以 maker fee（0.02%/side）取代 taker fee（0.06%/side）。

與純 GLFT 策略的關鍵差異：

- ML 方向過濾：只在 ML 預測方向一致時開倉
- Maker fee：假設使用 limit order（fee_rate=0.0002）
- 更低的 entry edge 門檻（因手續費更低）

## 信號邏輯

以下 ML 方向條件適用於已完成訓練的模型；未訓練時保留純 GLFT 訊號路徑，
不使用 ML 方向過濾。

1. **ML 方向預測**：FLAML AutoML 在時間預算內搜尋 learner 與超參數，預測 N 分鐘後的價格方向
   - ML=1（看漲）：只允許做多
   - ML=-1（看跌）：只允許做空
   - ML=0（信心不足）：不開倉

2. **GLFT 開倉條件**（與 ML 方向一致時）：
   - 價格偏離 EMA 超過 `min_entry_edge`
   - 偏離方向符合 ML 預測方向
   - 公允價格 EMA 暖機期內不開倉

3. **平倉條件**（與純 GLFT 相同）：
   - Profit target: 偏離回歸 `profit_target_ratio` 比例
   - Strategy stop-loss: 偏離擴大超過 `strategy_sl`
   - Max holding: 超過 `max_holding_bars` 強制平倉

參數搜尋與正式訊號生成都保留完整行情時序。特徵暖機或缺值造成無法預測時，
該根 K 棒不開新倉；既有持倉仍逐根累計持有時間並檢查出場。特徵過濾只決定
哪些列能提供預測，不會從交易評估中刪除 K 棒。監督式訓練仍只使用特徵與
未來標籤皆完整的列；訓練資料上的搜尋績效不代表樣本外表現。
波動率缺值也會阻止新進場，即使已有 ML 方向預測；不將未知波動率當成極低波動。

## 參數說明

### ML 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| prediction_horizon | 5 | ML 預測方向的未來 N 分鐘 |
| feature_lookback | 60 | 特徵工程回看窗口（bars） |
| ml_time_limit | 300 | AutoML 搜尋時間預算（秒） |
| ml_estimator_list | ["lgbm", "xgboost", "catboost", "rf", "extra_tree"] | FLAML 搜尋的 learner 清單 |
| ml_metric | "accuracy" | AutoML 搜尋的評分指標 |
| confidence_threshold | 0.55 | ML 預測信心門檻 |

### GLFT 核心參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| gamma | 0.0 | 風險厭惡係數（%-space） |
| kappa | 1000.0 | 訂單到達強度（%-space） |
| ema_window | 15 | 公允價格 EMA 窗口；本值及候選值均為整數 2～100000 |

### 進出場參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| min_entry_edge | 0.0012 | 最低進場偏離（>= fee_rate * 2） |
| profit_target_ratio | 0.75 | 獲利目標（entry deviation 的比例） |
| strategy_sl | 0.003 | 策略止損（超過 entry 偏離的額外容忍） |
| min_holding_bars | 5 | 最小持倉 bars（交易所合規） |
| max_holding_bars | 13 | 最大持倉 bars |

### 波動度與費率

| 參數 | 預設值 | 說明 |
|------|--------|------|
| vol_type | "implied" | 波動度估計（realized/parkinson/implied） |
| fee_rate | 0.0002 | Maker fee（limit order） |
| position_size | 3000.0 | 每筆交易大小 |

DVOL 檔案路徑由 `data.requirements.features` 宣告。策略參數只控制
`vol_type` 與交易邏輯，不保存 feature 資料路徑。

## 特徵工程

`DirectionFeatureEngineer` 產生約 50 個特徵：

- **收益率特徵**：1/5/15/30/60 分鐘 log return 及其 lag
- **滾動統計**：均值、標準差、偏度、峰度
- **EMA 偏離**：多個窗口的 close/EMA 偏離比
- **波動度**：rolling std、Parkinson estimator、DVOL
- **成交量**：volume ratio、rolling volume z-score
- **時間特徵**：hour/day-of-week cyclical encoding
- **技術指標**：RSI、MACD 柱狀圖、Bollinger %B、ADX

## YAML 配置範例

以下僅為設定節錄；執行時請以[完整 bundled 設定](../../src/tradingdev/domain/strategies/bundled/glft_ml_strategy/config.yaml)
為起點，依所選模式保留必要設定，並確認日期與資料路徑。
訓練需求由內建註冊策略決定，修改執行 YAML 不會改變支援模式；詳見
[訓練前置條件](../strategy_contract.md#training-before-scoring)。

```yaml
strategy:
  id: "glft_ml"
  version: "1.0.0"
  class_name: "GLFTMLStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/glft_ml_strategy/strategy.py"
  parameters:
    prediction_horizon: 5
    feature_lookback: 60
    ml_time_limit: 300
    ml_estimator_list: ["lgbm", "xgboost", "catboost", "rf", "extra_tree"]
    ml_metric: "accuracy"
    confidence_threshold: 0.55
    gamma: 0.0
    kappa: 1000.0
    ema_window: 15
    vol_type: "implied"
    min_holding_bars: 5
    max_holding_bars: 13
    min_entry_edge: 0.0012
    profit_target_ratio: 0.75
    strategy_sl: 0.003
    position_size: 3000.0
    fee_rate: 0.0002

data:
  source: "binance_vision"
  market_type: "futures/um"
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1m"
    features:
      - type: "dvol"
        source: "deribit"
        column: "dvol"
        path: "workspace/data/processed/btc_dvol_1m_2024_2025.parquet"
        raw_path: "workspace/data/raw/btc_dvol_1m_2024_2025.csv"
```

## 使用方式

此策略的訓練需求為可選（`optional`）：未訓練仍可使用純 GLFT 訊號。
一般回測與參數最佳化支援此路徑，但不會執行 `fit()`，而且設定不得包含
`validation`；可執行不代表 ML 模型已訓練或策略績效已獲驗證。

在這三種執行模式中，walk-forward 才會依訓練區間執行 `fit()`。隨附設定含
`validation`，請使用以下命令，並先確認訓練／測試日期、行情及 DVOL 資料路徑：

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/glft_ml_strategy/config.yaml --walk-forward
```

若要研究未訓練的純 GLFT 路徑，請另使用不含 `validation` 的設定。訓練需求及
執行模式的契約見
[Training Before Scoring](../strategy_contract.md#training-before-scoring)。

## Walk-Forward 結果（2024 train / 2025 test）

ML 方向預測在 OOS 等同隨機（50% 勝率），顯示 BTC 短期方向在
1-15 分鐘尺度上缺乏可被簡單 ML 模型捕捉的持續性 alpha。

## 與純 GLFT 策略的比較

| 面向 | GLFT | GLFT + ML |
|------|------|-----------|
| 方向決定 | 均值回歸（價格偏離 EMA） | ML 預測 + GLFT 偏離 |
| 手續費 | taker (0.06%/side) | maker (0.02%/side) |
| ML 依賴 | 無 | FLAML AutoML（lgbm / xgboost / catboost / rf / extra_tree） |
| OOS 邊際 | 穩定（train/test 一致） | ML 過擬合風險高 |
| 適用場景 | 低費率環境 | 需要有效 ML alpha |

## 參考文獻

- Gueant, O., Lehalle, C-A., Fernandez-Tapia, J. (2012). *Dealing with the inventory risk: a solution to the market making problem.*
