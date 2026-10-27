# KD Stochastic Oscillator 交叉策略

## 策略概述

本策略使用 **KD 隨機指標（Stochastic Oscillator）** 的 %K 與 %D 線交叉，結合超買超賣區域過濾，產生 BTC/USDT 永續合約的多空交易信號。

## 指標數學定義

### %K（快線）

```text
%K = SMA( (Close - Lowest Low(N)) / (Highest High(N) - Lowest Low(N)) × 100, smooth_K )
```

- `N`：回看期間（預設 14 根 K 線）
- `smooth_K`：%K 的平滑期數（預設 3）

### %D（慢線）

```text
%D = SMA(%K, D_period)
```

- `D_period`：%D 的平滑期數（預設 3）

兩條線的值域都在 0 ~ 100 之間。

## 信號邏輯

### 做多信號（signal = 1）

同時滿足以下條件：

1. **金叉**：%K 由下方向上穿越 %D（即當前 %K > %D 且前一根 %K ≤ %D）
2. **超賣區域**：交叉發生時 %K < oversold（預設 20）

直覺：價格在低位出現上升動能反轉。

### 做空信號（signal = -1）

同時滿足以下條件：

1. **死叉**：%K 由上方向下穿越 %D（即當前 %K < %D 且前一根 %K ≥ %D）
2. **超買區域**：交叉發生時 %K > overbought（預設 80）

直覺：價格在高位出現下降動能反轉。

### 無信號（signal = 0）

不滿足以上任一條件時，維持無信號狀態。

## 出場邏輯

本策略為 **反向信號出場**：

- 持有多單時，收到做空信號即平多並開空
- 持有空單時，收到做多信號即平空並開多
- 回測引擎會自動處理 signal 的狀態轉換

## 前視偏差防範

- 策略信號在 bar `i` 收盤時計算，使用的是直到 bar `i`（含）的歷史資料
- 回測引擎將信號做 `shift(1)` 處理，確保在 bar `i+1` 的開盤價才執行交易
- 這意味著實際進場價格是產生信號的下一根 K 線的開盤價

## 預設參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `k_period` | 14 | %K 計算的回看期間 |
| `d_period` | 3 | %D 的 SMA 平滑期數 |
| `smooth_k` | 3 | %K 的 SMA 平滑期數 |
| `overbought` | 80.0 | 超買門檻 |
| `oversold` | 20.0 | 超賣門檻 |

所有參數定義於 `src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml`，禁止寫死在程式碼中。

## Config Schema

以下僅節錄策略與資料需求；執行時請以[完整 bundled 設定](../../src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml)
為起點，依所選模式保留必要設定，並確認日期與資料路徑。
訓練需求由內建註冊策略決定，修改執行 YAML 不會改變支援模式；詳見
[訓練前置條件](../strategy_contract.md#training-before-scoring)。

```yaml
strategy:
  id: "kd_crossover"
  class_name: "KDStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/kd_strategy/strategy.py"
  parameters:
    k_period: 14
    d_period: 3
    smooth_k: 3
    overbought: 80.0
    oversold: 20.0

data:
  source: "binance_vision"
  market_type: "futures/um"
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
```

## 已知限制

1. **信號稀少**：要求交叉必須發生在超買/超賣區域，在趨勢行情中可能長時間無信號
2. **不適合強趨勢**：KD 作為震盪指標，在單邊趨勢中容易產生逆勢信號
3. **參數敏感**：超買超賣門檻的設定對信號頻率影響很大
4. **未考慮資金費率**：當前版本未納入永續合約的資金費率成本

## 執行方式

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
```

## 相關檔案

- 策略實作：`src/tradingdev/domain/strategies/bundled/kd_strategy/strategy.py`
- KD 指標：`src/tradingdev/domain/indicators/kd.py`
- 策略配置：`src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml`
- 策略 config 模型：`src/tradingdev/domain/strategies/bundled/kd_strategy/config.py`（`KDStrategyConfig`）
- 資料需求模型：`src/tradingdev/domain/data/requirements.py`（`DataRequirement`）
