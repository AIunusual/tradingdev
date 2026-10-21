# Safety-First Volume Strategy

## 策略概述

以**最小虧損**滿足**月度交易量目標**的策略。核心思想：只在安全的市場環境下進場交易，避開高風險時段。

適用場景：第三方造市服務，需達成指定交易量（如單邊 12.5M USDT/月），同時控制虧損。

## 三層架構

### Layer 1: Risk Gate (XGBoost)

判斷「現在進場是否安全」。

- **模型**: XGBoost 二元分類
- **Target 定義**:
  - safe=1: 未來 `target_holding_bars` 根 bar 內，**最佳方向**的 round-trip P&L > `-max_acceptable_loss_pct`
  - safe=0: 即使選對方向也會虧損超過門檻
- **決策**: `P(safe) >= risk_threshold` → 允許進場
- **特徵**: 波動率（多 horizon）、ATR、布林帶寬度、ADX、成交量異常、K 線結構、收益率偏度/峰度

### Layer 2: Direction (SMA Crossover)

Risk Gate 放行後決定做多或做空。

- **預設**: SMA 快線 > 慢線 → 做多，反之做空
- **可選**: 切換為 ML 方向模型（`use_ml_direction: true`）

ML 方向模型只對特徵齊備的 bar 產生預測，依原始列位置回填，與是否提供 `timestamp`
欄位無關；使用 DatetimeIndex 或一般索引也維持相同對齊。暖機段及中途因特徵缺值
被移除的 bar（例如平價區間造成技術指標無法計算），方向固定為 0；全部特徵皆缺值
時也回傳全 0，不呼叫方向模型。這與 SMA 路徑的 NaN 區間一致，FLAT 狀態下
`direction == 0` 不會進場；已持倉時仍依既有持倉規則處理。有效預測只用於產生它的
那根 bar，不會因中間缺列移動其他預測。若模型回傳的列數或來源索引與特徵列不一致，
執行會以 `RuntimeError` 停止，不會截短或猜測對齊。

### Layer 3: Holding Manager (狀態機)

管理持倉生命週期，確保符合交易所規則。

```text
FLAT ──(safe + direction)──► LONG/SHORT
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              min_holding   risk < thr    max_holding
              未到 → 維持   → 平倉        → 平倉
                    │
                    ▼
            direction 反轉
            → 反轉持倉 (2× volume)
```

## 參數說明

### Risk Gate

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `risk_model` | (see YAML) | XGBoost 超參數 |
| `risk_threshold` | 0.15 | P(safe) 需超過此值才允許進場 |
| `risk_threshold_candidates` | [0.05..0.25] | fit() 時自動搜尋最佳 threshold |
| `target_holding_bars` | 7 | 計算 safe target 時的假設持倉期 |
| `max_acceptable_loss_pct` | 0.0 | 不接受淨虧損；費後無法打平即標記為 unsafe |
| `fee_rate` | 0.0011 | 單邊手續費 + 滑點 |

執行 threshold 搜尋時，`fit()` 以驗證資料回測候選值，選擇扣除回測手續費
與滑點後 `total_pnl` 最高且為有限數值的 `risk_threshold`；若有效結果皆虧損，
選擇虧損最小者。不可用與非有限數值不參與比較；全部無有效結果時，保留
設定的 `risk_threshold`。此搜尋不以 `monthly_volume_target` 作為硬性篩選條件。

### Direction

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_ml_direction` | false | true: 使用 XGBoost 方向模型 |
| `sma_fast` | 5 | SMA 快線週期；整數 2～100000 |
| `sma_slow` | 20 | SMA 慢線週期；整數 2～100000 |

### Holding Management

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_holding_bars` | 5 | 最短持倉（5 min @1m）— 符合交易所規則 |
| `max_holding_bars` | 30 | 最長持倉（30 min @1m） |

### Training

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `lookback_candidates` | [360, 720, 1440] | 搜尋最佳特徵 lookback |
| `retrain_interval` | 720 | 每 720 根 bar（12h @1m）重訓模型 |
| `validation_ratio` | 0.2 | 驗證集比例 |

### Position & Volume

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `position_size` | 3000.0 | 每筆持倉大小 |
| `monthly_volume_target` | 12500000 | 月度單邊交易量目標 |

## 交易所規則合規

**限制**: ≤3 min 交易占比 < 60%, ≤5 min 交易占比 < 75%

**合規機制**:

- `min_holding_bars >= 5`（@1m timeframe = 5 分鐘）
- 狀態機強制執行：持倉期間內不可提前退出（僅緊急 SL 例外）
- 較大的 position_size (3000 USDT) 減少所需交易次數

## YAML 配置範例

```yaml
strategy:
  id: "safety_first_volume"
  version: "1.0.0"
  class_name: "SafetyVolumeStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/safety_volume_strategy/strategy.py"
  parameters:
    risk_model:
      n_estimators: 300
      max_depth: 4
      learning_rate: 0.05
    risk_threshold: 0.15
    risk_threshold_candidates: [0.05, 0.10, 0.15, 0.20, 0.25]
    target_holding_bars: 7
    max_acceptable_loss_pct: 0.0
    fee_rate: 0.0011
    use_ml_direction: false
    sma_fast: 5
    sma_slow: 20
    min_holding_bars: 5
    max_holding_bars: 30
    lookback_candidates: [360, 720, 1440]
    retrain_interval: 720
    position_size: 3000.0
    monthly_volume_target: 12500000

backtest:
  # ... (see src/tradingdev/domain/strategies/bundled/safety_volume_strategy/config.yaml)
  signal_as_position: true    # signal=0 → 平倉
  re_entry_after_sl: false    # SL 後不自動重入場
  stop_loss: 0.03             # 3% 緊急 SL

data:
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1m"
    features: []
```

## 使用方式

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/safety_volume_strategy/config.yaml --walk-forward
```

## 與 XGBoost Direction Strategy 的比較

| | XGBoost Direction | Safety-First Volume |
|---|---|---|
| 目標 | 預測方向 | 預測安全性 |
| 進場頻率 | 每根 bar | 安全時才進場 |
| 持倉大小 | 200 USDT | 3,000 USDT |
| 持倉時間 | < 1 min | 5-30 min |
| SL/TP | 緊密（0.4-0.5%） | 寬鬆（3% 緊急） |
| signal=0 | 不操作 | 平倉 |
| 交易所合規 | 不符合 | 符合 |
