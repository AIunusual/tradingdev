# GLFT Market-Making Strategy

## 策略概述

基於 **Guéant–Lehalle–Fernandez-Tapia (2013)** 最優做市模型的交易策略。核心思想：根據市場波動率和庫存風險動態調整進出場閾值，在安全時進場交易，並透過利潤目標和止損機制控制風險。

與 Safety-First Volume Strategy 的最大區別是：本策略**不需要 ML 模型**，完全基於解析公式計算，運行速度快且可解釋性強。

適用場景：第三方造市服務，需達成指定交易量（如單邊 12.5M USDT/月），同時控制虧損。

## 理論基礎

### 原始 GLFT 模型

GLFT 模型解決了做市商的庫存風險問題，給出最優報價：

- **Reservation price** (庫存調整公允價格):

  ```text
  r(t, q) = s(t) - q · γ · σ² · τ
  ```

- **Optimal spread** (最優報價差):

  ```text
  δ*(t) = γ · σ² · τ + (2/γ) · ln(1 + γ/κ)
  ```

其中：

- `s(t)` = 中間價 (mid-price)
- `q` = 庫存方向 (+1=多頭, -1=空頭, 0=空倉)
- `γ` = 風險厭惡係數 (越大越保守)
- `σ` = 波動率 (per-bar)
- `τ` = 時間上限 (time horizon)
- `κ` = 訂單到達強度

### Pure %-Space Half-Spread 公式

本實作使用純百分比空間的 GLFT **半價差 (half-spread)** 公式，即原始 optimal spread 的一半（`δ*/2`），因為我們是將單側偏離度 `|deviation|` 與閾值比較：

```text
half_spread = γ · σ² · τ / 2 + (1/γ) · ln(1 + γ/κ)
```

γ 和 κ 均為無量綱的 %-space 參數。當 BTC 價格從 $40K 變動到 $100K 時，spread 只隨波動率 (σ) 變化，不受價格水平影響。

在本實作中，**τ 固定為 `max_holding_bars`**（而非原始論文中隨持倉時間遞減的剩餘期）。原因是本策略只在進場時計算一次 half-spread 作為閾值，持倉期間的出場由獨立的 profit target、stop-loss 和 max holding 規則控制。

### 信號型適配

由於我們的框架是信號驅動（非限價單），適配方式如下：

1. 使用 **EMA** 作為公允價格 `s(t)` 的估計；EMA 以前 N 期 SMA 起算，暖機期內公允價格為 NaN，不進場
2. 計算價格偏離度：`deviation = (close - EMA) / EMA`
3. 計算正規化 half-spread 作為進場閾值
4. 利潤目標和止損控制出場

```text
FLAT ──(|deviation| > half_spread)──► LONG/SHORT
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   min_holding      profit_target /     max_holding
                   未到 → 維持      SL → 平倉           → 強制平倉
```

### 進場邏輯

1. 計算 GLFT half-spread：`glft_hs = γ·σ²·τ/2 + spread_const`
2. 取 `max(glft_hs, min_entry_edge)` 作為進場閾值
3. `deviation < -threshold` → 做多（價格大幅低於公允價值）
4. `deviation > threshold` → 做空（價格大幅高於公允價值）
5. 可選動量防護：要求 deviation 正在收斂（價格回歸 EMA）
6. 可選趨勢過濾：只順勢開倉；慢 EMA 暖機期內沒有趨勢估計，不進場

### 出場邏輯

持倉後按優先級檢查以下出場條件：

| 優先級 | 條件 | 參數 | 說明 |
|--------|------|------|------|
| 0 | 最短持倉 | `min_holding_bars` | 持倉不足門檻 → 強制不平倉（交易所規則） |
| 1 | 最長持倉 | `max_holding_bars` | 持倉滿上限 → 強制平倉 |
| 2 | 策略級止損 | `strategy_sl` | deviation 反向超過入場偏離 → 止損 |
| 3 | 利潤目標 | `profit_target_ratio` | deviation 回歸至目標水平 → 獲利了結 |

利潤目標計算：`target = entry_dev × (1 - profit_target_ratio)`

- ratio=1.0 → target=0（等待完全回歸到 EMA）
- ratio=0.5 → target=entry_dev 的 50%（回歸一半即出場）
- ratio=1.5 → target 超過 EMA（等待 overshoot）

### 關鍵行為

- **高波動 → 寬 spread → 減少交易** (自我調節)
- **低波動 → 窄 spread → 更多交易** (增加流動性)
- **價格水平無關**：純 %-space 公式確保 spread 只隨 σ 變化
- **止損控制尾部風險**：避免持有到 max_hold 的大額虧損
- **利潤目標鎖定收益**：不再過早出場切割利潤

## 參數說明

### Core GLFT (%-space)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma` | 500.0 | 風險厭惡係數 (γ)，越大 spread 越寬 |
| `kappa` | 1000.0 | 訂單到達強度 (κ)，越大 spread 越窄 |
| `ema_window` | 21 | EMA 計算窗口，用於估計公允價格；整數 2～100000 |

### Volatility

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `vol_window` | 30 | 滾動波動率估計窗口（`implied` 時不使用） |
| `vol_type` | "implied" | `"realized"` / `"parkinson"` / `"implied"` |

### Implied Volatility (DVOL)

當 `vol_type: "implied"` 時，策略使用 DataFrame 中的 `dvol` 欄位作為波動度來源，而非從歷史價格估計。DVOL 檔案路徑由 `data.requirements.features` 宣告，策略參數不保存資料路徑。

**DVOL 簡介**：

- Deribit 的 BTC 隱含波動度指數，類似傳統市場的 VIX
- 衡量市場預期的未來 30 天年化波動度（單位：%）
- 由選擇權市場價格反推（forward-looking）
- 公開 API，免 API Key，支援 1 分鐘解析度

**轉換公式**（年化 % → per-bar sigma）：

```text
sigma_per_bar = DVOL / 100 / sqrt(525960)
```

其中 `525960 = 365.25 × 24 × 60`（年分鐘數）。

### Multi-Timeframe

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `signal_agg_minutes` | 1 | 信號聚合時間（分鐘）；1=不聚合，5=5 分鐘 EMA |
| `signal_agg_minutes_candidates` | [1] | Grid search 候選值 |

當 `signal_agg_minutes > 1` 時，EMA 在聚合後的 N 分鐘 K 棒上計算，再映射回 1 分鐘解析度；每根 1 分鐘 bar 只使用已收完的聚合 K 棒，第一根聚合 K 棒收完之前為 NaN，不進場。執行仍在 1 分鐘精度。效果：偏離度更大、信號更穩定。

### Entry

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_entry_edge` | 0.0015 | 最低進場偏離閾值（須 >= `fee_rate × 2`） |
| `momentum_guard` | false | 動量防護：只在 deviation 收斂時進場 |
| `trend_ema_window` | 0 | 趨勢過濾慢 EMA；0=停用，啟用時為整數 2～100000 |

### Exit

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `profit_target_ratio` | 1.0 | 利潤目標比率 (1.0=完全回歸到 EMA) |
| `strategy_sl` | 0.01 | 策略級止損閾值（0=停用） |

### Holding Management

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_holding_bars` | 5 | 最短持倉（5 min @1m）— 符合交易所規則 |
| `max_holding_bars` | 30 | 最長持倉（30 min @1m）— 對應 GLFT 的 horizon T |

### Dynamic Sizing（動態倉位）

啟用 `dynamic_sizing: true` 時，倉位大小根據進場偏離度動態調整——偏離度越大表示均值回歸機會越好，因此分配更大倉位。

```text
weight = |deviation| / edge_for_full_size
weight = clamp(weight, min_position_size / position_size, 1.0)
actual_size = position_size × weight
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `dynamic_sizing` | true | 啟用動態倉位 |
| `min_position_size` | 10000.0 | 最小倉位（USDT） |
| `edge_for_full_size` | 0.005 | 達到全倉的偏離度門檻 |
| `edge_for_full_size_candidates` | [0.003, 0.005, 0.008] | Grid search 候選值 |

目前 bundled config 將 `min_position_size` 與 `position_size` 都設為 10,000 USDT，
因此實際上維持固定名目倉位；若要讓倉位依偏離度縮放，需把
`min_position_size` 調低。

### fit() Grid Search

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma_candidates` | [0, 0.1, 1] | 搜尋最佳 γ |
| `kappa_candidates` | [250, 500, 750] | 搜尋最佳 κ |
| `ema_window_candidates` | [5, 15, 30] | 搜尋最佳 EMA 窗口；各值為整數 2～100000 |
| `max_holding_bars_candidates` | [6, 8, 13] | 搜尋最佳持倉上限 |
| `min_entry_edge_candidates` | [0.0012, 0.0015, 0.002, 0.003] | 搜尋最佳進場門檻 |
| `trend_ema_candidates` | [0, 200] | 搜尋趨勢過濾窗口；各值為 0 或整數 2～100000 |
| `profit_target_ratio_candidates` | [0.5, 0.75, 1.0] | 搜尋最佳利潤目標 |
| `target_metric` | "total_volume" | 優化目標指標 |
| `min_monthly_pnl` | -1500 | 月均 PnL 約束門檻（USDT） |

#### 約束優化

`fit()` 依指標目錄宣告的方向選擇參數：例如 `total_volume` 取最大值，
`max_drawdown_amount` 取最小值。目標值不可用或非有限數值的組合不參與比較。

若 `min_monthly_pnl` 不是 `null`，先排除 `daily_pnl_mean` 不可用、非有限，
或估計月均 PnL（`daily_pnl_mean * 30`）低於門檻的組合；通過約束後才比較目標。
沒有有效且符合約束的組合時，搜尋以 `ValueError` 失敗。
目前 bundled YAML 的目標為 `total_volume`，因此在符合損益約束下最大化交易量。

### Position & Volume

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `position_size` | 10000.0 | 每筆最大持倉大小 |
| `monthly_volume_target` | 12,500,000 | 月度單邊交易量目標 |
| `fee_rate` | 0.0005 | 單邊手續費（taker） |

## 交易所規則合規

**限制**: ≤3 min 交易占比 < 60%, ≤5 min 交易占比 < 75%

**合規機制**:

- `min_holding_bars >= 5`（@1m timeframe = 5 分鐘）
- 狀態機強制執行：持倉期間內不可提前退出
- 較大的 position_size (10,000 USDT) 減少所需交易次數

## 歷史 Walk-Forward 驗證結果

本節保留既有文件中的設定、最佳參數、績效表與分析結論，尚未依現行引擎及
指標口徑重新回測、核對；不可視為新版輸出或目前策略效能的驗證。

目前 volume 模式沒有初始資金基準，`total_return`、`annual_return`、
`sharpe_ratio` 與百分比 `max_drawdown` 均不適用，應以 `total_pnl` 與
`max_drawdown_amount` 等金額指標評估。下方初始資金與報酬百分比僅為歷史紀錄。
新版 `total_volume` 是實際進場與出場名目成交額的合計，不能直接與單邊月目標比較。
本節「單邊」沿用既有文字，未核實其與新版成交量的對應；交易量達標、虧損控制
及泛化判斷都須以新回測重新評估。

### 設定

- **訓練期**: 2024-01-01 ~ 2024-12-31（366 天）
- **測試期**: 2025-01-01 ~ 2025-12-31（365 天）
- **初始資金**: 100,000 USDT
- **優化目標**: `total_volume`（約束月均 PnL >= -1500 USDT）

### 最佳參數（Grid Search 結果）

| 參數 | 搜尋範圍 | 最佳值 |
|------|----------|--------|
| `gamma` | [0, 0.1, 1] | **0.0** |
| `kappa` | [250, 500, 750] | **750.0** |
| `ema_window` | [5, 15, 30] | **15** |
| `max_holding_bars` | [6, 8, 13] | **13** |
| `min_entry_edge` | [0.0012, 0.0015, 0.002, 0.003] | **0.0012** |
| `trend_ema_window` | [0, 200] | **0**（停用） |
| `profit_target_ratio` | [0.5, 0.75, 1.0] | **1.0** |
| `edge_for_full_size` | [0.003, 0.005, 0.008] | **0.008** |

**參數解讀**：

- `gamma=0` → GLFT half-spread 退化為 `1/κ = 1/750 ≈ 0.13%`，但被 `min_entry_edge=0.12%` 取代，實際由 min_entry_edge 主導進場
- `ema_window=15` → 15 分鐘 EMA 作為公允價格，較快響應價格變動
- `max_holding_bars=13` → 最多持倉 13 分鐘，配合 5 分鐘最短持倉，實際持倉 5~13 分鐘
- `profit_target_ratio=1.0` → 等待價格完全回歸 EMA 才出場
- `edge_for_full_size=0.008` → 偏離 0.8% 才用全倉，多數交易使用較小倉位

### 績效指標

| 指標 | 訓練期 (2024) | 測試期 (2025) |
|------|---------------|---------------|
| **Total Return** | -18.04% | -12.37% |
| **Total PnL (USDT)** | -18,036 | -12,372 |
| **Annual Return** | -17.99% | -12.37% |
| **Sharpe Ratio** | -25.48 | -24.17 |
| **Max Drawdown** | 18.04% | 12.37% |
| **Win Rate** | 52.46% | 50.74% |
| **Profit Factor** | 0.6071 | 0.6142 |
| **Total Trades** | 19,831 | 14,978 |
| **Total Volume (USDT)** | +29,992,151 | +21,799,951 |

### Daily PnL 統計

| 統計量 | 訓練期 (2024) | 測試期 (2025) |
|--------|---------------|---------------|
| Mean | -49.28 | -33.90 |
| Std | 56.41 | 34.42 |
| Min | -760.17 | -278.16 |
| Max | +37.82 | +0.70 |
| Median | -36.42 | -25.00 |

### 結果分析

**交易量達標**：

- 訓練期月均交易量：29,992,151 / 12 ≈ **2,499,346 USDT/月**（舊文標示：單邊）
- 測試期月均交易量：21,799,951 / 12 ≈ **1,816,663 USDT/月**（舊文標示：單邊）
- 距離 12.5M USDT/月目標仍有差距，需要更低的進場門檻或更大倉位

**虧損控制**：

- 訓練期年化虧損 -17.99%，恰好在 -18% 約束邊界
- 測試期年化虧損 -12.37%，較訓練期改善（OOS 泛化尚可）
- Win Rate ~50-52%，但 Profit Factor < 1，表示虧損交易的平均虧損大於獲利交易的平均獲利

**泛化性**：

- 測試期虧損較訓練期少（-12.37% vs -18.04%），非過擬合表現
- 測試期交易量較訓練期少 27%，可能是 2025 年市場波動結構不同所致
- Daily PnL Max 在測試期幾乎為零（+0.70），說明 2025 年幾乎沒有單日正收益

## YAML 配置範例

以下僅為設定節錄；執行時請以[完整 bundled 設定](../../src/tradingdev/domain/strategies/bundled/glft_strategy/config.yaml)
為起點，依所選模式保留必要設定，並確認日期與資料路徑。
訓練需求由內建註冊策略決定，修改執行 YAML 不會改變支援模式；詳見
[訓練前置條件](../strategy_contract.md#training-before-scoring)。

```yaml
strategy:
  id: "glft_market_making"
  version: "1.0.0"
  class_name: "GLFTStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/glft_strategy/strategy.py"
  parameters:
    # Core GLFT (%-space)
    gamma: 500
    kappa: 1000
    ema_window: 21

    # Volatility
    vol_window: 30
    vol_type: "implied"

    # Holding management
    min_holding_bars: 5
    max_holding_bars: 30

    # Grid search candidates
    gamma_candidates: [0, 0.1, 1]
    kappa_candidates: [250, 500, 750]
    ema_window_candidates: [5, 15, 30]
    max_holding_bars_candidates: [6, 8, 13]
    target_metric: "total_volume"

    # Entry
    min_entry_edge: 0.0015
    min_entry_edge_candidates: [0.0012, 0.0015, 0.002, 0.003]
    trend_ema_window: 0
    trend_ema_candidates: [0, 200]
    momentum_guard: false

    # Exit
    profit_target_ratio: 1.0
    profit_target_ratio_candidates: [0.5, 0.75, 1.0]
    strategy_sl: 0.01

    # Multi-timeframe
    signal_agg_minutes: 1
    signal_agg_minutes_candidates: [1]

    # Position & volume
    position_size: 10000.0
    monthly_volume_target: 12500000
    fee_rate: 0.0005

    # Dynamic sizing
    dynamic_sizing: true
    min_position_size: 10000.0
    edge_for_full_size: 0.005
    edge_for_full_size_candidates: [0.003, 0.005, 0.008]

    # Constrained optimization
    min_monthly_pnl: -1500

validation:
  train_start: "2024-01-01"
  train_end: "2024-12-31"
  test_start: "2025-01-01"
  test_end: "2025-12-31"

backtest:
  symbol: "BTC/USDT"
  timeframe: "1m"
  fees: 0.0005
  signal_as_position: true
  re_entry_after_sl: false
  stop_loss: 0.01
  mode: "volume"

data:
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

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/glft_strategy/config.yaml --walk-forward
```

## 與其他策略的比較

| | Safety-First Volume | GLFT Market Making |
|---|---|---|
| 進場機制 | XGBoost P(safe) | 解析 spread 閾值 |
| 方向決定 | SMA 交叉 / ML | 價格偏離 EMA 方向 |
| ML 依賴 | 重度 (XGBoost) | 無 |
| 出場機制 | 風險分數下降 | 利潤目標 + 止損 + 最大持倉 |
| 參數優化 | Lookback + threshold 搜尋 (AUC-ROC) | Grid search (γ, κ, EMA, pt_ratio) via backtest |
| 滾動重訓 | 每 N bars 重訓 XGBoost | 不需要 (解析模型) |
| 計算成本 | 高 (ML 訓練/推理) | 低 (EMA + std + for-loop) |
| 理論基礎 | 風險分類 | 最優做市 (GLFT) |
| 波動適應 | 作為特徵輸入 ML | 直接影響 spread 公式 |
| 價格獨立 | 否（需 ML 重訓） | 是（純 %-space 公式） |

## 參考文獻

- Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). *Dealing with the Inventory Risk: A Solution to the Market Making Problem.* Mathematics and Financial Economics, 7(4), 477-507.
