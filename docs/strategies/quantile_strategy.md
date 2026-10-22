# Quantile Regime Volume Strategy

Source:
`src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py`

Config:
[Bundled YAML](../../src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml)

## Config Schema

```yaml
strategy:
  id: "quantile_volume"
  class_name: "QuantileStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py"
  parameters:
    horizon: 30
    horizon_candidates: [15, 30]
    min_entry_edge_candidates: [0.25, 0.30, 0.35, 0.40]
```

## Intent

This bundled strategy trains an XGBoost regime classifier. It classifies each
bar into `long_only`, `short_only`, `both`, or `neither`, then trades only when
one direction has a clear edge.

## Signal Logic

- `fit()` trains regime classifiers for candidate horizons and searches entry
  edge plus sizing parameters.
- `generate_signals()` requires a completed `fit()`. Without a trained model it
  raises `RuntimeError` rather than training on the frame it is about to score:
  training on the evaluation segment would let earlier signals use later data.
  A `fit()` call without a backtest engine skips training, so it does not
  satisfy this requirement.
- The state machine skips regimes where `P(both)` or `P(neither)` is too high.
- Dynamic sizing scales exposure by directional edge when enabled.
- Signals follow the project convention: `1` long, `-1` short, `0` flat.

## Running

A simple backtest cannot train this strategy. `generate_signals()` has no
training pass of its own, so a config without a `validation` block fails with
`RuntimeError` instead of reporting metrics; it does not silently fall back to
training on the frame it is scoring.

Use walk-forward validation to train on each fold's training window before
scoring its test window. Prepare the market and feature data described below,
and check the dates and paths in the linked bundled YAML before running:

```bash
uv run python -m tradingdev \
  --config src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml \
  --walk-forward
```

The split settings (`train_start`, `test_start`, `purged_split`,
`label_horizon`, ...) are documented in
[Strategy Walk-Forward Settings](../strategy_contract.md#strategy-walk-forward-settings).
A config that declares `validation` cannot be run as a simple backtest, and
`--walk-forward` against a config without it fails with `Config has no
validation section for walk-forward.`

For MCP, use `prepare_walk_forward`. Simple backtests and MCP optimization do
not support this strategy's training requirement; removing `validation` does
not make an untrained strategy executable. A rejected preparation returns a
failure message to the caller without starting a formal job. See the
[training contract](../strategy_contract.md#training-before-scoring) for the
failure semantics.

## Data Requirements

The config declares OHLCV plus two feature sources:

```yaml
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
      - type: "funding_rate"
        source: "binance"
        column: "funding_rate"
        path: "workspace/data/processed/btc_funding_rate_2025.parquet"
```

Feature paths are declared only in `data.requirements`, not in
`strategy.parameters`. `DataService.inspect_dataset(config_path)` reports
whether each feature path exists, how many rows it has, and how many missing
values are present in the declared column.

## Lifecycle

This is a bundled strategy under
`src/tradingdev/domain/strategies/bundled/quantile_strategy/`, so it is
git-versioned and treated as promoted by `StrategyService`. Generated strategy
states (`draft`, `validated`, `runnable`, `promoted`) apply only to runtime
artifacts under `workspace/generated_strategies/`.

## Key Parameters

- `horizon_candidates`: prediction horizons explored in `fit()`.
- `min_entry_edge_candidates`: directional edge thresholds.
- `edge_for_full_size_candidates`: dynamic sizing scale candidates.
- `retrain_interval` and `train_window`: rolling retrain cadence and context.
- `target_metric` and `min_monthly_pnl`: optimization objective and guardrail.
