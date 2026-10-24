# XGBoost Direction Strategy

Source:
`src/tradingdev/domain/strategies/bundled/xgboost_strategy/strategy.py`

Config:
`src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml`

## Config Schema

This is a partial snippet. Start from the [complete bundled
config](../../src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml),
including its `backtest` and `validation` settings, and check dates and data paths.
Training requirements come from the registered strategy; editing an execution
YAML does not change its supported modes. See the
[training contract](../strategy_contract.md#training-before-scoring).

```yaml
strategy:
  id: "xgboost_direction"
  class_name: "XGBoostStrategy"
  source_path: "src/tradingdev/domain/strategies/bundled/xgboost_strategy/strategy.py"
  parameters:
    lookback_candidates: [360, 720, 1440]
    retrain_interval: 720
    target_horizon: 5
    signal_threshold_candidates: [0.0, 0.35, 0.40, 0.45, 0.50]

data:
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1m"
    features: []
```

## Intent

This bundled strategy trains an XGBoost direction model and uses rolling
prediction to produce high-frequency long/short signals in volume mode. It is
designed for research on market-making volume targets, not live execution.

## Signal Logic

- `fit()` searches candidate lookback windows by validation accuracy.
- Optional threshold search evaluates `signal_threshold_candidates` in ascending
  order using the backtest engine. It selects the lowest threshold with finite,
  non-negative `total_pnl` after fees and slippage. Unavailable or non-finite
  results are skipped; if no candidate qualifies, it retains the configured
  `signal_threshold`.
- `generate_signals()` requires a fitted model and rolling retrains according to
  `retrain_interval`.
- Signals follow the project convention: `1` long, `-1` short, `0` flat.

## Data Requirements

The strategy declares market OHLCV only:

```yaml
data:
  requirements:
    market:
      symbol: "BTC/USDT"
      timeframe: "1m"
    features: []
```

Use `inspect_dataset(config_path)` before running long validations to verify
the workspace or `TRADINGDEV_DATA_ROOT` cache.

## Usage

Training is required before this strategy can produce signals. Simple backtests
and parameter optimization are rejected before market data is loaded. Use
walk-forward with a valid `validation` section defining training and test
periods; removing that section does not enable the unsupported modes.

The bundled config includes `validation`. Before running it, check the config
path, symbol and timeframe, and ensure the loaded date range and cached OHLCV
cover both training and test periods. From the repository root:

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml --walk-forward
```

For MCP, use `get_strategy` and `inspect_dataset(config_path)` to inspect the
configuration and data, then call `prepare_walk_forward`. A rejected simple
backtest or optimization directs the client to this tool; valid configuration
and sufficient data are still required before preparation can succeed. See
[Training Before Scoring](../strategy_contract.md#training-before-scoring) for
the execution contract.

## Lifecycle

This is a bundled strategy under
`src/tradingdev/domain/strategies/bundled/xgboost_strategy/`, so it is
git-versioned and treated as promoted by `StrategyService`. Generated strategy
states (`draft`, `validated`, `runnable`, `promoted`) apply only to runtime
artifacts under `workspace/generated_strategies/`.

## Key Parameters

- `lookback_candidates`: candidate training windows.
- `retrain_interval`: bars between rolling retrains.
- `target_horizon`: direction prediction horizon.
- `signal_threshold_candidates`: thresholds explored during `fit()`.
- `monthly_volume_target`: research target used by volume-mode analysis.
