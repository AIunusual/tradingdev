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
  source: "binance_vision"
  market_type: "futures/um"
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
  It rejects the call if no candidate has sufficient mature training and
  validation samples, preserving any previously fitted state. Use more history
  or a shorter prediction horizon; see the
  [dataset and model input contract](../ml_research.md#dataset-and-model-inputs).
- Optional threshold search evaluates `signal_threshold_candidates` in ascending
  order using the backtest engine. It selects the lowest threshold with finite,
  non-negative `total_pnl` after fees and slippage. Unavailable or non-finite
  results are skipped; if no candidate qualifies, it retains the configured
  `signal_threshold`.
- `generate_signals()` requires a fitted model and rolling retrains according to
  `retrain_interval`.
- A bar with unavailable features emits no entry signal; it does not reuse the
  preceding bar's prediction. Cooldown counts all source bars, including gaps,
  and threshold selection uses the same timing. See the
  [dataset and model input contract](../ml_research.md#dataset-and-model-inputs).
- Scored data must carry chronological timestamps. The
  [training contract](../strategy_contract.md#training-before-scoring) covers the
  accepted input, the exception and the empty-frame rules.
- Signals follow the project convention: `1` long, `-1` short, `0` flat.

## Declarative Rolling Schemes

`rolling_gen` replaces the rolling retrainer with a named window geometry. It is
off unless enabled, and while it is off the classic loop above is what runs. The
schemes, their optional fields and the causal boundary are described once in the
[ML research contract](../ml_research.md#rolling-retraining-schemes); this
strategy needs only the switch:

```yaml
strategy:
  parameters:
    rolling_gen:
      enabled: true
      scheme: roll_ex
```

## Data Requirements

The strategy declares market OHLCV only:

```yaml
data:
  source: "binance_vision"
  market_type: "futures/um"
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

For MCP, use `get_strategy` to inspect the configuration, then call
`prepare_walk_forward` with the required
[research choices](../strategy_contract.md#research-inputs-and-simulation-policy).
Bundled example dates do not replace the user's period selection.
A rejected simple backtest or optimization directs the
client to this tool. Preparation checks configuration and execution mode offline;
it does not require cached history, load market data, fit a model or construct folds.
Sufficient history and usable training windows are checked during the formal
execution after approval. For a separate request to inspect existing data,
`inspect_dataset(config_path)` can inspect the local cache without downloading it;
this is not a preparation prerequisite or proof that training will succeed. See
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
