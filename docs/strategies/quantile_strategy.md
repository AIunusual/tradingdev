# Quantile Regime Volume Strategy

Source:
`src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py`

Config:
[Bundled YAML](../../src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml)

## Config Schema

This is a partial snippet. Start from the [complete bundled
config](../../src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml),
including its `backtest` and `validation` settings, and check dates and data paths.
Training requirements come from the registered strategy; editing an execution
YAML does not change its supported modes. See the
[training contract](../strategy_contract.md#training-before-scoring).

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

Warm-up or unavailable features produce a flat signal and reset the position,
even before the minimum holding period. When features return, a new entry must
pass the entry conditions again. Indicator gaps can also make subsequent bars
unavailable, so missing data may cause extra exits or periods without exposure.
See the [scoring contract](../strategy_contract.md#training-before-scoring) for
the alignment and state-handling requirements.

## Running

A simple backtest cannot train this strategy. Preparation rejects that mode
before loading market data and explains that walk-forward validation is required.
Removing `validation` does not bypass this requirement. Direct Python calls to an
untrained `generate_signals()` also fail instead of training on the frame being scored.

Use walk-forward validation to train on each fold's training window before
scoring its test window. Prepare the market and feature data described below,
and check the dates and paths in the linked bundled YAML before running:

```bash
uv run python -m tradingdev \
  --config src/tradingdev/domain/strategies/bundled/quantile_strategy/config.yaml \
  --walk-forward
```

The split settings (`train_start`, `test_start`, `mode`,
`label_horizon`, ...) are documented in
[Strategy Walk-Forward Settings](../strategy_contract.md#strategy-walk-forward-settings).
A config that declares `validation` cannot be run as a simple backtest, and
`--walk-forward` against a config without it fails with `Config has no
validation section for walk-forward.`

Each fold reports metrics for its test window and for its training window. The
training-window figures are in-sample: the strategy is fitted on that window and
then scores the same bars, so they describe fit rather than out-of-sample
performance. Only the test-window metrics are held out. See the
[training contract](../strategy_contract.md#training-before-scoring) for the
scoring requirements.

For MCP, provide the required
[research choices](../strategy_contract.md#research-inputs-and-simulation-policy)
and ask the assistant to use walk-forward validation. Bundled example dates do not
replace the user's period selection. Strategy discovery
reports that training is required and walk-forward is the supported mode.
If the assistant tries a simple backtest or optimization, preparation returns
the reason and points it to `prepare_walk_forward`, with validation settings as a
prerequisite. A rejected preparation starts no formal job. The assistant must
still present a successful replacement plan for your confirmation. See the
[training contract](../strategy_contract.md#training-before-scoring) for the
failure semantics.

## Data Requirements

Training and scoring inputs need a `timestamp` column or `DatetimeIndex` with
non-missing, unique, strictly increasing times. Times are normalized to UTC;
values without a timezone are interpreted as UTC. Invalid time identities are
rejected because their history cannot be ordered reliably.

The config declares OHLCV plus two feature sources:

```yaml
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

An omitted or empty horizon, entry-edge or full-size-edge candidate list uses
the corresponding single configured value. With `dynamic_sizing: false`, the
full-size-edge candidate list is ignored and only `edge_for_full_size` is used.
Defaults therefore evaluate at least one combination; a finite eligible
objective is still required after evaluation to complete `fit()`.
