# Strategy Contract

Generated strategies are runtime artifacts managed by MCP. They are saved under
`workspace/generated_strategies/<strategy_id>/revisions/<revision_id>/` with
`strategy.py`, `config.yaml`, and `metadata.json`. Source and base config are
immutable through the service; metadata records the selected revision's lifecycle
and validation evidence. `current.json` in the strategy directory points to the
most recently saved revision.
Bundled strategies are engineering-maintained code under
`src/tradingdev/domain/strategies/bundled/`.

Generated code must pass the repository's Ruff and strict Mypy rules from
`pyproject.toml`, including the Pydantic plugin and explicit dependency overrides.
The installed wheel carries this same policy. Caller-local configuration cannot
relax it. Mypy follows imports silently so diagnostics target the generated file;
unknown imports and missing annotations are still rejected.

## Python Contract

`random_seed` is a single top-level YAML setting: a strict integer in
`0..4294967295` or `null` (independent entropy). `backtest.random_seed` is rejected.
Use `tradingdev.domain.randomness.get_random()` or `get_numpy_rng()` for run-local
Python and NumPy generators; `get_seed()` supplies an explicit seed to a
third-party model's `random_state` when needed. These functions require an active
execution context, including the strategy constructor. Validation and dry-run each
start a fresh context; a backtest or walk-forward run shares one stream throughout
its constructor and evaluation/folds. Each optimization trial and its out-of-sample
evaluation starts its own context with the recorded run seed, independent of trial
completion order. Nested contexts restore their parent on success or failure.
These APIs do not seed arbitrary global RNG calls, third-party engines, or GPU
operations. Separate concurrent strategy executions must each open a context;
strategy-created threads or processes must explicitly initialize their own streams.

`get_strategy_contract(section="core")` returns the required Python/signal rules,
base interface and revision lifecycle. Each reply identifies its selected section
and lists `available_sections` with their purposes. The `content` object contains
only that complete section: request `section="research"` for the research-input
schema and shared simulation policy, `section="evidence"` for parameter proposals,
source claims and formula recording, `section="example"` for the Python/YAML pair,
or `section="configuration"` for the executable YAML schema and financial-setting
notes. Research schema and policy are projected from their domain definitions.
Unknown sections fail; a missing core reference does not produce an empty
or partial contract. The default no longer embeds examples and the entire schema.

`get_strategy_contract(section="configuration").content.config_schema` describes
the complete YAML shape. Fixed
configuration sections reject unknown fields; strategy parameters remain dynamic.
Validation and dry-run validate this same run configuration before strategy code
executes and return `effective_config` for checking the request against defaults.
A valid configuration does not by itself prove that every natural-language
requirement has been fulfilled.

`list_data_sources` returns registered source descriptions, including asset classes,
symbol examples, local timeframe choices, applicable `data.market_type` values and
coverage limitations. Choose a name for `data.requirements.market.source` from that
catalog. A `null` timeframe list means the adapter does not advertise an exhaustive
local list, not that every interval is supported. Preparation rejects unknown
source names and choices excluded by an explicit local timeframe or market-type
list, without constructing a crawler. Actual availability is checked when the
approved execution obtains its data. Discovery and plan preparation do not
download market data.
The market requirement's symbol and timeframe must match the backtest fields,
which determine the actual data request. Direct CLI/Python preparation rejects
contradictory declarations; MCP request overrides update both together.

Generated code must:

- inherit `tradingdev.domain.strategies.base.BaseStrategy`;
- expose a constructor that can be called with YAML `strategy.parameters`;
- accept optional `backtest_engine` or `parallel_config` when it needs application
  execution context;
- implement `generate_signals(df)` and return a new pandas DataFrame;
- preserve the input DataFrame without mutation;
- include a `signal` column containing only `1`, `-1`, or `0`;
- avoid network, subprocess, destructive filesystem, dynamic import, `eval`, and
  `exec`.

In signal mode, each row specifies a position state: `1` long, `-1` short,
`0` flat. Repeat the state while it should persist. An entry pulse followed by
zero closes the position on the next signal change. `signal_as_position` has no
effect in this mode. In either mode, `-1` selects a short position; it is not a
sell-only instruction to close a long. Both modes shift signals by one bar and
execute signal-driven entries and exits at the following bar's open, falling back
to close when open is absent. Stop-loss and take-profit rules are separate.

Signal execution explicitly fixes the critical vectorbt 1.2.0 sizing, reversal,
order-cost and stop defaults under the versioned execution policy. An omitted
`position_size` uses available capital after execution costs; a specified amount
requests that quote-currency notional, with partial fills allowed when cash is
insufficient. Zero opens no positions. Repeated same-direction signals do not
add to a position. Fixed-notional shorts do not lock cash, so this is not a
guarantee of unleveraged exposure. Neither mode models complete margin,
liquidation, borrowing or funding-payment rules, or provides a leverage setting.

Signal stops use the entry bar's **close** as their reference and, on later bars,
inspect only the range between **open and close**, not the market `high`/`low`
columns. Stop-loss takes priority over take-profit and ordinary signals. A gap
across a threshold fills at open; otherwise the threshold price is used. The
fixed `stoplimit` policy applies no exit slippage to these stop orders, while
commission still applies. A zero stop is a zero-distance threshold, not disabled.
Stops do not execute on the entry bar; after a stop exit, a later signal-state
change is needed for another entry. `re_entry_after_sl` does not affect signal
mode. These are historical simulation assumptions, not exchange execution
guarantees or isolation from every possible global vectorbt setting.

The published SMA example is explicitly long-only: emit `1` on every row where
both averages are finite and the fast average exceeds the slow average, otherwise
emit `0`, including equality and warm-up. Adapt this example to the user's actual
direction, entry, holding and exit rules rather than inheriting its trading intent.

Backtest execution requires finite, strictly positive prices. Signal mode checks
`close` and any supplied `open` across all bars, using `close` for execution when
the `open` column is absent. On bars with logged order attempts (filled, ignored
or rejected), execution-record validation additionally requires nonmissing `high`
and `low` values to be finite and strictly positive. Absent or missing values
(including NaN) are recorded as `null` in those market fields, without filling
them from `close`. This additional high/low check does not apply to bars without
logged order attempts. Volume mode checks OHLC across all bars, using `close` for
absent `open`, `high`, or `low` columns.

When supplied, the `timestamp` column or DatetimeIndex must contain valid,
nonmissing, unique, increasing timestamps; they are normalized to UTC.

Volume mode applies optional `size_weight` values to the following bar with
`shift(1).fillna(1.0)`. The resulting weights must be finite and strictly positive;
zero is not a trade-suppression flag. The weight multiplies the base notional
only when opening a new position, including reversal and stop re-entry; changes
while holding do not rebalance the existing position. An omitted or null
`position_size` uses 200 quote-currency units; explicit zero opens no positions
and creates no zero-quantity trades. Emit `signal=0` to prevent new entries; set
`signal_as_position: true` if zero should also close an existing position. With
the default `false`, zero retains an existing position. An opposite nonzero
signal closes the current position and opens one in the opposite direction.

Volume stops first check an existing position against the current bar's high/low,
using entry price as the reference; stop-loss wins if both thresholds are hit,
and fills use the threshold price. A zero stop disables it. New positions are
not checked for stops on their entry bar. With `re_entry_after_sl: true`, a
nonzero shifted signal re-enters at the stop bar's close unless the monthly loss
breaker has triggered. With `false`, no new position opens on that bar; the
simulator stays flat until a later shifted signal differs from the stop bar's
signal, including a change to zero. An opposite signal on the stop bar itself
does not open a position and becomes the signal that must subsequently change.
A new month resets the monthly breaker but does not clear this signal wait;
signal changes during a monthly pause can clear the wait, while the monthly
pause still prevents entry. Fees and separate notional slippage charges apply
on each entry and exit, including a final close at the last bar's close.

The confirmation text, chat summary and HTML show the same backend-resolved
execution assumptions. They describe simulator policy, not proof that strategy
code or an ML model implements the requested idea, avoids look-ahead, or passed
preflight checks; completed checks are recorded separately.

Validation, dry-run, backtest and walk-forward use the same constructor binding:
YAML parameters become keyword arguments. Missing required parameters and names
the constructor cannot accept fail validation. `backtest_engine` and
`parallel_config` are injected by the application and cannot be overridden in
YAML. Saving a revised draft
creates a new revision and loads that source, including
same-size edits made within one filesystem timestamp interval. An existing
revision is never replaced by a later save.

Execution submission captures every declared constructor default, excluding the
injected execution context, as an explicit keyword value in
`manifest.strategy_execution`. The saved revision YAML remains unchanged;
`manifest.config.strategy.parameters` may expand to the effective captured
parameters for an MCP experiment. Defaults must be finite JSON values;
unsupported defaults require explicit serializable parameters or a revised
strategy. Generated code must express configurable values as constructor
parameters rather than derive hidden defaults from environment state. This
captures declared arguments, not arbitrary Python behavior or dependency versions.

Allowed import roots for generated strategies are intentionally small:

- Python standard library: `__future__`, `collections`, `dataclasses`,
  `datetime`, `enum`, `math`, `statistics`, `typing`.
- Runtime libraries: `numpy`, `pandas`, `talib`, `typing_extensions`.
- Project APIs: `tradingdev`.

Prefer `tradingdev.domain.indicators` (`sma`, `ema`, `rsi`, `macd`,
`bollinger_bands`, `atr`, `adx`, `stochastic`) for standard indicators. This
facade calls the official TA-Lib Python wrapper, preserves the input index,
and exposes named result fields for indicators with multiple outputs. Bundled
strategies and feature engineering must use this facade rather than call
TA-Lib directly. Bollinger Bands use population standard deviation.

The facade preserves TA-Lib's native warm-up NaNs and NaN propagation, including
NaNs that can extend to the end of the output after an interior missing value.
Short inputs therefore remain unready for indicators whose full lookback has
not elapsed. Keep signals flat until every value required by the signal is
finite; do not backfill indicators from future rows or substitute zero for
unavailable indicator values. OHLC inputs must have identical indexes and
lengths, and infinite input values are rejected. Empty inputs produce empty
float64 outputs with the same index.

Periods must be integers other than booleans and at most 100,000. SMA, EMA,
RSI, ADX, Bollinger Bands, and MACD fast/slow periods require at least 2 bars;
ATR, MACD signal, and Stochastic periods allow 1. Bollinger Bands expose only
population standard deviation: the previous `ddof` parameter is removed.
The `std` argument to `bollinger_bands` must be finite and in the inclusive
range `0 <= std <= 3e37`. Negative values, values above `3e37`, NaN, and
positive or negative infinity raise `ValueError`.

Generated strategies may also import `talib` directly. Use its Function API
with float64 NumPy arrays, unpack multi-output tuples, and restore the input
index when constructing pandas Series. The output order is `line, signal,
histogram` for `MACD`, `upper, middle, lower` for `BBANDS`, and `k, d` for
`STOCH`; these are not pandas-ta DataFrames with parameter-encoded column names.
The same warm-up and missing-value rules apply. The import allowlist rejects
`pandas_ta`; existing generated strategies using it must be revised and
validated again. Static validation does not enforce the numerical rules above.

The public `indicator_column` helper has also been removed. Remove its imports
and calls, replacing pandas-ta column-name lookups with the facade's named
result fields, such as `macd(close).histogram` or `bollinger_bands(close).upper`.
When calling `talib` directly, unpack its output tuples in the order described
above instead. The facade's result objects expose named fields and are not
tuples to unpack.

Ruff treats `tradingdev` as first-party code. Separate its imports from
third-party `numpy`, `pandas`, and `talib` imports with a blank line.

Recommended imports:

```python
from tradingdev.domain import indicators
from tradingdev.domain.indicators.kd import KDIndicator
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.shared.utils.logger import setup_logger
```

### Training Before Scoring

Strategies whose signals require a fitted model must establish that model from
training history before scoring evaluation data. An untrained strategy must not
fit on the entire evaluation frame and then replay signals over that frame:
earlier signals would use information from later rows. Strategies that do not
require training need no additional `fit()` step.

`list_strategies` page items and `get_strategy` return `execution_capabilities` for bundled,
generated and legacy strategies. `training_requirement` is `required`, `optional`
or `unknown`. `supported_execution_modes` is `["walk_forward"]` for required
training, all three modes (`backtest`, `walk_forward`, `optimization`) for optional
training, and `null` for unknown capabilities. These describe intrinsic support,
not model readiness, configuration validity, lifecycle permission or successful
validation evidence. A supported walk-forward mode still requires valid `validation`
settings; bundled strategies may provide a separate `walkforward_config.yaml`.
Conversely, a config with `validation` still cannot be used for a simple backtest
or optimization just because its strategy intrinsically supports those modes.

The bundled catalog reads `strategy.training_requirement` from its registered
YAML without importing strategy code. Quantile, XGBoost direction and Safety
Volume require training; KD, GLFT and GLFT ML declare optional training. GLFT ML
retains its existing pure-GLFT fallback when untrained. The declaration is
discovery metadata, excluded from normalized execution configuration; caller YAML
and parameter overrides cannot replace the registered capability. Generated and
legacy source remains `unknown` even if it defines `fit()`, declares this YAML
field or has passed validation/dry-run. No generated source is executed merely
to discover its capabilities.

Strategy discovery uses the shared [discovery page contract](run_artifacts.md#discovery-pages).
Its compact items omit full metadata, YAML requirements and recent runs; read
`get_strategy` for a selected identity instead of assuming the list contains
all configuration or validation evidence.

Simple-backtest and optimization preparation reject registered strategies that
require training before data loading or a formal job. The application error code
is `strategy_requires_training`. The CLI uses the same mode check; MCP returns
`success: false`, `error`, `execution_guidance` (strategy ID, capabilities,
`recommended_mode: "walk_forward"`, `required_configuration: ["validation"]`)
and `next_tool: "prepare_walk_forward"`. Guidance survives the supervised
preflight response; it is not a ready plan or authorization to start a job.
The MCP adapter adds the tool name while domain/application checks use execution
modes. Satisfy the validation prerequisites, prepare the replacement, then obtain
fresh user confirmation. Removing `validation`, waiting for an implicit background
training job, or using the separate ML `train_model` tool does not prepare this
strategy instance. Other preparation errors retain their own codes.

As a final strategy-level guard, bundled `QuantileStrategy.generate_signals()`
raises `RuntimeError` when its regime model, feature engineer, or training history
is absent. Its `fit(train_df)` requires an injected backtest engine; calling it
without one skips training and does not satisfy model readiness. This guard
checks readiness, not the correctness of all
training windows or predictions. Use the walk-forward execution path for this
strategy; the simple-backtest and optimization paths do not fit it before scoring.

Scoring a window that overlaps the training history must not duplicate that
history or replay future bars into it. `QuantileStrategy.generate_signals()`,
`XGBoostStrategy.generate_signals()` (through `RollingRetrainer`) and the
SafetyVolume Risk Gate loop keep only the training rows that strictly precede
the scored frame. Scoring the training frame itself therefore contributes no
history, so its prediction contexts stay chronological instead of continuing
from the tail of one copy into the head of the next.

This context correction does not make training-frame scores out-of-sample:
the initial model has already been fitted on that training window. Walk-forward
`train_metrics` remain in-sample diagnostics; use the held-out `test_metrics`
to assess generalization.

Whether the two frames overlap can only be decided from timestamps, so a scoring
request with a non-empty scored frame requires a time identity, whether or not
the frames overlap: that frame, and any non-empty training history, must carry a
`timestamp` column or a `DatetimeIndex` with non-missing, unique and strictly
increasing values. Either representation can be used independently for each
frame. Internal scoring copies carry a UTC `timestamp` column, including when
the input uses a `DatetimeIndex`; the original frames are unchanged. Naive
values are interpreted as UTC. A frame with no
identity, or with duplicated or decreasing timestamps, raises `ValueError`.

An empty scored frame produces no signals. `RollingRetrainer` and the
SafetyVolume Risk Gate return before validating either frame, while
`QuantileStrategy.generate_signals()` validates both frames before returning an
empty result.

When warm-up or immature horizon labels leave a rolling training slice with no
usable observations, scoring retains the fitted model until observations mature.

Quantile predictions are mapped back by source row. Its feature warm-up and
missing indicator values may remove rows; recursive indicators can also leave subsequent rows
unavailable. A bar without
features emits `signal=0` and resets the position, holding count, entry price,
and size weight. Once features become available, entry conditions are evaluated
again. A single state machine processes every input row, including unscorable
rows and chunk boundaries; output length and index match the input.
Quantile parameter-search evaluation uses the same gap handling on the full
source frame.
Its scoring features omit targets, so a missing future label does not prevent
prediction on an otherwise usable tail bar.
Quantile model probability arrays must contain one finite four-class vector per
scored feature row; malformed model output is rejected instead of treated as missing
features. Float32 and float64 model outputs are normalized to float64.
`QuantileFeatureEngineer.transform()` always preserves surviving input index
labels; callers that need positional mapping supply a positional index.

See the [Quantile guide](strategies/quantile_strategy.md#running) for commands,
data prerequisites and execution-mode restrictions.

## Research Inputs and Simulation Policy

MCP `save_strategy`, `prepare_backtest`, `prepare_comparison`, `prepare_walk_forward`, and
`prepare_optimization` require `research: ResearchSpec`. The assistant must obtain
four user choices: strategy outline, instrument, period, and bar timeframe. A vague
outline is acceptable; missing choices must be requested rather than copied from
examples or inferred from available data. The period occupies two input fields.
The backend enforces completeness and revision binding, not whether a human
actually supplied those choices.

For new research, send `kind: "new"` and all five fields:

```json
{
  "kind": "new",
  "strategy_outline": "Buy after a fast moving average crosses above a slow one; exit on the reverse crossing.",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-07-01T00:00:00Z"
}
```

Text must be nonempty. Dates accept ISO dates or timestamps, normalize to UTC,
and interpret a missing timezone as UTC. A date without a time means midnight;
ordinary backtest bounds do not automatically expand to cover the final day.
Rows are selected by candle timestamp with `start_date <= timestamp <= end_date`,
including either endpoint when matching data exists; this does not define the
final candle's close or execution time.
`end_date` must follow `start_date`. Unknown fields and explicit `null` choices
are rejected. Optional fields must be omitted to request their defaults.

The shared `research-v1` simulation policy supplies omitted `init_cash: 10000.0`,
`fees: 0.0006`, and `slippage: 0.0005`. Capital is in quote currency; fees and
slippage are decimal rates. These are simulation assumptions, not live exchange
fees. Explicit finite values take priority; capital must be positive and costs
nonnegative, including zero. Numeric strings and booleans are rejected. Volume
mode has no capital-return basis: omitted capital stays `null`, and an explicit
capital value does not change that engine's semantics. The resolver does not
synchronize similarly named strategy parameters with backtest execution costs.

`annualization` accepts `"market"` (the new-research default), `"unavailable"`,
or a positive finite factor for observed UTC daily returns. Market policy chooses
365 only when the selected provider's registered asset classes are exclusively
crypto. Mixed providers, including Yahoo Finance even for a crypto symbol, require
an explicit factor or `"unavailable"`; symbol spelling does not identify a trading
calendar. The latter saves `backtest.periods_per_year: null`. A factor does not
enable daily statistics for weekly, monthly or unrecognized bar frequencies.

Data configuration must explicitly select `data.source` or
`data.requirements.market.source`. Providers with market-type choices also require
an explicit supported `data.market_type`; research preparation does not silently
select Binance or its futures market. Source capabilities come from
`list_data_sources`, without downloading data. Bundled YAML includes its source
and market type as template settings, not as evidence of user-selected research
dates or costs.

`kind: "inherit_selected_revision"` enables inheritance only from an explicitly
selected immutable generated revision. Preparation requires its
`revision_id`; saving a revision requires its `strategy_id` and
`expected_revision_id`. Bundled strategies and implicit current selection cannot
supply inheritance. An older revision without a saved outline requires an explicit
`strategy_outline`. An omitted annualization choice can retain the selected
factor or unavailable setting when symbol, provider and market type match; after a market change, the
resolver requires an applicable market policy or an explicit new choice.

Saving uses the shared builder to produce a complete YAML from authoring settings
and research choices. Research owns its outline, symbol, timeframe, dates, capital,
costs and annualization factor; duplicate owned fields in authoring YAML must
agree. Strategy parameters, feature requirements, validation and other engine
controls remain authoring settings. Execution preparation uses the same resolver
to replace template research values without modifying the base revision.
`backtest_overrides` is for other engine controls; research-owned keys are rejected
there, even when their values agree. Put those choices in `research` instead.
An existing YAML cost is not an implicit request
to inherit it. Resolved values and their origins are frozen in the config;
[artifact provenance](run_artifacts.md#research-settings-and-provenance) records the
policy and selected revision without duplicating the effective values.

The old top-level `symbol`, `timeframe`, `start_date`, and `end_date` arguments are
not accepted by MCP preparation tools. Optimization still requires its separate
training/test calendar dates, and the resolved research period must equal
`train_start` at 00:00:00 UTC through `test_end` at 23:59:59.999999 UTC. A mismatch
is rejected, not silently expanded. CLI and direct Python execution with a complete
YAML remain separate explicit execution interfaces; they do not require an MCP
intake session or automatically reconstruct research provenance.

## YAML Contract

The following is a complete execution example. MCP authoring may omit the
research-owned fields that the builder supplies; if included, they must agree
with the required `research` input.

```yaml
strategy:
  id: "sma_crossover"
  version: "0.1.0"
  class_name: "SmaCrossoverStrategy"
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  init_cash: 10000.0
  periods_per_year: 365.0
  risk_free_rate: 0.0
  required_return: 0.0
  mode: "signal"

data:
  source: "binance_vision"
  market_type: "futures/um"  # explicitly choose "spot" for spot data
  requirements:
    market:
      # source selects the market data crawler registered in
      # domain/data/crawlers/registry.py:
      #   "binance_vision"          - crypto, data.binance.vision
      #   "binance_api"              - crypto, ccxt
      #   "yahoo_finance"            - stocks/ETFs/futures/indices/FX,
      #                                Yahoo symbols ("AAPL", "ES=F", "^GSPC")
      # omitted -> uses the explicitly selected top-level data.source
      source: "binance_vision"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
```

`save_strategy` assigns `strategy.id`, `strategy.revision_id`, and
`strategy.source_path` to the saved revision. These identity fields are managed
by the service; callers need not supply them. `strategy.version` remains optional
descriptive configuration, not the execution revision identifier.
`strategy.display_name` is an optional human-readable name, separate from the
internal ID; generated saves supply it from an explicit name or the outline.
The saved config also contains resolved `research` provenance. Neither field is a
constructor parameter, and changing a generated strategy's display name requires
a new revision.

Performance analysis uses QuantStats for return/risk metrics and VectorBT for
closed-trade statistics. `periods_per_year` is the explicit annualization factor
for observed UTC **daily returns**, independent of the strategy's bar timeframe:
365 for this continuously traded crypto example, or an appropriate trading-day
count for another market. In a direct execution config, omitting it leaves annualized
metrics unavailable. The guided research resolver supplies or explicitly disables
it according to the preceding policy before execution.
`risk_free_rate` and `required_return` are annual decimal rates, converted to
daily rates using that factor. Calendar aggregation uses observed dates; missing
market data is not silently filled with zero returns. Maximum drawdown retains
bar-level resolution. `max_drawdown` is a nonnegative fraction;
`max_drawdown_amount` is a nonnegative amount in the portfolio's quote currency.
Calmar uses daily-return drawdown, exposed separately as `daily_max_drawdown`.
Daily observations require a declared bar timeframe of one day or finer. Weekly,
monthly, multi-day or unrecognized timeframes leave annualized metrics,
`daily_max_drawdown`, and daily/monthly PnL statistics unavailable. Multi-day
marks cannot locate daily closes or allocate PnL across month boundaries; no
daily equity is fabricated. `periods_per_year` always means days per year, so
setting it to 52 does not enable weekly annualization. Total return, total PnL,
bar-level drawdown and trade statistics remain available. Frequency validation
uses the declared timeframe, not gaps caused by weekends or missing data.
Optimization rejects objectives requiring these unavailable observations before
creating a job.
Volume mode has no initial capital, so capital-return metrics are unavailable;
use amount-based PnL and drawdown rather than interpreting missing returns as zero.
`total_trades`, `win_rate`, `profit_factor`, and `trade_expectancy` refer to closed
trades after entry and exit costs. Open trades are counted separately.

Run and job responses contain summaries. A metric omitted from a summary may still
be computed and saved: use `get_metric_catalog` for current definitions and
optimization directions, and `get_run_metrics(run_id, metric_ids, scope)` to read
saved values and their definition/settings snapshots. Use each run's
`available_scopes` for fold and trial results. Missing values include an explicit
reason and must not be interpreted as zero or a request to rerun the strategy.

Feature sources are explicit:

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
      - type: "funding_rate"
        source: "binance"
        column: "funding_rate"
        path: "workspace/data/processed/btc_funding_rate_2025.parquet"
```

### Strategy Walk-Forward Settings

The top-level YAML `validation` mapping configures strategy backtests through
`WalkForwardConfig` and `WalkForwardValidator`. It is separate from the ML
`train_model(validation=...)` tool described below: it does not select an ML
scheme by name or infer a label horizon from an ML dataset.

Only MCP `prepare_walk_forward` accepts optional
`validation_config: WalkForwardConfig`. When supplied, it replaces the complete
validation mapping for that execution, using schema defaults for omitted settings;
it does not merge with the selected configuration. Omission or `null` preserves
the selected validation settings, including a bundled walk-forward template where
applicable. `prepare_backtest` and `prepare_optimization` reject this argument.
The effective mapping is captured in the manifest without modifying the source
revision, its validation evidence or current pointer.

| Setting | Default | Effect |
| --- | --- | --- |
| `label_horizon` | `0` | A positive value declares how many source rows a training label looks ahead and enables the pre-fold label-overlap audit. Values below 1 leave that audit disabled with the plain splitter. |
| `purged_split` | `false` | Selects the opt-in anchored purged splitter when true; it requires `label_horizon >= 1`. |
| `embargo_bars` | `0` | Extra rows removed before each test window in addition to the label-horizon purge; used only when `purged_split` is true. |
| `n_splits` | `1` | Number of automatic test folds; use a positive integer and enough data for nonempty windows. |

With `purged_split: false`, all four `train_start`, `train_end`, `test_start`
and `test_end` values select one explicit date split. Otherwise the plain
splitter uses automatic windows with `train_ratio: 0.8` and `expanding: false`
by default. Each boundary is an inclusive UTC calendar day, so the four dates
are converted to UTC first (naive values are treated as UTC), then their time
of day is discarded. Their UTC days must satisfy
`train_start <= train_end < test_start <= test_end`: a train
window ending on the day the test window starts would train on bars the test
set also scores, so it is rejected. Supplying only some of the four fields is
rejected as well, instead of silently falling back to automatic windows. The
active training and test windows, including the full final UTC day, must fit
inside the resolved research start/end timestamps. Execution preparation rejects
out-of-range windows rather than clipping them or extending the research period.
For the requested period, supply suitable explicit windows or replace the
validation mapping with an automatic split. Bundled examples with
explicit windows use `23:59:59.999999` on their final backtest day; a date-only
research end still means midnight, not the end of that day.
The splitter re-checks the loaded frame, raising `WalkForwardSplitError` when the
timestamps are not ascending, a window selects no rows, or the selected train
and test rows share a bar.
A positive `label_horizon` checks the selected windows before any fold's
strategy fitting or backtest runs. If training labels reach the test window,
the full validation raises `WalkForwardLeakageError`; this audit detects the
problem without changing the windows. Thus setting only a positive horizon on
adjacent plain train/test windows will block them rather than purge them.
`embargo_bars` has no effect in this plain mode, including with explicit dates.

To remove overlapping training labels automatically, add this mapping at the
root of a strategy YAML whose loaded history is long enough for all folds:

```yaml
validation:
  n_splits: 3
  label_horizon: 5
  embargo_bars: 2
  purged_split: true
  target_metric: "sharpe_ratio"
```

The purged splitter divides the loaded rows into `n_splits + 1` contiguous,
near-equal blocks. The first is training-only; each later block is tested once,
using all preceding blocks as anchored training history after removing the
`label_horizon + embargo_bars` gap immediately before its test window. In this
full-validation splitter, explicit train/test dates, `train_ratio` and `expanding` do not control
folds, even if present in the YAML. The schema still parses supplied dates as
datetimes, but does not require all four or validate their order in purged mode.
These inactive dates also do not trigger the research-period containment check.
Omit those inactive settings to keep the
request clear. The backtest's data bounds still determine the history supplied
to the splitter.

Preparation rejects a missing, zero or negative purged horizon, a negative purged
embargo, a nonpositive automatic fold count, and an ordinary automatic split ratio
outside `(0, 1)`. The splitter repeats its applicable checks during execution and
also rejects data-dependent problems such as a horizon at least as long as the
input, too many blocks, or a gap that leaves no training rows. The YAML schema
rejects unknown field names but does not enforce every numerical range or fold
geometry; successful YAML parsing alone does not establish a usable validation run.
The plain splitter's `label_horizon: 0` default is an audit opt-out, not evidence
that a strategy's labels are safe. Declare the actual forward horizon when the
strategy uses such labels. This gate audits row-window geometry, not arbitrary
feature code or a strategy's independently constructed labels.

MCP confirmation preparation validates configuration and execution mode without
loading market data, constructing data-dependent folds, fitting a model or running
any fold through the backtest engine. The confirmation shows the effective mode
and identifies inactive settings; it does not claim that the requested history
can form usable folds. The formal strategy walk-forward execution applies the
complete pre-fold audit after approval and before running its folds.
A nonpositive horizon means the audit is disabled, not that labels are safe.
This differs from the ML tool's timestamp/label-interval validation described below.

## ML Research Contract

The ML MCP tools are a separate synchronous research interface. They select
registered feature builders and estimators without generating a strategy,
preparing an execution plan or starting a backtest worker. A saved model is not
a runnable strategy revision and does not bypass the Python signal contract or
strategy lifecycle. User workflow is documented in the
[README](../README.md#研究機器學習訊號); file identities and persistence rules are in
[ML Research Artifacts](run_artifacts.md#ml-research-artifacts).

`get_ml_contract` returns schema version 1 discovery information: named feature
sets with ordered columns, model kinds with default parameters, validation
schemes, processor types, IC methods, quantile limits, calendar modes, missing
policies and rankable metrics. Use these returned names rather than constructing
arbitrary model classes. `processor_types` advertises domain building blocks
only: `build_dataset`, `build_panel` and `train_model` do not accept or execute a
processor pipeline. In particular, training does not implicitly fit a scaler
because its name appears in discovery.

### Dataset and Model Inputs

`build_dataset` reads existing yearly OHLCV files for the requested symbol and
timeframe, selecting available rows within the requested period. Missing yearly
files or an empty selected period fail the request; success does not establish
complete bar-by-bar coverage. Use `ensure_data` separately when a download is
needed. Building a dataset does not download, refresh an old partial-year market
cache or remove market files. The feature builder preserves rows and emits unavailable values
during warm-up. The service computes `forward_return = close[t+horizon] /
close[t] - 1` and `target = sign(forward_return)`, then drops incomplete feature
or label rows and reports their count. `horizon` is a positive number of source
bars, not elapsed calendar days. OHLCV and bookkeeping columns are not added to
the model's declared feature list.

XGBoost and SafetyVolume's optional direction model label their training frames
through `FeatureEngineer`: negative, zero and positive forward returns map to
`-1`, `0` and `1`. GLFT-ML uses `DirectionFeatureEngineer`, where `1` means a
positive return and `0` means a non-positive return. Both omit rows whose label
endpoint is unavailable, including the final horizon bars and missing interior
prices. Inference without targets retains those rows when their features are
complete. Quantile and SafetyVolume's risk model use their separate target
builders; they also leave outcomes beyond the supplied frame unavailable.

`FeatureEngineer.transform()` and `RiskFeatureEngineer.transform()` preserve
surviving source index labels; there is no index-reset option. Callers needing
row positions supply a positional index.
SafetyVolume maps optional ML directions to those source rows. XGBoost rolling
prediction skips a bar whose own features are unavailable, rather than reusing
an earlier prediction. Its threshold search evaluates the complete source frame,
leaves unavailable rows without entry signals, and measures cooldown in source
bars, including gaps.

SafetyVolume's rolling Risk Gate also predicts only from the current source
bar's features. If they are unavailable, it keeps the neutral `P(safe)=0.5`
instead of assigning an earlier bar's probability. Its state machine still uses
the configured risk threshold and holding rules; a missing risk feature does not
unconditionally flatten a position. Threshold search continues to evaluate its
filtered validation feature rows, unlike Quantile's full-frame gap handling.

The XGBoost and FLAML direction model training APIs require at least one labelled
training row. A supplied validation frame must also be non-empty. Empty inputs
raise `ValueError` before model state changes or external training starts,
preserving any previously fitted model. XGBoost's lookback search requires at
least 10 mature training rows and 5 mature validation rows per candidate, after
feature and label filtering. It skips insufficient candidates and raises
`ValueError` if none qualifies, without installing a default lookback or refitting
the model. Rolling retraining retains its previous model when the available
window has fewer than 10 complete labelled rows.

Both `build_dataset` and `build_panel` pass their date bounds to
`DataService.load_cached`. Bounds without a timezone are interpreted as UTC;
bounds with an offset are converted to UTC. Rows are selected with
`start_date <= timestamp <= end_date`, including both endpoints. A date-only
`end_date: "2024-01-31"` therefore stops at `2024-01-31T00:00:00Z`, not at the
end of January 31. For all hourly bars dated within January, use explicit bounds,
for example these `build_dataset` arguments:

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59.999999Z",
  "feature_set": "returns",
  "horizon": 1
}
```

The selected end bounds the source bars, so trailing rows whose forward label
needs later prices are still dropped. Reversed bounds fail; the loader does not
extend them to obtain missing labels. This timestamp contract differs from a
strategy walk-forward's explicit date split and optimization's inclusive UTC
calendar-day bounds. Explicit walk-forward dates first convert offsets to UTC,
then discard the time of day; the range is
`UTC midnight of start day <= timestamp < UTC midnight after end day`.
For example, `2024-01-02T03:00:00+08:00` names the entire January 1 UTC day.
The confirmation document displays these effective UTC boundaries, including
the exclusive upper bound, alongside the original settings.
Do not copy the ML timestamp example into optimization, which accepts only
`YYYY-MM-DD` boundaries.

`build_panel` requires at least two distinct symbols. It sorts symbol identity,
builds each symbol's features and labels independently, then aligns the labelled
frames. `intersection` keeps common timestamps. `union` uses `missing_policy`:
`drop` removes dates that any symbol cannot supply, `error` rejects gaps, and
`forward_fill` carries prior observations up to `max_staleness` shared calendar
positions (`null` leaves that carry unbounded). Leading gaps cannot be filled
from future observations. Read the returned calendar coverage, dropped dates
and filled-row counts; filled rows are carried observations, not new trades.
`calendar.dropped_dates` and each `calendar.coverage[].missing` list contain at
most the first 50 dates; `n_dropped_dates` and each `n_missing` remain exact.
The saved manifest uses the same bounded calendar summary.

`train_model(dataset_id, model_kind, parameters, validation, ...)` resolves
effective model defaults plus declared parameter overrides; unknown parameter
names are rejected. It trains only the dataset's ordered feature columns plus
`target`. Standalone ML training uses the model's `random_state` parameter,
independently of the top-level strategy YAML `random_seed`.
Feature importance is sorted descending using each estimator's native
`feature_importances_` values. It is not a common cross-framework scale or a
universal gain measure.

The final persisted artifact is fitted on every dataset row, and
`in_sample_accuracy` evaluates those same training rows. Optional validation
fits additional models for its folds; it does not replace the artifact with a
fold model or persist an out-of-fold prediction stream. Preserve the returned
validation evidence if it is needed later.

MCP `predict` verifies the saved dataset and model blob, requires an exact ordered
feature-column match, then returns one page of `signal` and `forward_returns` in
dataset row order with their `model_id`, `dataset_id` and horizon. `offset` starts
at zero; `limit` defaults to 100 and accepts 1–500. `n_rows` is the complete
dataset count, not the page length; `next_offset=null` marks the final page.
`MLService.predict` retains complete arrays for backend and Python callers.
MCP evaluation does not require retrieving every preview page. Prediction does not establish
whether the requested data or its label interval was used for fitting or
parameter selection. Evaluating the final artifact on its training dataset is
in-sample even when the training response includes validation scores. A separate
test dataset must be reserved before selection, with training label intervals
ending before that test period, to support a chronological out-of-sample claim.

### Fold Validation and Signal Interpretation

Omitting `validation` skips fold validation. A supplied scheme uses the saved
label horizon for purging and an optional nonnegative `embargo` gap; callers
cannot replace the dataset's horizon with a more convenient value.
`n_folds` and `min_train_bars` must leave usable training and test data after
exclusions; unsupported schemes or insufficient data fail the request.

Folds are built on sorted distinct timestamps, not the panel's physical row
order. Every symbol on one timestamp shares the same training/test role.
`min_train_bars` counts distinct training timestamps, whereas returned
`train_bars`, `test_bars`, `purged_bars` and `embargoed_bars` count dataset rows.
The service retains each label's original `label_start` and `label_end` before
feature-row removal or calendar alignment. In addition to the horizon-based
gap, it excludes training dates whose label intervals touch the test interval;
one symbol's overlap excludes that date for all symbols. This also checks test
labels extending into the training window on the right in K-fold validation.
Datasets without the label intervals must be rebuilt before validation.

A panel containing forward-filled observations is rejected for fold validation:
its carried label describes the original observation, not a new sample at the
filled timestamp. Rebuild with `intersection` or `missing_policy="drop"` for
validation. Such panels can still be fitted without validation and passed to
`predict`; this is not evidence of an out-of-sample test.

- `purged_walk_forward` expands training through dates before each test window.
- `purged_kfold` trains on retained data outside each test window: later data for
  the first fold, earlier data for the last, and available data on either side
  for interior folds. This is retrospective cross-validation, not predictions
  that could have been made at the test date.

The response reports each fold's retained training/test rows, purged and
embargoed rows, accuracy, IC, Rank IC and hit rate, plus the mean of available
fold scores. Panel fold metrics are calculated separately in timestamp order
for each symbol, then averaged across symbols with available values; rolling
IC windows never cross from one symbol into another. Unavailable scores remain
`null`. `artifact_trained_on: all_bars`
distinguishes the final artifact from the fold models. `leakage_audit_clean`
describes the implemented split and label-overlap checks only; it cannot prove
that features, research choices or external inputs are free of future knowledge.

MCP `evaluate_signal(model_id, dataset_id, window=60, n_quantiles=5, record=False)`
evaluates a verified saved model on the complete single-symbol dataset. The
backend produces the signal, obtains saved forward returns in the same row
order and adopts the dataset horizon; the MCP input does not accept arrays or
a caller-selected horizon. A preview's `offset` or `limit` never restricts
evaluation. There must be enough observations for the chosen IC `window` and
`n_quantiles` (2–10). It returns Pearson and Spearman IC statistics, hit rate, turnover,
prediction strata and simplified return summaries. These signal statistics are
separate from backtest performance metrics and do not model transaction costs,
slippage or order execution. Full-sample strata and their quantile means are
descriptive comparisons, not fitted trading thresholds or out-of-sample evidence.
Ties can reduce the actual number of strata; do not infer their count from the
requested `n_bins` alone.

The simplified long/short and long-only streams use thresholds computed only
from earlier signals. The first `n_quantiles` observations are warm-up, and at
least two distinct past signal values are needed. With few distinct signals,
historical minimum/maximum select the tails; otherwise expanding quantiles do.
Only original positions `0, horizon, 2*horizon, ...` can open a trade, so the
holding intervals do not overlap. Other rows contribute zero. Cumulative values
compound these holding-interval returns; means include the inactive rows.
Returns are indexed by decision row, not realization time, and do not form a
mark-to-market equity curve. A constant signal cannot form a long/short book and
is rejected.

The separate Python `MLService.evaluate_signal` accepts equal-length, finite
numeric signal and forward-return arrays. Without a dataset ID, that caller is
responsible for chronological single-asset alignment and the correct horizon;
omission defaults to one bar. Whenever `dataset_id` is supplied, including by
the MCP backend, evaluation requires a single-symbol dataset,
all of its rows, exact saved forward returns in order, and a matching horizon
(or omission to adopt the saved value), including when `record=False`.
Dataset rows must be strictly chronological, and saved label end times must
show non-overlapping holding intervals on the simulation grid. Datasets without
label intervals must be rebuilt for evaluation.
Panel evaluation is rejected; build a single-symbol dataset for evaluation.
Do not concatenate different assets into one time series. `record=True`
additionally requires the dataset ID for Python array callers.
The MCP path verifies which saved model produced the signal; it does not prove
a held-out evaluation. Python array callers and existing records may use a
caller-declared `model_id` label. Recording and candidate-replacement semantics are defined in the
[artifact contract](run_artifacts.md#recorded-signal-evaluations).
`leaderboard` ranks only a metric with a declared direction and, by default,
only matching measurement contexts. Its optional `dataset_id` filter selects a
verified dataset version and checks that recorded data context matches its
manifest; differences in evaluation settings still require compatible context.
`allow_mixed=True` opts into incomparable
contexts and must not be presented as a controlled comparison. Retain its
`caveats`, skipped entries and candidate counts when interpreting the winner;
trying more candidates increases selection bias.

MCP `leaderboard` accepts `offset=0` and `limit=20` (1–100); it does not accept
`top_k`. The complete filtered candidate set is compared and ranked before
paging. `ranked` and `skipped` each return at most `limit` entries at the same
offset; `n_ranked` and `n_skipped` describe their complete counts. Follow the
shared `next_offset` until `null`, even if one list has no entries on a page.
Mixed-context rejection reports the full context count, limited examples and
filter guidance, not every context's text.

Model prediction unpickles trusted Python objects. Generated-strategy import
checks do not apply to those objects, and a matching hash is not a sandbox or
proof of a trustworthy publisher. Do not load model files from an untrusted
workspace.

## Lifecycle

1. `save_strategy(code, yaml_config, research, strategy_id=None,
   expected_revision_id=None, display_name=None)`: saves a new draft after resolving
   research and checking config consistency. Omit `strategy_id` to allocate
   `strategy_<uuid>` automatically. A supplied ID without an expected revision also
   means creation and must have an unused namespace. Revising requires both IDs and
   a matching current revision under the lifecycle lock. Successful saves return
   `strategy_id` and `revision_id` and update that strategy's current pointer.
   Even identical content creates a new revision without validation evidence.
2. `validate_strategy`: runs syntax, static policy, restricted import checks,
   ruff, mypy, inheritance, constructor, and the shared signal-contract gate on
   a short fixture for the selected revision. It returns that `revision_id` and
   structured diagnostics with `level`, `code`, `phase`, `message`, and optional
   `fix`.
3. `dry_run_strategy`: accepts only `validated` strategies, re-runs the same
   signal-contract gate on a longer fixture, returns `signal_analysis`, and
   marks the strategy runnable when it passes.
4. `promote_strategy`: strategy tool that marks a runnable generated strategy
   as promoted (owned by `StrategyService`).
5. `save_research_evidence`: save the checked trading rules/formulas, parameter
   descriptions, proposal origins and limitations. Use the host's actual web
   capability for primary references; record only sources actually retrieved or
   supplied. Without retrieval, label proposals unverified and clarify unsupported
   ranges. A fixed set of alternatives does not authorize return optimization.
6. `prepare_backtest` / `prepare_comparison` / `prepare_walk_forward` / `prepare_optimization`: resolve
   the required research input, capture an execution manifest, check code and
   configuration offline, and prepare a
   plain-language confirmation and HTML artifact using the saved `evidence_id`.
   Every effective parameter must have an origin; missing bindings return exact
   targets and values for correction through a new record. Only runnable or promoted
   generated strategies, and promoted bundled strategies, are eligible. Preparation creates a plan,
   without creating a formal job or run.
7. Present `confirmation_summary` and a clickable link to the complete HTML,
   then end the conversation turn. Wait for the user to read it and explicitly
   approve, decline or request changes in a later turn. Do not paste the full
   settings, search grid or validation evidence into the conversation.
8. `submit_execution_decision(plan_id, decision)` relays that decision:
   `approve` submits the captured manifest, `decline` cancels, and `revise` cancels
   with `confirmation_adjustment_requested`. Edits require a new plan, a fresh
   code check and another user confirmation. Initial task instructions, silence,
   questions or ambiguous replies must not be treated as approval.

The backend trusts the caller to relay the user's actual decision. It cannot
independently prove that a human supplied it. Plan integrity, expiry, executable
source checks and submission deduplication still apply. No native MCP elicitation
capability is required; the former `request_execution_confirmation` tool is removed.
The complete text and HTML remain saved plan artifacts; only the short summary
and artifact location are returned by preparation and plan lookup. The CLI remains
an explicit execution interface without this conversation workflow.

The lifecycle gate is enforced again at execution time inside `BacktestService`,
so the CLI and subprocess workers cannot bypass it, and the config's `source_path`
must match the selected revision's source.

Strategy naming is optional. Supply a name through `display_name` or the YAML's
`strategy.display_name`; if both are supplied, they must agree. When neither is
supplied, saving uses the first line of the resolved outline, truncated to 80
characters. The name is stored once in `strategy.display_name`, covered by the
config hash and
available in HTML technical details. Presentation titles remain model-authored
page titles. `request_summary` is no longer a save input; metadata projects it
from the resolved outline.

Existing namespaces include immutable revisions, incomplete directories and legacy
source, metadata or config files. Explicit creation collisions return
`strategy_id_exists`; bundled IDs return `reserved_strategy_id`. Anonymous
creation retries up to eight candidate IDs while checking each under its lifecycle
lock, then returns `strategy_id_generation_failed`. None of these collisions
revises the existing strategy. A stale expected revision returns
`strategy_revision_conflict`; invalid revision contents or IDs return
`strategy_revision_invalid`. Supplying an expected revision without a strategy ID
returns `invalid_revision_request`. Research/config resolution errors return
`invalid_research_spec`; syntax and YAML errors keep their specific save codes.
Successful saving is not code validation: subsequent checks still return repairable
diagnostics and determine executable status.

Pass the `revision_id` returned by saving to get, validate, dry-run, promote,
and execution tools. Except for explicit research inheritance and revision saves,
omitting it selects current once at the start of that operation. Validation and
dry-run persist evidence to that selected revision,
even if another save changes current while checks are running. A new draft B
does not revoke revision A's evidence, and B cannot inherit it.

Generated execution requires matching source/config hashes and successful
validation and dry-run records for the selected revision. Editing revision files
directly invalidates them; repair through a new save. An unknown explicit
revision never falls back to current. Submitted jobs, worker configs, completed
runs and strategy source artifacts remain tied to the selected revision when
current changes, including while a prepared plan awaits confirmation.

`cleanup_strategy_drafts(strategy_id, revision_ids=None, apply=False)` previews
retention decisions. Applying requires an explicit nonempty `revision_ids` list
and user authorization for those deletions; previewing alone never authorizes
removal. Only intact, non-current drafts with no persisted references are eligible.
Validated, runnable and promoted revisions remain, even when unused. All jobs
(including failed or cancelled ones), completed runs and orphaned manifests
protect their referenced sources. Unverifiable historical identities or malformed
records block cleanup; ambiguous old records with no revision also block it unless
their stored config establishes a legacy flat source for this strategy.

Responses report `applied` and each revision's `outcome` (`eligible`, `protected`,
`deleted`, `failed` or `missing`) with reasons. Apply rechecks eligibility under
the same per-strategy lifecycle lock used for publication, validation and status
updates. A changed preview can therefore become protected. Integrity failures,
symlinks and unexpected files prevent deletion. An apply containing protected
or failed items reports `success: false`; other eligible items can still be
deleted. Filesystem deletion is not transactional and an I/O failure may leave
a partial directory. A missing generated strategy returns `strategy_not_found`
before creating its lifecycle lock. Cleanup rechecks current after acquiring the
lock and also rejects the request if the strategy disappeared in between.
For an existing strategy, missing explicit revision IDs are reported as `missing`,
making a repeat apply safe. Omitting IDs on apply returns `cleanup_revision_ids_required`;
an untrustworthy reference scan or lock failure returns `strategy_cleanup_blocked`.
No background cleanup, automatic revision merging or migration occurs.

Each preparation fixes an execution manifest containing that strategy
identity, source hash, effective config with defaults, and any optimization
search specification. Effective constructor values, including nested bundled
parameter/fit-model defaults, are stored separately in `strategy_execution` and
consumed without filling new defaults in the worker. The preparation and approval
responses return the same `manifest_hash`, which also appears on job status and
completed runs. Approval rechecks captured source and configuration integrity.
Workers recheck the manifest
against the submitted job and recheck strategy eligibility before execution.
Changing a separate runtime YAML after submission does not change the job;
submit a new job to change execution settings. The run's `config.yaml` is an
inspection projection, while `manifest.json` is the execution authority.

Runtime symbol, timeframe, and dates are separate from the revision's base
config. For generated strategies, ordinary backtest and walk-forward execution
configs may change `strategy.parameters` for experiments on the same runnable
revision. All other saved `strategy` fields, including
identity and any descriptive or constructor settings such as `description`,
`version` or `fit`, must remain unchanged. Changing those fields requires saving
and checking a new revision, even if identity and parameters are unchanged.
The comparison excludes the execution-managed `source_hash` and separately
verifies that `source_path` resolves to the selected revision's source. Copy the
saved config and apply market/date/cost changes outside the `strategy` section.
Direct Python `apply_run_overrides` and supplied `apply_backtest_overrides` remove
research provenance from the effective config because they do not resolve the
new settings through research intake. The saved revision and its metadata remain
unchanged; missing origins must not be presented as inherited user choices.
MCP `prepare_backtest`, `prepare_walk_forward`, and `prepare_optimization` accept
a `parameters` mapping;
overrides apply to the captured effective base parameters, including constructor
defaults. When both existing and supplied values are mappings, they merge
recursively; other values replace the specified parameter. At each merge level,
supplied keys must already exist in the captured mapping. A constructor accepting
`**kwargs` does not allow an override to introduce new keys. Unknown keys return
`invalid_execution_request` for backtest/walk-forward preparation or
`invalid_optimization_request` for optimization preparation, without creating
a job. CLI configs instead provide
their complete experiment parameters, which must satisfy constructor binding
and the execution checks.
The base source, YAML, lifecycle evidence and current pointer remain unchanged.
Each run stores its own fixed parameters and constructor settings in its manifest.
`prepare_comparison` instead accepts `candidates`: one to eight distinct ordered
sets, each with `label`, `rationale` and a `parameters` override mapping. Labels
are stripped of leading and trailing whitespace and must remain nonempty and
unique across the comparison; outer whitespace cannot distinguish two labels.
It has no global `parameters` argument. Research, selected revision, data, costs and seed
are shared; each candidate freezes its complete effective constructor. All
candidates pass the two synthetic contract fixtures before the whole bundle is
presented for one approval. After approval, they use fresh strategies/engines and
independent capital against one saved dataset snapshot. Fixed comparison performs
no adaptive search or automatic winner selection. It supports ordinary backtest
strategies only; training-required or walk-forward research uses its existing
workflow. The report selector `comparison_job_id` retains all candidates and their
outcomes, including failures or candidates that were never run.
Short and long signal-contract fixtures check the effective generated execution
settings at submission and execution, including when MCP `parameters` is omitted
or `null`; a failure rejects the experiment without changing the revision's
status. Static checks remain bound to the verified source.
Optimization search candidates override only their selected parameters, retaining
all other captured base parameters and saved strategy settings. Nested candidates
recursively override only the specified fields of the fixed effective base;
other nested fields retain their captured values. Validation evidence covers
the base parameters; generated optimization candidates also pass the short and
long signal-contract fixtures before execution. Optimization fixes candidate lists,
metric, calendar training/test ranges, the metric's minimization or maximization
direction in the manifest. Version 5 does not contain an in-worker confirmation
wait policy: the user approves the complete search before its worker starts.
A config with `validation` settings cannot also request
optimization; choose walk-forward or supply a config using the optimization
training/test split alone. The complete request does not freeze imported Python
dependencies, the engine environment, market data, or RNG state, and therefore
does not guarantee fully reproducible results. Bundled strategies remain
Git-managed and promoted with `revision_id: null`.

All four MCP preparation tools require an `evidence_id` returned by
`save_research_evidence`;
callers do not supply historical sample budgets or minimum-history declarations.
The saved draft contains `presentation` with a `title`, `summary`,
and `parameter_descriptions`, which maps the union of captured constructor and
effective candidate leaves to a distinct readable `label`, a `description`, and
optional `unit`. Keys are
JSON pointers relative to `strategy_execution.constructor_kwargs`: for example,
`/fast_period` for a generated strategy or `/config/k_period` for a bundled strategy.
Research binding rejects missing or extra descriptions and incomplete provenance
with `research_evidence_binding_invalid`. Save corrected evidence and prepare
again using the new ID. The assistant supplies this technical mapping; the user
reviews readable names, values and units.

Evidence capacity is checked both when saving and after binding. The canonical
JSON draft must fit within 512 KiB. Preparation allows at most 1,024 targets
(optimization counts the base plus every search-axis option, not Cartesian
trials) and 4,096 parameter bindings summed across targets. The complete expanded
`BoundResearchEvidence`, including the saved record, resolved values and origins,
must independently fit within 512 KiB. A draft can therefore save successfully
but fail preparation. Reduce the proposed candidate/search options or the
strategy's parameter structure, or shorten explanations to fit; retain complete
descriptions and origins for every effective parameter. Never omit required
bindings or silently split the user's requested comparison to evade these limits.

Optional `backtest_overrides` capture other engine settings in the same plan;
research supplies capital, costs and annualization choices.
The short conversation summary uses `summary`, not `title` or the saved name,
and includes resolved capital/cost sources and the annualization basis. HTML
retains the title, saved name, complete settings, policy and inheritance details.
The surrounding research draft also contains trading conditions, formulas, variable
units, source claims and limitations. These client-authored explanations do not
prove equivalence to the strategy's Python or ML implementation.

The preflight receipt records offline code/configuration checks and the fixed
synthetic signal fixtures used for generated strategies. It does not contain market
windows, trades or performance. Preparation neither loads real market data nor
runs the backtest engine or model training. Outside fixed comparisons, bundled strategies receive configuration,
execution-mode and parameter checks without calling `fit` or `generate_signals`;
training-dependent signal checks are explicitly deferred. The confirmation lists
market availability, data-dependent fold geometry, training and trade/performance
behavior as unverified until formal execution. `get_execution_plan` reads the
fixed confirmation, while `cancel_execution_plan` cancels an unsubmitted plan.
Changing requested parameters requires a new preparation and approval.

Legacy flat source/metadata/config files are not migrated or overwritten. They
cannot execute using their old lifecycle state. `list_strategies` discovers
root-level `<id>.json` names separately from current revisions, without parsing
legacy metadata or blocking bundled/current strategies. These entries have
`kind: legacy`, `status: revision_required`, `revision_id: null`, and
`code: strategy_revision_required`, with explicit resaving instructions.
`get_strategy(id)` reads the fixed `generated_strategies/<id>.py` and
`configs/<id>.yaml` locations as source/YAML recovery material; it does not follow
paths or trust lifecycle evidence from legacy metadata. Missing, unreadable,
non-UTF-8, or symlinked source/config files return `strategy_revision_invalid`;
their legacy entry remains discoverable. Restore the files or provide replacement
content to `save_strategy` with the required research input and a new identity
(omit `strategy_id`, or choose another unused ID), then validate/dry-run the newly
returned revision. The legacy namespace cannot be reused by a new save.
Validate, dry-run, and promote reject legacy entries with
`strategy_revision_required`; execution still rejects them as non-executable.
Recovery preserves the legacy files and discovery entry. If a workspace already
has a current revision under a legacy ID, discovery uses that current revision;
this does not authorize a new save to bypass namespace checks. Explicit revision
lookups never fall back to legacy files.
If a legacy entry shares a bundled ID, discovery lists both kinds and default
source lookup/execution selects bundled. Read the legacy source explicitly with
`get_strategy(strategy_id, legacy=true)` and save it under a different,
non-reserved ID; saving over bundled IDs returns `reserved_strategy_id`.
The explicit legacy query never falls back to bundled/current, and combining it
with `revision_id` returns `strategy_revision_invalid`.
Historical run records with no revision retain `revision_id: null`.

## Security Model

Validation currently uses static policy checks, including an import allowlist,
as a first layer. It rejects known unsafe imports (`os`, `sys`, `subprocess`,
`socket`, `requests`, `httpx`, `ccxt`, `shutil`, `pathlib`), dynamic execution
calls (`eval`, `exec`, `__import__`), raw `open`, and common destructive file
operations such as `unlink`, `remove`, `rmtree`, and `write_text`.
Static-policy or quality-gate errors stop validation before generated code is
imported or executed.

This is not a full process sandbox. `validate_strategy` and `dry_run_strategy`
still import and execute generated Python to check class loading, constructor
behavior, BaseStrategy inheritance, input immutability, signal values, and smoke
dataframe output.

Preparation is another code-execution boundary. MCP preparation runs under a
supervised subprocess with a 60-second budget covering imports, constructor/default
capture, generated signal-contract checks, parameter binding and manifest serialization.
`StrategyLoader.resolve_execution` and `BacktestService.capture_execution` load
generated module-level Python and create strategy instances. Generated strategies
also run 80-row and 240-row contract fixtures with the captured settings, including
when `parameters` is omitted or `null`. Optimization checks signals for the captured
base and validates every candidate's parameter binding; it does not run candidate
signals, engines or performance calculations. Module-level code can execute more than once. These actions
can have effects even when preparation fails without creating a formal job.

The supervised timeout and process cleanup bound the program checks; they do not
isolate filesystem or network access. Validation, dry-run, and direct Python/CLI
preparation continue to execute generated Python in their calling process. A
workspace path is an output location, not a security boundary.

For scheduled jobs, workers later repeat the generated execution-contract checks,
then construct and execute the strategy using the captured settings; generated
optimization candidates also undergo these checks before their trial runs.
These execution steps are separate from validation, dry-run and plan preparation.
The approval gate controls formal submission; it does not make earlier generated
code execution safe. None of these checks provides a process sandbox. Generated
strategies must therefore be reviewed as runtime code at all of these boundaries
until a dedicated sandboxed execution layer is added.

`signal_analysis` includes row count, signal distribution, missing-signal count,
transition count, active-signal ratio, and timestamp bounds. Use it to debug
strategies that technically pass static checks but produce unusable signals.
