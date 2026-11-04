# Strategy Contract

This contract explains strategy authoring, signal semantics and revision eligibility.
For preparation and approval, read [Execution Plans](execution_plans.md); for saved
execution identities and history, read [Run Artifacts](run_artifacts.md).
Generated strategies live in the user workspace; bundled strategies remain
Git-managed under `src/tradingdev/domain/strategies/bundled/`.

## Contract References

Read the reference for the task at hand. Repository links are for developers;
MCP clients use the listed tools, which do not require access to server files.
`get_strategy_contract` returns one complete section and advertises the others in
`available_sections`. Unknown sections or unavailable required source fail;
restore the reference before authoring rather than guessing a partial contract.

| Task | Maintained definition | MCP entry |
| --- | --- | --- |
| Implement a strategy | [BaseStrategy](../src/tradingdev/domain/strategies/base.py), including optional hooks | `get_strategy_contract(section="core")` |
| Select YAML fields, types and defaults | [BacktestRunConfig](../src/tradingdev/domain/backtest/schemas.py), including imported data and validation models | `get_strategy_contract(section="configuration")` |
| Supply research choices and simulation assumptions | [ResearchSpec](../src/tradingdev/domain/research.py) and [research_policy_payload](../src/tradingdev/domain/research_policy.py) | `get_strategy_contract(section="research")` |
| Start from a Python/YAML pair | [strategy_contract_payload](../src/tradingdev/domain/strategies/templates.py) | `get_strategy_contract(section="example")` |
| Check permitted imports and calls | [static_policy_payload](../src/tradingdev/domain/strategies/validator.py) | `get_strategy_contract(section="core").content.static_policy` |
| Select an installed indicator | [indicator_reference](../src/tradingdev/domain/indicators/reference.py) | `get_strategy_contract(section="indicators")`, then pass `indicator` |

Schemas describe shape and validation, not proof that a strategy fulfills a
natural-language request. The following sections explain semantics that field
definitions alone cannot establish.

## Python Contract

Generated code must:

- inherit `tradingdev.domain.strategies.base.BaseStrategy`;
- expose a constructor that can be called with YAML `strategy.parameters`;
- accept optional `backtest_engine` or `parallel_config` when it needs application
  execution context;
- implement `generate_signals(df)` and return a new pandas DataFrame;
- preserve the input DataFrame without mutation;
- preserve every input row in order, the exact index (including its metadata),
  and the original timestamp/OHLCV columns; use separate columns for features;
- return uniquely named columns and a `signal` value on every row: finite real
  `1`, `-1`, or `0`; warm-up rows use zero, not missing values or dropped rows;
- avoid network, subprocess, destructive filesystem, dynamic import, `eval`, and
  `exec`.

The optional `rolling_trace()` hook defaults to `None`. A returned mapping is
saved as an independent snapshot for each scored frame; it must describe that
frame's actual signals. Use the [ML retraining contract](ml_research.md#rolling-retraining-schemes)
for schedule semantics and [performance artifacts](performance_artifacts.md) for
the payload and saved locations.

The shared frame checker enforces these structural rules on synthetic fixtures
and the actual frames used by simple backtests, fixed comparisons, optimization
trials and both sides of walk-forward folds. Invalid frames fail before their
engine call. Execution supplies a copy so ordinary DataFrame mutation by a failed
candidate cannot alter the captured market data for later candidates. Indicator
columns may still contain warm-up NaNs. These checks do not prove the absence of
future-data use, agreement with a prose strategy, or safe Python execution.

### Signal and Execution Semantics

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

The generated example is long-only and holds while finite fast SMA exceeds slow
SMA, otherwise flat, including equality and warm-up. Adapt its direction, holding
and exit rules to the user's idea rather than inheriting the example's intent.

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

Volume's `monthly_max_loss` measures the change in cumulative P&L equity during
each UTC calendar month, including costs already charged and unrealised P&L
changes. The baseline is zero for the first observed month, then the last close
equity of the previous observed month. Missing months are not filled; price gaps
and changes in carried positions belong to the next observed month. Closing or
reversing a carried position does not charge its prior-month P&L or entry costs
again, while trade records still retain the entire trade's P&L and costs.
The breaker checks after a stop exit before re-entry, and after ordinary trading
at the bar's close. A loss reaching the threshold closes any remaining position
at that close and pauses trading until the next observed month. Exit costs can
increase the final loss beyond the threshold; this is not a guaranteed loss cap.
Without timestamps, the monthly breaker is unavailable.

The confirmation text, chat summary and HTML show the same backend-resolved
execution assumptions. They describe simulator policy, not proof that strategy
code or an ML model implements the requested idea, avoids look-ahead, or passed
preflight checks; completed checks are recorded separately.

### Constructor Binding

Validation, dry-run, backtest and walk-forward use the same constructor binding:
YAML parameters become keyword arguments. Missing required parameters and names
the constructor cannot accept fail validation. `backtest_engine` and
`parallel_config` are injected by the application and cannot be overridden in
YAML. Revised source and base configuration require a new immutable revision;
see [Lifecycle](#lifecycle).

Execution submission captures every declared constructor default, excluding the
injected execution context, as an explicit keyword value in
`manifest.strategy_execution`. The saved revision YAML remains unchanged;
`manifest.config.strategy.parameters` may expand to the effective captured
parameters for an MCP experiment. Defaults must be finite JSON values;
unsupported defaults require explicit serializable parameters or a revised
strategy. Generated code must express configurable values as constructor
parameters rather than derive hidden defaults from environment state. This
captures declared arguments, not arbitrary Python behavior or dependency versions.

### Indicators and Code Quality

Discover the installed `tradingdev.domain.indicators` catalog, then request the
selected indicator for its signature, named result fields and transitive local
checks. Bundled strategies and feature engineering use this facade instead of
calling TA-Lib directly. Missing source is an error to resolve before authoring;
the catalog does not recommend profitable parameters.

The facade preserves TA-Lib's native warm-up NaNs and NaN propagation, including
NaNs that can extend to the end of the output after an interior missing value.
Short inputs therefore remain unready for indicators whose full lookback has
not elapsed. Keep signals flat until every value required by the signal is
finite; do not backfill indicators from future rows or substitute zero for
unavailable indicator values. OHLC inputs must have identical indexes and
lengths, and infinite input values are rejected. Empty inputs produce empty
float64 outputs with the same index.

Numerical limits and warm-up details are available in each selected API reference,
derived from its implementation rather than a second handwritten parameter table.
Generated strategies may use the direct TA-Lib Function API when permitted by the
current import policy, but must manage dtype, tuple outputs and index restoration
themselves. Prefer the facade's named result fields; they are not tuples. For old
or unknown APIs, discover the installed catalog rather than guessing replacement
names. Source/type checks alone do not establish numerical or trading correctness.

Generated code must pass the repository's Ruff and strict Mypy policy in
`pyproject.toml`, also shipped in the wheel; caller-local settings cannot relax it.
Mypy follows imports silently, but unknown imports and missing annotations remain
errors. Ruff treats `tradingdev` as first-party: separate its imports from NumPy,
pandas and TA-Lib with a blank line.

### Randomness

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

The bundled catalog reads registered YAML capabilities without importing strategy
code. Caller YAML and overrides cannot replace them. Generated and legacy sources
remain `unknown`, even after validation or when they define `fit()`. Read
`get_strategy` for a selected identity's configuration; discovery pages are compact.

Simple backtest and optimization reject training-required strategies before data
loading or job creation with `strategy_requires_training`. MCP includes capabilities
and guidance toward `prepare_walk_forward`; satisfy its `validation` prerequisites,
prepare a replacement and obtain fresh confirmation. Removing `validation` or
calling the separate ML `train_model` tool does not train this strategy instance.
Registered declarations are maintained in the [bundled catalog](../src/tradingdev/domain/strategies/catalog.py);
strategy-specific readiness and fallback behavior belong in each strategy's guide.

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

An empty scored frame produces no signals. `RollingRetrainer` and the SafetyVolume
Risk Gate return before validating either frame; Quantile validates both time
identities before returning. For Quantile's readiness and unavailable-feature
behavior, read its
[signal logic](strategies/quantile_strategy.md#signal-logic).

When implementing or debugging Quantile alignment and model output, read
[`QuantileStrategy.generate_signals`](../src/tradingdev/domain/strategies/bundled/quantile_strategy/strategy.py)
and [`QuantileFeatureEngineer.transform`](../src/tradingdev/domain/ml/features/quantile_features.py).
Features retain surviving input index labels; callers needing positional mapping
supply a positional index. Model output requires one finite four-class vector
per scored feature row, normalized to float64; malformed output fails rather than
being treated as a feature gap.

When warm-up or immature horizon labels leave a rolling training slice with no
usable observations, scoring retains the fitted model until observations mature.
The optional `rolling_gen` engine is stricter during overlapping replay: it
discards the initial model and waits for a successful refit on preceding rows.
If none becomes available it fails explicitly. Full-fold parameter selection
still makes training replay an in-sample diagnostic; see
[rolling retraining](ml_research.md#rolling-retraining-schemes).

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

Use the shared policy reference above for current omitted capital and cost values.
They are simulation assumptions, not live fees: capital is in quote currency and
costs are decimal rates. Explicit valid numbers take priority, including zero
costs; strings and booleans are rejected. Volume mode has no capital-return basis:
omitted capital stays `null`, and explicit capital does not change its semantics.
Strategy parameters with similar names are not synchronized with execution costs.

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
select Binance or its futures market. Source capabilities come from `list_data_sources` without downloading data.
A null timeframe list means no exhaustive local list is advertised, not universal
support. Preparation rejects unknown sources and choices outside an explicit
local timeframe/market-type list; actual availability is checked during execution.
The market requirement's symbol/timeframe must match the backtest request. Direct
CLI/Python preparation rejects contradictions; MCP overrides update both together. Bundled YAML includes its source
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
      # Discover supported sources and market types with list_data_sources.
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

Use `periods_per_year` for observed UTC daily returns, independent of bar timeframe;
omitting it in a direct config leaves annualized metrics unavailable. The research
resolver instead supplies or explicitly disables it under its policy. Annual
`risk_free_rate` and `required_return` are decimal rates. A factor cannot make
weekly or monthly bars into daily observations; volume mode has no capital returns.
Definitions, units, sampling and unavailable-value semantics are maintained in
[Performance Artifacts](performance_artifacts.md#metric-interpretation).
Read `get_metric_catalog` for current definitions and optimization directions,
or `get_run_metrics` for saved values and their definition/settings snapshots.
Summary omission does not mean a metric was not computed; missing is not zero.

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

Feature values join market bars on exact timestamps. Earlier matched values may
fill subsequent gaps, but leading gaps remain missing rather than using later
observations. Strategies must handle these missing values without backfilling
from the future; unmatched feature timestamps do not seed the fill.

For automatically managed DVOL, use `source: deribit` and omit both `path` and
`raw_path`. Exact UTC query bounds and acquisition time determine reuse; changing
the research period cannot reuse an unqueried range. A `raw_path` requires an
explicit `path`. Existing explicit local files retain the join behavior above
and do not automatically refresh when the period changes. See
[Automatic DVOL Queries](run_artifacts.md#automatic-dvol-queries).

### Strategy Walk-Forward Settings

The top-level YAML `validation` mapping configures strategy backtests through
`WalkForwardConfig` and `WalkForwardValidator`. It is separate from the ML
[`train_model(validation=...)` contract](ml_research.md#fold-validation-and-signal-interpretation):
it does not select an ML scheme by name or infer a label horizon from an ML dataset.

Only MCP `prepare_walk_forward` accepts optional
`validation_config: WalkForwardConfig`. When supplied, it replaces the complete
validation mapping for that execution, using schema defaults for omitted settings;
it does not merge with the selected configuration. Omission or `null` preserves
the selected validation settings, including a bundled walk-forward template where
applicable. `prepare_backtest` and `prepare_optimization` reject this argument.
The effective mapping is captured in the manifest without modifying the source
revision, its validation evidence or current pointer.

[WalkForwardConfig](../src/tradingdev/domain/validation/config.py) and the generated
`configuration` contract supply accepted fields and defaults. Each mode accepts
only its own fields; unused settings, even null/default values, are rejected.

| Mode | Window semantics |
| --- | --- |
| `automatic` | Rolling or expanding windows from the loaded row count. |
| `explicit` | One training/test pair of inclusive UTC calendar days. |
| `purged` | Anchored expanding windows with a label-horizon purge and optional gap. |

`target_metric` controls fold logging/summary, not a strategy training objective.
In automatic and explicit modes, zero `label_horizon` disables the overlap audit;
a positive value declares how many source rows a training label looks ahead.
New configurations require an explicit mode; the removed `purged_split` input is
not accepted. Historical manifests remain readable without rewriting their settings
or hashes, but execution requires valid new settings.

Automatic `train_ratio` means the initial training window divided by that window
plus one test window, before integer rounding. For `n` rows, `k` folds and ratio
`r`, the initial training size is `floor(n * r / (r + k * (1 - r)))`, evaluated
exactly from the decimal ratio saved in JSON before flooring. The remaining
rows form `k` consecutive, nonoverlapping test windows through the last loaded
row; boundaries divide those rows proportionally with integer floor, so test
sizes differ by at most one. Rolling windows retain the initial training size;
expanding windows retain training start at row zero. Each training window ends
immediately before its test window. If any training or test window would be
empty, splitting fails rather than dropping folds or unused tail rows.
For 100 rows, 5 folds and ratio 0.8, training initially uses rows `[0, 44)` and
tests cover `[44, 55)`, `[55, 66)`, `[66, 77)`, `[77, 88)`, `[88, 100)`.
Re-running automatic configurations may change results compared with older
versions that omitted the tail or rounded an integer boundary incorrectly;
saved historical results are not rewritten.

Explicit dates are converted to UTC (naive dates mean UTC), then their time of day
is discarded. Their UTC days must satisfy
`train_start <= train_end < test_start <= test_end`. All four dates are required.
The complete UTC days must fit inside the requested research period; preparation
rejects rather than clips or extends them. Bundled examples therefore end their
backtest period at `23:59:59.999999`; a date-only research end still means midnight.
The loaded frame must have ascending timestamps, nonempty windows and no shared
train/test bars. Invalid data-dependent windows raise `WalkForwardSplitError`.

A positive horizon audits all selected windows before fitting or scoring any fold.
Overlapping labels raise `WalkForwardLeakageError`; automatic and explicit modes
check without changing their windows. Adjacent windows with forward labels will
therefore fail. To remove overlapping training labels automatically:

```yaml
validation:
  mode: purged
  n_splits: 3
  label_horizon: 5
  embargo_bars: 2
  target_metric: "sharpe_ratio"
```

Purged mode divides the loaded rows into `n_splits + 1` contiguous near-equal blocks.
The first is training-only; later blocks are tested once, with all preceding blocks
as training history after removing `label_horizon + embargo_bars` rows immediately
before the test window. It rejects explicit dates, `train_ratio` and `expanding`.
The backtest's data bounds still determine the supplied history. Insufficient data,
too many blocks or a gap leaving no training rows fail during formal execution.
Schema validation cannot establish data-dependent coverage.

Confirmation preparation checks code and configuration without loading historical
data, building data-dependent folds, training or backtesting. The confirmation
shows only the selected mode's effective settings. After approval, formal execution
performs the complete pre-fold audit. Zero horizon is an audit opt-out, not proof
that labels are safe. The audit checks row-window geometry, not arbitrary feature
code or independently constructed labels. Declare the actual forward horizon.

This differs from the ML tool's
[timestamp/label-interval validation](ml_research.md#fold-validation-and-signal-interpretation).

## ML Research Contract

Standalone ML research has its own [input, validation and storage contract](ml_research.md).
A saved model is not an executable strategy revision; strategy training
prerequisites remain in [Training Before Scoring](#training-before-scoring).

## Lifecycle

Use this section for revision ownership and execution eligibility. The execution
plan contract owns preparation and approval; storage details belong to run artifacts.

### Revision Identity and Eligibility

1. `save_strategy` creates an immutable source/base-YAML revision in `draft` state.
   Omit `strategy_id` for automatic naming. An explicit ID without
   `expected_revision_id` also means creation and must use an unused namespace.
   Revising requires both IDs and a matching current revision under the lifecycle
   lock. Even identical content creates a new revision without validation evidence.
2. `validate_strategy` checks syntax, static restrictions, Ruff, Mypy, inheritance,
   constructor/configuration and the signal contract on a short fixture.
3. `dry_run_strategy` requires `validated`, repeats the signal contract on a longer
   fixture and marks passing revisions `runnable`. Inspect diagnostics,
   `effective_config` and `signal_analysis`; synthetic success does not prove
   trading intent, sufficient historical warm-up or future-data safety.
4. `promote_strategy` optionally marks a runnable revision promoted. Execution
   accepts runnable/promoted generated revisions and promoted bundled strategies.

For exact inputs, responses and diagnostic fields, read the MCP schemas or
[strategy response models](../src/tradingdev/app/contracts/strategy.py).
[StrategyService](../src/tradingdev/app/strategy_service.py) owns save, validation,
promotion and executable-state checks. Static/quality failures stop checks before
generated Python executes; runtime checks have the [security boundary](#security-model).

Names are optional and separate from technical identity. `display_name` and
YAML `strategy.display_name` must agree when both are given. If only one is given,
that name is used; if neither is given, the first outline line supplies the name,
limited to 80 characters. It is part of the base config hash, so renaming requires
a revision. Presentation titles are independent.
`request_summary` is derived from the outline, not a save input.

Creation never overwrites bundled IDs, legacy files, incomplete directories or
existing revisions. Collisions fail without changing current; anonymous allocation
retries under the lock. A stale expected revision returns `strategy_revision_conflict`.
Failed current-pointer publication may leave a complete unreferenced revision;
publication and pointer replacement are separate filesystem operations.

Use the returned `revision_id` for all subsequent operations. Except for explicit
inheritance and revision saves, omission selects current once per operation.
Checks record evidence on that selected revision even if current later changes.
A new draft cannot inherit or revoke another revision's evidence. Executability
requires matching source/base-config hashes and successful validation plus dry-run.
Direct edits invalidate the revision; repair through a new save. Explicit unknown
revisions never fall back to current.

`BacktestService` enforces eligibility again for CLI and workers, including the
selected source path. Jobs and results stay bound to the chosen revision after
current changes. Bundled strategies are Git-managed with `revision_id: null`.

### Execution and Parameter Experiments

For formal execution, save research evidence, prepare the requested plan, present
its summary and complete HTML, then wait for explicit user approval in a later
turn before submission. Preparation itself executes generated code but does not
load market history, train or run a backtest. The complete workflow, preflight
coverage and failure recovery live in [Execution Plans](execution_plans.md).
Evidence fields and authoring guidance are available through
`get_strategy_contract(section="evidence")`; binding and capacity semantics live
in [Research Evidence Records](run_artifacts.md#research-evidence-records).

Parameter experiments reuse a runnable revision without changing its base YAML,
source, evidence or current pointer. Generated execution may change
`strategy.parameters`; every other saved strategy declaration, including
description, version and fit settings, remains revision-bound. Source path and
execution-managed source hash receive separate integrity checks. Changing a
revision-bound declaration requires a new save and checks.

Effective constructor defaults and parameter overrides are fixed in the execution
manifest. Its [override and hash rules](run_artifacts.md#execution-manifest) explain
recursive merging, omitted/null/empty mappings, CLI settings and optimization
candidates. Each generated execution checks its effective settings on short and
long fixtures, including when no parameters are supplied; failure does not change
the source revision's lifecycle. Static checks remain bound to verified source.

Use [fixed comparison](run_artifacts.md#fixed-comparison-evidence) for proposed
alternatives sharing one research/data context. Adaptive optimization requires its
own explicit request; neither workflow permits rewriting revision evidence.
Direct Python override helpers remove research provenance from their execution
copy; [provenance rules](run_artifacts.md#research-settings-and-provenance) explain
manual YAML edits and missing origins.

### Draft Cleanup

`cleanup_strategy_drafts` previews by default. Apply requires explicit nonempty
revision IDs and authorization for those deletions. Only intact, non-current,
unreferenced drafts are eligible; validated/runnable/promoted revisions remain.
All jobs, including failed/cancelled jobs, runs and orphaned manifests protect
referenced sources. Malformed or ambiguous history blocks cleanup; an old record
without a revision blocks unless its config establishes this strategy's flat
legacy source. No background cleanup, deduplication or migration occurs.

Apply rechecks under the lifecycle lock; a preview can become protected. Integrity
failures, symlinks or unexpected files prevent deletion. An apply can delete some
eligible revisions while reporting failure for others; filesystem deletion is not
transactional and I/O failure may leave a partial directory. Missing requested
revisions are reported explicitly, making retries safe. An absent strategy fails
without creating its lock; disappearance while acquiring the lock is rechecked.
Read each outcome and reason, not only the overall success flag. Exact response
fields and refusal codes are in [StrategyCleanupService](../src/tradingdev/app/strategy_cleanup_service.py)
and its [response models](../src/tradingdev/app/contracts/strategy_cleanup.py).

### Legacy Recovery

Flat source/metadata/config files are preserved and discovered as legacy entries
requiring a new revision; their old lifecycle evidence cannot authorize execution.
`get_strategy` reads fixed legacy source/YAML paths as recovery material, without
following metadata paths. Missing, unreadable, non-UTF-8 or symlinked files fail
explicitly while the entry remains discoverable. Restore readable files or supply
replacement content, save under a new unused ID with required research choices,
then validate and dry-run. Recovery does not reuse the reserved legacy namespace
or delete its files.

A current revision takes discovery precedence over a same-ID legacy entry. A
bundled/legacy collision is listed separately but defaults to bundled lookup and
execution; use `get_strategy(strategy_id, legacy=true)` to recover the legacy
source under another ID. Explicit legacy lookup never falls back to another kind
and cannot be combined with `revision_id`. Explicit revision lookups never fall
back to legacy files. Historical runs without revisions retain `revision_id: null`.

## Security Model

The [static policy reference](#contract-references) supplies the enforced import
and call restrictions. Static-policy or quality failures stop validation before
generated code is imported or executed.

This is not a full process sandbox. `validate_strategy` and `dry_run_strategy`
import and execute generated Python in a supervised subprocess to check class
loading, constructor behavior, BaseStrategy inheritance, input immutability,
signal values, and smoke dataframe output. Each runtime check has a 60-second
budget, excluding earlier static/quality checks and subsequent process cleanup;
direct application callers may set `StrategyService(contract_timeout_seconds=...)`.
An exception, abnormal exit such as `SystemExit`, or timeout produces failure
diagnostics instead of ending or indefinitely blocking the calling service.
Successful evidence must match the selected revision and fixture request.
Temporary files are removed only after the supervisor's owned process group stops;
unverified cleanup retains the directory and reports its path. Cleanup failure
never produces passing validation/dry-run evidence. As with MCP preparation, this
POSIX process-group guarantee excludes descendants that create a separate session
and external forced termination of the supervisor; it is not an OS isolation boundary.

Preparation is another code-execution boundary: module imports, constructor/default
capture and synthetic signal checks can have effects even when no formal job is
created. Module-level code may run more than once. MCP preparation uses the
supervised budget and cleanup described in [Execution Plans](execution_plans.md).
Direct Python/CLI preparation executes in the calling process. Workers later
repeat generated execution-contract checks, including optimization candidates,
before their actual runs.

Supervision does not isolate filesystem or network access, and a workspace is
only an output location. User approval controls formal submission; it does not
make earlier code execution safe. Review generated Python as runtime code at
every boundary until a dedicated isolation layer exists.
