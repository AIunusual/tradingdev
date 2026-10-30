# Performance Artifacts

Saved metric definitions, observation streams and integrity rules. Use
[Run Artifacts](run_artifacts.md#mcp-lookup) for discovery and query entry points,
and [Reports](reports.md) for derived HTML presentation.

## Performance Scopes and Provenance

`performance.json` schema version 1 records `run_id`, `manifest_hash`,
`default_scope`, `scopes`, and `definitions`. Each scope contains `values` and
`metadata`, its engine `mode`, `kind`, `split`, optional fold/trial index, and
parameter values. Definition snapshots describe unit, provider, meaning,
applicability, summary selection and optimization direction. They are saved with
the result; queries do not substitute the current catalog.

| Run type | Scopes | Default |
| --- | --- | --- |
| Backtest | `full` | `full` |
| Walk-forward | `fold/<index>/train`, `fold/<index>/test`, `test_summary` | `test_summary` |
| Optimization | `trial/<index>/train`, `test` | `test` |

New backtest and walk-forward scopes record the manifest's fixed, effective
constructor parameters, including defaults, in `parameters`. This includes
`test_summary`: its parameters describe the initial execution settings, not an
aggregation of learned fold settings. Generated strategies use their constructor
arguments; bundled strategies use the constructor's `config` object. Actual
values reported by `strategy.get_parameters()` are saved separately in
`metadata.strategy_parameters` for the full backtest or each train/test fold.
Walk-forward captures each fold's values independently after fitting; later
folds cannot overwrite earlier snapshots. The summary retains those values in
`metadata.fold_metadata`.

A bundled ML strategy that enables its rolling switch also reports
`metadata.rolling_trace`: the rolls it actually ran, each with its training
window, labelled rows, whether a model was fitted, and how many bars it scored.
It is saved beside `strategy_parameters` for a full backtest, for each
optimization training trial, and for the optimization's selected `test`
evaluation; it is absent when the switch is off, so other strategies' metadata is
unchanged. Walk-forward folds record `metadata.strategy_parameters` but not the
trace: fold capture is not implemented.

| Field | Meaning |
| --- | --- |
| `scheme` | The declared scheme name. |
| `rtype` | Effective window shape: `ROLL_SD` slides, `ROLL_EX` expands. |
| `step` | Effective bars between retrains, after defaults were resolved. |
| `window_size` | Effective sliding length, or null for an expanding window. |
| `min_train_bars` | Labelled rows a window must offer before a refit is attempted. |
| `predict_lookback` | Bars of context given to each scored bar. |
| `label_horizon` | Bars a label reads ahead; rows past it are dropped before fitting. |
| `bars` | Bars to score after rows overlapping the training history are removed. |
| `bars_scored` | Bars it actually scored; missing features leave the remainder. |
| `unscored_bars` | Positions that could not be scored; empty when every bar was. |
| `rolls` | One entry per roll, in order. |

Each `rolls` entry carries `index`; `retrain_bar`, its position in the scored
frame where the model was fitted; `train_start` and `train_end`, half-open
positions in the concatenated history-plus-scored frame, ending at the bar being
scored; `test_start` and `test_end`, the interval that model was assigned to
score; `n_train_rows`, the labelled rows the window offered; `trained`, false when
the window was too small and the previous model was kept; and `bars_scored`, the
subset of its interval that was actually scored. Positions are indices in the
concatenated frame rather than timestamps, so a bar's model is found by the roll
whose interval contains it, unless the bar appears in `unscored_bars`.

Optimization keeps its existing meaning: `parameters` contains the trial's
searched values (or the selected values for `test`), while
`metadata.strategy_parameters` contains the strategy's complete reported values.
Previously published artifacts are read unchanged. In older artifacts, full and
summary parameters may be empty and fold parameters may contain fitted values;
queries do not infer missing values from current strategy files.

Indices are zero-based. Optimization's `selected_train_scope` identifies the
winning training trial. A `backtest` scope contains scalar values; the
`fold_summary` scope contains `mean`, `std`, `min`, `max` and `valid_count` per
metric. Fold summaries are descriptive distributions, not portfolio returns over
the concatenated test periods, and have no fabricated observation series.

Metadata records provider versions, annualization and rate settings, daily return
sampling, bar drawdown sampling, UTC calendar aggregation, cost model, execution
context, and unavailable-value reasons. Examples include `not_applicable`,
`missing_annualization`, `missing_timestamps`, `no_trades`, `zero_denominator`,
`unbounded`, `unsupported_daily_sampling` (bars coarser than daily), and
`unknown_bar_frequency`. Numeric JSON values remain finite; unavailable values
use `null`.

New runs use QuantStats for return/risk statistics and retain VectorBT's
closed-trade statistics. Sharpe uses the configured risk-free rate and Sortino
uses the separately configured required return, both converted to an observed-day
rate before calling QuantStats. Negative rates retain the same interpretation.
The `excess_return_adjustment` setting records this conversion. QuantStats's
automatic price detection cannot accept some valid nonnegative returns above
100%, and its drawdown interface requires timestamps. Narrow numerical fallbacks
preserve the metric definitions in these cases; `metadata.calculation_fallbacks`
records each affected metric's actual `provider` and `reason`, with its package
version in `metadata.providers`. Missing dates are never fabricated for these
calculations. No observed drawdown makes Calmar unavailable, and positive returns
with no downside make Sortino unbounded, rather than a finite zero. Historical
Empyrical results retain their original saved definitions, providers and values.

Annualized metrics, `daily_max_drawdown`, and daily/monthly PnL statistics require
a declared bar frequency of one day or finer. Coarser or unrecognized frequencies
retain bar observations, total return/PnL, bar drawdown, observed date/month counts
and trade statistics. Their `return_sampling` and `calmar_drawdown_sampling`
settings are `unavailable`, and converted per-period rates are `null`. Frequency
validation does not infer bar duration from missing dates or synthesize daily
equity. Existing saved results remain snapshots; this correction does not
recalculate previously published artifacts.

`observations.json` has the same run and manifest identity. Backtest scopes retain
initial capital, equity, per-bar returns, UTC ISO timestamps and normalized trade
records. Volume returns and missing timestamps remain `null`. Available equity,
return and timestamp arrays align by bar; trade indices refer to that scope's
arrays. Open marks are separate from executed exits. Raw observations are readable
through artifact lookup without Python pickle.

New signal-mode scopes additionally save `execution_records` and `account_history`.
Each execution record is one native VectorBT order attempt, including filled,
ignored and rejected outcomes. A reversal may be one fill closing a position and
opening the opposite position; it is not split into invented orders. Records keep
bar index/time, native IDs, requested size/type/direction/price/costs, actual filled
size/price/fees, original OHLC and cash/position/free-cash/debt/equity before and
after the attempt. An unlimited requested size is represented by a finite/null
value plus an explicit kind instead of emitting JSON Infinity. Unfilled attempts
never carry executed order details. Timestamps identify the bar, not an exact
intrabar fill or exchange order-book quote.

Before/after equity is cash plus signed position valued at the saved request price
before slippage. VectorBT's default log value is not updated after every fill;
the new records do not enable `update_value` or change simulation sizing to obtain
different log values. End-of-bar `account_history` comes from portfolio accessors
and contains cash, free cash, signed position, asset value, close price and equity.
These use VectorBT generic accounting, not exchange wallet, perpetual-contract
margin, liquidation or funding accounting. A negative position denotes short
base-asset quantity; it is not a stock share count. Market fields are the original
bar, so a daily record cannot supply intraday bid/ask prices.

The metadata's `observations` block declares availability and valuation/time
semantics. Old JSON missing these optional fields loads as `null` (`not_recorded`);
volume mode remains `null` (`unsupported_volume_accounting`). New signal runs with
no attempts save `[]`, meaning recorded but empty. No historical field is guessed
from paired trades. Filled transitions are checked against native orders and
accounting identities, and saved account/bar/timestamp/equity arrays must align.

`get_run_metrics` defaults to the run's saved default scope; omit `metric_ids` to
read that scope in full. Unknown scopes and metric IDs produce structured errors
with discovery information. Summary omission never affects calculation or storage.
`get_run`, `get_run_metrics`, and each `compare_runs` row also return
`display_metrics` alongside unchanged raw `metrics`. Display values use each
run's saved definition: fractions are percentages with two decimal places and a
`%` suffix (`493.045775` becomes `49304.58%`); other units remain numeric strings
in their saved units. Integers remain exact; other numbers use two decimal places.
Use these rounded values for presentation and raw metrics for calculations.
Fold statistics retain their shape and numeric, unscaled `valid_count`; nulls
remain null with unavailable reasons in metadata. Metric selection also selects
display values. Without trustworthy saved definitions, including legacy runs,
the display mapping is empty; current catalog units are never substituted.
Comparisons retain the requested values and report comparability and reasons;
differences in mode, definitions, provider versions, sampling, annualization,
capital or execution context cannot silently become equivalent rankings.

Detailed reads verify supported schema, run/manifest identity and recorded
artifact digest. Missing or corrupted artifacts are errors rather than a request
to rerun a strategy. Publication validates and encodes data before writing;
existing different content is not overwritten. Retrying a background job's same
result also checks all registered artifact files and hashes before reporting
success; missing or corrupted files are explicit errors. File publication and SQLite writes
are not a cross-system transaction. A per-run `.result-publication` marker prevents
concurrent publishers. Handled exceptions restore the prior files and remove new
run/artifact records. An externally terminated process can leave the marker;
queries report the publication as busy instead of treating incomplete output as
legacy data. Ordinary readers do not delete the marker or repair stored data.
Comparison recovery is an explicit exception: after the caller confirms that the
owning worker has stopped, `ComparisonService.reconcile` may use
`RunPublisher.verify_published` to recover an already complete child publication.
It verifies the approved parent/child identity, all registered artifact paths and
digests, and the complete performance/observations pair before removing a leftover
marker and recording success. Partial or corrupt output remains unavailable;
recovery does not rerun strategy code or load a pipeline pickle. The worker's own
synchronous failure handler can also reconcile through this verification path;
external recovery must first confirm worker cleanup.

Historical runs without performance artifacts remain readable through ordinary
run lookup, marked as lacking detailed provenance. `get_run_metrics` reports
`performance_artifact_unavailable`; reads never migrate, unpickle or recompute
those results. Historical values are not asserted to follow the new definitions.
