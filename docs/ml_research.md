# ML Research

Input, validation and persistence contracts for synchronous ML research.
For strategy execution and training prerequisites, use the
[strategy contract](strategy_contract.md#training-before-scoring).
Workspace locations are listed in [Run Artifacts](run_artifacts.md#workspace-layout).

## ML Research Contract

The ML MCP tools are a separate synchronous research interface. They select
registered feature builders and estimators without generating a strategy,
preparing an execution plan or starting a backtest worker. A saved model is not
a runnable strategy revision and does not bypass the Python signal contract or
strategy lifecycle. User workflow is documented in the
[README](../README.md#研究機器學習訊號); file identities and persistence rules are in
[ML Research Artifacts](ml_research.md#ml-research-artifacts).

`get_ml_contract` returns schema version 1 discovery information: named feature
sets with ordered columns, model kinds with default parameters, validation
schemes, rolling schemes with their window type and default cadence, processor
types, IC methods, quantile limits, calendar modes, missing policies and rankable
metrics. Use these returned names rather than constructing arbitrary model
classes. `processor_types` advertises domain building blocks
only: `build_dataset`, `build_panel` and `train_model` do not accept or execute a
processor pipeline. In particular, training does not implicitly fit a scaler
because its name appears in discovery.

### Dataset and Model Inputs

`build_dataset` and `build_panel` select cached OHLCV by source, effective
`market_type`, exact symbol and timeframe. Binance Vision defaults to `futures/um`
and also accepts `spot`; providers that ignore market type store it as `null`.
The effective source/market is returned in the build summary, saved in dataset
metadata and participates in its identity.
Each requested year slice requires a covering successful query acquired at least
seven days after that slice ended. Missing coverage, unsettled acquisition or an
empty selected period fails; success does not establish complete bar coverage.
Use `ensure_data` with the same source/market when a download is needed. For recent
intervals, wait until the seven-day boundary before querying again: an immediate
retry cannot make recent observations reusable. Legacy unidentified flat files
are preserved but excluded. Builds never download, refresh or remove market data.
See [Market Data Cache](run_artifacts.md#market-data-cache).
The feature builder preserves rows and emits unavailable values
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

Bundled GLFT ML aligns predictions back to every original market row after
feature filtering, in both parameter search and formal signal generation.
Missing predictions prevent new entries; existing positions still age and
evaluate exits on the full price timeline. Filtering supervised training rows
does not compress the bars used to evaluate trades.

`FeatureEngineer.transform()`, `DirectionFeatureEngineer.transform()` and
`RiskFeatureEngineer.transform()` preserve
surviving source index labels; there is no index-reset option. Callers needing
row positions supply a positional index.
SafetyVolume maps optional ML directions to those source rows. XGBoost rolling
prediction skips a bar whose own features are unavailable, rather than reusing
an earlier prediction. A scored frame in which no bar can be scored fails
instead of reporting an all-flat result; the
[rolling rules](#rolling-retraining-schemes) cover the context and the failure.
Its threshold search evaluates the complete source frame,
leaves unavailable rows without entry signals, and measures cooldown in source
bars, including gaps.

SafetyVolume's rolling Risk Gate also predicts only from the current source
bar's features. Unavailable estimates never become probabilities or reuse a neighbouring bar.
The [SafetyVolume guide](strategies/safety_volume_strategy.md#layer-1-risk-gate-xgboost)
defines entry, reversal and holding behavior, and the entirely-unscored failure.
Threshold search continues to evaluate its filtered validation feature rows,
unlike Quantile's full-frame gap handling.

The XGBoost, LightGBM, CatBoost and FLAML direction model training APIs require at
least one labelled training row. A supplied validation frame must also be non-empty. Empty inputs
raise `ValueError` before model state changes or external training starts,
preserving any previously fitted model. XGBoost, LightGBM and CatBoost also reject
validation targets whose classes are absent from the training window. Their label
encoders use training targets only; validation answers are never replaced with
a known class. A rejected fit preserves the previous model, feature columns and
class mapping. Select a training window covering the needed classes before retrying.
XGBoost's lookback search requires at
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
`signal_fingerprint` hashes the complete signal in dataset row order, regardless
of `offset` or `limit`, for comparison with evaluation and leaderboard records.
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

MCP `evaluate_signal` accepts `dataset_id`, optional `model_id`, `column`,
`threshold`, `direction="above"`, `window=60`, `n_quantiles=5` and `record=False`.
Supply either `model_id` for a verified saved model or both `column` and
`threshold` for a rule. Combining `model_id` with either rule input is rejected.
Rules accept `direction="above"` (default) or `"below"`; the condition is strict,
so a value equal to the threshold produces `-1`. Direction is unused for model
evaluation. Only `record=True` saves or replaces the same candidate's evaluation.
See method families below for rule restrictions. The backend evaluates the
complete single-symbol dataset, obtains saved forward returns in the same
row order and adopts the dataset horizon; the MCP input does not accept arrays or
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

Additional performance statistics use only the holding grid, including its cash
intervals, with the benchmark on the same grid. Annualised statistics require a
verified continuous crypto calendar; annual return also requires positive capital
throughout. Unavailable values stay `null`. See the
[recorded-metric contract](ml_research.md#recorded-signal-evaluations) for the
market policy and storage rules.

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
a held-out evaluation. Results expose `producer_source` separately from a
caller-declared `model_id` label, including when no record is saved. Recording,
legacy-source handling and candidate replacement are defined in the
[artifact contract](ml_research.md#recorded-signal-evaluations).
`leaderboard` ranks only a metric with a declared direction. A signal ranking also
requires, by default, only matching measurement contexts; its optional `dataset_id`
filter selects a verified dataset version and checks that recorded data context
matches its manifest; differences in evaluation settings still require compatible
context.
`allow_parameter_axes=True` permits window, horizon and quantile-count variations
under the [parameter-surface rules](ml_research.md#recorded-signal-evaluations);
it does not make those measurements comparable. `allow_mixed=True` opts into
incompatible contexts and must not be presented as a controlled comparison. Retain its
`caveats`, skipped entries and candidate counts when interpreting the winner;
trying more candidates increases selection bias.

The same tool ranks saved backtest scopes. The metric id selects the source: a
bare name or `signal.<metric>` ranks recorded evaluations, while `run.<metric>`
ranks saved scopes and requires `runs` as a list of selection objects, for example
`runs=[{"run_id":"example","scope":"full"}]`. An omitted or null `scope` selects
that run's default. The Python library uses tuples instead, as described in
[the run-ranking contract](run_artifacts.md#ranking-saved-run-scopes).
The response reports the qualified id, so
`metric="ic_mean"` comes back as `metric="signal.ic_mean"`; the submitted spelling
is not echoed. The two argument sets are exclusive: a signal ranking refuses
`runs`, and a run ranking refuses `dataset_id`, `symbol`, `timeframe`, `horizon`,
`feature_set`, `family`, `allow_mixed=True` and `allow_parameter_axes=True`, because
one call ranks one
source. Selecting the same measurement twice is refused as `duplicate_selections`,
in two layers. Identical requested pairs are refused before anything is loaded;
differently spelled selections are compared after they load and resolve, which
catches an omitted scope that names a scalar scope already selected. A selection
that cannot be loaded is skipped before that second comparison, so two spellings of
an unloadable scope appear as two skipped rows; neither is ranked. A run row carries
`run_id`, `scope`, the trading conditions its manifest recorded, `metric_source`
and the stored value; `context`, `families` and `parameter_axes` are `null` —
present but not applicable, because a named selection has no evaluation context or
method mix to report. An empty value would read as measured-and-nothing, which is a
different claim. Every row reports `source`, and one response never mixes the two: a
signal metric and a backtest metric share no definition, so they share no column.
The run-ranking semantics — saved scope parameters and metadata, manifest-level
conditions, definition checks and fold-summary skips — are in
[the run-ranking contract](run_artifacts.md#ranking-saved-run-scopes).

MCP `leaderboard` accepts `offset=0` and `limit=20` (1–100); it does not accept
`top_k`. For recorded evaluations the complete filtered candidate set is compared
and ranked before paging; to rank saved backtest scopes instead, pass
`run.<metric>` with explicit `runs` selections, and those scopes are ordered by the
same rules. `ranked` and `skipped` each return at most `limit` entries at the same
offset; `n_ranked` and `n_skipped` describe their complete counts. Follow the
shared `next_offset` until `null`, even if one list has no entries on a page. A run
selection is named explicitly and is usually small, so its page is normally the
whole list; the source-specific response fields are described in
[the run-ranking contract](run_artifacts.md#ranking-saved-run-scopes).
On a signal ranking, mixed-context rejection reports the full context count, limited
examples and filter guidance, not every context's text.

After inspecting saved candidates, discover the `ml` section with
`get_report_sections` and pass 1–50 distinct entry IDs to
`generate_report(ml_entry_ids=[...])`. This can produce an independent ML report
or attach independent evidence to selected runs/a fixed comparison. The report
uses the recorded candidates and never reruns prediction or training. Its default
ranking metric is `ic_mean`; `ml_ranking_metric` selects another directed metric,
and `ml_allow_parameter_axes=True` explicitly enables the same surface rules.
Missing or incompatible evidence fails instead of silently dropping candidates.
Deliver the returned report link and navigation instructions. The artifact saves
the selected content even if those leaderboard entries are later replaced; it
does not prove a model-to-run relationship or persist missing time series. Full
selection, visibility and retention semantics are in the
[report contract](reports.md#composable-offline-reports).

Leaderboard rows, including skipped candidates, retain `producer_source` and the
saved training dataset binding. Use them together under the
[recorded-evaluation interpretation rules](ml_research.md#recorded-signal-evaluations).
A model label or a `full` scope alone does not establish who produced the signal,
and a confirmed producer does not establish a held-out evaluation.

Model prediction unpickles trusted Python objects. Generated-strategy import
checks do not apply to those objects, and a matching hash is not a sandbox or
proof of a trustworthy publisher. Do not load model files from an untrusted
workspace.

### Rolling Retraining Schemes

The bundled `XGBoostStrategy` and `SafetyVolumeStrategy` use one rolling
prediction core. Every scheduled refit uses only preceding, mature labelled
rows; every prediction uses the current source bar's own features. The
`rolling_gen` setting selects geometry, not a different execution engine.

With `enabled: false` (the default), automatic geometry uses the strategy's
`retrain_interval`, requires ten mature training rows and slides a training
window covering the largest of the retained initial history, ten times lookback,
and the raw history needed for ten mature observations. With `enabled: true`,
a named scheme and its overrides determine the training geometry:

```yaml
strategy:
  parameters:
    rolling_gen:
      enabled: true
      scheme: roll_sd    # or roll_ex
      step: 720          # optional; follows retrain_interval
      window_size: 7200  # roll_sd only; otherwise follows 10 x lookback
```

- `roll_sd` slides a fixed training window, clipped to available history.
- `roll_ex` keeps the first bar and expands the window with each retrain.

`get_ml_contract` advertises named schemes under `rolling_schemes`, from the
same registry used for execution. The scheme fixes the window shape;
`window_size` is invalid for an expanding scheme. Enabled unknown names or
conflicting shapes fail during configuration validation. Named schemes default
to `lookback + 10` mature labelled rows unless `min_train_bars` is supplied.
Disabled scheme overrides have no effect.

Prediction context is owned by the feature pipeline in **both** modes.
`minimum_context_rows` counts the fewest rows, including the scored bar, that
can produce every feature. `context_rows` is the preferred length, including
seed-decay margins. Direction features include RSI/MACD; risk features include
the fixed 60-bar volatility window, lagged returns and ATR/ADX smoothing.
Short lookbacks are supported without shortening the fixed feature requirements.
The preferred history reduces initialization effects but does not guarantee a
uniform output error against full-history computation. Available history can be
shorter than the preference and still score once the minimum is reached.

An explicit `predict_lookback` counts **preceding** bars; the engine adds the
scored bar. It must be at least `minimum_context_rows - 1`. An unset value uses
`context_rows - 1`, without a separate lookback-multiplier floor. Both bundled
strategy configurations validate every lookback candidate before market-data
loading; `fit()` rechecks the same rules before any model training. Direct engine
callers are validated before rolling refits or scoring.
Insufficient actual history or missing/degenerate values can still leave a bar
unavailable even with valid geometry.

The schedule, execution and saved trace consume one resolved `RollingGeometry`.
For Python inspection, use `FeatureContext.for_features(fe)` and
`resolve_geometry(scheme, lookback=..., retrain_interval=..., context=...,
history_bars=..., label_horizon=...)`, then
`enumerate_rolls(geometry, n_scored_bars=..., history_bars=...)`.
Pass `None` as the scheme for automatic geometry. History counts must describe
rows strictly preceding the scored frame, as determined by
`align_scoring_history`; the prediction engine performs that alignment itself.
These entry points live in
[rolling_gen](../src/tradingdev/domain/ml/rolling_gen/__init__.py) and
[feature context](../src/tradingdev/domain/ml/features/context.py).
Schedules describe planned attempts, not proof that a refit or prediction ran.

An initial model is available only if its entire training frame precedes the
scored frame. Overlapping replay discards it and waits for a successful refit on
preceding rows in either geometry mode. If no model becomes available during a
nonempty run, execution fails explicitly. Training replay remains an in-sample
diagnostic because full-fold parameter selection has not been undone.
A refit retains an available model when too few mature observations survive;
it never extends its window into the future. The core independently excludes
labels whose horizon reaches outside the training window and checks feature
source-row and timestamp identity before fitting or predicting.

`rolling_outcome` returns a `RollingOutcome` with class-labelled probabilities
aligned to the input scored frame's index, per-bar availability and roll
attribution. A scored distribution assigns zero to a class absent from that
fitted model. An unavailable bar has NaN in every class and a reason distinguishing
insufficient history, incomplete features and no causal model. Risk consumers
read class `1` as safe; direction consumers interpret `-1`, `0`, `1` as direction.
There is no risk probability stored under a directional column name.
Wrong prediction rows, duplicate or unexpected classes, nonfinite, out-of-range
or unnormalized probabilities fail instead of being treated as feature gaps.

A nonempty interval with no scored bars fails with a model-availability,
history or feature diagnostic. An empty interval yields no predictions. Direction
consumers emit no entry for unavailable bars; cooldown is applied **after**
prediction and never determines which bars are scored. SafetyVolume applies its
[Risk Gate](strategies/safety_volume_strategy.md#layer-1-risk-gate-xgboost)
holding policy. Neither consumer borrows a neighbouring score or substitutes a
numerical probability for an unavailable estimate.

Both geometry modes record `rolling_trace` in metric metadata. Each roll reports
its training window, mature row count, whether it fitted a model and how many
bars it scored. Windows too short for even one mature feature row are skipped
before transformation and record zero measured training rows. Unscored positions
and reason counts remain available in the trace. An unsuccessful scoring call
clears the strategy's previous trace rather than exposing stale evidence.
Saved traces are independent snapshots for each walk-forward train/test frame.
SafetyVolume's optional ML direction also uses this core and records its own
trace under `direction`; both models obey the same replay causality boundary.
These bundled strategies still require training and support walk-forward
execution; the trace hook does not add simple backtest or optimization support.
Metric queries return compact summaries and a reference to the complete saved
artifact. [Performance artifacts](performance_artifacts.md#performance-scopes-and-provenance)
defines trace fields, coordinates and paged evidence retrieval.

This is not fold validation. `validation_schemes` cut one dataset into
train/test folds to estimate the training procedure; `rolling_gen` decides how
an already fitted strategy refits and predicts bar by bar. Date anchoring is not
implemented: declaring `train_start` is rejected instead of silently ignored.

## ML Research Artifacts

ML datasets, feature matrices, trained models and signal evaluations use the
workspace directories listed in [Run Artifacts](run_artifacts.md#workspace-layout). They do not create background jobs, backtest runs
or SQLite artifact registrations. Dataset and model IDs refer to their own
stores; they are not run IDs.
Reports generated from these records are separate registered artifacts; their
publication and retention are defined in [Reports](reports.md#composable-offline-reports).

### Immutable Datasets

`MLDatasetStore` publishes schema version 2 datasets as a complete directory
containing `dataset.parquet` and `manifest.json`. The manifest records:

- The single symbol or ordered panel symbols, timeframe, requested date range,
  source, effective `market_type` (`null` when the provider ignores it) and dataset
  kind. The effective market is also returned in dataset and panel summaries.
- Feature-set name, ordered feature columns, build-time feature-code fingerprint,
  label column, label horizon and dropped-row count.
- For panels, the calendar settings, coverage summary and date count. As in the
  tool response, dropped/missing date lists each contain at most 50 dates;
  `n_dropped_dates` and each `n_missing` remain exact.
- `schema_version`, `rows`, the SHA-256 `content_hash` of the exact Parquet bytes,
  and `dataset_id`.

The Parquet frame retains `label_start` and `label_end`: the original source-bar
timestamps used to calculate each row's forward return, captured before feature
rows are dropped or a panel calendar is aligned. Panel labels are computed on
each symbol's own bars. Forward filling carries these timestamps together with
the original label; it does not turn that label into a newly observed outcome
at the filled calendar date. Fold validation refuses panels with carried
observations and requires rebuilding with intersection or dropped missing dates.

`dataset_id` is `ds_` followed by the first 16 hexadecimal characters of the
SHA-256 of the sorted JSON manifest, excluding `dataset_id` itself. Thus metadata,
feature code and the saved data bytes all participate in identity. Rebuilding
after an input or feature-code change publishes a different version rather than
replacing the previous dataset. Cache-hit status is response metadata only and
does not change dataset identity.

Publication holds `.publish.lock` across the existence check and same-directory
rename of a completed temporary directory. An existing ID is reused only after
its data and manifest pass verification and its manifest matches the proposed
one; a collision or corrupt existing version fails without replacing it.
Temporary publication directories are removed on success and failure. The lock
file remains available to coordinate later writers.

Every dataset load validates the ID syntax, schema, manifest identity, Parquet
content hash and row count before exposing a frame. Hashing and parsing use the
same bytes. Missing, incomplete or altered versions fail rather than rebuilding
silently. Earlier dataset schemas are unsupported and must be rebuilt; reads do
not migrate, delete or rewrite their files. These hashes detect inconsistent
content, not a writer able to replace both the manifest and its data.

Schema 2 alone does not establish that label intervals are present. Fold
validation additionally requires both interval columns; single-symbol
evaluation requires `label_end` and chronological `timestamp` rows to check its
holding intervals. Earlier schema 2 datasets missing the required columns must
also be rebuilt for those operations. The generic store does not infer missing
intervals or modify an old version. Validation and evaluation semantics are
defined in the [ML research contract](ml_research.md#fold-validation-and-signal-interpretation).

### Feature Matrix Cache

`ml_feature_cache/<cache_key>.parquet` stores a feature matrix, with a JSON
sidecar describing the symbol, timeframe, input-data fingerprint, feature set,
code fingerprint, feature and stored columns, row count, content hash and last
use time. Labels are constructed after the cache lookup, so changing the label
horizon can reuse the same feature computation.

The cache key includes bar values, symbol, timeframe, feature set and feature
code. The default code fingerprint hashes relative source paths and bytes from
both `domain/ml/features/` and `domain/indicators/`, registered column declarations,
and the loaded NumPy, pandas, TA-Lib Python and TA-Lib C library version strings.
It does not depend on the installation's absolute path or a Git checkout. Changes
to these sources or versions produce a different cache identity; previous files
are retained but are not reused for the new computation. Version strings do not
detect local binary patches that keep the same version, or arbitrary changes to
third-party runtime configuration. An application caller may
explicitly supply that fingerprint. Dataset manifests retain the fingerprint
used at build time, so later evaluation does not substitute the currently
installed feature code's identity.

A cache hit requires matching computation metadata and feature columns, a
matching Parquet hash, and matching stored columns and row count. Sidecars must
contain a JSON object with string identity fields and SHA-256 hash, a nonnegative
integer row count, and distinct string lists for stored and feature columns;
feature columns must be present in the stored columns. A present last-use time
must be finite and numeric. Malformed metadata, missing files, hash or shape
mismatches and invalid Parquet content are cache misses, allowing the service
to recompute and replace the disposable cache entry. The matrix is parsed from
the same bytes whose hash was verified. Recomputing does not replace source
market files or immutable dataset versions.

Each cache directory has a persistent `.feature-cache.lock` shared by processes.
It serializes complete writes and eviction, reads through their recency update,
and key/statistics snapshots, so an in-progress publication cannot be swept as
an orphan. The default lock wait is 30 seconds; expiration raises a timeout
instead of reporting a cache miss. Matrix and sidecar publication use unique
temporary files in the cache directory, with cleanup on success or failure;
cleanup errors follow the I/O policy below. The lock file is retained and excluded from retention bytes,
as are temporary files. This lock covers the disposable feature cache only.

Verification reads and cache publication propagate permission and other storage
I/O failures; these are not reported as misses. Updating recency is best effort
and logs a warning if it fails, as does an unsuccessful eviction deletion.
The default retention limits are 256 entries and 512 MiB, counting matrices and
sidecars. Successful reads update recency; writes sweep incomplete pairs and evict least
recently used entries. The just-written entry is protected, so a matrix larger
than the byte budget remains stored and logs a warning. Failed deletions may
also leave the cache above its limits. These are retention targets, not a hard
disk quota; feature-cache eviction does not delete the immutable datasets that
already contain the computed features.

### Trained Models

`models/<model_id>.json` holds schema 2 metadata; the matching `.bin` holds the
pickled model wrapper. Schema 1 records still load. Metadata records model name,
framework, kind, effective parameters, ordered features, training-row count,
available feature importance, the dataset the final artifact was fitted on and
full blob SHA-256. At build time, the ID is computed as a 16-character
SHA-256 prefix over name, parameters, ordered features and blob hash. It
identifies model content, not a training execution or saved validation result.

Save and load verify the blob hash and recompute the ID from its identity fields.
Loading also checks that the metadata ID matches the requested filename. Framework,
kind, training-row count and feature importance are not identity fields; changes
to those fields may still go undetected. Metadata is not compared with the
serialized estimator, and the ID neither authenticates publishers nor proves all
metadata unchanged. Prediction checks exact ordered dataset features and unpickles
the same bytes that passed hash verification.

Schema 2 adds `training_dataset_id`, the dataset the final artifact was fitted on.
The dataset ID is derived from the dataset manifest, so it identifies the rows, the
request that built them and the code that did it. It is deliberately absent from the
model ID: recording it must not rename a model that already exists. Legacy records
without this field read with an empty value, which stays unknown and is never inferred
from the model, its metrics or the current dataset registry. Either supported schema
preserves an explicit stored binding. An empty value is not "trained on nothing".

The field records where the artifact was fitted. It does not make any evaluation
held out; [Recorded Signal Evaluations](#recorded-signal-evaluations) states how a
leaderboard record compares it with the dataset that record scored.

Writers for one model ID hold `.<model_id>.lock`. A new blob is staged beside its
destination and atomically published; an existing blob must have the same hash and
is never truncated or replaced. Metadata is published last by atomic replacement,
so concurrent readers see complete records and can continue using an existing model
during repeated training. Temporary files are removed on success and failure; the
lock file is retained. A failed first publication removes its newly created blob
if no metadata was published. This is not a crash or power-loss transaction:
an interrupted, unpublished blob can be recovered by retrying the same content.

Because the binding is not part of the model ID, a repeated save of the same content
may update it: an existing artifact whose training dataset is unknown gains the one
now supplied, including a schema 1 record read as unknown. Once recorded it is
immutable — a save that would name a different dataset for the same content is
rejected instead of overwriting, because leaderboard rows already written kept the
old value.

The schema 1 filenames remain unchanged. `ModelArtifactStore.load_with_bytes`
returns the verified metadata and bytes together; `load` returns only metadata.
Saving no longer accepts an alternate `model_id`, and the unchecked
`load_model_bytes` helper is removed.

CatBoost removes random `model_guid` and wall-clock `train_finish_time` metadata
before serialization, avoiding identities driven by those incidental values.
This does not guarantee identical blobs across dependency versions or environments.
Pickled wrappers are tied to the training environment, not a portable model
exchange format. After a Python or ML dependency upgrade, retrain in the new
environment and use the returned model ID. Schema 1 does not record dependency
versions or enforce compatibility; a valid blob hash cannot establish that an
older pickle is loadable in the current environment.

Pickle requires trusted workspace files. Hashing is not authentication or a
sandbox; a writer able to replace metadata and blob can supply executable code.
The final artifact fits all dataset rows even when training returns fold scores;
its ID or predictions on those rows do not establish out-of-sample evidence.

### Recorded Signal Evaluations

MCP `evaluate_signal(dataset_id, model_id, record=True)` generates the complete
signal from a verified saved model and single-symbol dataset, then records its
evaluation. Without `model_id` it records a threshold rule instead, from one
feature column of that dataset. Preview pages are not persisted prediction
artifacts and are never the evaluation's input. The separate Python
`MLService.evaluate_signal` accepts arrays and requires a verified dataset when
recording; `MLService.evaluate_rule` is the rule equivalent.
Whenever `dataset_id` is supplied, including without recording, the signal must
cover all its rows, the supplied forward returns must exactly match the saved
`forward_return` values in row order, and an explicit horizon must equal the
saved horizon; omitting it adopts the saved value. The saved interval endpoints
must support non-overlapping trades on the simulation grid. Panel datasets are
refused for evaluation. Non-finite signal or return values are rejected, rather
than dropped before recording. Without `record=True`, evaluation returns its
result without a leaderboard write.

`ml_leaderboard/leaderboard.jsonl` contains one JSON record per `entry_id`, sorted
by ID. Each record preserves finite metrics, observation count, model and dataset
IDs, a display label, the candidate's saved settings, the hash of the
signal that was measured, the declared method family (`ml_model`, `rule`
or `other`) with a readable `method_name` for candidates that have no model, the
dataset its model was trained on when the artifact records one, and
an evaluation context: symbol or panel symbols,
timeframe, requested period, horizon, IC window, quantile count, feature set,
build-time code fingerprint and dataset content hash. For a model row, the fitted
parameters are copied from the model artifact, so the row still says how that
model was configured after the artifact is retrained; a rule row carries the
settings its caller declared. Other signals may also declare settings. Records
without declared settings or a loadable model artifact have empty parameters. Older records missing the new fields
read as `parameters={}` and `signal_fingerprint=""`; reads do not reconstruct them.
Empty parameters are not fitted defaults, and an empty fingerprint is not
reproduction evidence. Re-evaluate and record to obtain that evidence.
Comparison uses the saved context; changed data or feature code does not silently
inherit an old context.

`producer_source` records the actual evaluation entry point, independently of the
declared method family or model label. Its valid values come from
`ProducerSource` in `domain/ml/leaderboard.py` and the typed evaluation/leaderboard
response schemas in `app/contracts/ml.py`:

- `backend_model`: `evaluate_prediction` generated the scored sequence from the
  validated saved model artifact.
- `backend_rule`: `evaluate_rule` built the scored threshold rule from the saved
  dataset feature.
- `caller_array`: the public Python `evaluate_signal` received caller-supplied
  arrays, even if they match model predictions or declare a model/rule identity.
- `unknown`: the stored record predates source recording. Missing source fields
  read this way and remain unknown when serialized again.

Public callers cannot select a backend source flag. Invalid source values are
rejected. Reads do not infer origin from today's model registry, `model_id`,
`method_family` or `signal_fingerprint`; reproducing the same values is not proof
of who originally produced them. This field describes the service execution path,
not human authorization, authentication of externally modified workspace files,
or a held-out evaluation. It does not add a ranking eligibility rule. The existing
leaderboard `source` tag still distinguishes signal measurements from backtest runs.

`training_dataset_id` is empty when no model is named, and when the
artifact cannot be read as a valid record: it is absent (`FileNotFoundError`), or
`load` raises `ValueError`, which covers unreadable JSON, an unsupported schema
version, and an identity or hash mismatch. Those cases are recorded as unknown; any
other error propagates, so the recording fails and no entry is written. Equality with
`dataset_id` means the recorded score is in-sample only when the signal came from
that artifact — the MCP path above — and not when `MLService.evaluate_signal`
evaluated an array the caller supplied. Such a record under a declared `model_id`
says where that artifact was fitted without establishing that it produced the
signal. `producer_source` distinguishes these cases; `unknown` preserves the
historical uncertainty instead of upgrading an old record to a verified origin.
Re-running the model and matching `signal_fingerprint` shows that the recorded
values are reproducible from that model; it does not show that the original signal
came from it, because the fingerprint covers signal values only and a caller can
supply the same values.
Inequality proves nothing about the rows: a dataset identity also covers the
requested period and the build-time code fingerprint, so identical rows can carry
two identities, and genuinely different versions can still cover overlapping
periods. A training window and an evaluation window that share dates produce two
identities and no holdout. Older records are not backfilled, and a reader must not
substitute the evaluation dataset for a missing training one.

#### Method families

Not every candidate is a model. `method_family` describes the declared method,
so a hand-set rule can be ranked beside a fitted model. Use `producer_source`
separately to distinguish backend construction from a caller's claim:

- `ml_model`: a saved model. Its `parameters` are copied from the artifact, and a
  caller may not supply them.
- `rule`: a hand-set method with no fitting step. It must carry a `method_name`,
  and its `parameters` are declared by the caller, not read from an artifact.
- `other`: a signal whose construction is not declared, including every row
  written before families existed that carries no model ID. Rows that carry a
  model ID read as `ml_model`, because that ID is the caller's declaration of a
  fitted model; it is a label, not proof of what produced the signal.

The Python `MLService.evaluate_signal` array API accepts `method_family`,
`method_name` and `method_parameters`. Omitting the family uses `ml_model` when
a model ID is supplied, otherwise `other`; unknown families are rejected.
Names are trimmed and limited to 160 characters. A `rule` requires a nonblank
name. Declared parameter keys must be nonblank strings and values must be finite
numbers, excluding booleans; values are normalized to floats. Model candidates
require a model ID and reject a non-model family, a nonblank method name or
nonempty declared parameters: their settings come from the artifact. These
declarations describe the caller's claim, not verified signal provenance.
MCP accepts the narrower model-or-threshold inputs described above; it does not
accept arbitrary method declarations or signal arrays.

The family is identity, not a measurement condition, so it stays out of
`EvaluationContext`: rules and models measured on one dataset remain comparable,
which is the point of putting them in one table. `leaderboard` reports a
`families` count for the candidates its filters selected, including ones it could
not rank, and appends a caveat when more than one family is present. Family
labels do not establish selection history or out-of-sample validity.
Inspect provenance and measurement conditions; pass `family` to rank one kind.
Names and declared parameters alone do not prove reproducible construction.

Named candidates are identified by family, name and full-precision declared
parameters within the saved dataset, evaluation context and producer source.
Repeating those settings from the same source replaces that candidate; changing
a parameter creates another entry, even when rounded display names or resulting
signals match. Cross-source identity and legacy preservation follow the rules below.

The `evaluate_signal` tool without `model_id` builds a rule signal from one saved
feature column: `+1` where the column is above (or below) the threshold and `-1`
otherwise. These are signal values, not executed positions: the shared
simulator still requires `n_quantiles` prior observations and both historical
signal values, and opens positions only on the non-overlapping horizon grid.
A first switch on the final row can leave every simulated return zero. This
is signal research, not an always-invested threshold-strategy backtest. The
label, realised outcome and row bookkeeping (`forward_return`, `target`,
`timestamp`, `label_end`, `symbol`) are forbidden. Constant signals are
unsupported by the quantile evaluator; observing both values in the full sample
does not guarantee a trade. Choosing a threshold after viewing this dataset
introduces in-sample selection; the backend does not verify selection history.
A threshold sweep is stored as one
candidate per level, each carrying its level in `parameters`. It is not reported
as a parameter surface: `parameter_axes` covers only horizon, IC window and
quantile count, so the surface, its caveats and the plateau analysis built on
them do not apply to rule settings yet. Reading a sweep as a surface today would
compare separate candidates as if they were one measurement.

The recorded metrics include annualised return, maximum drawdown, Sharpe, alpha
and beta alongside the signal-quality ones. They describe the simulated
long/short book built from the predictions: costs are excluded and no order was
ever sent, so they are not backtest results. Alpha and beta regress that book
against holding the measured asset over the same intervals, which makes them a
statement about market exposure rather than about an unrelated index. Annualised
figures use only the non-overlapping decision grid, including its cash intervals.
They require a crypto-only provider and consecutive label intervals matching the
declared timeframe and horizon, using the shared 365-day crypto policy. Without
that evidence (including Yahoo, gaps or no dataset), they are `null`. Annualised
return is also `null` if compounded capital becomes nonpositive; subsequent
losses cannot revive a bankrupt book into a profitable candidate.

A metric that comes out NaN or Inf is stored as an explicit `null`, never removed
from newly written records. Three states therefore stay distinguishable: a key that is
absent was never recorded, a key holding `null` is unavailable or non-finite, and a
finite value is an ordinary measurement. This matters for parameter surfaces. If
the run at `window=20` fails, deleting its score would leave `window=10` and
`window=30` looking adjacent, and any slope or flatness fitted along that
shortened axis would be wrong. A candidate with no score is skipped when ranking,
but its `skipped` row still reports its horizon, window and quantile count, and
the response's `parameter_axes` still lists the value it was measured at.
Older evaluations dropped unavailable values: a missing key there cannot establish
whether a metric was unmeasured or unavailable. Re-evaluate and record when that
distinction is needed; reading an old record does not recover missing evidence.

Consumers must read `null` as a gap at a known coordinate. A consumer whose own
type cannot represent null must keep the coordinate and record validity beside
the value: when a number is unavoidable, write `0` as a placeholder and mark the
row invalid. `0` is never a substitute for a missing score, because zero is a
perfectly plausible IC or Sharpe, so a slope, flatness or ranking computed over
placeholder zeros is meaningless until the validity mask is applied.

`domain/ml/report.py` provides the pure projection used by `ReportService` for
explicitly selected saved ML candidates. `ml_signal_scope` produces a scope
per recorded candidate: `kind` is `ml_signal` and `mode` is `signal`, so a reader
cannot mistake it for a backtest; `values` keeps the same three-state `null`
behaviour; `observations` is present and null, because a signal evaluation
records no orders and no equity curve. The three tunable settings are reported
in `metadata`, not merged into `parameters`: a fitted parameter dictionary may
legitimately contain a key of the same name, and merging would silently
overwrite one of them. `parameters` retains the model artifact settings or
caller-declared method settings, distinguished by `method_family`.
`metadata.producer_source` preserves the stored origin, including `unknown`,
without consulting the current model registry. Non-finite scores become `null`, as in leaderboard storage, while absent
metric keys remain absent. `ml_report_block` collects the whole candidate set,
including candidates that failed to rank, with block-level definitions, the
observed parameter axes, the skipped rows and their coordinates, and the ranking
caveats. It accepts the complete candidate list and a ranking metric, computing
the ranking and selection caveats together from those candidates rather than
accepting an independent or truncated ranking. `alpha` sets the family-wise
significance level. Duplicate entry IDs raise `ValueError`. Comparability uses
the leaderboard rules: settings must match by default; pass
`allow_parameter_axes=True` for a surface, which still requires one research
identity and one dataset version per horizon. Cross-horizon source equivalence
remains unverified and is disclosed in the caveats.

Signal metric definitions now carry the two fields a portfolio metric definition
already had, `modes` and `requires_annualization`, so one definition shape
serves both. `modes` is always `signal`; only annualised return, Sharpe and alpha
require annualisation. Newly built `SignalEvaluationBundle` definitions record
both fields. Older definitions without the flag load with
`requires_annualization=null` (unknown), not `false`; their original semantics
are not inferred from today's catalog. Explicit saved values remain unchanged.
The projection's definitions come from the current signal catalog because
leaderboard rows do not store definition snapshots; they are not historical
producer definitions recovered from those rows. Report selection, presentation
and snapshot publication are defined in [Reports](reports.md#composable-offline-reports).

`leaderboard(dataset_id=...)` verifies and selects one immutable dataset
version, so models on it can be compared separately from rebuilt versions.
Records must match its manifest's data context and row count; unknown or invalid
IDs fail. Other filters combine with this selector. Different IC windows or
quantile counts require `allow_parameter_axes=True` or `allow_mixed=True`; neither
bypasses dataset verification.

`allow_parameter_axes=True` is the narrower opt-in a sensitivity study needs: it
permits horizon, IC window and quantile count to differ under these constraints.
Instrument or panel symbols, timeframe, period, feature set and code fingerprint
must still match, and one horizon may hold only one dataset ID and content hash, because a new horizon
rebuilds its labels on its own dataset. Across horizons, provider and underlying
source-bar equivalence are not verified. Inspect dataset manifests and source
data separately before treating the sweep as a controlled comparison; the tool
reports this limitation in its caveats. A ranking across those axes describes a
parameter surface, not one comparable measurement, so the response reports the
observed values of each axis and appends a caveat saying so. Every row states its
own horizon, window and quantile count, so a point on the surface is readable
without a second lookup.

Entry identity binds the producer source, dataset and evaluation context to the
model ID, or to a named method's family/name/settings. Only unnamed candidates include a
signal fingerprint in their entry ID. The fingerprint uses a canonical JSON
array of finite floating-point values, preserving array and value boundaries;
every record also stores it separately as `signal_fingerprint`. A matching re-run
reproduces the recorded values; source interpretation remains governed by
`producer_source` above. Repeating a model/dataset/context entry from the same
source replaces its earlier metrics, including when the caller supplies different
signal values. Distinct unlabelled signals remain separate candidates.
Different producer sources coexist as separate entries even with the same model,
dataset and measurement context. Thus a caller array cannot overwrite a backend
evaluation, and a newly verified evaluation cannot replace or upgrade an old
`unknown` row. Legacy IDs and evidence are preserved without migration or inferred
provenance. The store also rejects an attempted replacement of an existing ID
with a different source. Ranked rows and reports display source alongside identity
so same-name entries remain distinguishable. Each distinct selected entry counts
as a candidate under the existing ranking and selection-bias rules; repeating the
same source identity does not add a candidate.

`LeaderboardStore.record` holds `.leaderboard.lock` across its read, update and
write. It flushes a unique temporary file beside the destination and atomically
replaces the JSON-lines file. `entries` reads without locking and sees a complete
old or new file. Temporary files are removed after successful and failed writes, and
the lock file is retained. Re-recording an entry returns `replaced=True` rather
than increasing the candidate count. Unreadable records fail the read instead
of being skipped and silently reducing that count.
