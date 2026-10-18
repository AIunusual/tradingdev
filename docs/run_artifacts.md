# Run Artifacts

TradingDev stores runtime metadata in `workspace/tradingdev.sqlite` and files
under `workspace/`.

## Workspace Layout

```text
workspace/
  tradingdev.sqlite
  .workers/
    <launch_token>/
      start.json
      ready.json
      stop
      finished.json
  generated_strategies/
    .locks/
      <strategy_id>.lock
    <strategy_id>/
      current.json
      revisions/
        <revision_id>/
          strategy.py
          config.yaml
          metadata.json
  configs/
  data/
    raw/
    processed/
  runs/
    <run_id>/
      manifest.json
      result.json
      performance.json
      observations.json
      config.yaml
      strategy.py
      dataset_fingerprint.json
      pipeline_result.pkl
  feature_requests/
  execution_plans/
    <plan_id>/
      confirmation.txt
      confirmation.html
      .approval.lock
  reports/
    <report_id>/
      report.html
      manifest.json
```

## SQLite Tables

- `jobs`: generic background job status, job type, pid,
  `created_at`/`started_at`/`ended_at`, error, and payload. Backtest-specific
  fields such as strategy, symbol, timeframe, date range, and config path are
  nullable columns mirrored in the JSON payload when present. The payload also
  records `process_create_time`, the supervisor's OS creation time in Unix seconds,
  and `worker_control_id`, its unique launch token. The PID belongs to the
  supervisor, not the strategy-executing child. New submissions also record
  `manifest_hash`, the expected digest of their fixed execution specification.
- `runs`: completed run metadata, metrics JSON, config hash, source hash,
  random seed, dataset id, artifact directory, selected `revision_id`, and
  execution `manifest_hash`.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.
- `execution_plans`: immutable plan payload, independent approval state,
  active confirmation token, submitted job ID, error, and the saved text/HTML
  digests. The plan payload includes its manifest, preflight evidence and document.

`job_id` and `run_id` are currently the same for completed backtest and
optimization jobs. `get_job_status(job_id)` returns the `run_id` once a run is
done. Generated strategy jobs and runs carry their selected `revision_id`;
bundled strategies and historical records without revision identity expose
`null`. A later save changes only the current pointer and does not change the
revision used by an existing job or completed run.

Draft cleanup scans raw job identities and payloads, run references and all stored
manifests, including manifests left without a database job. Failed and cancelled
jobs still retain their source references. A malformed or ambiguous history blocks
cleanup. Only explicitly selected non-current drafts without references can be
deleted; runnable sources and execution artifacts remain. Cleanup does not dedupe
or migrate old revisions. The per-strategy `.locks` files coordinate saves,
validation, status updates and cleanup; they stay outside deletable revision
directories and are never unlinked while another process might use their inode.

Successful MCP `prepare_backtest`, `prepare_walk_forward`, and
`prepare_optimization` responses include the required `manifest_hash` and a
`plan_id`; they do not create a job. `submit_execution_decision` with `approve`
returns the job ID after the caller relays the user's explicit consent.
Job status, job lists, and run
records carry the same digest; historical entries without an execution manifest
return `null`. Historical results remain readable, but old jobs cannot resume
through the new workers. Prepare a new plan to execute through MCP under a fixed
specification. The old MCP `start_backtest`, `start_walk_forward`,
`start_optimization` and `confirm_optimization` tools are removed.

Persisted legacy jobs with `status: pending_confirmation` are incompatible with
the current job response contracts and are not migrated. `get_job_status` and
`cancel_job` reject their responses at the MCP boundary; even one such record
causes the entire `list_jobs` response to fail validation. Preparing another plan
in the same workspace does not repair those records. After using the version that
created the jobs to stop its workers and server and verify cleanup, preserve the
old workspace at its original location and configure the new MCP server with a
fresh `--workspace` directory. Do not copy the old SQLite database or worker
control files into it. The new workspace does not automatically import strategies
or history; old result files remain in the preserved workspace. This is a new
workspace recovery path, not an in-place migration or resumption of old jobs.

`.workers/<launch_token>` stores the startup identity, worker startup response,
optional cancellation request (`stop`), and final cleanup acknowledgement. These
small records remain in a regular workspace so later sessions can verify cleanup
without trusting a potentially reused PID. Test workspaces remove them with the
rest of the temporary workspace after verified cleanup.

Cancellation of an active job without a complete control identity fails without
signalling or claiming that the process stopped. A queued job that has not started
a process can be cancelled directly. Status checks mark a started job failed when
its supervisor identity is missing or no longer matches; the first response
includes the same error persisted in the database. Spawn failures also persist
`failed`, an explanatory error and `ended_at` before propagating the exception.

## Execution Manifest

Before spawning a background worker, the service publishes
`runs/<job_id>/manifest.json` and records its hash in the job. Publication never
replaces a different existing manifest. Atomic publication applies to the manifest
file; file publication and the database job write do not form one transaction.
Schema version 5 contains:

- `kind`: `backtest`, `walk_forward`, or `optimization`.
- `config`: the effective strategy identity and source hash, backtest settings,
  optional walk-forward settings, parallel policy, seed settings, and data
  requirements with resolved absolute directories and feature paths. Typed
  execution defaults are materialized and dates are JSON strings.
- `strategy_execution`: the strategy kind (`bundled` or `generated`) and
  `constructor_kwargs`. Bundled `config` and optional `fit_config` objects include
  nested model defaults; generated keyword arguments include declared constructor
  defaults. Injected engine and parallel policy objects remain outside this JSON
  object and are supplied from the manifest's execution configuration.
- `optimization`: `null` for other kinds; otherwise the parameter ranges,
  optimization metric, `train_start`/`train_end`/`test_start`/`test_end`,
  catalog-defined `direction: maximize|minimize`, and `trial_timeout_seconds: 300`.
  This timeout is fixed at preparation and limits only the first training
  candidate used to estimate the formal search duration. Remaining candidates
  and held-out evaluation do not use that deadline. It is separate from the
  complete preflight-process budget. Parameter names are
  sorted; the caller's candidate order within each range is preserved.
  Confirmation timeout/polling fields are removed: the worker no longer owns
  the user-confirmation interaction.
- `manifest_hash`: SHA-256 of canonical JSON over all the preceding fields and
  `schema_version: 5`, excluding the hash field itself. Object keys are sorted;
  job IDs and creation timestamps are not part of the specification.

Typed backtest, walk-forward, parallel, and data defaults have the same digest
whether implicit or explicit. Strategy identity and non-parameter declarations
remain bound to the revision. Without an MCP `parameters` override, or with
`parameters: null`, `config.strategy.parameters` retains the input configuration's
parameter mapping while constructor defaults are expanded separately into
`strategy_execution`. CLI execution likewise retains the parameter mapping from
its submitted YAML, which may already contain experiment settings.

An explicit MCP `parameters` mapping, including `{}`, recursively overlays the
captured values and replaces `config.strategy.parameters` with the complete
effective experiment parameters, including defaults. For bundled strategies this
is the captured constructor's `config` mapping; for generated strategies it is
the captured constructor keyword mapping. The run's `config.yaml` projection
reflects that replacement. The revision's original YAML and evidence stay unchanged.
Overrides can target only keys already captured from base parameters or constructor
defaults; unknown keys at any recursively merged mapping level are rejected.

Without overrides, omitted versus explicit constructor defaults in the input
parameter mapping can produce different digests despite equal effective values.
An explicit empty override can also differ from an omitted override if it expands
defaults absent from that mapping. Conversely, explicit override mappings that
produce identical full parameters have the same digest when the rest of the
manifest is identical. The hash records the stored specification, not whether
the caller explicitly supplied a particular override key.

Nested optimization candidates override only their specified leaves, preserving
the remaining fixed values; candidate structure and bundled model contracts are
validated before creating a job without constructing each candidate. Preparation
also constructs generated strategies with the base execution parameters and runs
80- and 240-row signal-contract fixtures, even without parameter overrides.
MCP preparation places these checks in the supervised preflight subprocess and
includes them in the preparation time budget. Direct application-service/CLI
execution does not acquire that MCP-specific preparation deadline. Each generated candidate is checked again
with its own parameters in the worker; constructor or execution errors can still
occur during the trial. Unsupported defaults or
candidates requiring new implicit fields are rejected. Integer candidates for
float fields are accepted only when conversion preserves their numeric value;
string/boolean coercion and precision loss are rejected.
Non-finite numbers in execution configuration or search values are rejected instead of
being converted to `null`. Invalid resolved execution settings return
`invalid_execution_request` for backtest/walk-forward submissions or
`invalid_optimization_request` for optimization, without creating a job.
These codes also cover strategy identity mismatches, revision-file read failures
during binding, and failures of the repeated executable-state check during
preparation. Failure to select an executable strategy at the start of the request
instead returns `strategy_not_executable`.
The mode must match the presence of walk-forward
validation settings. Optimization has its own training/test split and rejects
configs containing `validation` with `invalid_optimization_request`.

Manifest decoding preserves stored JSON without reapplying today's config defaults.
Before execution, current runtime models must accept the saved config and effective
strategy settings without adding fields or changing values. New required/defaulted
fields or incompatible normalization cause execution to fail, not reinterpret the
request. Schema version 5 requires the version, effective strategy fields,
performance settings, the fixed optimization direction, and the canonical
top-level `random_seed`; versions 1/2/3/4 or unknown versions require a new submission. No manifest is migrated
in place, and historical result metadata remains readable.

Workers receive only a job ID and load this fixed manifest path. They verify the
supported schema, content digest, expected job hash, and strategy/mode identity
before execution. Plan confirmation and result persistence verify the
same binding. Missing, modified, or unsupported manifests fail execution;
plan confirmation refuses to submit invalid content. Changing the original YAML,
the current strategy pointer, or the neighboring `config.yaml` does not redefine
the submitted request. A changed request requires a new job.

`config.yaml` is a readable projection of `manifest.config`; workers do not use
it as their source of execution settings. `config_hash` still hashes the YAML
projection (including the execution's strategy identity, non-parameter declarations,
and parameter mapping described above); resolved constructor
settings are available in the manifest's `strategy_execution` object.
`manifest_hash` covers the complete execution request. The
artifact's `sha256` hashes the actual `manifest.json` file bytes, including its
own `manifest_hash` field, and therefore is a separate digest.

This specification fixes the request, not the contents of referenced data files,
installed dependencies, engine environment, or RNG state. Absolute paths and
fingerprints do not make those inputs immutable or guarantee identical results
on a later run. Integrity verification assumes the job's expected digest remains
trusted; it is not a security boundary against someone controlling both the
database and workspace files.

## Execution Plans and Confirmation Documents

`ExecutionPlan` schema version 1 binds a unique 32-character `plan_id`, the full
execution manifest, `original_config_path`, `PreflightReceipt`,
`ConfirmationDocument`, timezone-aware `created_at`/`expires_at`, and `plan_hash`.
The hash covers canonical JSON of every field except itself. The document and
preflight hashes must identify the same manifest; the document also identifies
the same plan. Plans currently expire one hour after preparation. The content
does not change when approval state changes.

`PreflightService` runs preparation and sample execution in a supervised child
with a default 60-second budget, including imports, configuration resolution,
generated signal-contract checks, sample data acquisition, engine execution and
result serialization. Process cleanup has its own deadline. Success is returned
only after verified worker cleanup. The request supplies
`minimum_history_bars` and `sample_bars` (default 1024, allowed 64–4096), with
the minimum required history bounded by that total budget. Every actual sample
window must satisfy the declared history requirement. A split execution needs
enough total budget for its training and test windows separately.

The sample uses the selected market-data pipeline and effective settings.
It searches bounded calendar windows within the requested dates, continuing
past empty market-closure windows until it reaches the bar budget or end date.
Cached Parquet data is read in batches, interpreting timezone-naive timestamps
as UTC just like ordinary loading. Feature caches project only the timestamp
and requested value column, retaining exact matches to selected market bars.
Join cardinality is checked before materializing duplicate-key expansions;
exact left joins and forward/backward fill keep the ordinary loading semantics.
Sampling does not rewrite source market or feature caches.

Each crawler implements `fetch_sample(..., max_rows=...)` independently of full
history loading, enforcing bounds before constructing sample frames and honoring
inclusive endpoints, including a single timestamp. Custom providers must also
implement this method. API paths use bounded pages or time windows; Yahoo and
Deribit streamed JSON responses have a 4 MiB ceiling. Streamed sample requests
reject redirects and extra HTTP compression before reading the body, avoiding
client buffering or decoding before the size checks. Binance Vision
prefers daily archives, falling back to a monthly archive on a missing daily
file. Downloads stream into spooled temporary files with identity HTTP encoding.
Each selected CSV is scanned in batches of at most 1024 rows, retaining only
the earliest requested bars even when stored rows are out of order. Later-date
archives are not read once the output budget is satisfied. Downloads are limited
to 16 MiB per archive and 32 MiB per `fetch_sample` call; declared CSV expansion
is limited to 128 MiB. Exceeding a limit fails preparation.
An archive is still downloaded in full before parsing, so sample bar counts do
not imply proportionate network traffic. These are application-level resource
controls, not an operating-system memory limit or a Python sandbox.
Backtests sample one window; walk-forward samples one train/test fold;
optimization samples its first candidate on both training and held-out periods.
Temporary sample results, data and tool caches live in an independent temporary
directory, outside regular run history. The child's `TMPDIR`, `TMP` and `TEMP`
also point inside that directory, containing standard temporary files created
by third-party model training without changing the parent's environment.
Successful, failed and timed-out workers
are stopped and verified before their files are deleted. If process cleanup
cannot be verified, the directory is retained and the operation reports failure;
file cleanup failure is also an error. No successful sample publishes a regular
job or run. This process supervision is not a complete generated-Python sandbox.

The receipt records:

- `manifest_hash`, elapsed seconds, requested/used sample bars and declared
  minimum history;
- `data_origin: market_data`, provider ID, and actual `windows` with role,
  start/end timestamps and row count;
- `checked_paths`: backend-issued check codes for configuration, signals,
  actual engine execution and serialization, plus generated signal-contract
  checks, walk-forward fitting or optimization candidate binding when applicable;
- observed trade count, optional native execution-record count and whether the
  sample exercised the trading path;
- tested/total fold or candidate counts, and explicit coverage warnings.

`market_data` means the configured data pipeline was used. Offline tests may
substitute the provider; their receipts do not prove live external-market
verification. Zero trades alone do not fail preparation: the document says the
engine completed but this sample did not cover the strategy's trade/cost path.
Optimization additionally requires the first candidate's sampled training
objective to be available and finite. Otherwise preparation fails with
`unrankable_sample_objective`, even if another candidate or the full training
period could yield a rankable value. Enlarge the sample within its allowed budget
or revise the settings and prepare again. A zero-trade sample can therefore fail
when the selected objective cannot be calculated without trades.
The document also states that remaining dates, folds and candidate combinations
may still fail during full execution. Sample success is not a profitability claim.

`ConfirmationPresentation` accepts only model-authored title, summary and
`parameter_descriptions`. Description keys are JSON pointers relative to captured
constructor arguments; each value contains a label, explanation and optional
unit, never an executable parameter value. Labels must distinguish values and
cover every scalar, null or empty-container leaf, including nested bundled
`config`/`fit_config` and the union of effective optimization candidate leaves.
Candidate presentation and strategy loading share the same pure parameter merge:
partial objects preserve unnamed captured fields, empty objects preserve an
existing object, and non-object values replace the complete entry. Each search
group displays its merged candidate values; it does not present a partial
override as a complete replacement. Missing or
extra descriptions return `invalid_confirmation_presentation` with
`required_parameter_paths`, allowing the model to repair wording without changing
the source revision. Model units do not rescale captured values.

The backend builds a typed document with fixed strategy, market, parameters,
execution, validation, optimization, preflight and limitations sections.
Non-applicable sections remain explicit. Known cost/rate semantics have backend
formatters; unknown effective setting fields fail preparation rather than being
silently omitted. Model explanations are labelled separately from backend
values and successful-check evidence. Parameter/settings keys and YAML are not
the user interface. Both text and HTML render the same document; technical
identity and file paths are outside the main explanation, in HTML details and
machine metadata. The HTML is offline, escapes text and disables scripts.

`execution_plans/<plan_id>/confirmation.txt` and `confirmation.html` preserve the
reviewable documents. The HTML is registered as `execution_confirmation` under
`<plan_id>:confirmation_html`, without a run ID. Reads verify plan identity and
both saved file digests. File publication and SQLite updates are separate
resources; publication rolls back its new directory on handled failure.

Plan states are `ready`, `awaiting_confirmation`, `submitting`, `submitted`,
`cancelled`, or `failed`. A retained `.approval.lock` and state comparison
serialize transitions; only one submission reservation can be active. The plan
stays `ready` while the user reads it. On approval, the service checks expiry and
current executable-source integrity before reserving and again before submitting.
An internal server-owned challenge binds the submission to the same plan content;
the client is responsible for truthfully relaying user consent. Repeating a
successful submission returns its original job ID rather than launching twice.
The deterministic job ID equals the plan ID. If worker launch succeeds but the
final `submitted` state write fails, the response still identifies that job and
warns about the pending state update. Plan lookup retains `submitting` and exposes
the job ID. A later confirmation request can reconcile an existing job only when
its manifest matches and it has a complete worker-control identity; it records
`submitted` and returns the same job without another launch. Failed launches or
queued records lacking that identity remain inspectable through the returned job
ID but are not reported as successfully started work.

Preparation and lookup return `confirmation_summary` and the complete HTML's
`html_path`/`artifact_id`. Full confirmation text remains in the saved plan; it is
not copied into ordinary tool responses. The assistant presents the short summary
and HTML link, ends the turn and waits for the user to review it. Reading the
artifact does not submit a job. A local path must be made available through the
client's file/artifact interface when the user is not on the server's machine.

`submit_execution_decision(plan_id, decision)` relays an explicit later user reply.
`approve` submits the fixed manifest; `decline` cancels without running;
`revise` cancels and returns `confirmation_adjustment_requested`, requiring a new
plan and fresh confirmation after changes. Questions, ambiguous replies and the
initial task request are not approval. Conversation waiting leaves the plan
`ready`; there is no five-minute form deadline, but the one-hour plan lifetime
still applies. A new plan must be prepared after expiry.

The server trusts the assistant to convey the user's decision; it cannot prove
human consent from a tool argument. Source and effective-setting checks, internal
submission reservations and duplicate-job prevention remain enforced. The former
native MCP form tool is removed. CLI commands remain explicit execution requests
and do not use this conversation workflow.

## MCP Lookup

- `list_jobs` / `get_job_status` / `cancel_job`: operational progress and
  cancellation.
- `list_runs` / `get_run`: completed result summaries and metric/scope discovery.
- `get_metric_catalog`: current definitions, units, applicability, summary choices,
  annualization requirements and optimization directions.
- `get_run_metrics`: complete or selected metrics in a saved scope, with definition
  snapshots, provider versions, settings and unavailable-value reasons.
- `compare_runs`: scoped metric comparison with explicit comparability reasons.
- `list_artifacts` / `get_artifact`: result JSON and other artifact lookup.
- `promote_strategy`: generated strategy artifact promotion.
- `record_feature_request`: structured unsupported feature requests.

The generated revision's `config.yaml` holds its base settings;
`runs/<run_id>/config.yaml` holds effective execution settings and includes
`strategy.revision_id`. Runtime market/date and parameter experiments
do not rewrite the base revision. Backtest and walk-forward requests may supply
`parameters`, recursively merged into the base parameters. CLI runtime configs
may change `strategy.parameters` while preserving other strategy declarations.
The same revision can therefore have many runs with different parameters and
manifest hashes, without duplicate generated source revisions. Both short and
long signal-contract fixtures check effective generated settings before execution.
The execution manifest additionally fixes runtime overrides and search settings
without rewriting the base revision.

Each completed background job writes files under `workspace/runs/<run_id>/` and records
matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `performance.json`: all metric values by scope, definitions and calculation
  metadata, stored as `performance_json`.
- `observations.json`: aligned timestamps, equity, per-bar returns and trade
  records by scope, stored as `observations_json`.
- `manifest.json`: the specification published at submission, registered after
  successful result persistence as `execution_manifest`. Its artifact metadata
  includes `manifest_hash` and `schema_version`. Use `list_artifacts` and
  `get_artifact(include_content=true)` to inspect the JSON after completion.
- `config.yaml`: effective config snapshot used for the run, stored as
  `config_snapshot` with `config_hash`. It projects the executed manifest config;
  persistence checks the worker's snapshot against that manifest and the job's
  digest rather than reloading a possibly changed YAML file. It includes the
  start tool's symbol, timeframe, and date range together with resolved defaults.
  Optimization includes the entire final calendar day; ordinary backtest bounds
  remain timestamps.
- `strategy.py`: generated or bundled strategy source snapshot when
  `strategy.source_path` is available, stored as `strategy_source` with
  source hash metadata. Generated source snapshots come from the selected
  immutable revision, not whichever revision is current at completion.
  The same hash is indexed in `runs.source_hash` for
  direct SQL lookup.
- `dataset_fingerprint.json`: dataset id, symbol, timeframe, date range, and a
  fingerprint hash, stored as `dataset_fingerprint`.
- `pipeline_result.pkl`: full `PipelineResult` for dashboard rendering, stored
  as `pipeline_result` for completed backtest and walk-forward jobs.

For background jobs, SQLite `runs.artifact_dir` must match the corresponding
`workspace/runs/<run_id>/` directory. `runs.config_hash` is the SHA-256 of the
serialized effective config snapshot, not the hash of the immutable revision's
base config. `runs.random_seed` records only the canonical
top-level `random_seed`; strategy-specific model parameters are not inferred as a
run seed. Run comparison,
dashboard rendering, and artifact lookup always read through SQLite first, then
resolve files from the recorded artifact paths.

CLI runs instead store their `pipeline_result` cache under
`workspace/data/processed/cache` (or `$TRADINGDEV_DATA_ROOT/processed/cache`),
while `runs.artifact_dir` points to
`workspace/runs/cli_<cache_key>_<execution_id>/`, where the
execution manifest is published and registered as an artifact. The pipeline
shares the same performance/observations JSON storage in that run directory;
the pickle is only needed for dashboard rendering. The pipeline
result embeds both its config snapshot and execution manifest. The run's config
hash, manifest hash, and revision identity come from the executed specification.
Saving uses this manifest without re-reading the original config file; editing
that file after execution does not prevent saving the original result. The
original `config_path` remains descriptive artifact metadata. Each save has a
fresh execution ID and a separate `<run_id>.pkl`; same-request reruns can preserve
different results. Clearing the pickle cache leaves historical performance JSON
readable; the cleared historical dashboard artifact is unavailable, while a new
execution writes its own complete cache.

The CLI cache fingerprint requires the executed manifest hash and combines it with
processed-data file size/mtime and a Git source-code fingerprint. There is no
YAML-based key fallback. New settings produce a different manifest and therefore
a different cache key. Completed results are loaded by run ID through the
registered artifact path, without recomputing a key from current YAML, data, or
code. The unused YAML-based `load_cached_result` and `save_cached_result` helpers
have been removed; this storage path does not automatically reuse prior backtests.
These file-stat and code fingerprints describe the execution inputs; they are not
immutable data or environment versions and do not guarantee full reproducibility.

Optimization `result.json` includes the best parameters, training metrics,
out-of-sample metrics, objective and fixed direction. Its complete numerical
projection remains in SQLite; `get_run` and completed `get_job_status` select a
summary at read time, so their `metrics` are not the entire `result.json` payload.
Parameters outside the search grid retain the effective base values captured in
the manifest, including constructor defaults and any `parameters` overrides
applied during preparation. Selection uses training results, and only the selected
parameters are evaluated out of
sample. Every training trial retains its full result in the performance and
observation artifacts; selection references the saved winning trial without
discarding other trials.

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
legacy data. The reader does not delete the marker or repair stored data.

Historical runs without performance artifacts remain readable through ordinary
run lookup, marked as lacking detailed provenance. `get_run_metrics` reports
`performance_artifact_unavailable`; reads never migrate, unpickle or recompute
those results. Historical values are not asserted to follow the new definitions.

## Historical Observation Queries

`find_runs` matches a historical effective parameter subset, strategy, symbol and
timeframe. Each scalar scope is a separate row; fixed constructor parameters from
the saved manifest are recursively overlaid with optimization trial parameters.
Fitted parameters remain separate metadata. Results expose parameter provenance
and completeness; incomplete legacy parameters are not assumed to match a filter.
Unreadable runs produce `issues` with `complete: false` alongside readable matches.

`get_run_trades` and `get_run_equity` accept `run_id`, optional `scope`, zero-based
`offset` and `limit` (1..500). Responses carry total/matched counts and `next_offset`.
Omitting scope chooses the saved default; an aggregate fold scope returns an error
and available scalar scopes. Trade filters include status, direction and entry
time; equity filters refer to bar time. Bounds are inclusive; a date-only upper
bound includes the entire UTC date, while timestamps select exact instants. Naive
times are UTC. When the scope has a timestamp array, individual trades with an
unknown entry time (for example, a legacy record without `entry_idx`) do not pass
an entry-time filter. If the entire scope's `timestamps` array is `null`, supplying
any time bound to either query returns `timestamps_unavailable`, rather than an
empty filtered page. Without time bounds, both queries can return observations
with unknown timestamps.

Trades retain an unmodified `record` and stable source-order `trade_id`, along with
normalized timestamps, prices, sizes, costs and PnL. An open position's mark is
separate from a real exit. Signal equity is account value; volume observations
are cumulative PnL, without fabricated account capital. Queries read verified
saved JSON only: no replay, current strategy loading, pickle, or data download.

`get_run_executions` reads `records` with optional filled/ignored/rejected status,
buy/sell side and timestamp filters. Side refers to filled order direction;
unfilled attempts have no fill side. `get_run_account_history` reads `states`
with timestamp filters. Both use the same scope, offset/limit and inclusive UTC
date rules, returning availability, accounting and valuation/time semantics.
`available` with zero rows means a recorded empty or filtered sequence; legacy
`not_recorded` and volume `unsupported_volume_accounting` return explicit
unavailable status with zero rows, never reconstructed history. Records remain
aligned to their original bar indices after filtering and paging.

## Composable Offline Reports

`get_report_sections` returns built-in section IDs and optional `standard`,
`comparison` and `trades` recipes. `generate_report(run_ids, sections, commentary)`
accepts one to eight distinct runs. Omitted/null sections select standard;
an explicit ordered list chooses visible sections. With `[]`, no optional sections
are rendered; fixed report/source identity metadata and any supplied commentary
remain visible. Unknown or duplicate sections fail. Up to 20 plain-text title/text notes
(20000 characters total) are escaped and labelled LLM commentary. They cannot
replace computed values or inject HTML. The client selects information and writes
interpretation; the server renders tables, charts and document structure.

Section selection determines the visible presentation without pruning the loaded
or embedded data. Every HTML report embeds the complete loaded payload as JSON,
including all scopes of the requested runs, saved metrics, trades, equity and
execution configuration, even for `sections=[]`.
Omitting a section does not remove its underlying data from the HTML; recipients
of the file can still read that embedded data.

The `executions` and `account_history` sections, also in `standard`, display these
two saved ledgers separately from paired trades. Each has independent search,
sorting and full CSV export, with nested market/before/after fields flattened for
execution CSV. Missing streams show their reason; `[]` shows a recorded empty
stream. The manifest records each stream's original count, preserving null versus
zero. Old report files retain their original template version.

All saved scopes are validated, including sources of omitted sections. Scalar
observations and aggregate fold statistics remain distinct. Missing or corrupt
sources fail generation. The report names selected/omitted sections rather than
claiming every suggested section is present. Included trade tables expose all
records, sorting, text search and local CSV export; SVG equity and drawdown charts
and scripts are embedded for offline use. UTC times, units, unavailable reasons,
provenance and known overlap/interpretation limits are supplied by the backend.
Paired-trade tables use Traditional Chinese column, status and direction labels,
with distinct closing and end-of-period valuation fields. Prices are quote currency
per asset unit; quantities are asset units, and PnL/fees are quote currency amounts.
These numeric cells display two decimal places, with a precision note below the
table; sorting uses the original values, including when displayed values are equal.
Expandable raw records, embedded observations and CSV exports retain their original
field names and values. Presentation changes create a new template version and report
identity without modifying earlier reports or backtest results.

The content digest includes source snapshots, template version, section order and
commentary. `reports/<report_id>/report.html` and `manifest.json` are registered as
`research_report_html` and `research_report_manifest` with IDs
`report:<report_id>:html` and `report:<report_id>:manifest`. A single-run report
is registered to that run; a multi-run report has no single `run_id` binding.
The response gives path, artifact IDs
and HTML SHA-256 without placing an unbounded document into the MCP response.
`get_artifact` can retrieve the files; local paths are not public URLs.

Publication prevents overwriting different content. Identical requests reuse an
intact registered report; missing/modified files for a registered report are errors.
Identical unregistered files may be registered on retry; conflicting registrations
are rejected rather than overwritten.
A per-report busy marker coordinates publishers. Handled write/registration
failures roll back newly created files and rows; files and SQLite are not a single
cross-resource transaction, and an external kill can leave a busy marker. Reads
do not repair old output or delete the marker. CLI and dashboard use the same
application service; LLM prose and report recipes require no new Python/HTML.

Dashboard sessions retain the generated report ID and expected HTML SHA-256.
On every rerun, `ReportService.get_report_download(report_id, expected_sha256=...)`
validates the registered HTML type, fixed report path and stored hash, then reads
and hashes the same bytes supplied to the download button. This in-process
interface returns HTML bytes on success, or an error without content; it does not
add document bytes to MCP responses. Missing, modified or unreadable files remove
the download from the dashboard and display an error. A failed generation also
clears the previous download, so an integrity error cannot fall back to a stale
file. Reads do not repair or regenerate the report.

## JSON Values

Before writing `result.json` or SQLite `runs.metrics`, metric values are
recursively normalized: `NaN`, `Infinity`, and `-Infinity` become JSON `null`,
including values inside nested objects and arrays such as optimization metrics.

Legacy results are also normalized in memory when `JobStore.load_result` reads a
result file or `SQLiteStore` reads run metrics. These reads do not rewrite the
stored file or database row. Consequently, `get_run`, `list_runs`, and completed
`get_job_status` responses expose legacy non-finite metric values as `null`.

In contrast, `get_artifact(include_content=true)` returns the original UTF-8 file
text without parsing or normalizing its contents. A legacy `result.json` may
therefore still contain `NaN` or `Infinity` even when result lookup returns `null`;
the content-equivalence statement above does not apply to these legacy artifacts.
The CLI displays missing or non-finite metrics as `N/A`, while keeping finite zero
values visible as zero.
