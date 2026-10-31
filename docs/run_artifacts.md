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
  ml_datasets/
    .publish.lock
    <dataset_id>/
      dataset.parquet
      manifest.json
  ml_feature_cache/
    .feature-cache.lock
    <cache_key>.parquet
    <cache_key>.json
  models/
    .<model_id>.lock
    <model_id>.json
    <model_id>.bin
  ml_leaderboard/
    .leaderboard.lock
    leaderboard.jsonl
  comparisons/
    <job_id>/
      manifest.json
      config.yaml
      ledger.json
      snapshot.parquet
      receipt.json
  runs/
    <run_id>/
      manifest.json
      result.json
      performance.json
      observations.json
      research_evidence.json
      config.yaml
      strategy.py
      dataset_fingerprint.json
      pipeline_result.pkl
  feature_requests/
  research_evidence/
    <evidence_id>.json
  submitted_research/
    <job_id>.json
  execution_plans/
    <plan_id>/
      confirmation.txt
      confirmation.html
      .approval.lock
  research_archives/
    <archive_id>/
      index.html
      research.json
      manifest.json
      reports/
      research.zip
  reports/
    <report_id>/
      report.html
      manifest.json
```

Comparison snapshots and receipts are published only after approval and data
loading. A failed or cancelled comparison may have neither `snapshot.parquet`
nor `receipt.json` if it stopped before data publication.

Strategy validation and dry-run use a separate system temporary directory named
`tradingdev-strategy-check-*`, outside the source workspace. It contains
`request.json`, the worker's `result.json` when available, `.workers/<launch_token>`
supervisor controls, and isolated `tmp`, `numba`, `matplotlib` and `mypy` paths as
needed. The request identifies the source workspace, fixed revision metadata and
fixture size; the response binds its evidence to the request's SHA-256 digest.
These files are runtime-check artifacts, not saved research runs or reports.

The directory is removed after the supervisor confirms its owned process group
has stopped, on both success and failure. If shutdown cannot be verified, the
directory is retained and its path is included in failure diagnostics. Failure
to remove the directory also returns a cleanup diagnostic with its path; neither
case grants passing validation or dry-run evidence. The process-group guarantee
excludes descendants that create a separate session and external forced
termination of the supervisor, as described in the
[strategy security model](strategy_contract.md#security-model).

## Research Evidence Records

`ResearchEvidenceService` saves client-authored explanations without fetching
sources, executing strategies or acquiring market data. Each
`research_evidence/<evidence_id>.json` is a canonical UTF-8 record, limited to
512 KiB, with a creation time, 32-character hexadecimal ID and content digest.
Publication does not replace an existing record; an identical record can be
retried. Reads check the ID, schema, digest and canonical bytes, rejecting unsafe
paths and symbolic links. These checks detect inconsistency, not an attacker
who can rewrite both the record and its digest; approved content must retain its
own expected digest.

The draft contains human parameter descriptions, trading conditions and formulas,
variable meanings, limitations, source declarations and parameter provenance.
Sources identify client-reported web retrieval or user-supplied references;
neither their presence nor a hash proves that browsing occurred or that a claim
is true. Reference-backed parameter declarations must cite explicit source claim
values matching the effective values, including JSON type. Every cited claim must
match each bound value; differently valued settings need separate declarations.
A source supporting only a formula cannot
also establish a suggested parameter value.

`bind_research_evidence` expands grouped provenance declarations against the
complete fixed execution, rejecting missing, overlapping or unknown targets and
parameter pointers. Values come from the manifest, not from a repeated client
copy. A single constructor uses target `base`; fixed comparisons use
`candidate_1` through `candidate_8`. Optimization also binds each search-axis
option's merged parameter subtree as `search_<group>_<option>`; the base value
alone does not explain the search grid. Bound content retains the complete record,
target values, execution identities and its own digest. These are explanation
bindings, not proof that the supplied formulas faithfully implement Python or ML.

`ApprovedResearch` freezes bound declarations together with one versioned execution
policy disclosure per distinct manifest. Submission and worker startup compare the
saved policy with the simulator's current policy before acquiring data or running
the strategy. Historical reads validate the saved content and identity without
replacing its wording with today's policy.

Execution policy `execution-v2` uses the observed month-boundary equity baseline
for Volume's monthly loss limit, as defined in the
[signal and execution contract](strategy_contract.md#python-contract).
Plans prepared under an older execution policy must be prepared and confirmed
again before submission; already saved results and their disclosures are not
rewritten or recalculated.

`submitted_research/<job_id>.json` retains the complete target and approved snapshot
before worker launch; the job stores its expected `research_evidence_hash`. The
canonical envelope is limited to 4 MiB, with its size checked before confirmation.
Successful runs receive an independently readable `research_evidence.json`, registered
as artifact type `research_evidence` with its byte hash, parent job ID and approved
content hash. Publication rollback and comparison recovery include this artifact.
A failed comparison can retain its parent disclosure even when no run was published.
Removing the original draft, confirmation HTML or submitted copy does not invalidate
a completed run's registered copy. Missing or changed declared evidence is an error;
legacy runs and direct callers that never supplied evidence remain explicitly
`not_recorded`, rather than receiving reconstructed explanations.

Historical scope reads verify parameter membership before attaching provenance.
Optimization binds each training trial to its Cartesian search position, including
duplicate-valued options, and the selected test scope to the saved winning trial.
Walk-forward constructor settings remain distinct from fitted model parameters.
Walk-forward reads require the complete numbered train/test fold set and summary,
ordered disjoint test windows, and training observations before their test window.
Active explicit windows are checked using inclusive UTC calendar days. Automatic
fold checks do not reconstruct the original dataset or prove its split geometry.

## SQLite Tables

- `jobs`: generic background job status, job type, pid,
  `created_at`/`started_at`/`ended_at`, error, and payload. Backtest-specific
  fields such as strategy, symbol, timeframe, date range, and config path are
  nullable columns mirrored in the JSON payload when present. The payload also
  records `process_create_time`, the supervisor's OS creation time in Unix seconds,
  and `worker_control_id`, its unique launch token. The PID belongs to the
  supervisor, not the strategy-executing child. New submissions also record
  `manifest_hash`, the expected digest of their fixed execution specification.
  Field updates read, validate and merge within one SQLite `BEGIN IMMEDIATE`
  transaction, including the indexed columns and JSON payload. Execution identity
  remains immutable. The first persisted terminal state (`done`, `failed`,
  `cancelled`, `estimation_timeout`) wins; later lifecycle updates cannot replace
  its status, timestamps or error. Late process identity fields may still be
  attached without restoring an earlier lifecycle snapshot. Polling and
  cancellation condition their writes on an active state and report the state
  actually saved; completion during cancellation is not reported as cancellation.
  Status and list responses compute `elapsed_seconds` from creation to the saved
  terminal `ended_at`; only active jobs use the query time. Missing, invalid,
  timezone-less or reversed timestamps return `elapsed_seconds: null` with
  `elapsed_unavailable_reason: missing_or_invalid_timestamps`, without inventing
  a duration or rewriting legacy records.
- `runs`: completed run metadata, owning `job_id`, metrics JSON, config hash, source hash,
  random seed, dataset id, artifact directory, selected `revision_id`, and
  execution `manifest_hash`.
- `artifacts`: run and non-run artifact metadata, path, sha256, and metadata JSON.
- `events`: job-scoped structured events.
- `execution_plans`: immutable plan payload, independent approval state,
  active confirmation token, submitted job ID, error, and the saved text/HTML
  digests. The plan payload includes its manifest, preflight evidence and document.

`job_id` and `run_id` are currently the same for completed standalone backtest,
walk-forward and optimization jobs. `get_job_status(job_id)` returns the `run_id`
once such a run is done. A fixed comparison has a parent job and separate candidate
run IDs recorded in its ledger; the parent has no performance run of its own.
Generated strategy jobs and runs carry their selected `revision_id`;
bundled strategies and historical records without revision identity expose
`null`. A successful explicit revision save changes only that strategy's current
pointer and does not change the
revision used by an existing job or completed run.

Run publication is handled by `RunPublisher`, separately from job lifecycle
updates. `RunPublicationBinding` supplies the owning `job_id`, the distinct
`run_id`, and the expected execution manifest established by the submission
owner. The publisher verifies comparison membership against the saved parent
bundle and ledger; the binding alone does not prove that a child was approved.
The publisher verifies the existing
job, strategy/revision, executed manifest, configuration, source and performance
identity. Each run keeps its own `manifest.json` and run-scoped artifact IDs.
Artifacts registered directly by `RunPublisher` retain the owning job ID in their
metadata. `PerformanceStore` instead registers `performance_json` and
`observations_json` with only `schema_version` and `manifest_hash` in metadata;
their owning job is obtained through the artifact's `run_id` and the run record's
`job_id`. Child runs require complete
performance artifacts. Publication locking, rollback and retry checks cover
the run files and lineage. `JobStore.save_result` retains the single-run
interface and always constructs a binding with `run_id == job_id`.
Job creation and child publication share the run-ID submission lock; a new job
cannot reuse an existing run ID, and a child cannot occupy an existing job ID.
The lock file remains for concurrent waiters and same-thread nested submission.

Draft cleanup scans raw job identities and payloads, run references and all stored
manifests, including manifests left without a database job. Failed and cancelled
jobs still retain their source references. A malformed or ambiguous history blocks
cleanup. Only explicitly selected non-current drafts without references can be
deleted; runnable sources and execution artifacts remain. Cleanup does not dedupe
or migrate old revisions. The per-strategy `.locks` files coordinate saves,
validation, status updates and cleanup; they stay outside deletable revision
directories and are never unlinked while another process might use their inode.

Successful MCP `prepare_backtest`, `prepare_comparison`, `prepare_walk_forward`, and
`prepare_optimization` responses include the required `manifest_hash` and a
`plan_id`; they do not create a job. `submit_execution_decision` with `approve`
returns the job ID after the caller relays the user's explicit consent.
Job status and job-list items carry the parent digest. Single-run `get_run`
records carry that same digest; comparison children instead carry their individual
execution digests and retain the parent job ID. Historical entries without an execution manifest
return `null`. Historical results remain readable, but old jobs cannot resume
through the new workers. Prepare a new plan to execute through MCP under a fixed
specification. The old MCP `start_backtest`, `start_walk_forward`,
`start_optimization` and `confirm_optimization` tools are removed.

Persisted legacy jobs with `status: pending_confirmation` are incompatible with
the current job response contracts and are not migrated. `get_job_status` and
`cancel_job` reject their responses at the MCP boundary; even one such record
causes a `list_jobs` page containing that record to fail validation. Preparing another plan
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
its supervisor identity is missing or no longer matches, with a comparison-specific
exception: comparison status checks must first confirm worker cleanup. If cleanup
cannot be verified, the status check does not itself change job or candidate
states and returns `detail_error.code = worker_cleanup_failed`. Explicit
cancellation likewise does not change lifecycle states on cleanup failure and
returns `code = worker_cleanup_failed`. Both reread the saved job, so a concurrent
worker completion is preserved and reported: `status = done` can appear alongside
the cleanup diagnostic. Missing control identity during cancellation instead
returns `worker_identity_unavailable`. When a status check does mark a job failed, its
first response includes the same error persisted in the database. Spawn failures
also persist `failed`, an explanatory error and `ended_at` before propagating the
exception.

## ML Research Artifacts

Dataset, feature-cache, model and leaderboard identities and retention are defined
in [ML Research](ml_research.md#ml-research-artifacts). They use separate stores,
not backtest run IDs or the SQLite artifact index.

## Research Settings and Provenance

Research intake is not a separate persistent session. The shared resolver writes
one effective config before a generated revision or execution manifest is saved.
The required request modes, cost defaults and annualization rules are described
in the [research-input contract](strategy_contract.md#research-inputs-and-simulation-policy).

The canonical values are stored at their execution locations:

- `research.strategy_outline`: the resolved research outline;
- `backtest.symbol`, `timeframe`, `start_date`, `end_date`, `init_cash`, `fees`,
  `slippage`, and `periods_per_year`: the effective choices and simulation settings;
- `strategy.display_name`: the human-readable name supplied during authoring or
  generated from the outline, separate from `strategy.id` and `revision_id`.

`research` uses metadata schema version 1. Besides its outline, it contains
`policy_version`, `field_sources`, optional `selected_revision`, and
`annualization_reason`. `field_sources` covers exactly the JSON pointer for the
outline and each of the eight backtest fields above. Values are `request`,
`project_default`, or `selected_revision`; these describe resolution paths, not
proof of human input or approval. No second copy of the market, period, cost or
annualization values is stored in this metadata. An inherited field requires a
selected-revision identity containing `strategy_id`, `revision_id`, and its
`config_hash` when recorded. Generated revision metadata's `request_summary` is
a derived outline projection, not an independent save input or configuration
authority.

The config hash and manifest hash cover these saved values and origins. Equal
numeric costs with different origins or policy versions can therefore have
different hashes. Reusing a saved manifest does not reapply current policy defaults;
a new research request resolves its policy before producing a new manifest.
Direct execution configs and older artifacts can have no research provenance;
confirmation displays that absence rather than fabricating request origins.
Direct Python override helpers remove research metadata from the effective config
when applying request-level market/date or backtest settings, since they cannot
preserve the original field origins for that new execution. They do not rewrite
the saved revision or its metadata.
Direct YAML loading through CLI or `BacktestService.run_config` does not use these
override helpers: it preserves supplied research metadata without rebuilding its
origins. When manually editing research-owned settings in an execution YAML copy,
remove the entire top-level `research` mapping to avoid recording stale origins.
Keep the generated strategy identity and revision-bound declarations unchanged;
the saved source revision remains intact. Only changing strategy parameters does
not require removing research metadata.

Strategy creation and revision publication share the per-strategy lifecycle lock.
Omitted strategy IDs allocate a new `strategy_<uuid>` namespace; explicit IDs
without `expected_revision_id` also mean creation. Existing revision directories,
legacy files and incomplete publications reserve their namespace. Revising requires
the explicit expected current revision and verifies its stored content before
publication. A stale revision or creation collision does not move current.
Recovery from legacy flat files creates a new identity and leaves the original
files in place. As before, revision-directory publication and current-pointer
replacement are separate filesystem operations: a failed pointer update can leave
a complete unreferenced revision while the previous current pointer remains.

## Fixed Comparison Evidence

`ComparisonManifest` fixes one to eight ordered ordinary-backtest candidates,
including their labels, proposal rationale and complete execution manifests.
Only effective constructor parameters may differ; strategy identity, research,
data settings, execution assumptions and seed must match. Repeated effective
parameter sets are rejected. Its digest binds every candidate before execution.
`prepare_comparison` resolves research once and captures complete constructor
values from one selected revision. Candidate overrides never rewrite the base
revision. The plan contains this bundle, a separate program receipt for each
candidate and one confirmation covering the entire set. Each candidate, including
bundled strategies, passes both fixed synthetic fixtures; the whole preparation
shares one supervised time limit. No market data, engine or training runs before
approval. Training-required and walk-forward comparisons are not supported.

`ComparisonStore` uses `comparisons/<job_id>/manifest.json` and `ledger.json`.
Alongside them, `config.yaml` projects the shared configuration without selecting
one candidate's parameters; each child's manifest contains its complete settings.
The ledger reserves `<job_id>_candidate_N` run IDs and retains every candidate's
child manifest hash, status and failure reason. It progresses from `pending` to
`running`, then `succeeded`, `failed` or `cancelled`; untouched candidates can
become `not_run`. Identical retries preserve terminal outcomes. A reserved run ID
does not prove a result exists; the execution coordinator must verify published
child results before recording success. Metrics remain in each run's artifacts.

After approval, `snapshot.parquet` and `receipt.json` retain the actual merged
data bytes and their SHA-256, data binding, source request ID and comparison hash.
The coordinator verifies the snapshot's data binding against the accepted
execution; the storage adapter validates identity, JSON and file integrity.
This receipt does not change the approved manifest. Initial ledger publication
precedes the manifest completion marker; a missing ledger beside an existing
manifest is corruption and cannot reset completed candidates. Failed snapshot
publication rolls back only files created by that attempt. Draft cleanup scans
comparison children, including orphaned manifests, and blocks ambiguous history.

`ComparisonService` loads one merged dataset after approval and executes each
candidate independently against fresh decodes of that snapshot. It creates no
parent performance run or combined portfolio. Individual strategy failures retain
their reason and allow remaining candidates to run; shared source/data failures
stop the batch and mark untouched candidates `not_run`. Terminal candidates are
not replayed. The publisher binds each successful child to its parent, approved
child manifest and snapshot receipt. Recovery after publication but before ledger
update verifies the complete registered artifacts before recording success.

Cancellation and worker-death reconciliation happen only after worker cleanup is
verified. Parent `done` means all candidates were attempted, not that every one
succeeded: the compact `comparison` status includes `outcome`, counts, dataset
identity and every candidate's ID, label, run ID, child hash, status and reason.
`outcome=no_success` is explicit when none succeeded. Status reads receipt metadata
without loading the Parquet frame; execution and report creation verify actual
snapshot bytes. Constructor maps and proposal rationales stay in the confirmation
and report rather than being repeated on every poll.

## Execution Manifest

For a single execution, before spawning a background worker, the service publishes
`runs/<job_id>/manifest.json` and records its hash in the job. Publication never
replaces a different existing manifest. Atomic publication applies to the manifest
file; file publication and the database job write do not form one transaction.
Schema version 5 contains:

- `kind`: `backtest`, `walk_forward`, or `optimization`.
- `config`: the effective strategy identity and source hash, backtest settings,
  optional walk-forward settings, parallel policy, seed settings, and data
  requirements with resolved absolute directories and feature paths. Typed
  execution defaults are materialized and dates are JSON strings. Guided research
  also includes the resolved `research` metadata and the saved strategy display
  name; these are not executable constructor arguments.
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
whether implicit or explicit when the rest of the specification, including any
research provenance, is identical. Strategy identity and non-parameter declarations
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
MCP preflight maps `StrategyNotExecutableError` raised during research preparation
to `strategy_not_executable`, including initial strategy selection, identity
mismatches, revision-file read failures during binding, and repeated
executable-state checks during capture. Caught configuration, type, YAML and I/O
errors use the invalid-request codes above; execution-mode refusals retain their
capability-specific codes and guidance.

Direct Python `JobService` and `OptimizationService` preparation methods instead
return `strategy_not_executable` for failed initial selection and their respective
invalid-request code for binding or subsequent capture failures. The lower-level
`BacktestService` direct YAML/config methods raise exceptions rather than returning
these structured service codes.
The mode must match the presence of walk-forward
validation settings. Optimization has its own training/test split and rejects
configs containing `validation` with `invalid_optimization_request`.
MCP `prepare_walk_forward` alone accepts `validation_config` to replace the entire
effective validation mapping before research resolution and manifest capture.
There is no merge with base validation settings; omitted or `null` input preserves
them. Active explicit train/test windows use inclusive UTC calendar days and must
fit within the research period. New settings require `mode: automatic`, `explicit`
or `purged` and reject fields belonging to another mode. Historical flat settings
remain readable with their recorded mode precedence; they cannot be submitted as
new execution inputs. See the [strategy split contract](strategy_contract.md#strategy-walk-forward-settings).
Failed containment does not change the requested dates or publish a
plan; the caller must supply suitable windows or automatic-fold settings.

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

## Market Data Cache

OHLCV cache v2 uses an exact identity: provider, effective market type, original
symbol (including case and punctuation), and timeframe. Providers that ignore
market type use `null` for it. `requirements.market.source` overrides
`data.source`. Request IDs additionally hash the full UTC start/end instants
and feature requirements; they identify requests, not immutable data contents.
Load, cached-only ML reads, sampling, availability and inspection share this
identity and honor configured raw/processed roots. The inventory tool uses the
default root; config inspection inventories the configured root.

Each root contains `cache-v2/<identity-sha256>/`. Raw CSV and processed Parquet
files use immutable generation IDs; `<year>.json` in the processed directory
points to the current generation and records identity, successful query bounds
and acquisition time. These bounds describe the provider query, not the first
or last returned candle. Empty successful queries have receipts too. Inspection
exposes query evidence and observed rows separately. `rows` is `null` when no
generation can be read; otherwise it counts the loaded annual generations,
including successful empty ones. This count can be partial when other requested
years are missing or unreadable. Callers must also inspect `exists` and `errors`;
a known row count does not establish that every requested year was read or that
the requested interval has complete observations.

The application queries each requested UTC year through its end or acquisition
time, whichever is earlier. A receipt is reusable only if it covers the requested
year slice and was acquired at least seven days after that slice ended. Recent
intervals refresh on every load; merely waiting seven days after an earlier
download does not make it reusable. Future request endpoints fail. Successful
queries do not prove complete observations or prevent later provider corrections.
Cached-only reads require reusable evidence and fail without downloading when it
is absent; empty observations cannot serve as ML input.

A per-identity/year file lock serializes refreshes. The storage adapter stages
CSV, Parquet and metadata, publishes immutable files, then atomically replaces
the receipt. Failures leave the previous receipt and generation readable and
remove unpublished attempt files; older or narrower query evidence cannot
replace newer evidence. Published generations remain available to concurrent
readers and are not automatically collected. Lock files may remain after release.
Legacy flat files are preserved but excluded because their source cannot be
proven; fetching again creates v2 data without relabeling or deleting them.

## Automatic DVOL Queries

Without an explicit feature `path`, `app/dvol_data.py` manages Deribit DVOL
queries in the configured raw/processed roots. The cache identity separates
DVOL from OHLCV and records provider, currency and timeframe. A
`query-<sha256>.json` receipt hashes the exact UTC start/end instants, so changing
either bound, including within one year, cannot reuse an unqueried interval.
Successful empty queries are recorded independently of observed timestamps.
The same seven-day acquisition rule, future-time rejection, file locks and
immutable atomic publication used for OHLCV apply here. Failed refreshes preserve
the previous query and data; old year-only files are retained but not inferred
to cover any automatic query.

Prepared execution configs retain `path: null` and `raw_path: null` for automatic
DVOL, using the already resolved data roots. Preparation must not convert an
automatic query into an unchecked local file. Explicit feature paths remain local
file inputs when present, with existing exact-join/forward-fill behavior; they
are not refreshed merely because a requested period changes. Missing explicit
DVOL files can be downloaded from Deribit to the specified path. A `raw_path`
requires an explicit `path`; omit both to use managed caching.

All automatic DVOL downloads require `source: deribit`, including pathless
managed queries and downloads to missing explicit paths. Existing explicit
local files may use another source label; they are read without downloading.

Inspection exposes `query` and `query_reusable` for managed DVOL. An unqueried
interval has no path or query, unknown row count and `query_reusable: false`;
a successful empty query has zero rows and its receipt. Explicit local files
have `null` query fields because no automatic query evidence is claimed.
Sampling reuses only a receipt for the original requested interval that satisfies
the reuse policy. Otherwise it fetches bounded selected timestamps into the
sample directory, leaving source caches unchanged and publishing no evidence
that the full research interval was queried.

## Bounded Historical Sampling

`DataService.load_sample` and crawler `fetch_sample` remain independent data
utilities; MCP confirmation preparation does not call them. Sampling uses the
selected market-data pipeline and effective settings and searches bounded calendar
windows within the requested dates, continuing past empty market-closure windows until it reaches the bar budget or end date.
Market requests normalize their bounds to UTC before selecting reusable cache receipts
or provider intervals. An offset timestamp near New Year can therefore select
the preceding or following UTC year. Reversed intervals fail before data access;
one inclusive instant remains valid for the data utilities.
Cached Parquet data is read in batches, interpreting timezone-naive timestamps
as UTC just like ordinary loading. Feature caches project only the timestamp
and requested value column, retaining exact matches to selected market bars.
Join cardinality is checked before materializing duplicate-key expansions;
Both sampling and ordinary loading use exact left joins followed by forward fill
from earlier matched values. Leading gaps remain missing; later observations
never fill earlier bars. Unmatched feature timestamps do not seed the fill.
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
is limited to 128 MiB. Exceeding a limit fails the sampling operation.
An archive is still downloaded in full before parsing, so sample bar counts do
not imply proportionate network traffic. These are application-level resource
controls, not an operating-system memory limit or a Python sandbox.

## Execution Plans and Confirmation Documents

See [Execution Plans](execution_plans.md) for offline preparation, saved confirmation
documents, explicit user decisions, expiry and submission deduplication.
A plan fixes the manifest; preparing it does not start a formal job or backtest.

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

### Discovery Pages

MCP `list_strategies`, `list_runs`, `list_jobs`, `list_available_data` and
`list_artifacts` return an object with `items`, `offset`, `limit`, `total` and
`next_offset`, rather than a bare list. `offset` is a nonnegative integer;
`limit` defaults to 20 and accepts 1–100. Booleans and numeric strings are not
valid page bounds. `total` counts all matching items before paging, while an
offset at or beyond the total returns an empty page. Follow `next_offset` until
it is `null`; do not treat a first page as the complete inventory.

| Tool | Filters applied before paging | Order |
| --- | --- | --- |
| `list_strategies` | `kind`, `strategy_id` | strategy ID, kind, revision ID |
| `list_runs` | `strategy_id`, `revision_id` | creation time descending, run ID ascending |
| `list_jobs` | `status`, `job_type`, `strategy_name` | creation time descending, job ID ascending |
| `list_available_data` | `source`, `market_type`, `symbol`, `timeframe` | source, market type, symbol, timeframe |
| `list_artifacts` | `run_id`, `artifact_type` | creation time descending, artifact ID ascending |

Filters are exact matches, including symbol case and punctuation.
Ordering is deterministic for unchanged data; separate page calls
are not a snapshot across concurrent writes. Restart at offset zero if the
inventory changes before claiming complete discovery.

Strategy pages retain identities, lifecycle status and execution capabilities;
`get_strategy` provides source, configuration and full metadata. Run pages
contain identities, not metrics or growing scope catalogs; use `get_run` and
scoped metric/history tools for details. Artifact pages omit producer metadata;
`get_artifact` retrieves it. These are record-count bounds and deliberately
small projections, not a universal byte or token budget for arbitrary fields.
The internal Python service list APIs remain complete lists for their existing
application callers; only the MCP discovery contract uses these pages.

### Reading Artifact Text

`get_artifact` returns metadata by default. With `include_content=true`, it
returns one original UTF-8 text page: `offset` counts Unicode code points and
`limit` defaults to 8192, with a maximum of 32768. `content_page` reports
`offset`, `limit`, `total_characters`, `next_offset` and `content_sha256`.
Follow `next_offset` until it is `null`; concatenate every page before parsing
JSON or hashing a whole report. Use structured result/trade queries when only
selected observations are needed, rather than reading the entire artifact.

Every content request streams the complete file to validate UTF-8, count
characters and hash its original bytes, while retaining only the requested
page. `content_sha256` identifies the bytes observed by that read; it is not a
claim that the file still matches the registered `artifact.sha256`. Send the
first page's `content_sha256` as `expected_sha256` on subsequent requests to
reject changed content. Reading pages does not rewrite files or normalize JSON.

Unknown IDs return `artifact_not_found`; missing files return
`artifact_file_missing`; other file I/O failures return
`artifact_file_unreadable`. Invalid UTF-8 returns `artifact_not_text`, including
invalid bytes outside the requested page. Observed mutation or a mismatched
expected digest returns `artifact_content_changed`. Invalid page bounds or
paging options without `include_content=true` return `invalid_artifact_request`;
the MCP input schema may reject malformed arguments before service execution.
Metadata lookup does not read file content and remains available for binary
artifacts. A server-local path is not a public download URL.

### Completed Run Files

The generated revision's `config.yaml` holds its base settings and, for guided
authoring, research provenance and its human-readable name;
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

Each successfully published background run, including a comparison child, writes
files under `workspace/runs/<run_id>/` and records matching SQLite metadata:

- `result.json`: serialized metrics, stored as `result_json`.
- `performance.json`: all metric values by scope, definitions and calculation
  metadata, stored as `performance_json`.
- `observations.json`: aligned timestamps, equity, per-bar returns and trade
  records by scope, stored as `observations_json`.
- `manifest.json`: the approved execution specification, registered after
  successful result persistence as `execution_manifest`. For a single job, the
  file is published at submission; a comparison child's file is published from
  its approved parent bundle when saving that child's result. Its artifact metadata
  includes `manifest_hash` and `schema_version`. Use `list_artifacts` and
  `get_artifact(include_content=true)` to inspect the JSON after completion.
- `config.yaml`: effective config snapshot used for the run, stored as
  `config_snapshot` with `config_hash`. It projects the executed manifest config;
  persistence checks the worker's snapshot against that manifest and the job's
  digest rather than reloading a possibly changed YAML file. It includes the
  resolved research symbol, timeframe, and date range together with fixed defaults
  and their recorded origins when available.
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
  as `pipeline_result` for completed backtest and walk-forward runs, including
  successful ordinary-backtest comparison children.

For background runs, SQLite `runs.artifact_dir` must match the corresponding
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

The CLI cache fingerprint requires the executed manifest hash and combines it
with the processed-data file's content hash and the executed strategy revision
and source hash. It does not derive code identity from the current Git checkout
or use file size/mtime as data identity. There is no YAML-based key fallback.
New settings produce a different manifest and therefore
a different cache key. Completed results are loaded by run ID through the
registered artifact path, without recomputing a key from current YAML, data, or
code. The unused YAML-based `load_cached_result` and `save_cached_result` helpers
have been removed; this storage path does not automatically reuse prior backtests.
These fingerprints describe the execution inputs; they do not archive immutable
copies of all data or environment dependencies and do not guarantee full
reproducibility.

Training trials share the same loaded frame values, column dtypes and index
metadata, including timezone and timestamp precision. Worker transfer uses
joblib's DataFrame serialization without a lossy JSON conversion; the first trial
and subsequent candidates therefore evaluate the same data. Out-of-sample
evaluation preserves the corresponding loaded test frame as well.

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

See [Performance Artifacts](performance_artifacts.md) for scope identities, saved
metric definitions and units, observations, unavailable values and publication
integrity. Historical reads use saved evidence rather than rerunning strategies.

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

## Research Notes and Archives

Editable research summaries live in the SQLite `research_notes` table. Versioned
exports live under `research_archives/`; they do not modify runs or execution
evidence. See the [notebook contract](reports.md#research-notebooks-and-archives)
for schemas, capacity, publication integrity and HTTP/stdio download behavior.

## Composable Offline Reports

See [Reports](reports.md) for section selection, detail navigation, complete exports
and immutable HTML publication. Report delivery includes the link and instructions
for opening the selected results; tool success alone does not establish delivery.

## JSON Values

Before writing `result.json` or SQLite `runs.metrics`, metric values are
recursively normalized: `NaN`, `Infinity`, and `-Infinity` become JSON `null`,
including values inside nested objects and arrays such as optimization metrics.

Legacy results are also normalized in memory when `JobStore.load_result` reads a
result file or `SQLiteStore` reads run metrics. These reads do not rewrite the
stored file or database row. Consequently, `get_run` and completed
`get_job_status` responses expose legacy non-finite metric values as `null`.

In contrast, `get_artifact(include_content=true)` returns pages of the original
UTF-8 file text without parsing or normalizing its contents. A legacy `result.json` may
therefore still contain `NaN` or `Infinity` even when result lookup returns `null`;
the content-equivalence statement above does not apply to these legacy artifacts.
The CLI displays missing or non-finite metrics as `N/A`, while keeping finite zero
values visible as zero.
