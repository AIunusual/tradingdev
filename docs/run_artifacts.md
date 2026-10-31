# Run Artifacts

This contract explains saved identities, authority, retention and historical reads.
TradingDev stores indexed runtime metadata in `workspace/tradingdev.sqlite` and
files under `workspace/`. For authoring use [Strategy Contract](strategy_contract.md);
for approval use [Execution Plans](execution_plans.md).

## Artifact References

Use the relevant definition when implementing a writer, reader or inspection tool.
These repository references serve developers; MCP clients discover input/output
schemas through `tools/list` and use the [lookup tools](#mcp-lookup).

| Information needed | Maintained definition |
| --- | --- |
| Execution fields, schema version and digest | [ExecutionManifest and OptimizationSpec](../src/tradingdev/domain/execution.py) |
| Comparison membership and candidate limits | [ComparisonManifest](../src/tradingdev/domain/comparison.py) |
| Research explanation fields and bindings | [ResearchEvidenceDraft, BoundResearchEvidence and bind_research_evidence](../src/tradingdev/domain/research_evidence.py); MCP `get_strategy_contract(section="evidence")` supplies authoring guidance |
| Runtime table columns and record updates | [SQLiteStore](../src/tradingdev/adapters/storage/sqlite.py), [ExecutionPlanStore](../src/tradingdev/adapters/storage/execution_plans.py) and [ResearchNoteStore](../src/tradingdev/adapters/storage/research_notes.py) |
| Publishing run files and registering artifact types | [RunPublisher](../src/tradingdev/app/run_publisher.py) and [PerformanceStore](../src/tradingdev/adapters/storage/performance.py) |
| Saved metrics, observations and reporting | [Performance Artifacts](performance_artifacts.md) and [Reports](reports.md) |

The following sections retain cross-file meaning and failure behavior; they are
not a second field-by-field schema. A digest detects inconsistency only while its
expected value remains trusted, not against a writer controlling both records
and files.

## Workspace Layout

Choose a location by the artifact's owner; a path alone does not prove a complete
result exists. Standard roots are resolved by
[WorkspacePaths](../src/tradingdev/adapters/storage/filesystem.py).

| Location under the workspace | Owner and use |
| --- | --- |
| `generated_strategies/<strategy_id>/revisions/<revision_id>/` | Immutable `strategy.py` and base `config.yaml`, plus lifecycle `metadata.json`; the strategy's `current.json` selects the latest saved revision. |
| `generated_strategies/.locks/` | Per-strategy lifecycle locks, outside deletable revisions. |
| `configs/`, `data/raw/`, `data/processed/` | Execution projections and configured data/cache roots; see [market cache](#market-data-cache). |
| `runs/<run_id>/` | [Completed run files](#completed-run-files); a standalone job's manifest is already published here at submission. |
| `comparisons/<job_id>/` | Shared manifest/config, ledger and post-approval data snapshot/receipt; see [comparison evidence](#fixed-comparison-evidence). |
| `.workers/<launch_token>/` | Supervisor startup, readiness, stop request and cleanup acknowledgement; see [worker identity](#job-state-and-worker-identity). |
| `research_evidence/`, `submitted_research/` | Draft explanations and approved job snapshots, described below. |
| `execution_plans/<plan_id>/` | Saved confirmation text/HTML and approval lock; see [Execution Plans](execution_plans.md). |
| `ml_datasets/`, `ml_feature_cache/`, `models/`, `ml_leaderboard/` | Independent [ML stores](ml_research.md#ml-research-artifacts). |
| `reports/`, `research_archives/`, `feature_requests/` | Derived reports, notebook exports and recorded unsupported requests; see [Reports](reports.md). |

Strategy validation/dry-run use separate `tradingdev-strategy-check-*` system
temporary directories, not research runs. Request/response evidence binds the
selected revision and fixture; caches and supervisor controls stay inside the
temporary directory. Cleanup happens only after owned processes stop. Unverified
shutdown or failed file removal returns failure with the retained path, never
passing evidence. Guarantees and exclusions are in the
[strategy security model](strategy_contract.md#security-model).

## Research Evidence Records

`ResearchEvidenceService` saves client-authored explanations without fetching
sources, executing strategies or acquiring market data. Each
`research_evidence/<evidence_id>.json` is a size-bounded canonical UTF-8 record
with creation time, identity and content digest.
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

Capacity is checked on both the saved draft and expanded binding, so a successful
save can still fail preparation. Optimization binds the base and each search-axis
option, not every Cartesian trial. Values also have independent finite/acyclic
JSON, depth, node, string and key limits. For exact budgets, read
`MAX_PARAMETER_TARGETS`, `MAX_PARAMETER_BINDINGS` and the models in
[research_evidence.py](../src/tradingdev/domain/research_evidence.py), plus
`MAX_RESEARCH_EVIDENCE_BYTES` and `bounded_json` in
[research_evidence_values.py](../src/tradingdev/domain/research_evidence_values.py).
The tool schema exposes field-level constraints. Reduce candidate complexity or
explanation size while retaining every required binding; do not silently split
the user's comparison or omit provenance to evade limits. Binding errors provide
required paths/values; save corrected evidence and prepare with the new ID.

Descriptions identify each effective constructor leaf with a distinct readable
label, explanation and optional unit. JSON pointers are relative to
`strategy_execution.constructor_kwargs` (for example `/fast_period` or
`/config/k_period`); their union covers base and effective candidates. Explanations
do not rescale values. The [confirmation contract](execution_plans.md) owns
presentation, approval and code-check coverage.

`ApprovedResearch` freezes bound declarations together with one versioned execution
policy disclosure per distinct manifest. Submission and worker startup compare the
saved policy with the simulator's current policy before acquiring data or running
the strategy. Historical reads validate the saved content and identity without
replacing its wording with today's policy.

Execution policy version and fixed simulator defaults are maintained in
[execution_policy.py](../src/tradingdev/domain/backtest/execution_policy.py); their
[signal/execution semantics](strategy_contract.md#signal-and-execution-semantics)
remain distinct from source-code validation. A policy change requires preparation
and confirmation again; historical results and disclosures are not rewritten.

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

The core index stores jobs, completed runs, registered artifacts and job events.
Plans and editable notes have separate table-owning stores; use the
[table definitions](#artifact-references) for exact columns and JSON payloads.
Schema initialization is not permission to migrate old execution evidence.

`job_id` and `run_id` are currently the same for completed standalone backtest,
walk-forward and optimization jobs. `get_job_status(job_id)` returns the `run_id`
once such a run is done. A fixed comparison has a parent job and separate candidate
run IDs recorded in its ledger; the parent has no performance run of its own.
Generated strategy jobs and runs carry their selected `revision_id`;
bundled strategies and historical records without revision identity expose
`null`. A successful explicit revision save changes only that strategy's current
pointer and does not change the
revision used by an existing job or completed run.

### Run Publication

`RunPublisher` verifies the owning job, strategy/revision, executed manifest,
configuration, source and performance identity before registering a result.
`RunPublicationBinding` identifies job, run and expected manifest, but comparison
children must also match the saved parent bundle and ledger; binding alone is not
approval evidence. Each child has its own manifest and run-scoped artifact IDs.
`JobStore.save_result` retains the single-run interface with `run_id == job_id`.

Publication locking, rollback and retries cover both files and lineage. Job
creation and child publication share a run-ID submission lock, preventing collisions
between jobs and child runs. Lock files remain for concurrent waiters. Files and
SQLite are not one transaction; complete verification and recovery rules are in
[Performance Artifacts](performance_artifacts.md#performance-scopes-and-provenance).
Artifact metadata differs by producer; resolve ownership through the registered
artifact's run and the run's `job_id`, not a presumed universal metadata field.

Draft cleanup scans raw jobs, run references and all stored manifests, including
orphans and failed/cancelled jobs. Ambiguous history blocks deletion. Lifecycle
locks stay outside revision directories and are not unlinked while waiters may
use them. Eligibility and partial-failure behavior are defined once in
[Draft Cleanup](strategy_contract.md#draft-cleanup).

Preparation returns a plan and manifest hash, not a job. Approval returns the
submitted job ID. Status/list responses use the parent hash; comparison child runs
carry their individual hashes and parent job ID. Missing historical manifest
identity stays `null`. Old jobs cannot resume through new workers: prepare a new
plan under a fixed specification. The removed `start_*` and `confirm_optimization`
MCP tools are not execution entry points.

### Job State and Worker Identity

Job field updates merge indexed columns and JSON payload within one SQLite
`BEGIN IMMEDIATE` transaction; execution identity is immutable. The first terminal
state wins, including its timestamps and error. Late process identity may be
attached without restoring an earlier lifecycle state. Poll/cancel operations
condition writes on active state and return the state actually saved, so concurrent
completion is not reported as cancellation.

Elapsed time ends at the saved terminal timestamp; only active jobs use query
time. Missing, invalid, timezone-less or reversed timestamps return an explicit
unavailable reason and `null`, without rewriting legacy records.

The job PID identifies the supervisor, not the strategy child. Its OS creation
time and unique `worker_control_id` bind a launch to `.workers/<launch_token>`;
do not trust a potentially reused PID alone. Control records remain in normal
workspaces for later cleanup verification. Temporary tests remove them only after
verified shutdown.

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

Cancellation requires a complete control identity before signalling a running
process; queued jobs without a process can be cancelled directly. Status checks
fail a started job whose supervisor identity is missing/mismatched. Comparisons
first require verified worker cleanup; failure leaves lifecycle states unchanged.
Cancellation likewise preserves states on cleanup failure. Both return the saved
job, so concurrent `done` may coexist with a cleanup diagnostic.

Status failure returns its persisted error immediately; launch failure saves the
error and terminal timestamp before propagating. For response placement and codes,
read [JobService](../src/tradingdev/app/job_service.py) and its
[job response models](../src/tradingdev/app/contracts/jobs.py). A cleanup diagnostic
does not prove the process stopped; resolve it before deleting its workspace.

## ML Research Artifacts

Dataset, feature-cache, model and leaderboard identities and retention are defined
in [ML Research](ml_research.md#ml-research-artifacts). They use separate stores,
not backtest run IDs or the SQLite artifact index.

## Research Settings and Provenance

Research intake is not a separate persistent session. The shared resolver writes
one effective config before a generated revision or execution manifest is saved.
The required request modes, cost defaults and annualization rules are described
in the [research-input contract](strategy_contract.md#research-inputs-and-simulation-policy).

Effective research values live in their execution locations (`backtest`, the
outline in `research`, and the name in `strategy.display_name`); provenance does
not store a second copy. For exact paths, version and field constraints, read
`RESEARCH_FIELD_PATHS`, `ResearchMetadata` and `ResearchRevisionIdentity` in
[research.py](../src/tradingdev/domain/research.py).

Origins distinguish request, project default and explicitly selected revision.
They describe resolution paths, not proof of human input or approval. Inheritance
retains the selected identity and config hash. Revision `request_summary` is a
derived outline projection, not independent configuration authority.

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

Revision creation, namespace collisions and current-pointer publication are
covered by [Revision Identity and Eligibility](strategy_contract.md#revision-identity-and-eligibility).

## Fixed Comparison Evidence

`ComparisonManifest` fixes a bounded ordered set of ordinary-backtest candidates,
including labels, rationale and complete manifests. Only effective constructor
parameters may differ; strategy/revision, research, data settings, execution
assumptions and seed must match. Labels are unique after trimming; repeated
effective parameter sets are rejected. Candidate count and field constraints
come from the [comparison definition](#artifact-references) and tool schema.

`prepare_comparison` captures the selected revision once without rewriting its
base. It accepts candidate-specific overrides, not a global `parameters` argument.
All candidates, including bundled strategies, pass both synthetic fixtures under
one shared preparation deadline. One confirmation covers the complete bundle;
no market data, engine or training runs first. Training-required and walk-forward
comparisons are unsupported. Fixed comparison performs no adaptive search or
automatic winner selection. See [Execution Plans](execution_plans.md) for receipts.

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
and report rather than being repeated on every poll. Generate reports with
`comparison_job_id` to retain every candidate, including failed or unexecuted ones.

## Execution Manifest

For a single execution, before spawning a background worker, the service publishes
`runs/<job_id>/manifest.json` and records its hash in the job. Publication never
replaces a different existing manifest. Atomic publication applies to the manifest
file; file publication and the database job write do not form one transaction.
The [ExecutionManifest definition](#artifact-references) owns the supported schema,
fields and digest calculation. `config` holds effective execution settings and
resolved absolute data/feature paths. `strategy_execution` separately captures
constructor values, including nested bundled config/fit defaults; injected engine
and parallel objects are supplied from execution configuration, outside this JSON.
Research origins and display names are saved metadata, not constructor arguments.

Optimization fixes candidate ranges, objective/direction, calendar train/test
bounds and the first-trial estimation timeout. The timeout limits only that
training trial; remaining candidates and held-out evaluation do not share it.
It is separate from MCP preflight's total process budget. Parameter names are
sorted while each caller-supplied option order is preserved. User approval occurs
before worker launch, not inside the worker.

`manifest_hash` covers canonical JSON including schema version and excluding the
hash itself. Job IDs and creation times are outside this specification.

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
Mappings merge recursively; other values replace the specified parameter. Empty
objects preserve existing mappings, and `**kwargs` does not permit new override keys.

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
For refusal codes and caller-specific error handling, use the relevant
[MCP execution tools](../src/tradingdev/mcp/tools/execution.py),
[JobService](../src/tradingdev/app/job_service.py) or
[OptimizationService](../src/tradingdev/app/optimization_service.py).
MCP preparation exceptions are mapped in
[`execute_preflight`](../src/tradingdev/adapters/execution/preflight_worker.py)
and forwarded by [PreflightService](../src/tradingdev/app/preflight_service.py).
The lower-level `BacktestService` raises exceptions rather than service responses.

The mode must match the presence of walk-forward
validation settings. Optimization has its own training/test split and rejects
configs containing `validation` with `invalid_optimization_request`.
Walk-forward replacement, mode-specific fields and inclusive calendar containment
follow [Strategy Walk-Forward Settings](strategy_contract.md#strategy-walk-forward-settings).
Invalid windows do not change the requested dates or publish a plan.

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

Custom crawlers must implement `fetch_sample(..., max_rows=...)` with bounds
enforced before frame materialization, including inclusive single-instant requests.
API sampling uses bounded pages/windows; streamed responses reject redirects and
extra HTTP compression before body reads. Binance Vision prefers daily archives,
falls back to monthly on missing daily files, and scans bounded CSV batches while
retaining earliest requested bars even if stored rows are unordered. It stops
before later-date archives once the row budget is satisfied.

For exact streaming, expansion and per-call budgets, inspect the selected
`fetch_sample` implementation through the [crawler modules](../src/tradingdev/domain/data/crawlers/)
and their shared [sampling helpers](../src/tradingdev/domain/data/crawlers/sampling.py).
Exceeding a limit fails; archives are downloaded in full before parsing, so bar
count is not proportional to traffic. These are application resource controls,
not OS memory limits or a Python sandbox.

## Execution Plans and Confirmation Documents

See [Execution Plans](execution_plans.md) for offline preparation, saved confirmation
documents, explicit user decisions, expiry and submission deduplication.
A plan fixes the manifest; preparing it does not start a formal job or backtest.

## MCP Lookup

Choose queries by purpose; exact arguments, bounds and output fields come from
the registered MCP schemas. Tool definitions and response models are maintained in
[`mcp/tools/`](../src/tradingdev/mcp/tools/) and
[`app/contracts/`](../src/tradingdev/app/contracts/).

| Need | Entry |
| --- | --- |
| Job progress or cancellation | `list_jobs`, `get_job_status`, `cancel_job` |
| Saved result identity and scopes | `list_runs`, `get_run`, `find_runs` |
| Current metric definitions or saved values/comparisons | `get_metric_catalog`, `get_run_metrics`, `compare_runs` |
| Selected saved observations | `get_run_trades`, `get_run_equity`, `get_run_executions`, `get_run_account_history` |
| Artifact metadata or original file text | `list_artifacts`, `get_artifact` |

### Discovery Pages

Growing discovery lists use [Page and PageRequest](../src/tradingdev/app/contracts/common.py).
Use each tool's schema and description for accepted filters, bounds and ordering.
Filters apply before paging; `total` counts all matches. An offset at or beyond
the total returns an empty page. Follow `next_offset` until `null` before claiming
complete discovery; a first page is not the full inventory.

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

`get_artifact` returns metadata by default. Set `include_content=true` for one
original UTF-8 text page. Offsets count Unicode code points, not bytes. Bounds and
page fields are defined by the [artifact tool](../src/tradingdev/mcp/tools/artifacts.py)
and [read_text_page](../src/tradingdev/adapters/storage/artifact_text.py).
Follow `content_page.next_offset` to `null` and concatenate all pages before parsing
JSON or hashing a whole report. Use structured queries for selected observations.

Every content request streams the complete file to validate UTF-8, count
characters and hash its original bytes, while retaining only the requested
page. `content_sha256` identifies the bytes observed by that read; it is not a
claim that the file still matches the registered `artifact.sha256`. Send the
first page's `content_sha256` as `expected_sha256` on subsequent requests to
reject changed content. Reading pages does not rewrite files or normalize JSON.

Missing/unreadable files, invalid UTF-8 anywhere in the file, changed content and
invalid paging requests fail explicitly rather than returning partial evidence.
Metadata remains available for binary artifacts and does not read their contents.
Exact diagnostics are maintained in [ArtifactService.get_artifact](../src/tradingdev/app/artifact_service.py).
A server-local path is not a public download URL.

### Completed Run Files

A revision's base YAML remains unchanged; a run's config projection contains its
fixed effective settings. [Execution Manifest](#execution-manifest) defines the
authority, overrides and hashes. For background runs, including comparison children,
the following files are independently registered under `runs/<run_id>/`:

| File | Purpose |
| --- | --- |
| `result.json` | Numerical result projection; normal job/run responses may select only a summary. |
| `performance.json`, `observations.json` | Complete saved metrics, definitions and observations by scope; see [Performance Artifacts](performance_artifacts.md). |
| `manifest.json` | Approved execution specification; single-job publication occurs at submission, child publication when saving its approved result. |
| `config.yaml` | Inspection projection checked against the executed manifest, without reloading caller YAML. |
| `strategy.py` | Source snapshot when a source path is available; generated source is from the selected immutable revision. |
| `dataset_fingerprint.json` | Dataset identity and fingerprint, not a complete archive of inputs. |
| `pipeline_result.pkl` | Dashboard cache for completed backtest/walk-forward runs, including successful backtest comparison children. |
| `research_evidence.json` | Independently readable approved explanation when recorded; see [Research Evidence Records](#research-evidence-records). |

For exact artifact type identifiers and metadata, use the [publishing references](#artifact-references)
or `list_artifacts`/`get_artifact`; do not infer registration from file presence.

For background runs, SQLite `runs.artifact_dir` must match the corresponding
`workspace/runs/<run_id>/` directory. `runs.config_hash` is the SHA-256 of the
serialized effective config snapshot, not the hash of the immutable revision's
base config. `runs.random_seed` records only the canonical
top-level `random_seed`; strategy-specific model parameters are not inferred as a
run seed. Run comparison,
dashboard rendering, and artifact lookup always read through SQLite first, then
resolve files from the recorded artifact paths.

CLI runs keep dashboard pickles under the processed-data `cache` directory, while
`runs.artifact_dir` points to `runs/cli_<cache_key>_<execution_id>/` with the
registered manifest and shared performance/observation JSON. Each save gets a
fresh execution ID and separate pickle; rerunning the same request can preserve
different results. Clearing the pickle leaves historical JSON readable but its
dashboard unavailable. No previous backtest is automatically reused.

The pipeline result embeds the executed config and manifest. Saving uses those
snapshots, not a later YAML read; `config_path` is descriptive metadata. The cache
key combines executed manifest, processed-file content and strategy revision/source
hashes, not current Git state, file size/mtime or a YAML fallback. Historical reads
resolve the registered run artifact without recomputing identity from today's
inputs. These fingerprints do not freeze all data or dependencies or guarantee
reproducibility. Inspect `compute_cache_key` in [cache.py](../src/tradingdev/shared/utils/cache.py)
and [ArtifactService](../src/tradingdev/app/artifact_service.py) when changing
dashboard cache publication or lookup.

Optimization trials share the same loaded frame values, dtypes and index/time
metadata; held-out evaluation preserves its corresponding test frame too.

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
