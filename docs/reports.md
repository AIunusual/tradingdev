# Reports

Offline report composition, navigation, export and publication contracts.
Reports read saved [performance artifacts](performance_artifacts.md) and
explicitly selected [ML evaluations](ml_research.md#recorded-signal-evaluations);
[Run Artifacts](run_artifacts.md#mcp-lookup) describes source lookup.

## Composable Offline Reports

`get_report_sections` returns built-in section IDs and optional `standard`,
`comparison`, `trades` and `ml_research` recipes. `generate_report(run_ids, sections, commentary)`
accepts one to eight distinct runs.
Saved ML candidates can instead be selected with `ml_entry_ids`, either alone or
alongside one of the run/comparison selectors. Select 1–50 distinct entry IDs from
recorded evaluations or leaderboard results; never infer a model-to-run relationship.
The ML block and its presentation are absent when no ML evidence is selected.
Newly generated reports include interpretation and diagnostic fields under template
version 24, so their identity differs from earlier templates even without ML.
Previously saved reports are not rewritten.
Alternatively, pass `comparison_job_id` with no `run_ids` (CLI:
`--comparison-job-id`, exclusive with repeatable `--run-id`). The parent must be
terminal with a complete ledger. This selector includes every approved candidate;
successes link to existing run detail views, while failures, cancellations and
unexecuted rows show reasons without fake metrics or detail links. Even `sections=[]`
retains the complete candidate table and effective parameters/rationales. Zero
successful runs still produce a report. The report identity includes the bundle,
ledger, verified snapshot/receipt and successful child artifacts. A failed or
cancelled job with only `not_run` candidates may have no snapshot if loading failed
before publication; missing evidence for any attempted candidate is an error.
Final conversational delivery may identify the complete comparison by its exact
parent job ID and result scopes, with the report link and detail-view instructions.
Interpret every candidate using readable parameter names; exact internal labels
and child run IDs remain in the report and need not be repeated in conversation.
The parent's binding and report completeness are mechanically checked; natural
language completeness, parameter/metric correspondence and navigation wording
also require semantic review. Single-run delivery still identifies its run ID.

`app/report_ml.py` reads one complete leaderboard file generation; missing IDs,
corrupt records and duplicate stored identities fail without publishing a partial
report. It does not load model pickle, retrain, rerun predictions or download data.
Leaderboard records carry the training dataset of the artifact they name, when it
records one. Equality with the evaluation dataset means the score is in-sample only
when the backend produced the signal from that artifact; under a caller-supplied
signal it says only where the named artifact was fitted. A different value does not
establish a holdout. A model entry with an empty value leaves the relationship
unknown. Only a rule produced by backend rule evaluation has its training dataset
labelled not applicable. Caller-supplied and unknown-source signals retain unknown
training history even when declared as rules, including whether an external model
produced them. Report commentary and final interpretation must
state that condition rather than infer a training relationship from model or dataset
identities alone.
The selected candidates must satisfy the
[ML comparison rules](ml_research.md#recorded-signal-evaluations).
The MCP/CLI options `ml_ranking_metric` (default `ic_mean`) and
`ml_allow_parameter_axes` (default `false`) select ranking and the explicit surface
opt-in. CLI uses `--ml-entry-id` repeatedly, `--ml-ranking-metric` and
`--ml-allow-parameter-axes`. Without selected IDs, non-default ML options are
invalid. Selecting the `ml` section without evidence is an error.

For either run/comparison selector, omitted/null sections select standard
(plus `ml` when ML candidates are also selected); ML-only requests default to `ml`.
An explicit ordered list chooses visible sections and their order within the detail
view. When selected, `overview` always belongs to the first navigation level.
With `[]`, no optional sections are rendered. Fixed report/source identity metadata,
research interpretation (or its missing-analysis notice), diagnostics for every
loaded backtest scope and any supplied commentary remain visible. Section selection
cannot hide interpretation or diagnostics. Unknown or duplicate sections fail.
Up to 20 plain-text title/text notes
(20000 characters total) are escaped and labelled LLM commentary. They cannot
replace computed values or inject HTML. The client selects information and writes
interpretation; the server renders tables, charts and document structure.

Report schema 3 / template 24 uses two navigation levels within one offline HTML
file when at least one detail section (`settings`, `metrics`, `benchmark`,
`equity`, `ledger`, or `trades`) is selected. The overview compares every loaded
run/scope, including effective parameters, date coverage, train/test role,
principal saved metrics, and the baseline symbol, its buy-and-hold return and the
strategy's excess return over it.
Interactive and static overview tables show explicit `查看明細 →` links in the
result-sequence column, with a separate navigation hint beside the table.
The interactive table also supports plain row clicks; keyboard focus alone does
not navigate. A result link opens
the matching detail view (`#result-<run-index>-<scope-index>`); a return link brings
the reader back to the overview. This keeps all optimization training trials and
the selected parameter's test result distinct. A report is generated explicitly
after completed or historical result inspection, not automatically by execution.
MCP instructions request a standard report when completing an execution with
performance/risk interpretation, unless the user asks for summary/metrics only
or opts out. This is a client workflow requirement, not an automatic worker step.
The final user-facing response must deliver the returned report link, identify
the selected runs/scopes or ML candidates and explain the selected views.
`generate_report` returns
`navigation_instructions` from the same section-aware definition used by the HTML
introduction; this text is also saved in the payload and manifest. The client must
include it with the final link. It does not advertise unselected detail sections or
trade links for a comparison with no successful results. Tool success and an
intermediate link do not establish delivery. Reports and their source runs remain
workspace artifacts after server shutdown; shutdown does not authorize cleanup.
If report generation or delivery fails, the client should preserve those sources
and retry reporting instead of running the strategy again.
Without `overview`, a result chooser links to the selected detail sections.
Without any detail sections, result links and result switching are absent;
selected `provenance` and `limitations` sections appear directly in the main view.
The overview, if selected, stays first; other sections keep their requested order.
Unknown result hashes leave the main view visible rather than opening empty details.
The standard recipe is `overview`, `research`, `settings`, `metrics`,
`benchmark`, `equity`, `ledger`, `trades`, `limitations`, `provenance`;
comparison is `overview`, `research`, `metrics`, `benchmark`, `equity`,
`provenance`; trades is `overview`, `research`, `ledger`, `trades`, `provenance`.
The `ml_research` recipe contains `ml`. ML evidence stays in the main view and is
independent of the run/scope detail switcher; its own template version lives in
the optional block so future ML presentation changes affect only ML reports.
The `research` section belongs to the main view. It displays saved trading formulas,
plain-language rules, parameter origins and source claims, limitations and versioned
execution assumptions. Scalar backtest scopes show their own parameter origins
and saved policy. Aggregate walk-forward summaries do not repeat per-fold origins
or execution assumptions; inspect their original train/test folds instead.
Shared research snapshots are stored once under their content hash in
the report's `research_documents`; runs reference that hash. A comparison with no
successful runs still retains its parent's disclosure. Explicitly omitting the
section changes visibility, not the saved payload or evidence validation. Missing
legacy explanations remain labelled as absent; reports never synthesize formulas
or infer that reference URLs were actually visited.
The old `executions` and `account_history` section IDs are invalid, with no alias
or compatibility shim. Their MCP history query tools and saved observations are
unchanged. Existing reports retain their original versions and remain untouched.

Section selection determines the optional presentation without pruning the loaded
or embedded data. Every HTML report embeds the complete loaded payload as JSON,
including all scopes of the requested runs, saved metrics, trades, equity,
execution configuration and selected ML evidence, even for `sections=[]`.
Omitting a section does not remove its underlying data from the HTML; recipients
of the file can still read that embedded data.

The HTML ML section uses one rule for default visibility: show a metric when
it is a summary metric and it declares an optimisation direction. Anything else
is stored in full but marked hidden by default. `beta` is the case
worth naming: it is a summary metric with no direction, so it describes exposure
rather than quality. A block reports `visible_metrics` and `hidden_metrics`
alongside the data. The HTML shows default metrics and lets readers expand other
metrics, candidate identities/settings and definitions, or download the entire
ML block as JSON. Null scores remain in each scope regardless of visibility;
HTML distinguishes unavailable scores, unrecorded keys and finite zero. The
ordering is not a guarantee of significance or out-of-sample performance.
ML tables retain saved numeric precision and unscaled fractional values; for
example, a hit rate of `0.2` remains `0.2`, not `20%`. Interpret units and meanings
using each metric's saved definition. The backtest tables' two-decimal and
percentage formatting rules and precision controls do not apply to ML tables.
ML JSON downloads use pre-encoded UTF-8 bytes, preserving numeric precision
without a browser JSON parse/stringify round trip.
Candidate details show the saved method family and name. Parameters come from
the model artifact or, for rules and other signals, the caller's declaration;
rule thresholds are not described as fitted model parameters. Family labels do
not establish training/selection history or out-of-sample validity.
The candidate's training dataset is shown when the artifact records one. Only a
saved backend rule origin establishes that model training does not apply;
caller-supplied and unknown-source signals retain unknown training history even
when their declared family is rule. Source interpretation follows the
[recorded evaluation contract](ml_research.md#recorded-signal-evaluations);
no missing source is inferred.

Selecting `metrics` also derives a native QuantStats HTML tear sheet for every
eligible scalar scope. It uses saved bar returns grouped by observed UTC date,
preserves the first return, does not fill missing dates and does not fetch market
or benchmark data. Each scope remains independent. The adapter records its
version, calculation/rendering package versions, saved settings, source artifact
hashes, input/daily-return digests and native calculation conventions. The native
HTML and metadata participate in report identity; `manifest.derived_analytics`
contains the metadata and explicit unavailable reasons, without duplicating HTML.
Original `performance.json` values and definitions are never replaced.

Native performance tables are rendered directly in the main HTML for text readers
without JavaScript, with percentage units written in headers instead of relying
on native CSS pseudo-elements. All scopes are expanded when JavaScript is disabled.
The full chart report is loaded from embedded JSON into a
sandboxed iframe when its scope is selected. A local download preserves the native
report together with a source banner and escaped JSON evidence. QuantStats' native
metric/chart names remain English; the surrounding explanation and saved metrics
are Traditional Chinese. Period win rates and profit factors are distinct from
closed-trade statistics. Native table Sharpe/Sortino use the saved risk-free rate;
native rolling Sharpe/Sortino charts use zero, while the saved Sortino metric uses
the separate configured required return. The native EOY table's `Return` sums daily
returns within each year; `Cumulative` compounds within each year, not across years.
Both columns use percentage units. These differences are disclosed.

Known unsupported data (including volume, fold summaries, missing dates/settings,
nonpositive capital, ambiguous large daily returns and negative risk-free rates
ignored by the native provider) produce explicit unavailable messages while saved
metrics remain readable. Unknown provider exceptions become `QuantStatsReportError`
and fail generation with `report_generation_failed`; process interrupts propagate.
The renderer
fixes SVG identifiers, removes generation dates and the external favicon, restores
plot state, and removes temporary output/new figures on success or failure. It
does not close pre-existing figures or silently substitute another HTML provider.

The `benchmark` section presents the baseline saved with each scalar scope: its
fixed identity (asset class, symbol, weighting and rationale), the requested provider and
effective market setting, whether the run's own frame was reused, the costs
applied, and how many of the scope's bars the baseline observed. A comparison
table puts the strategy's saved values beside the baseline's under the same
units, with a difference only where coverage is complete, both sides are finite
and the units match. The overview includes aligned/total bar counts and withholds
excess returns for incomplete or unknown coverage. Reused frames explicitly retain
the run's unverified cache provenance; they do not independently prove the provider.
When the baseline was normalized to one quote currency unit because the run had
no initial capital, its `amount` values are withheld instead of sitting beside
the strategy's realized amounts, and no `amount` difference is computed. A normalized equity chart
plots both paths as growth multiples of the same initial capital, positioned by
bar index, so a shorter baseline never stretches across unobserved bars. This
overlay requires positive recorded strategy initial capital; when it is missing
or nonpositive, including volume results, the comparison table remains but the
overlay is omitted. The
baseline is not selected by the client: there is no request parameter for it,
and the same market always yields the same reference. An undefined asset class
or an unusable baseline is reported with its reason and never as a zero return.
The schema 3 manifest has a `benchmarks` mapping keyed first by run ID and then
by scope ID. Each entry summarizes `status`, baseline `id` and `symbol`,
`asset_class`, `source`, `dataset_id`, `reason` and `coverage`. Scopes without
saved benchmark evidence are omitted, so a run can have an empty mapping.
These manifest entries contain discovery and coverage summaries; the HTML's
embedded scope payload retains the complete benchmark evidence, including
costs, metric values, timestamps, returns and equity series, even when the
benchmark section is not selected.

Reports generated from results saved before benchmark evidence existed say so
explicitly; they are not recomputed from current market data. Walk-forward
`test_summary` scopes have no single equity path and therefore no baseline;
optimization trials currently have none because the optimization worker does not
capture one. Fold summaries say so instead of suggesting a re-run.

The `ledger` section combines saved order attempts and end-of-bar account states
into one chronological event table. Within a bar, original execution order comes
before the closing valuation. Each row identifies its event type and original
record; closing valuations never masquerade as orders or fills. Missing streams
show their reason, while `[]` means a recorded empty stream. No missing ledger is
reconstructed from paired trades. The manifest preserves each source stream's
original count, including null versus zero.

Bokeh widgets and tables provide filled/ignored/rejected/valuation filtering and
column visibility checkboxes. The runtime is bundled in the HTML for offline use.
The same ledger rows and column definitions produce the interactive view, a complete
static HTML table and full-precision CSV/JSON downloads. Static content is readable
without executing JavaScript; LLM consumers can instead use embedded JSON or the
original MCP history queries. Filters, sorting and column visibility affect only
the interactive presentation, never the exported records or columns.

Ledger columns begin with time, event sequence, bar index, event type and execution
status. They expose market OHLC, requested size and its unit/type, request reference
price, filled asset quantity and price, requested fee/slippage rates, charged fees,
and account before/change/after values. Cash flows and signed position changes are
kept distinct from fees and notional size. A percentage or value request is not
labelled as an asset quantity; an all-available request is not converted to an
invented fixed size. Closing rows carry the saved close mark and account valuation,
leaving unrecorded order/fill fields empty. Prices are saved bar references rather
than bid/ask quotes or order-book data. Valuation basis remains explicit: execution
snapshots use the request reference price, while closing states use the bar close.
Before/change/after valuation rows identify whether their prior state is the last
execution in the same bar or the preceding saved close. Without a prior state the
change stays unavailable. Arithmetic checks use complete precision and documented
tolerance to compare cash, position, fees, slippage and equity where the inputs are
recorded. A matching check does not validate the strategy, market quotes or exchange
accounting, and close records do not invent missing debt values.
The `trades` section is supplementary paired-trade PnL rather than the reconciliation
ledger; a reversal can close one paired trade and open another in a single fill.

All saved scopes are validated, including sources of omitted sections. Scalar
observations and aggregate fold statistics remain distinct. Missing or corrupt
sources fail generation. The report names selected/omitted sections rather than
claiming every suggested section is present. Included record tables expose all
records and local CSV export; SVG equity and drawdown charts
and scripts are embedded for offline use. UTC times, units, unavailable reasons,
provenance and known overlap/interpretation limits are supplied by the backend.
Static record tables are embedded as semantic HTML inside `noscript` elements.
With JavaScript enabled, their DOM is materialized only when expanded, printed or
needed after a Bokeh loading failure. With JavaScript disabled they render directly.
This retains complete source-readable tables without eagerly creating every scope's cells.
Overview, settings, metrics, provenance and record tables use Traditional Chinese
field names, known metric names, units and enum explanations. Saved metric definitions
remain authoritative and are expandable in their original form; current display
labels never replace saved units or descriptions. Unknown codes remain visible with
an explicit fallback, and strategy-defined parameters retain their original names.
Paired-trade tables keep distinct closing and end-of-period valuation fields.
The unified ledger identifies before/change/after and closing balances explicitly.
These remain generic engine accounting, not exchange margin wallets.
Prices are quote currency
per asset unit; quantities are asset units, and PnL/fees are quote currency amounts.
Backtest numeric displays default to two decimal places; each overview, ledger and
paired-trade table has one full-precision checkbox shared by its interactive and static views.
The renderer supplies source decimal strings before browser number parsing; percent
text shifts decimal places exactly rather than multiplying a browser float. The switch
retains filters, sorting and column visibility, including lazy loading and printing.
Notional value is visible by default and included in the cash/position preset.
Readers must enable full precision before recalculating from quantities and prices.
Backtest fractional rates and returns display as percentages; identifiers and
integer counts retain exact integer form. Columns containing integers outside the JavaScript safe
range disable sorting to avoid ordering rounded values. Nullable numeric parameters
keep numeric formatting; truly mixed-type parameter columns use backend text for
all non-null values, preserving exact values in both interactive and static views. Fold statistics show
labelled mean/std/min/max and valid-fold counts instead of a raw JSON cell. Precision
notes accompany the tables; supported numeric sorting uses original values even when
displays are equal. Precision controls require JavaScript; static numeric cells also
carry full-value attributes and tooltips, and downloads always preserve original values.
CSV exports use Traditional Chinese headers with units and translated known enums.
Record columns follow a defined reading order rather than code-name sorting or
object insertion order: time/sequence, direction/status, prices/quantities, costs/PnL
and account balances. Ledger HTML/CSV share the same column definitions and preserve
before/change/after balance groups. Overview comparison, ledger and paired-trade
tables each provide CSV and JSON downloads. JSON contains the full table specification:
`columns`, `rows`, `full_precision_values` and applicable navigation, filtering and
presentation metadata. It uses stable code keys and full numeric precision, rather
than exporting only an array of raw ledger records; raw observations remain
separately embedded.
Paired-trade HTML and CSV begin with a display sequence and separate entry, closing
and valuation timestamps. Both then show direction/status, prices/quantities and
costs before PnL; source indices and original record JSON remain available later in
the same column order. Consumers relying on positional CSV columns must adjust.
All rows and columns remain present in export order, independent of screen filtering,
sorting or visibility; numeric values retain full precision. Rates remain
fractions (1 = 100%), timestamp displays/exports are normalized to UTC, and
missing values are empty cells. Ledger nested source fields are flattened with
distinct headers; original records retain source fields and timestamps. Paired-trade
closing time/price are empty for open trades, whose final mark uses separate valuation
columns; the source exit index still identifies that final valuation bar.
The renderer prepares UTF-8 BOM CSV bytes with CRLF records and quoted fields;
formula-leading text is prefixed with an apostrophe. Browsers download these embedded
bytes directly without parsing or rounding numeric values. Programs consuming old
code-based CSV headers must update their mappings for newly generated reports.
Expandable raw records and embedded observations retain original field names and
values. Presentation changes create a new template version and report
identity without modifying earlier reports or backtest results.

The payload and manifest both preserve the full `ml` block, including `entry_ids`,
the ML renderer's `template_version`, and `source_sha256` over the ordered selected
`LeaderboardEntry.to_record()` list (canonical sorted-key UTF-8 JSON plus newline).
That hash identifies the content read for this report, not proof of external data
provenance. The block contains all candidate metrics, parameters and context,
including the saved `producer_source`. Candidate tables and details display that
source so same-name evaluations from different origins remain distinguishable.
Legacy origin stays unknown; report generation does not consult the current model
registry to fill it in. Interpret the model's training dataset with the saved
source under the [ML evaluation contract](ml_research.md#recorded-signal-evaluations).
Source recording does not establish a held-out evaluation.
Subsequent re-recording or deletion does not change an existing report. A new
request selects the current stored version and obtains a new report ID when its
evidence differs. The block does not persist prediction or IC time series.

The content digest includes source snapshots, template version, section order and
commentary. `reports/<report_id>/report.html` and `manifest.json` are registered as
`research_report_html` and `research_report_manifest` with IDs
`report:<report_id>:html` and `report:<report_id>:manifest`. A report containing one
successful run is registered to that run; reports with zero or multiple successful
runs have no single `run_id` binding. Comparison report metadata also retains the
parent job identity.
ML report metadata and the response retain `ml_entry_ids`; the manifest embeds
the complete selected ML block. `scope_count` includes both run scopes and ML
candidates, while `available_scopes` remains the mapping of run scopes.
The response gives path, artifact IDs
and HTML SHA-256 without placing an unbounded document into the MCP response.
For HTTP, use `get_artifact_download` with the report artifact ID and SHA-256;
[reading artifact delivery](run_artifacts.md#reading-artifact-delivery) defines the
short-lived link, verification and refresh contract. `get_artifact` supports text
retrieval for verified client copies; local paths require actual file access.

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

## Research interpretation and diagnostic providers

`get_run_diagnostics(run_id, scope)` reads verified saved results without fetching
market data or rerunning a strategy. Omitted scope selects the saved default.
The initial provider adapts six recommendation checks from MaverickMCP commit
`da70d074dcdac5fb21e3b2e6383cfebf391f28b5` (MIT). This is a limited source
adaptation, not an installed Maverick server or its complete analysis engine.
The adapter and distributed license identify the source and changes. Native
TradingDev data remains authoritative; providers receive the original scope and
observations so later diagnostics can use more than these aggregate metrics.

Each response retains provider/version, findings, metric evidence, suggested
checks, evaluated rules, unavailable rules and limitations. The inherited
thresholds are heuristic review prompts, not user risk limits or causal evidence.
Missing metrics are not zero; aggregate folds are unavailable; portfolio-return
rules do not apply to volume mode. Trade counts mean closed trades and are not
normalized by research duration. Kelly/risk-reward checks lack a corresponding
saved metric and are explicitly unavailable. No finding is not a suitability
verdict. The provider does not diagnose market regimes or volatility sensitivity.

The normal MCP delivery supplies `analysis` to `generate_report`: `summary`,
`observations`, `hypotheses_and_limits`, `next_steps`, and `references` containing
exact `run_id`/`scope` pairs. Each text is nonblank and at most 4000 characters;
there are at most 32 distinct references, all belonging to the report. A report
with successful results needs at least one reference when analysis is supplied.
References identify what was interpreted, not proof of the prose's truth or
coverage of all scopes. The client must identify unanalysed scopes and explain
every comparison candidate, including failures. CLI accepts `--analysis-json`.
These references bind backtest scopes only. An ML-only report uses empty references
and names the selected entry IDs in its prose; it retains complete ML identities in
the ML evidence block. The backtest diagnostic provider is not an ML evaluator.

The opening HTML section shows this interpretation, followed by inspectable
provider evidence. Analysis and per-scope diagnostics are saved in the manifest
and content identity. Clients providing no interpretation receive an explicit
not-provided notice; the server never invents LLM prose. Supplementary commentary
remains separate. Specific loss-period claims require original evidence; market
condition claims need defined regimes and aligned data. Hypotheses and proposed
experiments remain distinct from observations and do not approve execution.

## Research notebooks and archives

A notebook is a client-authored research summary, not an audit transcript or
execution authorization. The server does not observe messages outside MCP calls.
It keeps one editable current record; exported snapshots preserve chosen versions.
Notebook summaries do not capture client messages. Recorded conversations use the
separate [conversation recording service](#recorded-conversations).

`save_research_note(draft, note_id?, expected_version?)` creates with both optional
fields absent. An update requires both and replaces the complete draft, with an
atomic version check. Read and reconcile after `research_note_conflict`; never
blindly retry with a guessed version. `get_research_notes()` lists metadata with
`offset`/`limit` (default 20, maximum 100); `note_id` reads a complete bounded record
and does not accept nondefault paging. `export_research_note(note_id,
expected_version)` snapshots that selected version without modifying the note or
executing a strategy. A concurrent later edit does not alter the selected snapshot.

The canonical schema is in `domain/research_notes.py`. Draft example:

```json
{
  "title": "BTC strategy research",
  "current_brief": "Compare saved results; a later 4h experiment is pending.",
  "entries": [
    {
      "title": "Backtest review",
      "summary": "Limited sample; volatility sensitivity remains untested.",
      "artifact_ids": []
    }
  ],
  "next_steps": "Agree on an independent period before executing another run."
}
```

Titles allow 120 characters, summary/brief/next-step text 2000 each, at most
24 entries and 10 references per entry. The complete persisted record is capped
at 16384 UTF-8 bytes, including backend ID/version/timestamps; condense long notes.
References are registered `research_report_html` or `execution_confirmation`
artifact IDs, verified on export. A note may be saved before reports exist, but
unknown, missing, changed or unsupported referenced artifacts prevent export.

`research_notes` in SQLite stores current note ID, version, title, update time and
JSON document. This additive table does not change existing runs. Export storage:

```text
research_archives/<archive_id>/
  index.html
  research.json
  manifest.json
  reports/1.html
  research.zip
```

The ZIP contains all these files except itself; zero reports is valid. Its reading
page escapes client text and uses relative report links. The manifest records
attachment IDs/types/SHA-256 and every content file hash; its SHA-256 is archive_id.
ZIP metadata is deterministic. At most 20 distinct reports and 64 MiB of content
(including notebook/manifest) or ZIP bytes are accepted. Only the explicit allowlist
is copied: no arbitrary paths, symlinks, credentials, raw datasets or whole-workspace
backup. Embedded report contents remain complete, so sharing includes all data
already embedded in those reports, not only initially visible sections.

Publication stages in a temporary directory, serializes identical archive IDs with
a file lock, then registers `research:<archive_id>:zip` as `research_archive_zip`.
An identical retry verifies existing bytes and registration; conflicting files are
never repaired or replaced. Failed publication removes only newly created files.
Lock files remain for concurrency coordination. No saved notes or archives expire
automatically. Downloads verify registered identity and the exact returned bytes;
they remain readable if original reports later disappear.

HTTP exports return a capability URL using the common
[reading artifact delivery contract](run_artifacts.md#reading-artifact-delivery).
The former archive-only `/research-download/` route is replaced by
`/artifact-download/`; deployments must expose that route and obtain fresh links.
Use `get_research_archive(archive_id)` to refresh a saved ZIP without reading its
original note or reports. Discover IDs through
`list_artifacts(artifact_type="research_archive_zip")`. CLI uses
`archive ARCHIVE_ID`. Missing or changed files fail rather than regenerate.
Stdio and CLI return server-local
`path`/`html_path`, with no fabricated remote URL; reading a saved archive returns
null `html_path` if any separately extracted reading file is missing, changed or
symlinked. The verified ZIP remains independently usable. Clients need file access or their
own transfer mechanism. Extract the entire ZIP and open index.html for offline use.

This archive is a reading package, not a restorable research workspace. Preserve
the original database/files for subsequent result queries. Client summary quality,
correct pending/completed distinctions and relevance need semantic review; hashes
and optimistic versions cannot prove that a user said or approved a statement.

## Recorded conversations

`app.conversation_service.ConversationService` records provider-neutral visible
conversation items supplied by client adapters. It is separate from editable
research summaries and does not observe arbitrary MCP clients. Canonical models
`ConversationRecord`, `ConversationTurn` and `ConversationItem`, with the record
limit `MAX_CONVERSATION_BYTES`, live in `domain/conversations.py`. The export limit
`MAX_EXPORT_BYTES` lives in `adapters/storage/conversation_exports.py`.

`create(provider, model)` creates an empty record. The adapter binds its source
thread before `start_turn(record, prompt)` persists the exact submitted input.
`save(record)` checks the selected version atomically and advances it; an adapter
must use the returned version. Source thread/turn identities, original inputs,
completed items, item order and finished turns cannot be rewritten. A reconnect
loads the existing record rather than creating a summary of earlier messages.
The SQLite `conversations` table is additive and does not change existing notes,
execution records or reports.

Items preserve visible text and selected public provider data, including tool
arguments, results, failures and message phases. Private reasoning, hidden
instructions and authentication material are outside the recording contract.
Adapters must persist partial output during a turn and mark missing or unsupported
evidence explicitly. `outcome` describes execution; `capture_complete` describes
record coverage. A failed model turn can have complete evidence, while an apparently
successful answer can have incomplete evidence. Neither field proves semantic
correctness or human authorization. Capacity errors retain the previous saved
version and must stop further recording rather than silently truncate.

`read(conversation_id)` returns the bounded record; `list(offset, limit)` returns
paged metadata. `export(conversation_id, expected_version)` produces an immutable
snapshot under `conversations/<export_id>/` with `index.html`, `conversation.json`
and a file-hash `manifest.json`. Its identity is the manifest SHA-256. HTML contains
selectable, escaped original text in turn/item order, expandable full tool records,
and clear partial/interrupted status. Assistant messages visibly distinguish
process updates from final replies using the recorded provider phase; missing or
unrecognized phases are labelled as unspecified, never inferred from message order.
Consecutive process updates and intervening tool records share one process block;
replies and unspecified assistant messages remain separate, preserving item order.
Message capture status is separate from that phase. It needs no network or scripts; the JSON link
needs its adjacent file. Exports contain no copied reports or full workspace data.
Sharing exposes all recorded messages and tool contents, including any sensitive
information submitted in those messages.

Publication checks existing snapshot bytes and artifact identity, rejects symlinks
and differing files, and removes only newly created files on failure. Repeated
exports reuse verified bytes when the record and rendered output are unchanged;
later turns or renderer changes produce a new snapshot without replacing earlier
exports. The HTML is registered as `conversation_html`, discoverable through the
existing artifact tools. Paths are local files, not public download URLs. No saved
conversation or export expires automatically; retain the workspace to resume.

The `tradingdev-conversation --workspace PATH` CLI provides `list --offset N
--limit N`, `read CONVERSATION_ID` and `export CONVERSATION_ID --expected-version N`.
It returns JSON and exits nonzero on invalid IDs, stale versions or storage errors.
This storage/reading interface does not itself start a model or import old chats.

### Codex CLI recording adapter

`adapters/clients/codex_session.py:record_turn` drives one text turn through the
installed Codex CLI's stdio app-server. `adapters/cli/codex.py` exposes
`tradingdev-codex`. A new managed conversation fixes the model, reasoning effort
and adapter protocol version in its immutable settings. Subsequent commands select
its ID explicitly; they cannot change those settings or import an arbitrary thread.
The adapter requires the initialize, config/read, thread/start, thread/resume,
thread/turns/list and turn/start methods, plus public item lifecycle events. It opts into the
experimental API; unsupported protocol versions fail rather than fall back to
scraping a UI or reading unrelated session files.

The adapter sets a dedicated working directory and per-conversation Codex SQLite
and log directories under `.conversation-clients/<conversation_id>/`. It uses the
existing Codex profile for authentication and native session persistence, without
copying credentials into exports. Native Codex session data may also reside in that
profile and is required for resume; the reading export alone cannot restore a live
model conversation. Codex's own internal state/logs are not the filtered reading
artifact and must not be included when sharing conversation exports.

A lifetime file lock prevents simultaneous recorder commands for the same ID.
Before resuming, the adapter pages source history, verifies turn identities and
compares complete saved items' text, phases, call arguments and results. Adjacent
tool-call groups are matched by item identity because native history can reorder
parallel calls; calls cannot move across visible message boundaries. Known native
text projections of successful tool results are checked against the full saved
wire result, including the retained prefix, suffix and exact omitted character
count. Omitted source content cannot be independently verified from that projection;
the original captured result remains complete and unchanged in the reading export.
Unknown projections or inconsistent retained evidence fail. Changed,
missing or still-running source history fails before a new input is sent. A stale
running local turn is sealed as interrupted with explicit missing-content status;
if its source ID is known and the source is terminal it can be followed by a new
turn, without rewriting or claiming to recover the partial turn. Unknown source
IDs require a new managed conversation.

Input is saved before turn/start; events arriving before its acknowledgement are
bound and persisted immediately, then checked against the acknowledged ID. Text
deltas save partial output; completed items supply authoritative final text. MCP
arguments, results, errors and progress remain in item order. Reasoning items and
private reasoning deltas are discarded. The adapter checks thread/turn identity,
visible item type and configured MCP server; unexpected events stop the capture
with an explicit issue. Retry errors can be recorded without treating them as
missing data. A terminal event alone does not make unfinished items complete.

Only the configured TradingDev MCP server is enabled for this client; other global
MCP servers, shell tools, plugins, web search, hooks, memories and extra agents are
disabled through per-process/thread overrides. Existing user configuration is not
rewritten. This integration does not add a sandbox guarantee to generated strategy
execution. Noninteractive MCP elicitations receive `cancel`, with both request and
client response saved; no user consent is inferred. Unsupported permission requests
receive an error and stop the turn.

The transport bounds individual JSON frames and uses nonblocking writes. Its direct
child remains unreaped until process-group teardown has been checked on success,
exception and timeout. TradingDev's independently managed research workers are not
cancelled merely because the client closes. Capture failures attempt to export the
last durably saved version. An already-persisted running turn is sealed as
interrupted/incomplete when that save succeeds. If a new input could not be saved,
for example because of capacity, it is absent from the export and earlier complete
turns remain labelled complete. Export failure returns the conversation identity
and reason for later inspection. Normal closure never
deletes a research workspace, conversation or export.
