# Reports

Offline report composition, navigation, export and publication contracts.
Reports read the saved [performance artifacts](performance_artifacts.md);
[Run Artifacts](run_artifacts.md#mcp-lookup) describes source lookup.

## Composable Offline Reports

`get_report_sections` returns built-in section IDs and optional `standard`,
`comparison` and `trades` recipes. `generate_report(run_ids, sections, commentary)`
accepts one to eight distinct runs.
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
For either selector, omitted/null sections select standard;
an explicit ordered list chooses visible sections and their order within the detail
view. When selected, `overview` always belongs to the first navigation level.
With `[]`, no optional sections
are rendered; fixed report/source identity metadata and any supplied commentary
remain visible. Unknown or duplicate sections fail. Up to 20 plain-text title/text notes
(20000 characters total) are escaped and labelled LLM commentary. They cannot
replace computed values or inject HTML. The client selects information and writes
interpretation; the server renders tables, charts and document structure.

Report schema 2 / template 21 uses two navigation levels within one offline HTML
file when at least one detail section (`settings`, `metrics`, `equity`, `ledger`,
or `trades`) is selected. The overview compares every loaded run/scope, including
effective parameters, date coverage, train/test role and principal saved metrics.
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
the runs/scopes and explain the selected detail views. `generate_report` returns
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
The standard recipe is `overview`, `research`, `settings`, `metrics`, `equity`,
`ledger`, `trades`, `limitations`, `provenance`; comparison is `overview`, `research`,
`metrics`, `equity`, `provenance`; trades is `overview`, `research`, `ledger`,
`trades`, `provenance`.
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

Section selection determines the visible presentation without pruning the loaded
or embedded data. Every HTML report embeds the complete loaded payload as JSON,
including all scopes of the requested runs, saved metrics, trades, equity and
execution configuration, even for `sections=[]`.
Omitting a section does not remove its underlying data from the HTML; recipients
of the file can still read that embedded data.

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
Numeric displays default to two decimal places; each overview, ledger and paired-trade
table has one full-precision checkbox shared by its interactive and static views.
The renderer supplies source decimal strings before browser number parsing; percent
text shifts decimal places exactly rather than multiplying a browser float. The switch
retains filters, sorting and column visibility, including lazy loading and printing.
Notional value is visible by default and included in the cash/position preset.
Readers must enable full precision before recalculating from quantities and prices.
Fractional rates and returns display as percentages; identifiers and integer counts
retain exact integer form. Columns containing integers outside the JavaScript safe
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

The content digest includes source snapshots, template version, section order and
commentary. `reports/<report_id>/report.html` and `manifest.json` are registered as
`research_report_html` and `research_report_manifest` with IDs
`report:<report_id>:html` and `report:<report_id>:manifest`. A report containing one
successful run is registered to that run; reports with zero or multiple successful
runs have no single `run_id` binding. Comparison report metadata also retains the
parent job identity.
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
