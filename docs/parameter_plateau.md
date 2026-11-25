# Saved parameter plateau analysis

This contract serves application and MCP implementers reading saved parameter
collections. The numeric algorithm remains owned by
[`domain.statistics.plateau`](../src/tradingdev/domain/statistics/plateau.py):
`trials_from_metrics`, `plateau_summary`, `plateau_scores` and `plateau_grid`.
Analysis does not search, train, backtest, change ranking membership or rewrite
source metrics.

## Source selection and completeness

[`PlateauService.analyze`](../src/tradingdev/app/plateau_service.py) selects
exactly one optimization `run_id` or an explicit set of signal `entry_ids`.
Metrics use `run.<id>` or `signal.<id>` matching that source. Run definitions
come from the saved performance bundle; signal definitions come from the catalog
at analysis creation, because saved signal entries do not retain historic
metric definitions. The derived artifact freezes that chosen definition.

[`plateau_sources`](../src/tradingdev/app/plateau_sources.py) owns source adapters:

- Optimization loads one integrity-checked `TradeHistoryService.load_run`
  snapshot and verifies its manifest, complete trial grid, scope conditions and
  selected training scope and observations using `verify_research_scopes`. Only
  `trial/<index>/train` contributes cells. The selected test and fold summaries
  never become grid points. Axes must exactly cover the declared saved search;
  missing or additional levels are rejected. A partial, unpublished or historical
  search without verifiable grid evidence is unavailable; completed trials are
  not reconstructed from a leaderboard, and worker failure persistence is unchanged.
  Coordinates use the manifest's effective constructor overlay, including captured
  defaults and recursively merged objects, rather than raw trial overrides.
- Signal reads one complete atomic leaderboard generation, then selects all
  specified entries, including rows with undefined or unrecorded metrics. Dataset
  bytes and manifest are verified against each entry's saved context. Selection is
  explicit and independent of rank pages. It describes that selected collection,
  not the full historical search: an absent cell means no record in this selection,
  never proof that no experiment was attempted elsewhere.
- Non-axis data, method declarations, training dataset and evaluation settings
  must agree. A numeric model setting or rule threshold can be an axis. Different
  dataset hashes across horizons alone establish neither comparability nor
  incompatibility: both raw-market inputs and ordered evaluation-row evidence must
  be verified. Missing evidence or a mismatch rejects this analysis, without
  changing the ordinary ranking inclusion policy. Within one horizon the saved
  dataset content must match. Recorded method names and producer identities remain
  declarations and evidence of the saved path, not proof of signal correctness.
  Backend threshold-rule names are checked against the saved numeric threshold
  using the producer's `ThresholdRule.name` format; column and comparison side
  stay fixed while the threshold can vary. Arbitrary caller method names are not
  rewritten or parsed as backend rules.

An executed evaluation with no usable target metric keeps an `undefined` cell and
its reason. A metric never recorded for that evaluation retains the distinct
`metric_not_recorded` reason. A failed or timed-out optimization that did not
publish its complete scopes is unavailable as a collection; it is never relabeled
as a completed undefined evaluation. No missing value becomes zero.

## Axes and calculation

Declare explicit finite numeric levels for each axis. Parameter axes use JSON
pointers into saved parameter objects, such as `/threshold` or
`/model/n_estimators`; signal evaluation settings use `horizon`, `window` and
`n_quantiles`. Numeric axes follow the domain plateau contract. Categorical values,
booleans, non-finite values and duplicate levels are rejected; do not invent a
numeric distance for categories. Every varying parameter must be declared, while
non-axis parameters remain fixed. Duplicate coordinates are rejected with source
identities for explicit selection; there is no hidden averaging or winner picking.

Capacity definitions are `MAX_CANDIDATES` in `plateau_sources`,
`MAX_GRID_CELLS` and `MAX_DIMENSIONS` in `plateau_service`. Exceeding them refuses
the collection rather than analyzing a truncated subset. `radius` is a positive
Chebyshev distance in sorted grid indices; `min_neighborhood` optionally limits
which available cells can become plateau-best. There is no default trade-count or
observation-count filter, and the neighborhood threshold is not a statistical
reliability claim.

The axes retain undefined intermediate levels and whole undefined layers. Available
neighbors exclude the point itself; scores use only available neighbors, with the
actual count exposed. Nonuniform grid spacing changes the meaning of one step.
Results warn when dimension makes the neighborhood less local. Tied metrics use
saved trial-index order for optimization and ascending entry ID for signals;
plateau-score ties retain the domain relative tolerance.

`value` and the best candidates' values retain the original metric and its unit.
`score`, `plateau_score`, best-score, neighborhood means and score-gap are on the
higher-is-better algorithm scale: minimizing negates the input. The response
keeps those names distinct. Plateau-best's own metric is distinct from its
neighbors' average score. Undefined statistics are null. Empty and all-undefined
selections still return their declared grid, counts and null best candidates.

A stable plateau does not establish out-of-sample validity, profitability or
removal of data-mining bias. Optimization train cells are in-sample. Scores depend
on radius, grid density and available neighbors, and cannot be compared as a
universal reliability measure across collections.

## Fixed artifacts and continuation

Every successful analysis publishes one derived JSON artifact via
[`PlateauStore`](../src/tradingdev/adapters/storage/plateau.py), under
`analyses/plateau/<analysis_id>.json`. The identity is the exact file SHA-256.
The artifact holds the complete grid, summary, frozen definition, settings,
limitations and selected source evidence. Source identity binds the verified run
manifest/performance and observation digest, or all selected signal records and
verified dataset manifests. Observations themselves remain in original run files.

`PlateauService.read(analysis_id, offset, limit)` verifies registration and exact
file content before returning a bounded page. Each page carries the same source
hash, settings, summary and artifact reference. Candidate updates, new candidates
or server restart do not change existing pages. Changes to source data require a
new analysis. The artifact can be read independently; the original sources and
older reports are preserved. Hashes detect inconsistency while the expected digest
is trusted, not an attacker controlling both files and database.

Publication uses a same-directory staging file and non-replacing link. Identical
retries verify existing bytes; changed artifacts fail without replacement. Staging
files are removed after successful, failed and timed-out writes. Artifact file
publication and SQLite registration are separate operations: registration failure
may leave a valid unregistered file, and an identical retry can register it. Reads
refuse absent or inconsistent registration, missing files and changed bytes.

Failure responses give an explanation and a next step. Resolve selection or saved
evidence first; failure never authorizes a new experiment automatically.
