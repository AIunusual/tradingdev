# Execution Plans

Preparation, confirmation and submission contracts. For the executable request
format, see [Execution Manifest](run_artifacts.md#execution-manifest); for
strategy authoring and revision eligibility, see [Strategy Contract](strategy_contract.md).

## Execution Plans and Confirmation Documents

`ExecutionPlan` schema version 3 binds a unique 32-character `plan_id`, the full
single execution or comparison manifest, `original_config_path`, the matching
`PreflightReceipt` or `ComparisonPreflightReceipt`, `ApprovedResearch`,
`ConfirmationDocument`, timezone-aware `created_at`/`expires_at`, and `plan_hash`.
The hash covers canonical JSON of every field except itself. The document and
preflight hashes must identify the same manifest; the document also identifies
the same plan. Plans currently expire one hour after preparation. The content
does not change when approval state changes.

`save_research_evidence(draft)` first stores the client-authored explanation and
returns a compact ID/hash receipt. All four MCP preparation tools require this
`evidence_id`, replacing their former inline `presentation` argument. Preparation
loads the record, binds every effective parameter and captures the current execution
policy; the plan retains its own full copy. Deleting or replacing the original draft
does not change an existing plan. Binding rejection returns
`research_evidence_binding_invalid` with `required_parameter_bindings` and
`required_parameter_paths`; corrections require a newly saved record.

`PreflightService` performs offline preparation in a supervised child with a
60-second default budget covering imports, configuration resolution, execution-mode
and parameter binding checks, generated signal-contract checks, and manifest
serialization. Process cleanup has its own deadline. The request contains only
`kind` and `arguments`. MCP and the worker share research-bound request DTOs;
`arguments.research` preserves explicit choices and always includes its `kind`.
Omitted optional choices are not serialized as `null`. The worker resolves the
research against the selected strategy before capturing the manifest; it cannot
inherit bundled example dates or silently change an optimization's requested
period. MCP preparation no longer accepts the former flat market/date arguments,
`minimum_history_bars` or `sample_bars`. It does not load cached or remote market
data, run a backtest
engine, train a model or serialize performance results.

Generated strategies use the existing fixed 80- and 240-row synthetic fixtures
with the captured base execution parameters. These check the returned DataFrame,
input preservation and the `signal` contract, including values in `{-1, 0, 1}`.
They do not prove the requested trading semantics, absence of look-ahead or
sufficient warm-up for real history. Bundled strategies receive configuration,
execution-mode and parameter checks without calling `fit` or `generate_signals`.
Training-dependent signal execution is deferred until formal training. Optimization
checks candidate bindings without evaluating any candidate's historical objective;
a successful plan does not imply that a rankable objective will be available.

The version 2 receipt records:

- `manifest_hash` and `elapsed_seconds`;
- `checked_paths`: `configuration`, `execution_capabilities`, `parameter_binding`
  and `manifest_serialization`, plus `signal_contract` for generated strategies
  and `candidate_binding` for optimization;
- `synthetic_fixture_rows`: the generated strategy fixture sizes, or an empty
  list for bundled strategies;
- `signal_check`: `synthetic_contract_passed`, `deferred_until_training` or
  `bundled_not_executed`, together with `training_requirement`;
- `parameter_combinations_checked`: the number of parameter combinations whose
  bindings were checked, not the number of backtests or performance trials;
- `not_checked`: `historical_data`, `backtest_engine`, `training` and `performance`.

There are no market windows, fold coverage, trade counts or profitability claims
in this evidence. A fixed comparison instead uses `ComparisonPreflightReceipt`
schema 1, binding the bundle hash and ordered candidate IDs to individual schema 2
receipts. Every comparison candidate has both synthetic fixtures, including bundled
strategies; the standalone bundled preparation behavior above is unchanged.
The confirmation explicitly states that data availability,
actual folds, model training, execution costs and performance remain unverified
until the user approves formal execution. Data or execution failures may therefore
occur after a successful preparation.

Temporary requests, responses and tool caches live in an independent directory,
outside regular run history. The child's `TMPDIR`, `TMP` and `TEMP` also point
inside that directory without changing the parent's environment. Successful,
failed and timed-out workers are stopped and verified before files are deleted.
If process cleanup cannot be verified, the directory is retained and the operation
reports failure; file cleanup failure is also an error. Preparation publishes no
regular job or run. These resource controls are not a generated-Python sandbox.

The plan schema is version 3; the confirmation-document schema remains version 2.
Older confirmation
plans must be prepared and approved again; they are not automatically migrated
or deleted. Existing run results and reports do not need to be recomputed.

Within a research draft, `ConfirmationPresentation` accepts model-authored title, summary and
`parameter_descriptions`. Description keys are JSON pointers relative to captured
constructor arguments; each value contains a label, explanation and optional
unit, never an executable parameter value. Labels must distinguish values and
cover every scalar, null or empty-container leaf, including nested bundled
`config`/`fit_config` and the union of effective optimization or fixed-comparison
candidate leaves. Fixed comparisons display every complete constructor, selection
rationale and candidate check evidence separately; common settings appear once.
Candidate presentation and strategy loading share the same pure parameter merge:
partial objects preserve unnamed captured fields, empty objects preserve an
existing object, and non-object values replace the complete entry. Each search
group displays its merged candidate values; it does not present a partial
override as a complete replacement. Missing or extra descriptions and duplicate
labels are rejected during binding with `research_evidence_binding_invalid`,
`required_parameter_bindings` and `required_parameter_paths`. Save corrected
evidence and prepare again; wording repairs do not require a new strategy revision.
Model units do not rescale captured values.

The backend builds a version 2 typed document with fixed strategy, market, parameters,
execution, validation, optimization, preflight and limitations sections.
Non-applicable sections remain explicit. Known cost/rate semantics have backend
formatters; unknown effective setting fields fail preparation rather than being
silently omitted. Model explanations are labelled separately from backend
values and successful-check evidence. Parameter/settings keys and YAML are not
the user interface. Both text and HTML render the same document; technical
identity and file paths are outside the main explanation, in HTML details and
machine metadata. The HTML is offline, escapes text and disables scripts.

When research provenance is present, market, period, capital, costs and annualization
fields identify their origins. The document separately displays the saved outline,
the observed-UTC-daily-return basis and unavailable sampling reasons. Technical
details include the saved strategy name, policy version, annualization decision,
and any selected revision/config hash. The model's presentation title remains a
separate HTML page title. Research provenance does not prove that the Python
implements the outline. Saved research adds conditions, actions, formulas, variable
units, parameter-value provenance and source claims to the same typed document.
These are client explanations, not a backend proof of equivalence with arbitrary
strategy code. The full document retains the versioned execution policy; its short
chat summary uses the policy's selected plain-language assumptions. Program checks
retain their stated coverage.

`execution_plans/<plan_id>/confirmation.txt` and `confirmation.html` preserve the
reviewable documents. The HTML is registered as `execution_confirmation` under
`<plan_id>:confirmation_html`, without a run ID. Reads verify plan identity and
both saved file digests. File publication and SQLite updates are separate
resources; publication rolls back its new directory on handled failure.

Plan states are `ready`, `awaiting_confirmation`, `submitting`, `submitted`,
`cancelled`, or `failed`. A retained `.approval.lock` and state comparison
serialize transitions; only one submission reservation can be active. The plan
stays `ready` while the user reads it. On approval, the service checks expiry and
current executable-source integrity and the saved execution policy before reserving
and again before submitting. A policy change requires a newly prepared plan;
historical plan lookup keeps the saved wording.
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

Preparation and lookup return `confirmation_summary`, `confirmation_link` and the
complete HTML's `html_path`/`artifact_id`. The service derives the ready-to-copy
Markdown `confirmation_link` from the same actual `html_path`, URL-encoding special
path characters without adding a URI scheme. Clients should copy it verbatim and
verify its destination instead of assembling plan IDs or paths. The raw `html_path`
remains available to machine clients. Full confirmation text remains in the saved
plan; it is not copied into ordinary tool responses. The short summary starts with the model's
strategy explanation, omits the title and saved name, and includes capital/cost
origins plus annualization settings, sampling basis and the inclusive bar-timestamp
boundaries of the research interval. The assistant presents the
short summary and HTML link, ends the turn and waits for the user to review it. Reading the
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
