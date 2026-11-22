"""Typed read contracts for the ML tool surface.

These contracts describe the *enumerable* ML vocabulary an LLM may choose
from (``get_ml_contract``) and the structured result of judging a prediction
(``evaluate_signal``). Every numeric field rejects non-finite values, so a
stored evaluation can never silently pack a NaN into a JSON boundary.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.ml.leaderboard import ProducerSource
from tradingdev.domain.ml.market_evidence import MarketComparison, RawMarketEvidence
from tradingdev.domain.optimization.selection import SelectionEvidence


class FeatureSetSummary(ContractModel):
    """One named, audited feature set the model may request by name."""

    name: str
    description: str
    n_columns: int
    columns: list[str]
    tags: list[str]


class ModelKindSummary(ContractModel):
    """One audited direction-model implementation and its default parameters."""

    name: str
    description: str
    framework: str
    default_parameters: dict[str, JsonValue]


class ValidationSchemeSummary(ContractModel):
    """A leak-aware train/test split scheme the model may request by name."""

    name: str
    description: str


class RollingSchemeSummary(ContractModel):
    """A rolling retraining geometry a bundled strategy may enable by name.

    Listed whether or not any strategy enables it: the menu describes what can
    be declared, while the strategy's own ``rolling_gen.enabled`` decides
    whether the classic retraining loop is replaced.
    """

    name: str
    description: str
    rtype: str
    """``ROLL_SD`` slides a fixed-length window; ``ROLL_EX`` expands one."""
    default_step: int | None
    """Cadence when the strategy declares none; null follows its own interval."""


class EvaluationSeriesLimits(ContractModel):
    """Read bounds an MCP client can choose between before requesting rows."""

    default_page_size: int
    max_page_size: int
    max_export_rows: int
    max_export_bytes: int
    overflow_read: str
    """Tool call to use when an artifact exceeds ``max_export_rows``."""


class MlContractResponse(ContractModel):
    """The full enumerable ML vocabulary; the menu for LLM-driven selection."""

    success: Literal[True]
    schema_version: Literal[1]
    feature_sets: list[FeatureSetSummary]
    model_kinds: list[ModelKindSummary]
    validation_schemes: list[ValidationSchemeSummary]
    rolling_schemes: list[RollingSchemeSummary]
    processor_types: list[str]
    ic_methods: list[str]
    n_quantiles_range: list[int]
    calendar_modes: list[str]
    missing_policies: list[str]
    leaderboard_metrics: list[str]
    """Metrics a leaderboard may rank on; each declares a direction."""
    method_families: list[str]
    """Kinds of method a candidate can declare, from fitted model to unnamed."""
    evaluation_series_limits: EvaluationSeriesLimits
    """Publishable bounds for discovery, row paging and full export."""
    guidance: dict[str, str]
    """On-demand interpretation and workflow details for the advertised tools."""


class StratumSummary(ContractModel):
    """One prediction quantile and its realised outcome (ascending by signal)."""

    bin: int
    count: int
    mean_prediction: float | None
    mean_forward_return: float | None
    std_forward_return: float | None


class SimulatedReturnsSummary(ContractModel):
    """Causal, non-overlapping holding returns; no costs or mark-to-market curve.

    Means include inactive zero rows. Quantile means describe the full sample.
    """

    long_short_mean: float | None
    long_only_mean: float | None
    long_short_cumulative: float | None
    long_only_cumulative: float | None
    quantile_means: list[float | None]


class RecordedEvaluation(ContractModel):
    """An evaluation stored for later ranking."""

    entry_id: str
    producer_source: ProducerSource
    """Actual signal execution path; unknown for records without this evidence."""
    replaced: bool
    """True when this replaced an earlier record for the same candidate."""
    context: str
    n_candidates: int
    """How many candidates are now recorded; the size of the comparison."""
    signal_fingerprint: str
    """Hash of the evaluated signal; compare it with a re-run to check reproduction."""
    method_family: str
    """Declared method family; producer_source separately records actual origin."""
    method_name: str
    """Readable rule name; empty for a model and for an unnamed signal."""
    training_dataset_id: str
    """Dataset the model was fitted on; empty when unknown or for no model.

    Equality with the evaluation dataset means the score is in-sample when the
    backend produced the signal from that artifact; under a caller-supplied
    signal it says only where the named artifact was fitted. Inequality proves
    nothing: a dataset identity also covers the requested period and the code
    fingerprint, and different versions can still overlap.
    """
    series_artifact_id: str
    """Content-bound pointwise series saved with this recorded evaluation."""


class SignalEvaluationResponse(ContractModel):
    """The full signal-quality result for one prediction series."""

    success: Literal[True]
    producer_source: ProducerSource
    """Signal execution path, distinct from a declared model label or holdout proof."""
    evaluation_id: str
    signal_fingerprint: str
    """Hash of the evaluated signal values in dataset row order.

    Stored separately from identity for models and named methods. Only unnamed
    candidates include it in their entry identity.
    """
    n_observations: int
    horizon: int
    window: int
    n_bins: int
    """Requested quantile count; tied predictions can produce fewer strata."""
    ic_methods: list[str]
    metrics: dict[str, float | int | None]
    strata: list[StratumSummary]
    strata_monotonicity: float | None
    long_short_spread: float | None
    simulated: SimulatedReturnsSummary
    series_artifact_id: str | None = None
    """Saved pointwise series, present only for a recorded evaluation."""
    recorded: RecordedEvaluation | None = None
    """Present only when the evaluation was recorded for the leaderboard."""


class PredictionEvaluationResponse(SignalEvaluationResponse):
    """Complete evaluation of a verified saved model and dataset pair."""

    model_id: str
    dataset_id: str


class RuleEvaluationResponse(SignalEvaluationResponse):
    """Evaluation of a hand-set threshold rule on a saved dataset.

    Carries the dataset and the readable rule name instead of a model ID,
    because no model was involved and the rule is what must be reproducible.
    """

    dataset_id: str
    rule: str
    """Readable rule identity, such as ``rsi_14 < 30``."""


class LeaderboardRow(ContractModel):
    """One placed candidate and the headline numbers behind its rank.

    The row is self-describing: the evaluation settings it was measured under,
    the candidate's own settings and the hash of its signal all travel
    with the metric, so a parameter surface or a reproduction check needs no
    second lookup.
    """

    rank: int
    source: Literal["signal"]
    """Which kind of measurement this row is; a run row carries ``run``.

    Required, not defaulted: a discriminated union reads the tag from the
    payload, so a default here would be accepted by the type and still fail at
    the boundary. Every producer states the source it actually measured.
    """
    entry_id: str
    series_artifact_id: str
    """Current pointwise series for this leaderboard row; empty on old records."""
    producer_source: ProducerSource
    """Saved signal execution path; caller/unknown model IDs are only labels."""
    label: str
    model_id: str
    dataset_id: str
    method_family: str
    """Declared method family: ml_model, rule or other; not proof of origin."""
    method_name: str
    """The rule's readable name; empty for a model and for an unnamed signal."""
    training_dataset_id: str
    """Dataset the model was fitted on; empty when unknown or for no model."""
    horizon: int
    window: int
    n_quantiles: int
    parameters: dict[str, JsonValue]
    """Artifact or declared method settings; empty does not imply defaults."""
    signal_fingerprint: str
    """Hash of the signal this row was measured on; empty on older records."""
    value: float
    tied: int
    """How many entries share this rank."""
    n_observations: int
    summary: dict[str, float | None]
    """Headline metrics; ``None`` marks an unavailable or non-finite score.

    The ranked metric is finite by definition, but the same candidate can have
    failed on another one. Keeping that key with a null says so; dropping it
    would hide the failure behind a shorter list.
    """


class LeaderboardSkipped(ContractModel):
    """A recorded candidate that could not be ranked, and why."""

    source: Literal["signal"]
    """Which kind of measurement this is; a run candidate carries ``run``."""
    entry_id: str
    series_artifact_id: str
    """Current pointwise series for this row; empty when none was recorded."""
    producer_source: ProducerSource
    """Saved signal execution path, retained even when its score is unavailable."""
    label: str
    method_family: str
    """Declared method family, retained even when the score is unavailable."""
    method_name: str
    training_dataset_id: str
    """Dataset the model was fitted on; empty when unknown or for no model."""
    reason: str
    horizon: int
    window: int
    n_quantiles: int
    """The settings this candidate was measured under.

    Reported so a failed run keeps its coordinates on the parameter grid.
    Without them the surviving points look adjacent, and a slope or flatness
    computed over them would be wrong.
    """


class RunConditions(ContractModel):
    """Run-level submitted conditions; scope observations remain in metadata."""

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    fees: float
    slippage: float


class RankedRunRow(ContractModel):
    """A placed saved run scope, labelled with the source it came from."""

    source: Literal["run"]
    rank: int
    tied: int
    value: float
    run_id: str
    scope: str
    strategy_id: str
    revision_id: str
    parameters: dict[str, JsonValue]
    metadata: dict[str, JsonValue]
    conditions: RunConditions
    metric_source: Literal["bundle", "legacy_summary"]
    """Whether the value came from the performance bundle or a legacy summary."""
    selection_evidence: SelectionEvidence | None = None
    """Training-search context for an optimization, with a saved evidence reference."""


class SkippedRunRow(ContractModel):
    """A selected run scope that could not be ranked, labelled with its source."""

    source: Literal["run"]
    candidate_id: str
    run_id: str | None = None
    scope: str | None = None
    reason: str


type RankedRow = Annotated[LeaderboardRow | RankedRunRow, Field(discriminator="source")]
type SkippedRow = Annotated[
    LeaderboardSkipped | SkippedRunRow, Field(discriminator="source")
]


class LeaderboardResponse(ContractModel):
    """One ranking on one metric, drawn from exactly one source.

    ``caveats`` is part of the answer, not decoration: the winner of a
    comparison is the most optimistic number in it, and the caveats say how
    much evidence would be needed to believe it.

    A signal metric and a backtest metric are different measurements that
    share no column, so a response ranks either recorded evaluations or saved
    run scopes. ``source`` names which, and the row models follow it; the
    signal-only fields are ``null`` on the run path — present but not
    applicable — rather than empty, which would read as measured-and-nothing.
    """

    success: Literal[True]
    metric: str
    direction: str
    source: Literal["signal", "run"]
    """Which source this ranking read; the rows carry the same tag.

    Required for the same reason the rows require it: a run ranking that
    defaulted to ``signal`` would mislabel every row it returned.
    """
    n_candidates: int
    """Every selected candidate, including ones that could not be ranked."""
    n_ranked: int
    """Candidates actually placed; the size of the comparison that decided."""
    context: str | None = None
    """Signal only: the shared evaluation context, or ``mixed``."""
    families: dict[str, int] | None = None
    """Signal only: how many candidates each method family contributed.

    A ranking that mixes fitted models with hand-set rules is not one kind of
    evidence, so the mix travels with the order.
    """
    parameter_axes: dict[str, list[int]] | None = None
    """Signal only: every observed value of horizon, window and n_quantiles.

    The grid coordinates behind the ranking: when an axis holds more than one
    value the rows span a parameter surface rather than one measurement.
    """
    market_comparison: MarketComparison | None = None
    """Signal only: evidence across all candidates for a sweep or strict check."""
    ranked: list[RankedRow]
    skipped: list[SkippedRow]
    caveats: list[str]

    @model_validator(mode="after")
    def validate_one_source(self) -> Self:
        """A ranking ranks one source, so a row tagged otherwise is refused.

        The tag on each row is what lets a reader trust the numbers beside it;
        a response whose header disagrees with its rows would make that tag a
        decoration instead of a fact.
        """
        stray = {row.source for row in [*self.ranked, *self.skipped]} - {self.source}
        if stray:
            raise ValueError(
                f"rows from {sorted(stray)} cannot appear in a {self.source} ranking"
            )
        return self


class LeaderboardPageResponse(LeaderboardResponse):
    """A page from both ranked and unavailable candidates; counts are totals."""

    n_skipped: int
    offset: int
    limit: int
    next_offset: int | None
    """Where the next page starts, or null when this page ends the ranking."""


class CalendarCoverageEntry(ContractModel):
    """How much of the requested calendar one symbol could supply.

    Measured before unusable dates were dropped, so it explains the drop
    rather than reporting full coverage after the fact.
    """

    symbol: str
    n_present: int
    n_calendar_dates: int
    n_missing: int
    coverage: float
    missing: list[str]
    """ISO dates without a bar; capped, so ``n_missing`` is the true count."""


class CalendarSummary(ContractModel):
    """The shared calendar a panel was built on, and what it cost.

    ``n_dates`` counts the calendar the panel actually uses; a coverage
    entry's ``n_calendar_dates`` counts the calendar that was *requested*
    before unusable dates were removed. The two differ by
    ``n_dropped_dates``.
    """

    mode: str
    missing_policy: str
    max_staleness: int | None
    n_dates: int
    start_date: str
    end_date: str
    n_dropped_dates: int
    dropped_dates: list[str]
    """Dates removed because some symbol could not supply them."""
    coverage: list[CalendarCoverageEntry]
    forward_filled: dict[str, int]
    """Per symbol, how many rows were carried from an earlier bar."""


class PanelSummary(ContractModel):
    """A built multi-symbol panel dataset and the calendar behind it."""

    dataset_id: str
    source: str
    market_type: str | None
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    feature_set: str
    horizon: int
    rows: int
    n_symbols: int
    n_dates: int
    dropped_rows: int
    """Per-symbol warm-up and label-tail rows dropped before alignment."""
    n_features: int
    feature_columns: list[str]
    label_column: str
    calendar: CalendarSummary
    content_hash: str
    raw_market: RawMarketEvidence
    path: str
    features_from_cache: bool
    """True when every symbol's features came from a verified cache entry."""


class BuildPanelResponse(ContractModel):
    """A multi-symbol panel stored under a deterministic id."""

    success: Literal[True]
    panel: PanelSummary


class FeatureImportanceEntry(ContractModel):
    """One feature's native model importance; scales differ across frameworks."""

    feature: str
    importance: float


class DatasetSummary(ContractModel):
    """A built, labelled feature dataset and how it was produced."""

    dataset_id: str
    source: str
    market_type: str | None
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    feature_set: str
    horizon: int
    rows: int
    dropped_rows: int
    n_features: int
    feature_columns: list[str]
    label_column: str
    content_hash: str
    raw_market: RawMarketEvidence
    path: str
    features_from_cache: bool
    """True when the feature matrix was reused instead of recomputed.

    Reuse is only possible when the bars, the feature set and the feature code
    all match a stored entry, so a hit never changes the dataset's content -
    it only saves the work.
    """


class BuildDatasetResponse(ContractModel):
    """A dataset stored under a deterministic id, ready for train_model."""

    success: Literal[True]
    dataset: DatasetSummary


class ValidationFold(ContractModel):
    """One purged fold: what it trained on, what it was judged on, what it scored.

    Counts are dataset rows; all symbols on one date are assigned together.
    Panel scores average the per-symbol scores, avoiding rolling across assets.
    """

    fold: int
    train_bars: int
    test_bars: int
    purged_bars: int
    embargoed_bars: int
    accuracy: float | None
    """Share of test bars whose direction was called correctly."""
    ic_mean: float | None
    rank_ic_mean: float | None
    hit_rate: float | None


class ValidationAggregate(ContractModel):
    """Fold scores averaged, so one lucky fold cannot stand in for the rest."""

    accuracy: float | None
    ic_mean: float | None
    rank_ic_mean: float | None
    hit_rate: float | None


class ValidationSummary(ContractModel):
    """Held-out fold scores of the training procedure.

    K-fold uses both temporal sides and is retrospective. Walk-forward trains
    only on earlier dates. Neither describes predictions of the saved artifact
    on its training rows: that artifact is fit on all rows after validation.
    """

    scheme: str
    label_horizon: int
    """Nominal purge width; actual label intervals can require more removal."""
    embargo: int
    n_folds: int
    purged_bars: int
    embargoed_bars: int
    leakage_audit_clean: bool
    """False would mean a split whose training labels reach into its own test
    window - which must not happen, and is reported rather than assumed."""
    artifact_trained_on: str
    folds: list[ValidationFold]
    aggregate: ValidationAggregate
    fold_predictions_available: bool | None = None
    """Whether this record carried an actual fold prediction sequence.

    ``null`` means a historical summary predates series publication, not an
    empty set of predictions.
    """
    fold_series_artifact_id: str | None = None
    """Content-bound series of the predictions made inside those folds."""


class TrainedModelSummary(ContractModel):
    """A trained model stored through the artifact registry."""

    model_id: str
    model_name: str
    framework: str
    kind: str
    dataset_id: str
    n_train_samples: int
    n_features: int
    parameters: dict[str, JsonValue]
    feature_importance: list[FeatureImportanceEntry]
    in_sample_accuracy: float | None
    """Fit on the training rows themselves; not an out-of-sample estimate."""
    validation: ValidationSummary | None = None
    """Purged out-of-sample estimate, present only when one was asked for."""
    path: str


class TrainModelResponse(ContractModel):
    """A trained, persisted model and its fingerprint identity."""

    success: Literal[True]
    model: TrainedModelSummary


class PredictionSummary(ContractModel):
    """A bounded preview; n_rows describes the complete saved dataset."""

    model_id: str
    dataset_id: str
    model_name: str
    framework: str
    n_rows: int
    horizon: int
    offset: int
    limit: int
    next_offset: int | None
    signal: list[float]
    forward_returns: list[float]
    """Preview returns aligned with signal; evaluate_signal uses all saved rows."""
    signal_fingerprint: str
    """Hash of the complete signal, not only the returned page."""


class PredictionResponse(ContractModel):
    """The signal produced by a stored model."""

    success: Literal[True]
    prediction: PredictionSummary


class EvaluationSeriesSummary(ContractModel):
    """One discoverable pointwise artifact, without loading its rows."""

    artifact_id: str
    kind: Literal["signal_evaluation", "fold_validation"]
    evaluation_id: str
    dataset_id: str
    entry_id: str
    row_count: int
    training_status: str
    holdout_relationship: str
    created_at: str


class EvaluationSeriesPageResponse(ContractModel):
    """A bounded discovery page of saved evaluation series."""

    success: Literal[True]
    items: list[EvaluationSeriesSummary]
    total: int
    offset: int
    next_offset: int | None


class EvaluationSeriesSignalRow(ContractModel):
    """One saved signal row; an absent IC is null with its reason beside it."""

    source_row: int
    timestamp: str
    signal: float | None
    realized_return: float | None
    pearson_ic: float | None
    pearson_availability_reason: str = ""
    spearman_ic: float | None
    spearman_availability_reason: str = ""
    availability_reason: str


class EvaluationSeriesFoldRow(ContractModel):
    """One fold prediction and the split that actually produced it."""

    fold: int
    source_row: int
    symbol: str
    """Dataset symbol; empty when the saved dataset has no symbol column."""
    timestamp: str
    prediction: float | None
    target: float | None
    forward_return: float | None
    train_start_row: int
    train_end_row: int
    eval_start_row: int
    eval_end_row: int
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str


class EvaluationSeriesExportResponse(ContractModel):
    """A complete, content-bound series; reading it never loads a model."""

    success: Literal[True]
    manifest: dict[str, JsonValue]
    rows: list[EvaluationSeriesSignalRow | EvaluationSeriesFoldRow]


class EvaluationSeriesReadResponse(ContractModel):
    """One bounded read surface: artifact discovery or one artifact's rows."""

    success: Literal[True]
    mode: Literal["discovery", "rows"]
    total: int
    offset: int
    next_offset: int | None
    items: list[EvaluationSeriesSummary] = Field(default_factory=list)
    artifact_id: str | None = None
    max_page_size: int | None = None
    max_export_rows: int | None = None
    max_export_bytes: int | None = None
    manifest: dict[str, JsonValue] | None = None
    rows: list[EvaluationSeriesSignalRow | EvaluationSeriesFoldRow] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def one_mode_only(self) -> Self:
        """A page cannot mix artifact summaries with another artifact's rows."""
        if self.mode == "discovery":
            if self.artifact_id or self.manifest is not None or self.rows:
                raise ValueError("a discovery page cannot carry row data")
        elif not self.artifact_id or self.manifest is None or self.items:
            raise ValueError("a row page needs its artifact and manifest")
        return self


class MlContractError(ErrorResponse):
    """Invalid ML request (e.g. mismatched or too-short signal arrays)."""
