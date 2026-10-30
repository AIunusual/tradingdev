"""Typed read contracts for the ML tool surface.

These contracts describe the *enumerable* ML vocabulary an LLM may choose
from (``get_ml_contract``) and the structured result of judging a prediction
(``evaluate_signal``). Every numeric field rejects non-finite values, so a
stored evaluation can never silently pack a NaN into a JSON boundary.
"""

from typing import Literal

from pydantic import JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse


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
    replaced: bool
    """True when this replaced an earlier record for the same candidate."""
    context: str
    n_candidates: int
    """How many candidates are now recorded; the size of the comparison."""
    signal_fingerprint: str
    """Hash of the evaluated signal; compare it with a re-run to check reproduction."""


class SignalEvaluationResponse(ContractModel):
    """The full signal-quality result for one prediction series."""

    success: Literal[True]
    evaluation_id: str
    signal_fingerprint: str
    """Hash of the evaluated signal values in dataset row order.

    Stored separately from entry identity for modelled candidates. Ad-hoc
    candidates without a model ID also include it in their entry identity.
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
    recorded: RecordedEvaluation | None = None
    """Present only when the evaluation was recorded for the leaderboard."""


class PredictionEvaluationResponse(SignalEvaluationResponse):
    """Complete evaluation of a verified saved model and dataset pair."""

    model_id: str
    dataset_id: str


class LeaderboardRow(ContractModel):
    """One placed candidate and the headline numbers behind its rank.

    The row is self-describing: the evaluation settings it was measured under,
    the candidate's own hyperparameters and the hash of its signal all travel
    with the metric, so a parameter surface or a reproduction check needs no
    second lookup.
    """

    rank: int
    entry_id: str
    label: str
    model_id: str
    dataset_id: str
    horizon: int
    window: int
    n_quantiles: int
    parameters: dict[str, JsonValue]
    """Fitted hyperparameters when available; empty does not imply defaults."""
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

    entry_id: str
    label: str
    reason: str
    horizon: int
    window: int
    n_quantiles: int
    """The settings this candidate was measured under.

    Reported so a failed run keeps its coordinates on the parameter grid.
    Without them the surviving points look adjacent, and a slope or flatness
    computed over them would be wrong.
    """


class LeaderboardResponse(ContractModel):
    """Recorded evaluations ranked on one metric.

    ``caveats`` is part of the answer, not decoration: the winner of a
    comparison is the most optimistic number in it, and the caveats say how
    much evidence would be needed to believe it.
    """

    success: Literal[True]
    metric: str
    direction: str
    n_candidates: int
    """Every stored candidate, including ones that could not be ranked."""
    n_ranked: int
    """Candidates actually placed; the size of the comparison that decided."""
    context: str
    parameter_axes: dict[str, list[int]]
    """Every observed value of horizon, window and n_quantiles.

    The grid coordinates behind the ranking: when an axis holds more than one
    value the rows span a parameter surface rather than one measurement.
    """
    ranked: list[LeaderboardRow]
    skipped: list[LeaderboardSkipped]
    caveats: list[str]


class LeaderboardPageResponse(LeaderboardResponse):
    """A page from both ranked and unavailable candidates; counts are totals."""

    n_skipped: int
    offset: int
    limit: int
    next_offset: int | None


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


class MlContractError(ErrorResponse):
    """Invalid ML request (e.g. mismatched or too-short signal arrays)."""
