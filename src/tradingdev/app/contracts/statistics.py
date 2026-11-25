"""Typed read contracts for single-run statistical analysis."""

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse


class StatisticsUnavailable(ContractModel):
    """One explicit absence: never a NaN, infinity, or invented zero."""

    reason: str
    next_step: str
    source: str
    detail: str | None = None


class StatisticsSettings(ContractModel):
    """The frozen settings actually used for the analysis."""

    capital_basis: Literal["account_equity", "cumulative_pnl"]
    init_cash: float | None
    mode: Literal["signal", "volume"]
    frequency: str
    periods_per_year: float | None
    risk_free_rate: float | None
    period_pnl_ddof: int
    return_std_ddof: int | None
    missing_periods: Literal["not_filled"]
    calendar_timezone: Literal["UTC"]
    gap_policy: Literal["exact_declared_bar_cadence"]
    terminal_close_policy: Literal["saved_as_executed"]
    settings_complete: bool


class StatisticsSource(ContractModel):
    """Identity of the one validated snapshot behind every returned value."""

    run_id: str
    scope: str
    kind: Literal["backtest", "fold_summary"]
    split: Literal["full", "train", "test"] | None
    mode: Literal["signal", "volume"]
    manifest_hash: str | None
    dataset_id: str | None
    source_hash: str
    selected_train_scope: str | None
    definition_ids: list[str]


class StatisticsOutcome[T](ContractModel):
    """A result or one reason it cannot be produced."""

    status: Literal["available", "unavailable"]
    result: T | None = None
    unavailable: StatisticsUnavailable | None = None


class TradeSummaryRecord(ContractModel):
    """Closed and marked-open trade groups stay separate."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    net_pnl: float
    win_rate: float | None
    profit_factor: float | None
    unavailable: dict[str, str]


class PerformanceSummaryRecord(ContractModel):
    """Independent descriptive summary on the run's original capital basis."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start: str
    end: str
    n_bars: int
    n_days: int
    n_months: int
    init_cash: float | None
    frequency: str
    periods_per_year: float | None
    risk_free_rate: float
    total_pnl: float
    observed_equity_change: float
    total_return: float | None
    max_drawdown: float | None
    max_drawdown_amount: float
    daily_pnl_mean: float | None
    daily_pnl_std: float | None
    monthly_pnl_mean: float | None
    monthly_pnl_std: float | None
    sharpe_ratio: float | None
    return_information_ratio: float | None
    closed_trades: TradeSummaryRecord
    marked_trades: TradeSummaryRecord
    unavailable: dict[str, str]


class BreakevenSummaryRecord(ContractModel):
    """Break-even payoff and turnover from saved trade records only."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float | None
    avg_loss: float | None
    payoff_ratio: float | None
    breakeven_win_rate: float | None
    win_rate: float | None
    win_rate_edge: float | None
    years: float
    traded_notional: float
    annual_traded_notional: float
    annual_turnover: float | None
    total_cost: float
    annual_cost: float
    cost_per_turnover: float | None
    annual_cost_drag: float | None


class InferenceConfig(ContractModel):
    """Fixed inference contract reported with every available or absent result."""

    algorithm_version: str
    seed: int
    reps: int
    block_size: int | Literal["auto"] | None
    effective_observations: int
    frequency: Literal["daily"]
    periods_per_year: float | None
    risk_free_rate: float | None
    source_hash: str
    scope: str
    source: str


class SharpeIntervalRecord(ContractModel):
    """Annualised Sharpe interval and the resampling it used."""

    sharpe: float
    lower: float
    upper: float
    confidence: float
    block_size: int
    reps: int
    valid_reps: int


class PSRRecord(ContractModel):
    """IID probabilistic Sharpe against a zero per-period hurdle."""

    psr: float
    benchmark_sr: float


class DSRRecord(ContractModel):
    """Selection-corrected PSR over a verified training search population."""

    dsr: float
    psr_vs_zero: float
    hurdle_annualised: float
    n_trials: int
    trial_sharpe_std_annualised: float | None
    trial_scope_count: int
    search_population: Literal["selected_training_scope"]


class InferenceOutcome[T](ContractModel):
    """One inference with its own method, fixed settings and absent reason."""

    status: Literal["available", "unavailable"]
    method: Literal[
        "stationary_bootstrap_percentile", "normal_iid_psr", "deflated_sharpe"
    ]
    result: T | None = None
    unavailable: StatisticsUnavailable | None = None
    config: InferenceConfig


class InferenceSection(ContractModel):
    """Each inference is independently available or unavailable."""

    interval: InferenceOutcome[dict[str, JsonValue]]
    psr: InferenceOutcome[dict[str, JsonValue]]
    dsr: InferenceOutcome[dict[str, JsonValue]]
    limitations: list[str]


class StatisticsArtifact(ContractModel):
    """Derived immutable artifact identity, separate from the source bundle."""

    artifact_id: str
    artifact_type: Literal["statistics_analysis"]
    sha256: str
    path: str


class StatisticsAnalysisResponse(ContractModel):
    """Bounded analysis of one saved run/scope; no raw observations are returned."""

    success: Literal[True]
    schema_version: Literal[1]
    algorithm_version: str
    source: StatisticsSource
    settings: StatisticsSettings
    # Detailed result records remain declared below for validators and docs;
    # the tool schema keeps one bounded JSON object per outcome so the MCP
    # discovery surface does not duplicate every field definition.
    summary: StatisticsOutcome[dict[str, JsonValue]]
    breakeven: StatisticsOutcome[dict[str, JsonValue]]
    inference: InferenceSection
    limitations: list[str]
    artifact: StatisticsArtifact | None = None


class StatisticsQueryError(ErrorResponse):
    """Expected failure with enough identity to choose another saved scope."""

    run_id: str | None = None
    scope: str | None = None
    available_scopes: list[str] = Field(default_factory=list)
    unavailable: StatisticsUnavailable | None = None
