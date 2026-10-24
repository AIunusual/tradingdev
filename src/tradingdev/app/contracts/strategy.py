"""Public response contracts for strategy discovery and lifecycle operations."""

from typing import Literal

from pydantic import Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.strategies.capabilities import StrategyCapabilities
from tradingdev.domain.strategies.schemas import StrategyStatus


class StrategyContractResponse(ContractModel):
    """Reference source and lifecycle instructions for generated strategies."""

    base_strategy_source: str
    example_strategy_code: str
    example_yaml_config: str
    api_reference: str
    lifecycle: str
    config_schema: dict[str, JsonValue]


class StrategyDiagnosticResponse(ContractModel):
    """One validation finding, including its original machine-readable code."""

    level: Literal["error", "warning"]
    code: str
    phase: str
    message: str
    line: int | None
    fix: str | None


class SignalAnalysisResponse(ContractModel):
    """Known signal statistics; early or external checks may provide a subset."""

    rows: int | None = None
    signal_distribution: dict[str, int] | None = None
    nan_count: int | None = None
    transition_count: int | None = None
    active_signal_ratio: float | None = None
    first_timestamp: str | None = None
    last_timestamp: str | None = None


class StrategyValidationRecord(ContractModel):
    """Persisted evidence of a validation or dry-run attempt."""

    revision_id: str
    checked_at: str
    success: bool
    diagnostics: list[StrategyDiagnosticResponse]
    signal_analysis: SignalAnalysisResponse
    effective_config: dict[str, JsonValue] | None = None


class GeneratedStrategyMetadata(ContractModel):
    """Known persisted metadata fields exposed for generated strategies."""

    strategy_id: str
    revision_id: str
    class_name: str
    artifact_type: Literal["generated_strategy"]
    status: StrategyStatus
    created_at: str
    updated_at: str
    request_summary: str
    source_path: str
    config_path: str
    source_hash: str
    config_hash: str
    validation: StrategyValidationRecord | None
    dry_run: StrategyValidationRecord | None


class StrategyRecentRun(ContractModel):
    """A recent run with extensible JSON metrics."""

    revision_id: str | None
    run_id: str
    job_id: str
    created_at: str
    dataset_id: str | None
    metrics: dict[str, JsonValue]


class BundledStrategySummaryMetadata(ContractModel):
    """Version, source, and configurable parameters of a bundled strategy."""

    revision_id: None
    version: str | None
    source_path: str | None
    parameters: dict[str, JsonValue]


class BundledStrategySummary(ContractModel):
    """One bundled strategy discovery entry."""

    strategy_id: str
    revision_id: None
    class_name: str
    description: str
    kind: Literal["bundled"]
    execution_capabilities: StrategyCapabilities
    status: Literal["promoted"]
    config_path: str
    metadata: BundledStrategySummaryMetadata
    data_requirements: dict[str, JsonValue] | None = Field(
        description="Unvalidated user config, preserved for inspection and repair."
    )
    recent_runs: list[StrategyRecentRun]


class GeneratedStrategySummary(ContractModel):
    """One generated strategy discovery entry."""

    strategy_id: str
    revision_id: str
    class_name: str
    kind: Literal["generated"]
    execution_capabilities: StrategyCapabilities
    status: StrategyStatus
    source_path: str
    config_path: str
    metadata: GeneratedStrategyMetadata
    data_requirements: dict[str, JsonValue] | None = Field(
        description="Unvalidated user config, preserved for inspection and repair."
    )
    recent_runs: list[StrategyRecentRun]


class BundledStrategySourceMetadata(ContractModel):
    """Paths and execution status of a bundled source artifact."""

    revision_id: None
    status: Literal["promoted"]
    source_path: str
    config_path: str


class LegacyStrategySummary(ContractModel):
    """A discoverable flat strategy that must be resaved before any checks or run."""

    strategy_id: str
    revision_id: None
    kind: Literal["legacy"]
    execution_capabilities: StrategyCapabilities
    status: Literal["revision_required"]
    code: Literal["strategy_revision_required"]
    message: str


class LegacyStrategyResponse(LegacyStrategySummary):
    """Read-only recovery material, without inherited validation evidence."""

    success: Literal[True]
    source_code: str
    yaml_config: str


class BundledStrategyResponse(ContractModel):
    """Source and configuration for a bundled strategy."""

    success: Literal[True]
    strategy_id: str
    revision_id: None
    kind: Literal["bundled"]
    execution_capabilities: StrategyCapabilities
    source_code: str
    yaml_config: str
    metadata: BundledStrategySourceMetadata


class GeneratedStrategyResponse(ContractModel):
    """Source, configuration, and lifecycle evidence for a generated strategy."""

    success: Literal[True]
    strategy_id: str
    revision_id: str
    kind: Literal["generated"]
    execution_capabilities: StrategyCapabilities
    source_code: str
    yaml_config: str
    metadata: GeneratedStrategyMetadata


class StrategySaveSuccess(ContractModel):
    """A newly saved draft, ready for validation."""

    success: Literal[True]
    message: str
    error: None
    strategy_id: str
    revision_id: str
    py_path: str
    yaml_path: str
    status: Literal["draft"]


class StrategySaveFailure(ErrorResponse):
    """A rejected save request with no generated artifact paths."""

    code: Literal[
        "invalid_strategy_id",
        "reserved_strategy_id",
        "syntax_error",
        "invalid_yaml",
        "invalid_strategy_config",
    ]
    message: Literal[""]
    strategy_id: str
    revision_id: None
    py_path: Literal[""]
    yaml_path: Literal[""]
    status: Literal["rejected"]


class StrategyStateError(ErrorResponse):
    """An operation rejected before validation because of the current state."""

    code: Literal["invalid_strategy_status"]
    strategy_id: str
    revision_id: str
    status: StrategyStatus


class StrategyCheckResult(ContractModel):
    """Evidence produced by a completed validation or dry-run check."""

    strategy_id: str
    revision_id: str
    diagnostics: list[StrategyDiagnosticResponse]
    signal_analysis: SignalAnalysisResponse
    effective_config: dict[str, JsonValue] | None = None


class StrategyValidationSuccess(StrategyCheckResult):
    """Validation succeeded; the next lifecycle step is a dry run."""

    success: Literal[True]
    status: Literal["validated"]


class StrategyValidationFailure(StrategyCheckResult):
    """Validation found errors; diagnostics describe the required repairs."""

    success: Literal[False]
    status: Literal["draft"]


class StrategyDryRunSuccess(StrategyCheckResult):
    """The validated strategy passed the longer fixture and can execute."""

    success: Literal[True]
    status: Literal["runnable"]


class StrategyDryRunFailure(StrategyCheckResult):
    """The longer fixture found errors without granting execution status."""

    success: Literal[False]
    status: Literal["validated"]


class StrategyPromoteSuccess(ContractModel):
    """A runnable generated strategy was promoted."""

    success: Literal[True]
    strategy_id: str
    revision_id: str
    status: Literal["promoted"]
