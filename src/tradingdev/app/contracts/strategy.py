"""Public response contracts for strategy discovery and lifecycle operations."""

from typing import Literal, Self

from pydantic import JsonValue, model_validator

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.strategies.capabilities import StrategyCapabilities
from tradingdev.domain.strategies.schemas import StrategyStatus


class StrategyContractSection(ContractModel):
    """Discoverable purpose of one complete contract section."""

    id: Literal["core", "example", "configuration"]
    description: str


class StrategyCoreContract(ContractModel):
    """Required authoring and lifecycle rules, without optional examples."""

    base_strategy_source: str
    requirements: list[str]
    lifecycle: list[str]


class StrategyExampleContract(ContractModel):
    """A complete Python/YAML example to adapt to a user's strategy."""

    example_strategy_code: str
    example_yaml_config: str


class StrategyConfigurationContract(ContractModel):
    """Executable configuration schema and interpretation of its financial fields."""

    config_schema: dict[str, JsonValue]
    notes: list[str]


class StrategyContractResponse(ContractModel):
    """A selected section, with references to the remaining contract material."""

    section: Literal["core", "example", "configuration"]
    available_sections: list[StrategyContractSection]
    content: (
        StrategyCoreContract | StrategyExampleContract | StrategyConfigurationContract
    )

    @model_validator(mode="after")
    def validate_section_content(self) -> Self:
        expected = {
            "core": StrategyCoreContract,
            "example": StrategyExampleContract,
            "configuration": StrategyConfigurationContract,
        }
        if not isinstance(self.content, expected[self.section]):
            raise ValueError("Contract content does not match its section")
        return self


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


class BundledStrategySummary(ContractModel):
    """One bundled strategy discovery entry."""

    strategy_id: str
    revision_id: None
    class_name: str
    description: str
    kind: Literal["bundled"]
    execution_capabilities: StrategyCapabilities
    status: Literal["promoted"]


class GeneratedStrategySummary(ContractModel):
    """One generated strategy discovery entry."""

    strategy_id: str
    revision_id: str
    class_name: str
    kind: Literal["generated"]
    execution_capabilities: StrategyCapabilities
    status: StrategyStatus


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


type StrategyKind = Literal["bundled", "generated", "legacy"]
type StrategySummary = (
    BundledStrategySummary | GeneratedStrategySummary | LegacyStrategySummary
)


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
