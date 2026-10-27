"""Typed read contracts for runs, artifacts, and feature requests."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.performance.artifacts import MetricDefinitionSnapshot


class MetricDefinitionRecord(MetricDefinitionSnapshot):
    """JSON boundary for snapshots, accepting JSON arrays for the tuple of modes."""

    model_config = ConfigDict(strict=False)


class DisplayedFoldStats(ContractModel):
    """Rounded fold statistics with an unscaled count of available observations."""

    mean: str | None
    std: str | None
    min: str | None
    max: str | None
    valid_count: int = Field(ge=0, strict=True)


type MetricDisplayValues = Annotated[
    dict[str, str | DisplayedFoldStats | None],
    Field(
        description="Ready-to-present rounded values; percent signs are included. "
        "Null is unavailable; empty without saved units. Raw metrics are unchanged."
    ),
]


class MetricDiscovery(ContractModel):
    """Discover stored detail without making summary selection a data filter."""

    details_available: bool
    provenance: Literal["performance_artifact", "legacy_metrics", "invalid_artifact"]
    available_metric_ids: list[str]
    available_scopes: list[str]
    default_scope: str | None
    selected_train_scope: str | None = None
    detail_error: ErrorResponse | None = None


class RunSummary(ContractModel):
    """Stable discovery identity; get_run retrieves metrics and scope discovery."""

    run_id: str
    job_id: str
    strategy_id: str
    revision_id: str | None
    dataset_id: str | None
    created_at: str


class RunRecord(MetricDiscovery):
    """Run identity and a compact view of the persisted default metric scope."""

    run_id: str
    job_id: str
    strategy_id: str
    revision_id: str | None = None
    manifest_hash: str | None = None
    config_hash: str | None
    source_hash: str | None
    random_seed: int | None
    dataset_id: str | None
    metrics: dict[str, JsonValue]
    display_metrics: MetricDisplayValues
    artifact_dir: str
    created_at: str


class RunResponse(ContractModel):
    """Successful lookup of a completed run."""

    success: Literal[True]
    run: RunRecord


class MetricCatalogResponse(ContractModel):
    """Current supported metric definitions; runs retain their own snapshots."""

    success: Literal[True]
    schema_version: Literal[1]
    definitions: list[MetricDefinitionRecord]


class MetricQueryError(ErrorResponse):
    """Invalid selection with discovery data to help the caller correct it."""

    run_id: str | None = None
    available_metric_ids: list[str] = Field(default_factory=list)
    available_scopes: list[str] = Field(default_factory=list)
    default_scope: str | None = None


class RunMetricsResponse(MetricDiscovery):
    """One recorded scope and its original definitions and calculation settings."""

    success: Literal[True]
    run_id: str
    manifest_hash: str | None
    scope: str
    kind: Literal["backtest", "fold_summary"]
    mode: Literal["signal", "volume"]
    split: Literal["full", "train", "test"] | None
    fold_index: int | None
    trial_index: int | None
    parameters: dict[str, JsonValue]
    metrics: dict[str, JsonValue]
    display_metrics: MetricDisplayValues
    metadata: dict[str, JsonValue]
    definitions: dict[str, MetricDefinitionRecord]


class MetricCompatibility(ContractModel):
    """Whether metric values share an interpretation, with explicit limitations."""

    comparable: bool
    reasons: list[str]


class RunComparison(MetricDiscovery):
    """Selected recorded metrics, with their context and provenance."""

    run_id: str
    strategy_id: str
    scope: str | None
    kind: Literal["backtest", "fold_summary"] | None
    mode: Literal["signal", "volume"] | None
    split: Literal["full", "train", "test"] | None
    metrics: dict[str, JsonValue]
    display_metrics: MetricDisplayValues
    metadata: dict[str, JsonValue]
    definitions: dict[str, MetricDefinitionRecord]


class CompareRunsResponse(ContractModel):
    """Values are not ranked; compatibility must be checked before comparison."""

    success: Literal[True]
    runs: list[RunComparison]
    comparable: bool
    metric_compatibility: dict[str, MetricCompatibility]
    context_differences: dict[str, dict[str, JsonValue]]


class ArtifactIdentity(ContractModel):
    """Persisted artifact identity and file location."""

    artifact_id: str
    run_id: str | None
    artifact_type: str
    path: str
    sha256: str | None
    created_at: str


class ArtifactRecord(ArtifactIdentity):
    """Artifact identity with metadata supplied by its producer."""

    metadata: dict[str, JsonValue]


class ArtifactContentPage(ContractModel):
    """A bounded code-point range tied to the observed raw file's SHA-256."""

    offset: int
    limit: int
    total_characters: int
    next_offset: int | None
    content_sha256: str


class ArtifactResponse(ContractModel):
    """Successful artifact lookup with an optional UTF-8 content page."""

    success: Literal[True]
    artifact: ArtifactRecord
    content: str | None = None
    content_page: ArtifactContentPage | None = None


class FeatureRequestMetadata(ContractModel):
    """The feature request recorded in an artifact's metadata."""

    request_id: str
    title: str
    description: str
    source_tool: str
    created_at: str
    status: Literal["open"]


class FeatureRequestArtifact(ArtifactIdentity):
    """An artifact with the fixed feature-request metadata contract."""

    artifact_type: Literal["feature_request"]
    metadata: FeatureRequestMetadata


class RecordedFeatureRequest(ContractModel):
    """Location of the newly recorded feature request."""

    success: Literal[True]
    request_id: str
    path: str


class RecordFeatureRequestResponse(ContractModel):
    """Successful creation of a local feature request artifact."""

    success: Literal[True]
    message: str
    feature_request: RecordedFeatureRequest
