"""Bounded parameter plateau requests and frozen analysis responses."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse, Page
from tradingdev.app.contracts.research import MetricDefinitionRecord

MAX_CANDIDATES = 1024
MAX_GRID_CELLS = 4096
MAX_DIMENSIONS = 6

type AxisLevels = Annotated[
    list[Annotated[float, Field(strict=True, allow_inf_nan=False)]],
    Field(min_length=1, max_length=MAX_GRID_CELLS),
]
type PlateauAxes = Annotated[
    dict[Annotated[str, Field(min_length=1, max_length=256)], AxisLevels],
    Field(min_length=1, max_length=MAX_DIMENSIONS),
]
type PlateauEntries = Annotated[
    list[Annotated[str, Field(min_length=1, max_length=256)]],
    Field(max_length=MAX_CANDIDATES),
]


class PlateauError(ErrorResponse):
    """Source, selection, calculation or publication could not be completed."""

    next_step: str


class PlateauArtifact(ContractModel):
    """Complete frozen collection; use the ordinary artifact delivery tools."""

    artifact_id: str
    sha256: str
    path: str


class PlateauBest(ContractModel):
    """Chosen saved candidate, keeping its original metric value and identity."""

    parameters: dict[str, float]
    identity: dict[str, str]
    value: float


class PlateauSummary(ContractModel):
    """Scores maximize; value retains original units even for minimized metrics."""

    best: PlateauBest | None
    plateau_best: PlateauBest | None
    best_score: float | None = Field(
        description="Highest direction-adjusted input score."
    )
    plateau_best_score: float | None = Field(
        description=(
            "Highest eligible mean neighbor score; not that candidate's own score."
        )
    )
    plateau_best_value: float | None = Field(
        description=(
            "Plateau-best candidate's own original metric value, in definition.unit."
        )
    )
    score_gap: float | None = Field(
        description=(
            "Best input score minus plateau-best candidate's own score; "
            "not the difference of neighborhood means."
        )
    )
    grid_distance: int | None = Field(
        description=(
            "Chebyshev distance in sorted grid indices between best and plateau-best."
        )
    )
    best_neighborhood_mean_score: float | None = Field(
        description="Mean available neighbor score around best, excluding best itself."
    )
    best_neighborhood_std: float | None = Field(
        description=(
            "Sample standard deviation (ddof=1) of best's available neighbor scores; "
            "null with fewer than two neighbors."
        )
    )
    best_neighborhood_size: int | None = Field(
        description=(
            "Available neighbor count around best; null without an available best."
        )
    )
    spikiness: float | None = Field(
        description=(
            "(best_score - best_neighborhood_mean_score) / best_neighborhood_std; "
            "null for undefined, zero spread or non-finite ratio. "
            "Unstable in small neighborhoods; not a significance test."
        )
    )
    dimension_warning: bool = Field(
        description="True with more than three numeric axes."
    )
    n_below_min_neighborhood: int = Field(
        description=(
            "Available candidates ineligible for plateau-best under the explicit "
            "neighbor-count threshold; zero without a threshold."
        )
    )


class PlateauCell(ContractModel):
    """A grid point; absent means no record in the explicit selected collection."""

    parameters: dict[str, float]
    identity: dict[str, str] | None
    status: Literal["available", "undefined", "absent"]
    reason: str | None
    value: float | None = Field(
        description="Original metric in definition.unit; null is never zero."
    )
    score: float | None = Field(
        description="Higher-is-better metric: minimizing negates value."
    )
    plateau_score: float | None = Field(
        description="Mean score of available neighbors, excluding this point."
    )
    neighborhood_size: int | None


class PlateauResponse(ContractModel):
    """One page from the same immutable collection as its summary and artifact."""

    success: Literal[True]
    schema_version: Literal[1]
    algorithm_version: str
    analysis_id: str
    source_hash: str
    source: dict[str, JsonValue]
    metric: str
    definition: MetricDefinitionRecord
    direction: Literal["maximize", "minimize"]
    axes: dict[str, list[float]]
    settings: dict[str, int | None]
    n_candidates: int
    n_available: int
    n_undefined: int
    n_absent: int
    summary: PlateauSummary
    derived_definitions: dict[str, str]
    limitations: list[str]
    artifact: PlateauArtifact
    cells: Page[PlateauCell]


def plateau_definitions() -> dict[str, str]:
    """Expose derived field semantics in every page and the frozen artifact."""
    return {
        f"{section}.{name}": field.description
        for section, model in (("summary", PlateauSummary), ("cells", PlateauCell))
        for name, field in model.model_fields.items()
        if field.description is not None
    }
