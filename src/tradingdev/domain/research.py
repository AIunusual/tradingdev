"""Explicit research intake and provenance carried by effective configurations."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema

ResearchText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]
ResearchSource = Literal["request", "project_default", "selected_revision"]
AnnualizationChoice = (
    Literal["market", "unavailable"]
    | Annotated[float, Field(gt=0, allow_inf_nan=False)]
)
RESEARCH_FIELD_PATHS = frozenset(
    {
        "/research/strategy_outline",
        *(
            f"/backtest/{field}"
            for field in (
                "symbol",
                "timeframe",
                "start_date",
                "end_date",
                "init_cash",
                "fees",
                "slippage",
                "periods_per_year",
            )
        ),
    }
)


def research_datetime(value: object) -> datetime:
    """Accept explicit ISO dates/timestamps, interpreting a missing timezone as UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError("Research dates must be ISO dates or timestamps") from exc
    else:
        raise ValueError("Research dates must be explicit ISO dates or timestamps")
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


class _ResearchFields(BaseModel):
    """None defaults record omission internally; explicit null is never a choice."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Reflect input validators without replacing Pydantic's generated schema.

        Python constructors know their concrete kind and retain that convenience
        default. A discriminated wire request must supply the tag. Optional None
        defaults represent an omitted choice, not an accepted explicit null.
        """
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        if handler.mode != "validation":
            return schema
        schema = handler.resolve_ref_schema(schema)
        properties = schema.get("properties", {})
        for name, field in cls.model_fields.items():
            if field.default is not None:
                continue
            property_schema = properties[name]
            alternatives = property_schema.get("anyOf", [])
            non_null = [item for item in alternatives if item != {"type": "null"}]
            if non_null:
                if len(non_null) == 1:
                    property_schema.pop("anyOf")
                    property_schema.update(non_null[0])
                else:
                    property_schema["anyOf"] = non_null
            property_schema.pop("default", None)
        if "kind" in properties:
            schema["required"] = list(
                dict.fromkeys([*schema.get("required", []), "kind"])
            )
            properties["kind"].pop("default", None)
        return schema

    @field_validator("start_date", "end_date", mode="before", check_fields=False)
    @classmethod
    def explicit_dates(cls, value: object) -> datetime:
        return research_datetime(value)

    @field_validator(
        "strategy_outline", "symbol", "timeframe", mode="before", check_fields=False
    )
    @classmethod
    def explicit_text(cls, value: object) -> object:
        if value is None:
            raise ValueError("Explicit null is not an omitted research choice")
        return value

    @field_validator(
        "init_cash",
        "fees",
        "slippage",
        "annualization",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def explicit_numbers(cls, value: object) -> object:
        if value in ("market", "unavailable"):
            return value
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(
                "Provide a finite number; null and booleans are not choices"
            )
        return value

    @model_validator(mode="after")
    def valid_period_and_factor(self) -> Self:
        start = getattr(self, "start_date", None)
        end = getattr(self, "end_date", None)
        if start is not None and end is not None and end <= start:
            raise ValueError("end_date must be after start_date")
        factor = getattr(self, "annualization", None)
        if isinstance(factor, float | int) and factor <= 0:
            raise ValueError("An explicit annualization factor must be positive")
        return self


class NewResearchSpec(_ResearchFields):
    """New research needs all four user choices; only simulation costs have defaults."""

    kind: Literal["new"] = "new"
    strategy_outline: ResearchText
    symbol: ResearchText
    start_date: datetime
    end_date: datetime
    timeframe: ResearchText
    init_cash: float | None = Field(default=None, gt=0)
    fees: float | None = Field(default=None, ge=0)
    slippage: float | None = Field(default=None, ge=0)
    annualization: AnnualizationChoice = "market"


class InheritResearchSpec(_ResearchFields):
    """Explicitly inherit omitted research choices from one selected fixed revision."""

    kind: Literal["inherit_selected_revision"] = "inherit_selected_revision"
    strategy_outline: ResearchText | None = None
    symbol: ResearchText | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    timeframe: ResearchText | None = None
    init_cash: float | None = Field(default=None, gt=0)
    fees: float | None = Field(default=None, ge=0)
    slippage: float | None = Field(default=None, ge=0)
    annualization: AnnualizationChoice | None = None


ResearchSpec = Annotated[
    NewResearchSpec | InheritResearchSpec, Field(discriminator="kind")
]


def research_payload(spec: ResearchSpec) -> dict[str, JsonValue]:
    """Transport explicit choices without inventing nulls or dropping the kind tag."""
    payload: dict[str, JsonValue] = spec.model_dump(mode="json", exclude_unset=True)
    payload["kind"] = spec.kind
    return payload


class ResearchRevisionIdentity(BaseModel):
    """Identity of the immutable YAML used for explicit inheritance."""

    model_config = ConfigDict(extra="forbid", strict=True)

    strategy_id: ResearchText
    revision_id: ResearchText
    config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ResearchMetadata(BaseModel):
    """Origins of resolved fields, never a proof that a human supplied the input."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    policy_version: ResearchText
    strategy_outline: ResearchText
    field_sources: dict[str, ResearchSource]
    selected_revision: ResearchRevisionIdentity | None = None
    annualization_reason: ResearchText

    @model_validator(mode="after")
    def inheritance_has_identity(self) -> Self:
        if self.field_sources.keys() != RESEARCH_FIELD_PATHS:
            raise ValueError(
                "field_sources must describe exactly the research-owned fields; "
                f"missing={sorted(RESEARCH_FIELD_PATHS - self.field_sources.keys())}, "
                f"unknown={sorted(self.field_sources.keys() - RESEARCH_FIELD_PATHS)}"
            )
        if (
            "selected_revision" in self.field_sources.values()
            and self.selected_revision is None
        ):
            raise ValueError(
                "Inherited field sources require a selected revision identity"
            )
        return self
