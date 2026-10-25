"""Shared response validation policy and explicit application failures."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100
type PageOffset = Annotated[int, Field(ge=0, strict=True)]
type PageLimit = Annotated[int, Field(ge=1, le=MAX_PAGE_LIMIT, strict=True)]


class ContractModel(BaseModel):
    """Reject undeclared fields and non-finite numbers at public boundaries."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ErrorResponse(ContractModel):
    """An expected application failure; code is stable, error explains context."""

    success: Literal[False]
    error: str
    code: str


class PageRequest(ContractModel):
    """Shared, strict bounds for growing discovery lists."""

    offset: PageOffset = 0
    limit: PageLimit = DEFAULT_PAGE_LIMIT


class Page[T](PageRequest):
    """One complete page; total counts matching items before offset/limit."""

    items: list[T]
    total: Annotated[int, Field(ge=0, strict=True)]
    next_offset: PageOffset | None

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        expected_count = min(self.limit, max(0, self.total - self.offset))
        if len(self.items) != expected_count:
            raise ValueError("Page items do not match the declared extent")
        expected_next = (
            self.offset + len(self.items)
            if self.offset + len(self.items) < self.total
            else None
        )
        if self.next_offset != expected_next:
            raise ValueError("Page next_offset does not match the declared extent")
        return self
