"""Compact acknowledgement of a saved research evidence record."""

from typing import Literal

from pydantic import Field

from tradingdev.app.contracts.common import ContractModel


class ResearchEvidenceSaved(ContractModel):
    """Stable identity for preparation without echoing the complete draft."""

    success: Literal[True]
    evidence_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_count: int = Field(ge=0, strict=True)
    parameter_declaration_count: int = Field(ge=0, strict=True)
