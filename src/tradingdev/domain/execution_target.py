"""One discriminated execution target for approval and job submission."""

from typing import Annotated

from pydantic import Field

from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest

type ExecutionTarget = Annotated[
    ExecutionManifest | ComparisonManifest, Field(discriminator="kind")
]
