"""Run one immutable strategy revision's signal check under a supervisor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
from tradingdev.domain.strategies.contract import SignalContractChecker
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import StrategyDiagnostic, StrategyMetadata


class StrategyCheckRequest(BaseModel):
    """The selected revision and exact synthetic fixture for this invocation."""

    model_config = ConfigDict(extra="forbid")
    source_workspace: str
    metadata: StrategyMetadata
    fixture_rows: int = Field(gt=0, strict=True)


class StrategyCheckResponse(BaseModel):
    """Structured evidence bound to the complete serialized request."""

    model_config = ConfigDict(extra="forbid")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: list[StrategyDiagnostic]
    signal_analysis: dict[str, Any] = Field(default_factory=dict)
    effective_config: dict[str, Any] | None = None


def check_revision(request_path: Path, result_path: Path) -> None:
    """Write a response only if the checker returns normally."""
    payload = request_path.read_bytes()
    request = StrategyCheckRequest.model_validate_json(payload)
    workspace = WorkspacePaths(Path(request.source_workspace))
    revisions = StrategyRevisionStore(workspace)
    revisions.verify(request.metadata)
    checker = SignalContractChecker(StrategyLoader(workspace_root=workspace.root))
    checked = checker.check(request.metadata, fixture_rows=request.fixture_rows)
    revisions.verify(request.metadata)
    response = StrategyCheckResponse(
        request_hash=hashlib.sha256(payload).hexdigest(), **checked
    )
    result_path.write_text(response.model_dump_json(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    check_revision(args.request, args.result)


if __name__ == "__main__":
    main()
