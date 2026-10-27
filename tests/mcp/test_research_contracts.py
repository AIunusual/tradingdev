"""Research tool contracts through in-process MCP dispatch and real storage."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp_types import CallToolResult
from pydantic import TypeAdapter, ValidationError

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.contracts.common import ErrorResponse, Page
from tradingdev.app.contracts.research import (
    ArtifactIdentity,
    ArtifactRecord,
    ArtifactResponse,
    CompareRunsResponse,
    FeatureRequestArtifact,
    RunResponse,
    RunSummary,
)
from tradingdev.app.feature_request_service import FeatureRequestService
from tradingdev.app.run_service import RunService
from tradingdev.mcp.tools import artifacts, feature_requests, runs

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_research_tools_validate_persisted_records_and_expected_errors(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    mcp = MCPServer("research-contract-test")
    runs.register(mcp, RunService(workspace=workspace, store=store))
    artifacts.register(mcp, ArtifactService(workspace=workspace, store=store))
    feature_service = FeatureRequestService(workspace=workspace, store=store)
    feature_requests.register(mcp, feature_service)

    for index in (1, 2):
        store.create_run(
            run_id=f"run_{index}",
            job_id=f"job_{index}",
            strategy_id="fixture",
            artifact_dir=tmp_path,
            metrics={"total_return": index / 10, "details": {"folds": [1, 2]}},
        )
    for artifact_id, content in (("text", b"report"), ("binary", b"\xff")):
        path = tmp_path / artifact_id
        path.write_bytes(content)
        store.create_artifact(
            artifact_id=artifact_id,
            run_id="run_1",
            artifact_type="fixture",
            path=path,
            metadata={"labels": ["fixture"], "optional": None},
        )
    store.create_artifact(
        artifact_id="missing_file",
        run_id="run_1",
        artifact_type="fixture",
        path=tmp_path / "absent.txt",
    )
    feature_service.record(title="A capability", description="Needed for research")

    async def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        return payload

    async def check() -> None:
        listed = await call("list_runs", {})
        records = Page[RunSummary].model_validate(listed).items
        assert {record.run_id for record in records} == {"run_1", "run_2"}
        response = RunResponse.model_validate(
            (await call("get_run", {"run_id": "run_1"}))["result"]
        )
        assert response.run.metrics["details"] == {"folds": [1, 2]}
        comparison = CompareRunsResponse.model_validate(
            (await call("compare_runs", {"run_ids": ["run_1", "run_2"]}))["result"]
        )
        assert comparison.runs[0].metrics == {"total_return": 0.1}

        listed_artifacts = await call("list_artifacts", {"run_id": "run_1"})
        records_artifacts = (
            Page[ArtifactIdentity].model_validate(listed_artifacts).items
        )
        assert len(records_artifacts) == 3
        artifact = ArtifactResponse.model_validate(
            (
                await call(
                    "get_artifact", {"artifact_id": "text", "include_content": True}
                )
            )["result"]
        )
        assert artifact.content == "report"
        requests = TypeAdapter(list[FeatureRequestArtifact]).validate_python(
            (await call("list_feature_requests", {}))["result"]
        )
        assert requests[0].metadata.title == "A capability"

        failures: list[tuple[str, dict[str, object], str]] = [
            ("get_run", {"run_id": "absent"}, "run_not_found"),
            ("compare_runs", {"run_ids": ["run_1"]}, "insufficient_runs"),
            (
                "compare_runs",
                {"run_ids": ["run_1", "absent"]},
                "run_not_found",
            ),
            ("get_artifact", {"artifact_id": "absent"}, "artifact_not_found"),
            (
                "get_artifact",
                {"artifact_id": "missing_file", "include_content": True},
                "artifact_file_missing",
            ),
            (
                "get_artifact",
                {"artifact_id": "binary", "include_content": True},
                "artifact_not_text",
            ),
        ]
        for name, arguments, code in failures:
            error = ErrorResponse.model_validate(
                (await call(name, arguments))["result"]
            )
            assert error.code == code
            assert error.success is False

    asyncio.run(check())


def test_research_tool_rejects_schema_drift_instead_of_returning_a_dict(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    service = RunService(workspace=WorkspacePaths(tmp_path / "workspace"))
    monkeypatch.setattr(service, "get_run", lambda _: {"success": True, "run": {}})
    mcp = MCPServer("research-drift-test")
    runs.register(mcp, service)
    with pytest.raises(
        UnexpectedToolError, match="^Error executing tool get_run$"
    ) as caught:
        asyncio.run(mcp.call_tool("get_run", {"run_id": "malformed"}))
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "run_id" in str(caught.value.__cause__)


def test_artifact_metadata_cannot_hide_unexpected_fixed_fields() -> None:
    payload = {
        "artifact_id": "artifact",
        "run_id": None,
        "artifact_type": "fixture",
        "path": "/unused",
        "sha256": None,
        "metadata": {"producer_extension": {"value": 1}},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    assert ArtifactRecord.model_validate(payload).metadata
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ArtifactRecord.model_validate(payload | {"unexpected": True})


def test_research_tool_rejects_service_response_without_success(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    service = RunService(workspace=WorkspacePaths(tmp_path / "workspace"))
    monkeypatch.setattr(service, "compare_runs", lambda *_, **__: {"runs": []})
    mcp = MCPServer("research-missing-success-test")
    runs.register(mcp, service)
    with pytest.raises(
        UnexpectedToolError, match="^Error executing tool compare_runs$"
    ) as caught:
        asyncio.run(mcp.call_tool("compare_runs", {"run_ids": ["first", "second"]}))
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert any(error["loc"] == ("success",) for error in cause.errors())
