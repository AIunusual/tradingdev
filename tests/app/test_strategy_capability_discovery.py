"""Discover training requirements without executing bundled or generated code."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import strategy as strategy_tools

if TYPE_CHECKING:
    import pytest


def test_bundled_capabilities_are_available_in_discovery_without_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        root = Path(directory)
        bundle = root / "bundled" / "training_fixture"
        bundle.mkdir(parents=True)
        (bundle / "strategy.py").write_text(
            "raise AssertionError('Discovery must not execute strategy code')\n"
        )
        (bundle / "config.yaml").write_text(
            "strategy:\n  id: training_fixture\n  class_name: TrainingFixture\n"
            "  training_requirement: required\n"
        )
        service = StrategyService(WorkspacePaths(root / "workspace"))
        monkeypatch.setattr(
            service, "_catalog", BundledStrategyCatalog(root / "bundled")
        )
        listed = service.list_strategies()
        selected = service.get_strategy("training_fixture")
        spec = service.resolve_executable("training_fixture")
        expected = {
            "training_requirement": "required",
            "supported_execution_modes": ["walk_forward"],
        }
        assert listed[0]["execution_capabilities"] == expected
        assert selected["execution_capabilities"] == expected
        assert spec.execution_capabilities.model_dump(mode="json") == expected


def test_generated_declarations_and_fit_method_do_not_prove_training_capabilities(
    tmp_path: Path,
) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        service = StrategyService(WorkspacePaths(Path(directory)))
        source = (
            "raise AssertionError('Discovery must not execute generated source')\n"
            "class TrainingFixture:\n    def fit(self, data):\n        pass\n"
        )
        saved = service.save_draft(
            "training_fixture",
            source,
            "strategy:\n  class_name: TrainingFixture\n"
            "  training_requirement: required\n",
        )
        assert saved.success
        listed = next(
            item
            for item in service.list_strategies()
            if item["strategy_id"] == "training_fixture"
        )
        selected = service.get_strategy("training_fixture", saved.revision_id)
        expected = {
            "training_requirement": "unknown",
            "supported_execution_modes": None,
        }
        assert listed["execution_capabilities"] == expected
        assert selected["execution_capabilities"] == expected
        assert selected["source_code"] == source


def test_mcp_discovery_exposes_capabilities_and_preserves_kd_walk_forward_support(
    tmp_path: Path,
) -> None:
    async def check(service: StrategyService) -> None:
        server = StrictMCPServer("capability-discovery")
        strategy_tools.register(server, service, Path("src/tradingdev"))
        listed = await server.call_tool("list_strategies", {})
        assert isinstance(listed, CallToolResult)
        assert listed.structured_content is not None
        entries = listed.structured_content["items"]
        quantile = next(
            item for item in entries if item["strategy_id"] == "quantile_volume"
        )
        kd = next(item for item in entries if item["strategy_id"] == "kd_crossover")
        assert (
            "walk_forward" in kd["execution_capabilities"]["supported_execution_modes"]
        )
        selected = await server.call_tool(
            "get_strategy", {"strategy_id": "quantile_volume"}
        )
        assert isinstance(selected, CallToolResult)
        assert selected.structured_content is not None
        response = selected.structured_content["result"]
        assert response["execution_capabilities"] == quantile["execution_capabilities"]

    with TemporaryDirectory(dir=tmp_path) as directory:
        asyncio.run(check(StrategyService(WorkspacePaths(Path(directory)))))
