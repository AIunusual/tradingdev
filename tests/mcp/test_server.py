"""Server composition and import side-effect tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.mcp.server import create_server, http_transport_security, main
from tradingdev.mcp.strict_server import StrictMCPServer

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def runtime_root(tmp_path: Path) -> Iterator[Path]:
    with TemporaryDirectory(dir=tmp_path) as temporary:
        yield Path(temporary)


def test_import_and_help_do_not_create_a_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env = {**os.environ, "TRADINGDEV_WORKSPACE": str(workspace)}
    for args in (
        ["-c", "import tradingdev.mcp.server"],
        ["-m", "tradingdev.mcp.server", "--help"],
    ):
        result = subprocess.run(  # noqa: S603
            [sys.executable, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert not workspace.exists()


@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
def test_main_passes_selected_transport_and_port_to_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    calls: list[dict[str, Any]] = []

    def run(_server: StrictMCPServer, transport: str = "stdio", **options: Any) -> None:
        calls.append({"transport": transport, **options})

    monkeypatch.setattr(StrictMCPServer, "run", run)
    arguments = ["--workspace", str(tmp_path / "workspace")]
    expected: dict[str, Any] = {"transport": transport}
    if transport != "stdio":
        arguments.extend(["--web", "--transport", transport, "--port", "18765"])
        expected.update(
            host="127.0.0.1",
            port=18765,
            transport_security=http_transport_security(18765),
        )
    if transport == "streamable-http":
        expected["json_response"] = False
    main(arguments)
    assert calls == [expected]


def test_main_passes_explicit_proxy_allowlists(
    runtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options: dict[str, Any] = {}

    def run(_server: StrictMCPServer, **kwargs: Any) -> None:
        options.update(kwargs)

    monkeypatch.setattr(StrictMCPServer, "run", run)
    main(
        [
            "--workspace",
            str(runtime_root / "workspace"),
            "--web",
            "--allow-host",
            "research.example",
            "--allow-host",
            "research.example:9443",
            "--allow-origin",
            "https://research.example",
            "--allow-origin",
            "https://portal.example:8443",
        ]
    )
    security = options["transport_security"]
    assert security.enable_dns_rebinding_protection is True
    assert {"research.example", "research.example:9443"} <= set(security.allowed_hosts)
    assert {"https://research.example", "https://portal.example:8443"} <= set(
        security.allowed_origins
    )
    assert options["host"] == "127.0.0.1"


def test_server_instances_keep_generated_strategies_in_their_workspace(
    runtime_root: Path,
) -> None:
    first_workspace = WorkspacePaths(runtime_root / "first")
    second_workspace = WorkspacePaths(runtime_root / "second")
    first = create_server(first_workspace)
    second = create_server(second_workspace)

    async def check() -> None:
        saved_result = await first.call_tool(
            "save_strategy",
            {
                "strategy_id": "isolated_strategy",
                "code": "class IsolatedStrategy: pass\n",
                "yaml_config": (
                    "strategy:\n  id: isolated_strategy\n"
                    "  class_name: IsolatedStrategy\n"
                    "data:\n  source: binance_vision\n  market_type: futures/um\n"
                ),
                "research": {
                    "kind": "new",
                    "strategy_outline": "Workspace isolation fixture",
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-08",
                },
            },
        )
        assert isinstance(saved_result, CallToolResult)
        assert saved_result.structured_content is not None
        assert isinstance(saved_result.structured_content, dict)
        assert saved_result.structured_content["result"]["success"] is True
        first_result = await first.call_tool(
            "get_strategy", {"strategy_id": "isolated_strategy"}
        )
        second_result = await second.call_tool(
            "get_strategy", {"strategy_id": "isolated_strategy"}
        )
        assert isinstance(first_result, CallToolResult)
        assert first_result.structured_content is not None
        assert isinstance(first_result.structured_content, dict)
        assert first_result.structured_content["result"]["success"] is True
        assert isinstance(second_result, CallToolResult)
        assert second_result.structured_content is not None
        assert isinstance(second_result.structured_content, dict)
        assert second_result.structured_content["result"]["success"] is False

    asyncio.run(check())
    assert list(
        (first_workspace.generated_strategies / "isolated_strategy" / "revisions").glob(
            "*/strategy.py"
        )
    )
    assert not (second_workspace.generated_strategies / "isolated_strategy").exists()


def test_revision_selectors_are_advertised_in_tool_schemas(runtime_root: Path) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    for name in (
        "get_strategy",
        "validate_strategy",
        "dry_run_strategy",
        "promote_strategy",
        "prepare_backtest",
        "prepare_comparison",
        "prepare_walk_forward",
        "prepare_optimization",
    ):
        schema = tools[name].input_schema
        assert "revision_id" in schema["properties"]
        assert "revision_id" not in schema.get("required", [])
    saved_schema = tools["save_strategy"].output_schema
    assert saved_schema is not None
    assert "revision_id" in saved_schema["$defs"]["StrategySaveSuccess"]["required"]


def test_execution_manifest_hash_is_required_for_prepared_and_accepted_plans(
    runtime_root: Path,
) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    for name, contract in (
        ("prepare_backtest", "PlanResponse"),
        ("prepare_comparison", "PlanResponse"),
        ("prepare_walk_forward", "PlanResponse"),
        ("prepare_optimization", "PlanResponse"),
        ("submit_execution_decision", "PlanStarted"),
    ):
        schema = tools[name].output_schema
        assert schema is not None
        accepted = schema["$defs"][contract]
        assert "manifest_hash" in accepted["required"]
        assert accepted["properties"]["manifest_hash"]["pattern"] == "^[0-9a-f]{64}$"


def test_confirmation_rules_are_shared_with_brief_tool_reminders(
    runtime_root: Path,
) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    assert server.instructions is not None
    instructions = " ".join(server.instructions.split())
    assert "Quote confirmation_summary verbatim" in instructions
    assert "Preserve every label, value, unit and train/test role" in instructions
    assert "answer or clarify and wait" in instructions
    assert "Copy confirmation_link verbatim" in instructions
    assert "its destination against html_path" in instructions
    assert "do not assemble or edit IDs or paths" in instructions
    assert 'section="research") before explaining date boundaries' in instructions
    assert "No code block, file:// or invented URI" in instructions
    assert "not a public URL" in instructions
    assert "End the turn after the user can read the HTML" in instructions
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert "get_artifact" in tools
    for name in (
        "prepare_backtest",
        "prepare_comparison",
        "prepare_walk_forward",
        "prepare_optimization",
    ):
        description = " ".join((tools[name].description or "").split())
        assert "confirmation_summary" in description and "verbatim" in description, name
        assert "confirmation_link" in description, name
        assert "HTML" in description and "end the turn" in description, name
        schema = tools[name].output_schema
        assert schema is not None
        assert {
            "plan_id",
            "confirmation_summary",
            "confirmation_link",
            "html_path",
        } <= set(schema["$defs"]["PlanResponse"]["required"])
    decision = " ".join((tools["submit_execution_decision"].description or "").split())
    assert "explicit later decision" in decision
    assert "questions and ambiguous replies are not consent" in instructions
    assert "new plan, presentation and fresh approval" in instructions
    assert "new plan and fresh approval" in decision
    assert "cannot prove the decision's human origin" in decision


def test_preparation_schema_separates_code_checks_from_historical_execution(
    runtime_root: Path,
) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    discovery = " ".join((tools["list_data_sources"].description or "").split())
    status_description = tools["get_job_status"].description or ""
    assert "generate_report(comparison_job_id=job_id)" in status_description
    assert "comparison_job_id" in tools["generate_report"].input_schema["properties"]
    comparison = tools["prepare_comparison"].input_schema
    assert "candidates" in comparison["required"]
    assert "parameters" not in comparison["properties"]
    assert {"label", "rationale", "parameters"} == set(
        comparison["$defs"]["ParameterCandidateInput"]["required"]
    )
    assert "local capabilities offline" in discovery
    assert "data after approval" in discovery
    assert "sample is available" not in discovery
    for name in (
        "prepare_backtest",
        "prepare_comparison",
        "prepare_walk_forward",
        "prepare_optimization",
    ):
        tool = tools[name]
        assert not {"sample_bars", "minimum_history_bars"}.intersection(
            tool.input_schema["properties"]
        )
        schema = tool.output_schema
        assert schema is not None
        receipt = schema["$defs"]["PreflightReceipt"]
        assert receipt["properties"]["schema_version"]["const"] == 2
        assert {
            "synthetic_fixture_rows",
            "signal_check",
            "training_requirement",
            "not_checked",
        } <= set(receipt["required"])
        assert not {
            "data_source",
            "windows",
            "trade_count",
            "execution_record_count",
            "trading_path_exercised",
            "tested_fold_count",
            "sample_bars_used",
        }.intersection(receipt["properties"])


def test_research_evidence_has_one_save_schema_and_compact_prepare_references(
    runtime_root: Path,
) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    saved = tools["save_research_evidence"]
    assert saved.annotations is not None
    assert (
        saved.annotations.read_only_hint,
        saved.annotations.destructive_hint,
        saved.annotations.idempotent_hint,
        saved.annotations.open_world_hint,
    ) == (False, False, False, False)
    assert saved.input_schema["required"] == ["draft"]
    assert "ResearchEvidenceDraft" in saved.input_schema["$defs"]
    assert saved.output_schema is not None
    compact = saved.output_schema["$defs"]["ResearchEvidenceSaved"]
    assert set(compact["properties"]) == {
        "success",
        "evidence_id",
        "content_hash",
        "source_count",
        "parameter_declaration_count",
    }
    for name in (
        "prepare_backtest",
        "prepare_comparison",
        "prepare_walk_forward",
        "prepare_optimization",
    ):
        schema = tools[name].input_schema
        assert "evidence_id" in schema["required"]
        assert "presentation" not in schema["properties"]
        assert "ResearchEvidenceDraft" not in schema.get("$defs", {})
        output = tools[name].output_schema
        assert output is not None
        assert (
            "required_parameter_bindings"
            in output["$defs"]["PlanRejected"]["properties"]
        )
