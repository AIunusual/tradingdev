"""Strategy response schemas and MCP lifecycle boundary regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
import yaml
from mcp.server import MCPServer
from mcp_types import CallToolResult
from pydantic import ValidationError
from tests.integration.execution_fixtures import research_input
from tests.research_fixtures import research_for_config

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.strategy import (
    BundledStrategySummary,
    GeneratedStrategyResponse,
    GeneratedStrategySummary,
    StrategyDryRunSuccess,
    StrategyValidationFailure,
)
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.templates import strategy_contract_payload
from tradingdev.mcp.tools.strategy import register

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategyDiagnostic

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "tradingdev"


def _server(tmp_path: Path) -> tuple[MCPServer, StrategyService]:
    service = StrategyService(WorkspacePaths(tmp_path / "workspace"))
    server = MCPServer("strategy-contract-test")
    register(server, service, _PACKAGE_ROOT)
    return server, service


def _call(
    server: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    result = asyncio.run(server.call_tool(name, arguments))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    assert isinstance(result.structured_content, dict)
    payload = result.structured_content["result"]
    assert isinstance(payload, dict)
    return payload


def _save(service: StrategyService, *, expected_revision_id: str | None = None) -> None:
    contract = strategy_contract_payload(_PACKAGE_ROOT, "example")["content"]
    assert service.save_draft(
        contract["example_strategy_code"],
        contract["example_yaml_config"],
        research_for_config(contract["example_yaml_config"]),
        strategy_id="contract_strategy",
        expected_revision_id=expected_revision_id,
    ).success


def test_contract_sections_are_discoverable_and_keep_required_rules(
    tmp_path: Path,
) -> None:
    server, _service = _server(tmp_path)
    result = asyncio.run(server.call_tool("get_strategy_contract", {}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    contract = result.structured_content
    assert contract["section"] == "core"
    assert {item["id"] for item in contract["available_sections"]} == {
        "core",
        "example",
        "configuration",
        "research",
        "evidence",
        "indicators",
    }
    content = contract["content"]
    assert set(content) == {
        "base_strategy_source",
        "requirements",
        "lifecycle",
        "static_policy",
    }
    assert "class BaseStrategy" in content["base_strategy_source"]
    assert "get_numpy_rng()" in " ".join(content["requirements"])
    assert "effective_config" in " ".join(content["lifecycle"])
    assert "security sandbox" in " ".join(content["requirements"])
    assert "-1 means short, never sell-only" in " ".join(content["requirements"])
    base_source = content["base_strategy_source"]
    assert 'In ``mode="signal"``, each row specifies a position state' in base_source
    assert "``signal_as_position`` has no effect in this mode" in base_source
    assert "Zero closes an existing position when ``signal_as_position=True``" in (
        base_source
    )
    assert "zero prevents new entries but retains an" in base_source
    assert "Both modes shift signals by one bar" in base_source

    sections = {}
    for item in contract["available_sections"]:
        assert item["description"]
        response = asyncio.run(
            server.call_tool("get_strategy_contract", {"section": item["id"]})
        )
        assert isinstance(response, CallToolResult)
        assert response.structured_content is not None
        selected = response.structured_content
        assert selected["section"] == item["id"]
        assert selected["available_sections"] == contract["available_sections"]
        sections[item["id"]] = selected["content"]
    schema = sections["configuration"]["config_schema"]
    assert schema["additionalProperties"] is False
    assert "random_seed" in schema["properties"]
    assert "random_seed" not in schema["$defs"]["BacktestConfig"]["properties"]
    assert schema["$defs"]["BacktestConfig"]["additionalProperties"] is False
    position_flag = schema["$defs"]["BacktestConfig"]["properties"][
        "signal_as_position"
    ]
    assert position_flag["default"] is False
    assert "Volume mode only" in position_flag["description"]
    assert "Signal mode always uses position states" in position_flag["description"]
    monthly_limit = schema["$defs"]["BacktestConfig"]["properties"]["monthly_max_loss"]
    assert monthly_limit["default"] == 1500.0
    for rule in (
        "UTC calendar month",
        "previous observed month's last close equity",
        "zero in the first month",
        "Missing months are not filled",
        "not prior-month P&L or entry costs again",
        "Trade records retain whole-trade P&L and costs",
    ):
        assert rule in monthly_limit["description"]
    assert set(sections["configuration"]) == {"config_schema", "notes"}
    configuration_notes = " ".join(sections["configuration"]["notes"])
    assert "consecutive test folds through the final loaded row" in configuration_notes
    assert "test sizes differ by at most one" in configuration_notes
    example = sections["example"]
    assert set(example) == {"example_strategy_code", "example_yaml_config"}
    assert yaml.safe_load(example["example_yaml_config"])["random_seed"] == 42
    research_notes = " ".join(sections["research"]["notes"])
    assert "start_date <= timestamp <= end_date" in research_notes
    assert "including both endpoints" in research_notes
    assert "Date-only values mean 00:00:00 UTC, not the whole final day" in (
        research_notes
    )


def test_contract_rejects_unknown_sections_and_mismatched_payloads(
    tmp_path: Path,
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    from tradingdev.app.contracts.strategy import StrategyContractResponse

    server, _service = _server(tmp_path)
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("get_strategy_contract", {"section": "all"}))
    wrong = strategy_contract_payload(_PACKAGE_ROOT, "example") | {"section": "core"}
    with pytest.raises(ValidationError, match="does not match"):
        StrategyContractResponse.model_validate(wrong)


def test_core_contract_missing_reference_fails_without_partial_response(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        strategy_contract_payload(tmp_path)
    # Examples do not pretend to contain the missing core reference.
    example = strategy_contract_payload(tmp_path, "example")
    assert example["section"] == "example"
    assert "base_strategy_source" not in example["content"]


def test_invalid_config_guidance_leads_to_the_configuration_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, service = _server(tmp_path)
    example = strategy_contract_payload(_PACKAGE_ROOT, "example")["content"]
    config = yaml.safe_load(example["example_yaml_config"])
    config["backtest"]["random_seed"] = 42
    service._revisions.create(
        "invalid_config", example["example_strategy_code"], config
    )

    def quality_diagnostics(_path: Path) -> list[StrategyDiagnostic]:
        return []

    monkeypatch.setattr(service, "_quality_gate_diagnostics", quality_diagnostics)
    result = _call(server, "validate_strategy", {"strategy_id": "invalid_config"})
    response = StrategyValidationFailure.model_validate(result)
    diagnostic = next(
        item for item in response.diagnostics if item.code == "invalid_execution_config"
    )
    assert diagnostic.fix is not None
    assert (
        'get_strategy_contract(section="configuration").content.config_schema'
        in diagnostic.fix
    )

    contract = asyncio.run(
        server.call_tool("get_strategy_contract", {"section": "configuration"})
    )
    assert isinstance(contract, CallToolResult)
    assert contract.structured_content is not None
    schema = contract.structured_content["content"]["config_schema"]
    assert "random_seed" in schema["properties"]
    assert "random_seed" not in schema["$defs"]["BacktestConfig"]["properties"]
    assert schema["$defs"]["BacktestConfig"]["additionalProperties"] is False


def test_strategy_discovery_validates_bundled_and_generated_metadata(
    tmp_path: Path,
) -> None:
    _server_instance, service = _server(tmp_path)
    _save(service)
    entries = service.list_strategies_page().model_dump(mode="json")["items"]
    assert any(item["kind"] == "bundled" for item in entries)
    for item in entries:
        if item["kind"] == "bundled":
            parsed = BundledStrategySummary.model_validate(item)
            assert parsed.status == "promoted"
        else:
            generated = GeneratedStrategySummary.model_validate(item)
            assert generated.revision_id
    response = GeneratedStrategyResponse.model_validate(
        service.get_strategy("contract_strategy")
    )
    assert response.metadata.status == "draft"
    assert response.source_code


@pytest.mark.parametrize(
    "requirements",
    [
        {},
        {"market": {"source": "binance_vision"}},
        {
            "market": {
                "source": "binance_vision",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
            },
            "features": [{"type": "unknown_feature", "settings": {"window": 5}}],
        },
    ],
)
def test_draft_research_completes_market_fields_and_rejects_invalid_features(
    tmp_path: Path, requirements: dict[str, object]
) -> None:
    repaired: dict[str, object] = {
        "market": {
            "source": "binance_vision",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        },
        "features": [],
    }
    with TemporaryDirectory(dir=tmp_path) as directory:
        server, _service = _server(Path(directory))
        expected_revision_id = None
        for current in (requirements, repaired):
            saved = _call(
                server,
                "save_strategy",
                {
                    "strategy_id": "repairable_draft",
                    "expected_revision_id": expected_revision_id,
                    "research": research_input(),
                    "code": "class DraftStrategy: pass\n",
                    "yaml_config": yaml.safe_dump(
                        {
                            "strategy": {"class_name": "DraftStrategy"},
                            "data": {
                                "source": "binance_vision",
                                "market_type": "spot",
                                "requirements": current,
                            },
                        }
                    ),
                },
            )
            if current.get("features"):
                assert saved["success"] is False
                assert saved["code"] == "invalid_research_spec"
                assert _service.load("repairable_draft") is None
                continue
            assert saved["success"] is True, saved
            expected_revision_id = saved["revision_id"]
            listed = asyncio.run(server.call_tool("list_strategies", {}))
            assert isinstance(listed, CallToolResult)
            assert listed.structured_content is not None
            entries = listed.structured_content["items"]
            assert isinstance(entries, list)
            assert any(item["kind"] == "bundled" for item in entries)
            draft = next(
                item for item in entries if item["strategy_id"] == "repairable_draft"
            )
            assert draft["status"] == "draft"
            assert "data_requirements" not in draft

            retrieved = _call(
                server, "get_strategy", {"strategy_id": "repairable_draft"}
            )
            assert retrieved["success"] is True
            yaml_config = retrieved["yaml_config"]
            assert isinstance(yaml_config, str)
            resolved = yaml.safe_load(yaml_config)["data"]["requirements"]
            assert resolved["market"] == repaired["market"]
            if "features" in current:
                assert resolved["features"] == current["features"]


def test_generated_strategy_rejects_undeclared_nested_metadata(
    tmp_path: Path,
) -> None:
    _server_instance, service = _server(tmp_path)
    _save(service)
    payload = service.get_strategy("contract_strategy")
    payload["metadata"]["unexpected_state"] = "unpublished"

    with pytest.raises(ValidationError, match="unexpected_state"):
        GeneratedStrategyResponse.model_validate(payload)


@pytest.mark.parametrize(
    "tool_name",
    ["get_strategy", "validate_strategy", "dry_run_strategy", "promote_strategy"],
)
def test_unknown_strategy_has_a_structured_stable_code(
    tmp_path: Path, tool_name: str
) -> None:
    server, _service = _server(tmp_path)

    result = _call(server, tool_name, {"strategy_id": "missing_strategy"})

    assert result["success"] is False
    assert result["code"] == "strategy_not_found"
    assert isinstance(result["error"], str)


@pytest.mark.parametrize(
    ("strategy_id", "code", "yaml_config", "expected_code"),
    [
        ("Invalid", "pass", "{}", "invalid_strategy_id"),
        ("invalid_python", "def broken(:", "{}", "syntax_error"),
        ("invalid_yaml", "pass", "strategy: [", "invalid_yaml"),
        ("invalid_mapping", "pass", "[]", "invalid_strategy_config"),
        ("invalid_section", "pass", "strategy: []", "invalid_strategy_config"),
        ("missing_class", "pass", "strategy: {}", "invalid_strategy_config"),
    ],
)
def test_save_rejections_preserve_context_and_report_specific_codes(
    tmp_path: Path,
    strategy_id: str,
    code: str,
    yaml_config: str,
    expected_code: str,
) -> None:
    server, _service = _server(tmp_path)

    result = _call(
        server,
        "save_strategy",
        {
            "strategy_id": strategy_id,
            "code": code,
            "yaml_config": yaml_config,
            "research": research_input(),
        },
    )

    assert result["success"] is False
    assert result["code"] == expected_code
    assert result["strategy_id"] == strategy_id
    assert result["status"] == "rejected"
    assert result["py_path"] == result["yaml_path"] == ""


def test_validation_diagnostics_remain_a_completed_check_result(tmp_path: Path) -> None:
    server, service = _server(tmp_path)
    contract = strategy_contract_payload(_PACKAGE_ROOT, "example")["content"]
    saved = service.save_draft(
        "import os\n",
        contract["example_yaml_config"],
        research_for_config(contract["example_yaml_config"]),
        strategy_id="contract_strategy",
    )
    assert saved.success

    result = _call(server, "validate_strategy", {"strategy_id": "contract_strategy"})
    response = StrategyValidationFailure.model_validate(result)

    assert response.success is False
    assert response.status == "draft"
    assert response.diagnostics[0].code == "banned_import"
    assert response.revision_id == saved.revision_id
    assert "error" not in result
    assert "code" not in result


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    [
        (
            '        result["signal"] = np.nan\n        return result',
            "invalid_signal_values",
        ),
        ("        return result.iloc[:-1].copy()", "signal_row_count_mismatch"),
        (
            "        result.index = result.index + 1000\n        return result",
            "signal_index_mismatch",
        ),
    ],
)
def test_invalid_frame_diagnostics_survive_typed_mcp_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
    expected_code: str,
) -> None:
    server, service = _server(tmp_path)
    example = strategy_contract_payload(_PACKAGE_ROOT, "example")["content"]
    saved = service.save_draft(
        example["example_strategy_code"].replace("        return result", replacement),
        example["example_yaml_config"],
        research_for_config(example["example_yaml_config"]),
        strategy_id="contract_strategy",
    )
    assert saved.success
    monkeypatch.setattr(service, "_quality_gate_diagnostics", lambda _path: [])
    result = _call(server, "validate_strategy", {"strategy_id": "contract_strategy"})
    response = StrategyValidationFailure.model_validate(result)
    assert not response.success and response.status == "draft"
    assert expected_code in {item.code for item in response.diagnostics}
    assert response.revision_id == saved.revision_id
    assert "error" not in result


def test_typed_lifecycle_returns_signal_evidence_and_rejects_wrong_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, service = _server(tmp_path)
    _save(service)

    def quality_diagnostics(_path: Path) -> list[StrategyDiagnostic]:
        return []

    monkeypatch.setattr(service, "_quality_gate_diagnostics", quality_diagnostics)
    rejected = _call(server, "dry_run_strategy", {"strategy_id": "contract_strategy"})
    assert rejected["code"] == "invalid_strategy_status"

    validated = _call(server, "validate_strategy", {"strategy_id": "contract_strategy"})
    assert validated["success"] is True
    assert validated["status"] == "validated"
    dry_run = _call(server, "dry_run_strategy", {"strategy_id": "contract_strategy"})
    response = StrategyDryRunSuccess.model_validate(dry_run)
    assert response.signal_analysis.rows == 240
    assert response.signal_analysis.nan_count == 0

    revalidation = _call(
        server, "validate_strategy", {"strategy_id": "contract_strategy"}
    )
    assert revalidation["code"] == "invalid_strategy_status"
    assert revalidation["status"] == "runnable"
    promoted = _call(server, "promote_strategy", {"strategy_id": "contract_strategy"})
    assert promoted["status"] == "promoted"
    repeated = _call(server, "promote_strategy", {"strategy_id": "contract_strategy"})
    assert repeated["code"] == "invalid_strategy_status"


def test_strategy_tool_annotations_describe_actual_effects(tmp_path: Path) -> None:
    server, _service = _server(tmp_path)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    expected = {
        "get_strategy_contract": (True, False, True, False),
        "list_strategies": (True, False, True, False),
        "get_strategy": (True, False, True, False),
        "save_strategy": (False, True, False, False),
        "validate_strategy": (False, True, False, True),
        "dry_run_strategy": (False, True, False, True),
        "promote_strategy": (False, True, True, False),
    }
    assert set(tools) == set(expected)
    for name, flags in expected.items():
        annotations = tools[name].annotations
        assert annotations is not None
        assert (
            annotations.read_only_hint,
            annotations.destructive_hint,
            annotations.idempotent_hint,
            annotations.open_world_hint,
        ) == flags
        schema = tools[name].output_schema
        assert schema is not None
        assert schema.get("type") == "object"
        definitions = schema.get("$defs", {})
        assert isinstance(definitions, dict)
        for model_schema in definitions.values():
            if isinstance(model_schema, dict) and model_schema.get("type") == "object":
                assert model_schema.get("additionalProperties") is False


@pytest.mark.parametrize(
    "tool_name",
    ["get_strategy", "validate_strategy", "dry_run_strategy", "promote_strategy"],
)
def test_explicit_missing_revision_never_selects_current(
    tmp_path: Path, tool_name: str
) -> None:
    server, service = _server(tmp_path)
    _save(service)
    result = _call(
        server,
        tool_name,
        {
            "strategy_id": "contract_strategy",
            "revision_id": "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa",
        },
    )
    assert result["success"] is False
    assert result["code"] == "strategy_not_found"


@pytest.mark.parametrize(
    "tool_name",
    ["get_strategy", "validate_strategy", "dry_run_strategy", "promote_strategy"],
)
def test_revision_tampering_is_a_structured_failure(
    tmp_path: Path, tool_name: str
) -> None:
    server, service = _server(tmp_path)
    _save(service)
    spec = service.load("contract_strategy")
    assert spec is not None
    Path(spec.source_path).write_text("def broken(:\n", encoding="utf-8")
    result = _call(
        server,
        tool_name,
        {"strategy_id": "contract_strategy", "revision_id": spec.revision_id},
    )
    assert result["success"] is False
    assert result["code"] == "strategy_revision_invalid"


def test_checking_an_older_revision_does_not_approve_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, service = _server(tmp_path)
    contract = strategy_contract_payload(_PACKAGE_ROOT, "example")["content"]
    saved = _call(
        server,
        "save_strategy",
        {
            "strategy_id": "contract_strategy",
            "code": contract["example_strategy_code"],
            "yaml_config": contract["example_yaml_config"],
            "research": research_for_config(contract["example_yaml_config"]).model_dump(
                mode="json", exclude_unset=True
            ),
        },
    )
    assert saved["success"], saved
    old = {"strategy_id": "contract_strategy", "revision_id": saved["revision_id"]}
    _save(service, expected_revision_id=str(saved["revision_id"]))

    def quality_diagnostics(_path: Path) -> list[StrategyDiagnostic]:
        return []

    monkeypatch.setattr(service, "_quality_gate_diagnostics", quality_diagnostics)
    for tool in ("validate_strategy", "dry_run_strategy", "promote_strategy"):
        result = _call(server, tool, old)
        assert result["success"] is True
        assert result["revision_id"] == saved["revision_id"]
    current = GeneratedStrategyResponse.model_validate(
        _call(server, "get_strategy", {"strategy_id": "contract_strategy"})
    )
    assert current.revision_id != saved["revision_id"]
    assert current.metadata.status == "draft"
    checked = GeneratedStrategyResponse.model_validate(
        _call(server, "get_strategy", old)
    )
    assert checked.metadata.validation is not None
    assert checked.metadata.dry_run is not None
    assert checked.metadata.validation.revision_id == saved["revision_id"]
    assert checked.metadata.dry_run.revision_id == saved["revision_id"]


def test_remote_indicator_catalog_resolves_one_complete_api(tmp_path: Path) -> None:
    server, _service = _server(tmp_path)
    catalog = asyncio.run(
        server.call_tool("get_strategy_contract", {"section": "indicators"})
    )
    assert isinstance(catalog, CallToolResult)
    assert catalog.structured_content is not None
    content = catalog.structured_content["content"]
    assert content["selected"] is None
    assert any(row["name"] == "bollinger_bands" for row in content["catalog"])
    response = asyncio.run(
        server.call_tool(
            "get_strategy_contract",
            {"section": "indicators", "indicator": "bollinger_bands"},
        )
    )
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    selected = response.structured_content["content"]["selected"]
    assert selected["result_fields"] == ["lower", "middle", "upper"]
    assert "0 <= std <= 3e37" in selected["source"]
    assert "_validate_period" in {
        item["name"] for item in selected["supporting_source"]
    }


@pytest.mark.parametrize(
    "section", ["core", "research", "evidence", "example", "configuration"]
)
def test_indicator_selection_cannot_be_silently_ignored(
    tmp_path: Path, section: str
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    server, _service = _server(tmp_path)
    with pytest.raises(ToolError, match="only valid"):
        asyncio.run(
            server.call_tool(
                "get_strategy_contract", {"section": section, "indicator": "macd"}
            )
        )


def test_core_import_policy_agrees_with_actual_static_validator(tmp_path: Path) -> None:
    from tradingdev.domain.strategies.validator import StrategyValidator

    policy = strategy_contract_payload(_PACKAGE_ROOT)["content"]["static_policy"]
    checker = StrategyValidator()
    source = tmp_path / "imports.py"
    for name in policy["allowed_import_roots"]:
        source.write_text(f"import {name}\n")
        assert checker.static_policy_scan(source) == []
    for name in policy["blocked_import_roots"]:
        source.write_text(f"import {name}\n")
        assert [item.code for item in checker.static_policy_scan(source)] == [
            "banned_import"
        ]
    source.write_text("import nonexistent_indicator_package\n")
    diagnostics = checker.static_policy_scan(source)
    assert diagnostics[0].code == "import_not_allowed"
    assert diagnostics[0].fix is not None
    assert ", ".join(policy["allowed_import_roots"]) in diagnostics[0].fix
