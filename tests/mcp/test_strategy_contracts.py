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


def _save(service: StrategyService) -> None:
    contract = strategy_contract_payload(_PACKAGE_ROOT)
    assert service.save_draft(
        "contract_strategy",
        contract["example_strategy_code"],
        contract["example_yaml_config"],
    ).success


def test_contract_advertises_seed_location_and_strict_config_schema(
    tmp_path: Path,
) -> None:
    server, _service = _server(tmp_path)
    result = asyncio.run(server.call_tool("get_strategy_contract", {}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    contract = result.structured_content
    schema = contract["config_schema"]
    assert schema["additionalProperties"] is False
    assert "random_seed" in schema["properties"]
    assert "random_seed" not in schema["$defs"]["BacktestConfig"]["properties"]
    assert schema["$defs"]["BacktestConfig"]["additionalProperties"] is False
    assert yaml.safe_load(contract["example_yaml_config"])["random_seed"] == 42
    assert "get_numpy_rng()" in contract["api_reference"]
    assert "effective_config" in contract["lifecycle"]


def test_strategy_discovery_validates_bundled_and_generated_metadata(
    tmp_path: Path,
) -> None:
    _server_instance, service = _server(tmp_path)
    _save(service)
    entries = service.list_strategies()
    assert any(item["kind"] == "bundled" for item in entries)
    for item in entries:
        if item["kind"] == "bundled":
            parsed = BundledStrategySummary.model_validate(item)
            assert parsed.status == "promoted"
        else:
            generated = GeneratedStrategySummary.model_validate(item)
            assert generated.metadata.source_hash
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
def test_draft_discovery_preserves_incomplete_requirements_for_repair(
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
        for current in (requirements, repaired):
            saved = _call(
                server,
                "save_strategy",
                {
                    "strategy_id": "repairable_draft",
                    "code": "class DraftStrategy: pass\n",
                    "yaml_config": yaml.safe_dump(
                        {
                            "strategy": {"class_name": "DraftStrategy"},
                            "data": {"requirements": current},
                        }
                    ),
                },
            )
            assert saved["success"] is True
            listed = asyncio.run(server.call_tool("list_strategies", {}))
            assert isinstance(listed, CallToolResult)
            assert listed.structured_content is not None
            entries = listed.structured_content["result"]
            assert isinstance(entries, list)
            assert any(item["kind"] == "bundled" for item in entries)
            draft = next(
                item for item in entries if item["strategy_id"] == "repairable_draft"
            )
            assert draft["status"] == "draft"
            assert draft["data_requirements"] == current

            retrieved = _call(
                server, "get_strategy", {"strategy_id": "repairable_draft"}
            )
            assert retrieved["success"] is True
            yaml_config = retrieved["yaml_config"]
            assert isinstance(yaml_config, str)
            assert yaml.safe_load(yaml_config)["data"]["requirements"] == current


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
        {"strategy_id": strategy_id, "code": code, "yaml_config": yaml_config},
    )

    assert result["success"] is False
    assert result["code"] == expected_code
    assert result["strategy_id"] == strategy_id
    assert result["status"] == "rejected"
    assert result["py_path"] == result["yaml_path"] == ""


def test_validation_diagnostics_remain_a_completed_check_result(tmp_path: Path) -> None:
    server, service = _server(tmp_path)
    contract = strategy_contract_payload(_PACKAGE_ROOT)
    saved = service.save_draft(
        "contract_strategy",
        "import os\n",
        contract["example_yaml_config"],
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
    contract = strategy_contract_payload(_PACKAGE_ROOT)
    saved = _call(
        server,
        "save_strategy",
        {
            "strategy_id": "contract_strategy",
            "code": contract["example_strategy_code"],
            "yaml_config": contract["example_yaml_config"],
        },
    )
    old = {"strategy_id": "contract_strategy", "revision_id": saved["revision_id"]}
    _save(service)

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
