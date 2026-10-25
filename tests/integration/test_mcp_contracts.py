"""Verify advertised contracts and error boundaries over real stdio MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate
from mcp import ClientSession

from tests.integration.execution_fixtures import execution_options

if TYPE_CHECKING:
    from mcp_types import ListToolsResult, PaginatedRequestParams

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_READ_ONLY_TOOLS = {
    "get_strategy_contract",
    "list_strategies",
    "get_strategy",
    "get_execution_plan",
    "list_available_data",
    "list_data_sources",
    "inspect_dataset",
    "list_jobs",
    "list_runs",
    "get_run",
    "get_metric_catalog",
    "get_run_metrics",
    "find_runs",
    "get_run_trades",
    "get_run_equity",
    "get_run_executions",
    "get_run_account_history",
    "get_report_sections",
    "compare_runs",
    "list_artifacts",
    "get_artifact",
    "list_feature_requests",
    "get_ml_contract",
    "predict",
    "leaderboard",
}

# readOnly, destructive, idempotent, openWorld: expected public behavior,
# intentionally independent of the server's annotation helper.
_MUTATING_HINTS = {
    "save_strategy": (False, True, False, False),
    "validate_strategy": (False, True, False, True),
    "dry_run_strategy": (False, True, False, True),
    "promote_strategy": (False, True, True, False),
    "cleanup_strategy_drafts": (False, True, True, False),
    "ensure_data": (False, True, False, True),
    "prepare_backtest": (False, False, False, True),
    "prepare_walk_forward": (False, False, False, True),
    "prepare_optimization": (False, False, False, True),
    "get_job_status": (False, True, True, False),
    "submit_execution_decision": (False, True, True, True),
    "cancel_execution_plan": (False, True, True, False),
    "cancel_job": (False, True, True, False),
    "record_feature_request": (False, False, False, False),
    "generate_report": (False, False, True, False),
    # Recording can replace an evaluation; dataset/panel builds can evict
    # feature caches, and training rewrites stored model artifacts. Stable
    # identities do not remove these operations' destructive side effects.
    "evaluate_signal": (False, True, True, False),
    "build_dataset": (False, True, True, False),
    "build_panel": (False, True, True, False),
    "train_model": (False, True, True, False),
}


def _assert_constrained_objects(node: object, location: str) -> None:
    """Allow typed dynamic maps, but never arbitrary dict[str, Any] objects."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" not in node:
            values = node.get("additionalProperties")
            assert isinstance(values, dict) and values, location
        for key, value in node.items():
            _assert_constrained_objects(value, f"{location}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_constrained_objects(value, f"{location}/{index}")


async def test_discovery_collects_output_schemas_across_pages(
    mcp_workspace: MCPWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_list_tools = ClientSession.list_tools
    cursors: list[str | None] = []
    expected_names: set[str] = set()

    async def paginated_tools(
        session: ClientSession, *, params: PaginatedRequestParams
    ) -> ListToolsResult:
        # Use real server schemas, split into pages at the client boundary.
        # The replacement accepts only the SDK's supported pagination API.
        result = await original_list_tools(session)
        expected_names.update(tool.name for tool in result.tools)
        cursors.append(params.cursor)
        assert params.cursor in {None, "next-page"}
        midpoint = len(result.tools) // 2
        assert midpoint > 0
        first_page = params.cursor is None
        return result.model_copy(
            update={
                "tools": result.tools[:midpoint]
                if first_page
                else result.tools[midpoint:],
                "next_cursor": "next-page" if first_page else None,
            }
        )

    monkeypatch.setattr(ClientSession, "list_tools", paginated_tools)
    async with mcp_workspace.connect() as client:
        assert cursors == [None, "next-page"]
        assert client.output_schemas.keys() == expected_names
        assert (await client.call("list_jobs"))["items"] == []


async def test_all_tools_advertise_constrained_output_schemas_and_hints(
    mcp_workspace: MCPWorkspace,
) -> None:
    expected_hints = {
        **dict.fromkeys(_READ_ONLY_TOOLS, (True, False, True, False)),
        **_MUTATING_HINTS,
    }
    malformed_payloads: list[dict[str, object]] = [
        {},
        {"result": {}},
        {"result": [{}]},
    ]
    async with mcp_workspace.connect() as client:
        tools = {tool.name: tool for tool in (await client.session.list_tools()).tools}
        # Counted from the inventory above rather than from a literal, so the
        # number cannot drift away from the hints that are actually declared.
        # A tool that is advertised but not listed still fails, on the
        # equality check just below.
        assert len(tools) == len(expected_hints)
        assert (
            not {
                "start_backtest",
                "start_walk_forward",
                "start_optimization",
                "confirm_optimization",
            }
            & tools.keys()
        )
        assert tools.keys() == expected_hints.keys() == client.output_schemas.keys()
        for name, tool in tools.items():
            assert tool.description and tool.description.strip(), name
            Draft202012Validator.check_schema(tool.input_schema)
            assert tool.input_schema.get("additionalProperties") is False, name
            assert tool.annotations is not None, name
            annotations = tool.annotations
            assert (
                annotations.read_only_hint,
                annotations.destructive_hint,
                annotations.idempotent_hint,
                annotations.open_world_hint,
            ) == expected_hints[name], name

            schema = client.output_schemas[name]
            assert schema == tool.output_schema
            Draft202012Validator.check_schema(schema)
            assert schema["type"] == "object", name
            assert schema.get("properties"), name
            _assert_constrained_objects(schema, name)
            # A fixed record or a list of fixed records must not degrade to an
            # arbitrary JSON container, including inside the SDK's result wrap.
            for malformed in malformed_payloads:
                with pytest.raises(ValidationError):
                    validate(instance=malformed, schema=schema)


async def test_application_errors_follow_advertised_output_schemas(
    mcp_workspace: MCPWorkspace,
) -> None:
    missing_strategy = {"strategy_id": "missing_strategy"}
    missing_job = {"job_id": "missing_job"}
    run_arguments = {
        **missing_strategy,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
        **execution_options({}),
    }
    failures: list[tuple[str, dict[str, Any], str, object]] = [
        (name, missing_strategy, "success", False)
        for name in (
            "get_strategy",
            "validate_strategy",
            "dry_run_strategy",
            "promote_strategy",
        )
    ]
    failures.extend(
        [
            ("get_run", {"run_id": "missing_run"}, "success", False),
            (
                "cleanup_strategy_drafts",
                {"strategy_id": "missing_strategy", "apply": True},
                "success",
                False,
            ),
            ("get_run_metrics", {"run_id": "missing_run"}, "success", False),
            (
                "get_artifact",
                {"artifact_id": "missing_artifact"},
                "success",
                False,
            ),
            (
                "compare_runs",
                {"run_ids": ["missing_a", "missing_b"]},
                "success",
                False,
            ),
            ("get_job_status", missing_job, "status", "not_found"),
            ("cancel_job", missing_job, "success", False),
            ("get_execution_plan", {"plan_id": "missing_plan"}, "success", False),
            (
                "submit_execution_decision",
                {"plan_id": "missing_plan", "decision": "approve"},
                "success",
                False,
            ),
            ("prepare_backtest", run_arguments, "success", False),
            ("prepare_walk_forward", run_arguments, "success", False),
            (
                "prepare_optimization",
                {
                    **missing_strategy,
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "param_ranges": {"fast_period": [3, 5]},
                    "optimization_metric": "sharpe_ratio",
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-03",
                    "test_start": "2024-01-04",
                    "test_end": "2024-01-08",
                    **execution_options({}),
                },
                "success",
                False,
            ),
            (
                "save_strategy",
                {
                    "strategy_id": "invalid_strategy",
                    "code": "def broken(:",
                    "yaml_config": "strategy: {}",
                },
                "success",
                False,
            ),
        ]
    )
    async with mcp_workspace.connect() as client:
        for name, arguments, field, expected in failures:
            # The harness checks the original structuredContent, before
            # unwrapping unions, and rejects MCP isError application replies.
            response = await client.call(name, **arguments)
            assert response[field] == expected, (name, response)
            assert isinstance(response["code"], str) and response["code"], name
            assert response.get("error") or response.get("message"), name
        assert (await client.call("list_jobs"))["items"] == []
        assert (await client.call("list_runs"))["items"] == []


async def test_missing_required_inputs_are_mcp_errors_without_side_effects(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        tools = (await client.session.list_tools()).tools
        required_tools = [tool for tool in tools if tool.input_schema.get("required")]
        assert {"save_strategy", "prepare_backtest", "record_feature_request"} <= {
            tool.name for tool in required_tools
        }
        for tool in required_tools:
            result = await client.session.call_tool(tool.name, {})
            assert result.is_error, (tool.name, result)
            assert result.content, tool.name
            assert result.structured_content is None, tool.name
        assert (await client.call("list_jobs"))["items"] == []
        assert (await client.call("list_runs"))["items"] == []
        assert await client.call("list_feature_requests") == []
        assert not list(mcp_workspace.workspace.rglob("*.py"))


async def test_cleanup_previews_then_deletes_only_explicit_unused_drafts(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        arguments = {
            "strategy_id": "cleanup_fixture",
            "code": "class Fixture: pass\n",
            "yaml_config": "strategy:\n  class_name: Fixture\n",
        }
        first = await client.call("save_strategy", **arguments)
        current = await client.call("save_strategy", **arguments)
        assert first["success"] and current["success"]
        preview = await client.call(
            "cleanup_strategy_drafts", strategy_id="cleanup_fixture"
        )
        assert preview["success"] and not preview["applied"]
        outcomes = {
            item["revision_id"]: item["outcome"] for item in preview["revisions"]
        }
        assert outcomes == {
            first["revision_id"]: "eligible",
            current["revision_id"]: "protected",
        }
        preserved = await client.call(
            "get_strategy",
            strategy_id="cleanup_fixture",
            revision_id=first["revision_id"],
        )
        assert preserved["success"]
        rejected = await client.call(
            "cleanup_strategy_drafts", strategy_id="cleanup_fixture", apply=True
        )
        assert not rejected["success"]
        assert rejected["code"] == "cleanup_revision_ids_required"
        deleted = await client.call(
            "cleanup_strategy_drafts",
            strategy_id="cleanup_fixture",
            revision_ids=[first["revision_id"]],
            apply=True,
        )
        assert deleted["success"] and deleted["applied"]
        assert deleted["revisions"] == [
            {"revision_id": first["revision_id"], "outcome": "deleted", "reasons": []}
        ]
        missing = await client.call(
            "get_strategy",
            strategy_id="cleanup_fixture",
            revision_id=first["revision_id"],
        )
        assert not missing["success"]
        protected = await client.call(
            "cleanup_strategy_drafts",
            strategy_id="cleanup_fixture",
            revision_ids=[current["revision_id"]],
            apply=True,
        )
        assert not protected["success"]
        assert protected["revisions"][0]["reasons"] == ["current_revision"]
        selected = await client.call("get_strategy", strategy_id="cleanup_fixture")
        assert selected["success"] and selected["revision_id"] == current["revision_id"]


@pytest.mark.parametrize("apply", [False, True], ids=["preview", "apply"])
async def test_unknown_cleanup_strategy_is_error_without_creating_lifecycle_lock(
    mcp_workspace: MCPWorkspace,
    apply: bool,
) -> None:
    strategy_id = "cleanup_unknown_apply" if apply else "cleanup_unknown_preview"
    generated = mcp_workspace.workspace / "generated_strategies"
    async with mcp_workspace.connect() as client:
        before = set(generated.rglob("*"))
        response = await client.call(
            "cleanup_strategy_drafts",
            strategy_id=strategy_id,
            revision_ids=["00000000000040008000000000000000"] if apply else None,
            apply=apply,
        )
        assert response["success"] is False, response
        assert response["code"] == "strategy_not_found", response
        assert not (generated / ".locks" / f"{strategy_id}.lock").exists()
        assert set(generated.rglob("*")) == before

        rejected = await client.call(
            "cleanup_strategy_drafts",
            strategy_id=strategy_id,
            apply=True,
        )
        assert rejected["success"] is False, rejected
        assert rejected["code"] == "cleanup_revision_ids_required", rejected
        assert set(generated.rglob("*")) == before
