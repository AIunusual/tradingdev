"""Strategy lifecycle MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.strategy import (
    BundledStrategyResponse,
    BundledStrategySummary,
    GeneratedStrategyResponse,
    GeneratedStrategySummary,
    LegacyStrategyResponse,
    LegacyStrategySummary,
    StrategyContractResponse,
    StrategyDryRunFailure,
    StrategyDryRunSuccess,
    StrategyPromoteSuccess,
    StrategySaveFailure,
    StrategySaveSuccess,
    StrategyStateError,
    StrategyValidationFailure,
    StrategyValidationSuccess,
)
from tradingdev.domain.strategies.templates import strategy_contract_payload
from tradingdev.mcp.schemas import SaveStrategyInput

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server import MCPServer

    from tradingdev.app.strategy_service import StrategyService


def register(mcp: MCPServer, service: StrategyService, package_root: Path) -> None:
    """Register strategy lifecycle tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_strategy_contract() -> StrategyContractResponse:
        """Read source and YAML requirements before drafting a new strategy."""
        return StrategyContractResponse.model_validate(
            strategy_contract_payload(package_root)
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_strategies() -> list[
        BundledStrategySummary | GeneratedStrategySummary | LegacyStrategySummary
    ]:
        """List strategies; read and explicitly resave any legacy entries."""
        return [
            BundledStrategySummary.model_validate(item)
            if item.get("kind") == "bundled"
            else LegacyStrategySummary.model_validate(item)
            if item.get("kind") == "legacy"
            else GeneratedStrategySummary.model_validate(item)
            for item in service.list_strategies()
        ]

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_strategy(
        strategy_id: str,
        revision_id: str | None = None,
        legacy: bool = False,
    ) -> (
        BundledStrategyResponse
        | GeneratedStrategyResponse
        | LegacyStrategyResponse
        | ErrorResponse
    ):
        """Read source/YAML; omitted revision_id selects current once.

        Legacy source is read-only recovery material: save it as a new draft,
        then validate/dry-run the returned revision before execution.
        Set legacy=true to read a legacy entry sharing a reserved bundled ID;
        save that source under a different ID. Do not combine with revision_id.
        """
        response = service.get_strategy(
            strategy_id, revision_id=revision_id, legacy=legacy
        )
        if not response["success"]:
            return ErrorResponse.model_validate(response)
        if response["kind"] == "bundled":
            return BundledStrategyResponse.model_validate(response)
        if response["kind"] == "legacy":
            return LegacyStrategyResponse.model_validate(response)
        return GeneratedStrategyResponse.model_validate(response)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def save_strategy(
        strategy_id: str,
        code: str,
        yaml_config: str,
        request_summary: str = "",
    ) -> StrategySaveSuccess | StrategySaveFailure:
        """Save a draft revision; use its revision_id for checks and execution."""
        payload = SaveStrategyInput(
            strategy_id=strategy_id,
            code=code,
            yaml_config=yaml_config,
            request_summary=request_summary,
        )
        saved = service.save_draft(
            payload.strategy_id,
            payload.code,
            payload.yaml_config,
            request_summary=payload.request_summary,
        )
        response = {
            "success": saved.success,
            "message": "Draft strategy saved." if saved.success else "",
            "error": saved.error,
            "strategy_id": saved.strategy_id,
            "revision_id": saved.revision_id,
            "py_path": saved.source_path,
            "yaml_path": saved.config_path,
            "status": saved.status,
        }
        if saved.success:
            return StrategySaveSuccess.model_validate(response)
        return StrategySaveFailure.model_validate({**response, "code": saved.code})

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    def validate_strategy(
        strategy_id: str,
        revision_id: str | None = None,
    ) -> (
        StrategyValidationSuccess
        | StrategyValidationFailure
        | StrategyStateError
        | ErrorResponse
    ):
        """Check a draft or validated strategy; repair diagnostics or dry-run next.

        Pass the revision_id returned by save_strategy to bind the evidence.
        The smoke check executes generated Python without a security sandbox.
        """
        response = service.validate(strategy_id, revision_id=revision_id)
        if "error" in response:
            if "status" in response:
                return StrategyStateError.model_validate(response)
            return ErrorResponse.model_validate(response)
        if response["success"]:
            return StrategyValidationSuccess.model_validate(response)
        return StrategyValidationFailure.model_validate(response)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    def dry_run_strategy(
        strategy_id: str,
        revision_id: str | None = None,
    ) -> (
        StrategyDryRunSuccess
        | StrategyDryRunFailure
        | StrategyStateError
        | ErrorResponse
    ):
        """Check a validated strategy on a longer fixture before backtesting.

        Use the revision_id that passed validation.
        This executes generated Python without a security sandbox. Repair any
        diagnostics through save_strategy and validation before trying again.
        """
        response = service.dry_run(strategy_id, revision_id=revision_id)
        if "error" in response:
            if "status" in response:
                return StrategyStateError.model_validate(response)
            return ErrorResponse.model_validate(response)
        if response["success"]:
            return StrategyDryRunSuccess.model_validate(response)
        return StrategyDryRunFailure.model_validate(response)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def promote_strategy(
        strategy_id: str,
        revision_id: str | None = None,
    ) -> StrategyPromoteSuccess | StrategyStateError | ErrorResponse:
        """Promote the selected revision only after dry-run marks it runnable."""
        response = service.promote(strategy_id, revision_id=revision_id)
        if not response["success"]:
            if "status" in response:
                return StrategyStateError.model_validate(response)
            return ErrorResponse.model_validate(response)
        return StrategyPromoteSuccess.model_validate(response)
