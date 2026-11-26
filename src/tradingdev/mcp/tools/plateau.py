"""MCP analysis of saved parameter collections, without execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp_types import ToolAnnotations
from pydantic import Field

from tradingdev.app.contracts.common import PageLimit, PageOffset  # noqa: TC001
from tradingdev.app.contracts.plateau import (
    PlateauAxes,
    PlateauEntries,
    PlateauError,
    PlateauResponse,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.plateau_service import PlateauService


def register(mcp: MCPServer, service: PlateauService) -> None:
    """Expose one creation and one immutable continuation entry."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def analyze_parameter_plateau(
        metric: str,
        axes: PlateauAxes,
        run_id: str | None = None,
        entry_ids: PlateauEntries | None = None,
        radius: Annotated[int, Field(ge=1, strict=True)] = 1,
        min_neighborhood: Annotated[int | None, Field(ge=1, strict=True)] = None,
        limit: PageLimit = 20,
    ) -> PlateauResponse | PlateauError:
        """Analyze saved parameter stability and save a fixed collection.

        Select run_id + run.<metric> (all trial/*/train only), or explicit entry_ids
        + signal.<metric>. Parameter axes are JSON pointers (/window, /threshold).
        Bare horizon/window/n_quantiles are signal-evaluation settings only.
        Include every numeric level, even undefined.
        Run axes must match the saved search; signal axes describe the selection.
        Fixed conditions must match; cross-horizon verifies raw inputs/evaluation
        rows. Duplicate coordinates, categories and incomplete evidence are refused.
        Radius uses grid indices; min_neighborhood limits plateau-best eligibility.
        No new experiments or sample filter. Values retain units; minimizing negates
        score only. Stability is no out-of-sample/profit proof. Response limitations
        define scores; get_parameter_plateau continues analysis_id. Deliver artifact
        via get_artifact_download or verified host file delivery.
        """
        payload = service.analyze(
            metric=metric,
            axes=axes,
            run_id=run_id,
            entry_ids=entry_ids,
            radius=radius,
            min_neighborhood=min_neighborhood,
            limit=limit,
        )
        return (PlateauResponse if payload["success"] else PlateauError).model_validate(
            payload
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_parameter_plateau(
        analysis_id: str,
        offset: PageOffset = 0,
        limit: PageLimit = 20,
    ) -> PlateauResponse | PlateauError:
        """Read a hash-verified page of one fixed plateau collection after creation.

        Follow cells.next_offset; total includes undefined/absent cells. Summary,
        source_hash and artifact stay fixed across pages, restarts and updates.
        Missing/tampered artifacts fail. Artifact retains complete source evidence.
        """
        payload = service.read(analysis_id, offset=offset, limit=limit)
        return (PlateauResponse if payload["success"] else PlateauError).model_validate(
            payload
        )
