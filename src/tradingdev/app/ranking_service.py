"""Rank saved backtest scopes by the metrics their run already recorded.

Ranking reads; it never recomputes. The values come from the performance bundle the
user's own development run saved, the direction and unit come from the metric
catalogue, and the trading conditions come from the execution manifest that was
published with that run. A run whose numbers or conditions cannot be read is
reported as skipped with the reason, never re-run and never zero-filled.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.performance.artifacts import MetricDefinitionSnapshot
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.ranking import Candidate, MetricDefinition, rank_candidates
from tradingdev.domain.run_candidates import (
    RunCandidate,
    build_run_candidate,
    to_ranking_candidate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.adapters.storage.sqlite import SQLiteStore

_RUN_PREFIX = "run."


def resolve_metric(metric: str) -> MetricDefinition:
    """Resolve a source-qualified backtest metric id to its direction and unit.

    Signal metrics are rejected here: they are ranked through the leaderboard, and
    a signal id can never equal a backtest id because both are source-qualified.
    """
    if not metric.startswith(_RUN_PREFIX):
        raise ValueError(
            f"backtest ranking needs a 'run.' metric id, not {metric!r}; "
            "signal metrics are ranked through the leaderboard"
        )
    name = metric[len(_RUN_PREFIX) :]
    definition = METRIC_CATALOG.get(name)
    if definition is None:
        raise ValueError(f"unknown backtest metric: {name}")
    if definition.optimization_direction is None:
        raise ValueError(
            f"metric '{name}' has no optimisation direction and cannot rank candidates"
        )
    return MetricDefinition(
        name=metric,
        direction=definition.optimization_direction,
        unit=definition.unit,
    )


class RankingService:
    """Order saved backtest scopes without recomputing anything."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._runs = RunService(workspace=self._workspace, store=store)

    def rank_run_scopes(
        self,
        metric: str,
        selections: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Rank ``(run_id, scope)`` pairs by one stored backtest metric.

        Each pair is one candidate, so several parameter sets of a single run can be
        compared side by side. A pair whose run, bundle, manifest or trading
        conditions cannot be read is skipped with that reason; the rest are ranked.
        """
        definition = resolve_metric(metric)
        candidates = []
        skipped: list[dict[str, str]] = []
        for run_id, scope in selections:
            try:
                candidate = self._load_candidate(run_id, scope, metric)
            except ValueError as exc:
                skipped.append(
                    {"candidate_id": f"run:{run_id}#{scope}", "reason": str(exc)}
                )
                continue
            candidates.append(candidate)

        result = rank_candidates(candidates, definition, top_k=top_k)
        return {
            "success": True,
            "metric": result.metric,
            "direction": result.direction,
            "unit": definition.unit,
            "ranked_total": result.ranked_total,
            "ranked": [
                {
                    "rank": item.rank,
                    "tied": item.tied,
                    "value": item.value,
                    "run_id": item.candidate.payload.run_id,
                    "scope": item.candidate.payload.scope,
                    "strategy_id": item.candidate.payload.strategy_id,
                    "revision_id": item.candidate.payload.revision_id,
                    "parameters": dict(item.candidate.payload.parameters),
                    "metadata": dict(item.candidate.payload.metadata),
                    "conditions": {
                        "symbol": item.candidate.payload.conditions.symbol,
                        "timeframe": item.candidate.payload.conditions.timeframe,
                        "start_date": item.candidate.payload.conditions.start_date,
                        "end_date": item.candidate.payload.conditions.end_date,
                        "fees": item.candidate.payload.conditions.fees,
                        "slippage": item.candidate.payload.conditions.slippage,
                    },
                    "metric_source": item.candidate.payload.metric_source,
                }
                for item in result.ranked
            ],
            "skipped": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "run_id": item.candidate.payload.run_id,
                    "scope": item.candidate.payload.scope,
                    "reason": item.reason,
                }
                for item in result.skipped
            ]
            + skipped,
            "caveats": [
                *result.caveats,
                "Ranking is conditional on each scope's saved metadata, including "
                "provider versions, settings and observation periods; it does not "
                "establish a controlled comparison. Conditions describe the run's "
                "submitted period, not each scope's observed period.",
            ],
        }

    def _load_candidate(
        self, run_id: str, scope: str, metric: str
    ) -> Candidate[RunCandidate]:
        """Build one candidate from stored values; raise ValueError with the reason."""
        run_payload = self._runs.get_run(run_id)
        if run_payload.get("success") is False:
            raise ValueError(f"unknown run: {run_id}")
        run = run_payload["run"]
        metrics_payload = self._runs.get_run_metrics(run_id, scope=scope)
        if metrics_payload.get("success") is False:
            raise ValueError(
                f"{run_id} {scope}: "
                f"{metrics_payload.get('error') or metrics_payload.get('code')}"
            )
        if metrics_payload["kind"] == "fold_summary":
            raise ValueError(
                f"{run_id} {scope}: fold summary cannot be ranked; select a fold scope"
            )
        name = metric[len(_RUN_PREFIX) :]
        recorded = metrics_payload["definitions"].get(name)
        if recorded is not None:
            current = MetricDefinitionSnapshot.model_validate(
                asdict(METRIC_CATALOG[name])
            ).model_dump()
            differences = [
                key for key, value in current.items() if recorded.get(key) != value
            ]
            if differences:
                raise ValueError(
                    f"{run_id} {scope}: recorded metric definition differs from "
                    f"the current catalog: {', '.join(differences)}"
                )
        manifest_hash = run.get("manifest_hash")
        if not manifest_hash:
            raise ValueError(f"{run_id}: no execution manifest recorded")
        try:
            manifest = ExecutionManifestStore(self._workspace).load(
                run_id, expected_hash=manifest_hash
            )
        except ManifestError as exc:
            raise ValueError(f"{run_id}: {exc}") from exc
        backtest_section = manifest.config.get("backtest")
        strategy_section = manifest.config.get("strategy")
        if not isinstance(backtest_section, dict) or not isinstance(
            strategy_section, dict
        ):
            raise ValueError(
                f"{run_id}: execution manifest lacks backtest or strategy settings"
            )
        try:
            backtest = BacktestConfig.model_validate(backtest_section)
        except ValueError as exc:
            raise ValueError(
                f"{run_id}: execution manifest has no usable backtest conditions"
            ) from exc
        candidate = build_run_candidate(
            run_id=run_id,
            scope=scope,
            strategy_id=str(run["strategy_id"]),
            revision_id=str(run.get("revision_id") or ""),
            parameters=metrics_payload["parameters"],
            metadata=metrics_payload["metadata"],
            symbol=backtest.symbol,
            timeframe=backtest.timeframe,
            start_date=backtest.start_date.isoformat(),
            end_date=backtest.end_date.isoformat(),
            fees=backtest.fees,
            slippage=backtest.slippage,
            metric_source="bundle",
        )
        return to_ranking_candidate(candidate, metrics_payload["metrics"])
