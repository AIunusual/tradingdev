"""Read immutable training-search evidence from published optimization runs."""

from typing import TYPE_CHECKING, Literal

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT
from tradingdev.app.contracts.optimization import (
    SelectionEvidencePage,
    SelectionMetric,
    SelectionTrial,
)
from tradingdev.app.pagination import paginate
from tradingdev.domain.optimization.grid_search import finite_metric_value
from tradingdev.domain.optimization.selection import (
    AvailableSelectionEvidence,
    SelectionEvidenceReference,
    UnavailableSelectionEvidence,
)
from tradingdev.domain.performance.artifacts import PerformanceArtifacts
from tradingdev.domain.research_scope import (
    verify_optimization_selection,
    verify_research_scopes,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.optimization.selection import SelectionEvidence
    from tradingdev.domain.performance.artifacts import (
        OptimizationSelection,
        PerformanceBundle,
    )


class PublishedSelectionEvidence:
    """Project one archived search into a ranking summary or a page of values."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self._store = store
        self._manifests = ExecutionManifestStore(workspace)
        self._performance = PerformanceStore(workspace, store)

    def describe(
        self,
        manifest: ExecutionManifest,
        bundle: PerformanceBundle,
        digest: str,
        scope: str,
    ) -> SelectionEvidence | None:
        if manifest.optimization is None:
            if bundle.selected_train_scope is not None:
                return UnavailableSelectionEvidence(
                    reason="No verified optimization manifest for these scopes"
                )
            return None
        try:
            reference, summary = self._resolve(manifest, bundle, digest)
        except PerformanceArtifactError as exc:
            return UnavailableSelectionEvidence(reason=str(exc))
        role: Literal["selected_test", "selected_train", "other_train"] = (
            "selected_test"
            if scope == summary.test_scope
            else "selected_train"
            if scope == summary.selected_train_scope
            else "other_train"
        )
        return AvailableSelectionEvidence(
            scope_role=role, reference=reference, summary=summary
        )

    def page(
        self,
        run_id: str,
        performance_sha256: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> SelectionEvidencePage:
        run = self._store.get_run(run_id)
        if run is None or not run.get("manifest_hash"):
            raise PerformanceArtifactError(
                "selection_evidence_unavailable",
                f"No published manifest-bound search for {run_id}",
            )
        bundle, digest = self._performance.load_with_hash(run_id)
        if digest != performance_sha256:
            raise PerformanceArtifactError(
                "selection_evidence_mismatch",
                "Performance SHA-256 differs from the ranking reference",
            )
        try:
            manifest = self._manifests.load(run_id, expected_hash=run["manifest_hash"])
        except ValueError as exc:
            raise PerformanceArtifactError(
                "selection_evidence_invalid", str(exc)
            ) from exc
        job = self._store.get_job(str(run["job_id"]))
        if job is not None and job.get("status") != "done":
            raise PerformanceArtifactError(
                "selection_evidence_unavailable",
                f"Optimization job status is {job.get('status')}",
            )
        reference, summary = self._resolve(manifest, bundle, digest)
        metric_ids = tuple(dict.fromkeys((summary.objective, "sharpe_ratio")))

        def project(index: int) -> SelectionTrial:
            scope_id = f"trial/{index}/train"
            scope = bundle.scopes[scope_id]
            unavailable = scope.metadata.get("unavailable")
            metrics = {}
            for name in metric_ids:
                recorded = scope.values.get(name)
                value = finite_metric_value(recorded)
                reason = (
                    unavailable.get(name) if isinstance(unavailable, dict) else None
                )
                if value is not None:
                    reason = None
                elif name not in scope.values:
                    reason = "metric_not_recorded"
                elif recorded is not None:
                    reason = "metric_outside_numeric_range"
                elif not isinstance(reason, str) or not reason:
                    reason = "reason_not_recorded"
                metrics[name] = SelectionMetric(value=value, reason=reason)
            settings = scope.metadata.get("settings")
            providers = scope.metadata.get("providers")
            fallbacks = scope.metadata.get("calculation_fallbacks")
            return SelectionTrial(
                trial_index=index,
                scope=scope_id,
                parameters=scope.parameters,
                metrics=metrics,
                calculation_settings=settings if isinstance(settings, dict) else None,
                provider_versions=providers if isinstance(providers, dict) else None,
                calculation_fallbacks={
                    name: fallbacks[name] for name in metric_ids if name in fallbacks
                }
                if isinstance(fallbacks, dict)
                else None,
            )

        context = bundle.scopes[summary.selected_train_scope].metadata[
            "execution_context"
        ]
        assert isinstance(context, dict)
        return SelectionEvidencePage(
            reference=reference,
            summary=summary,
            training_execution_context=context,
            definitions={
                name: bundle.definitions[name]
                for name in metric_ids
                if name in bundle.definitions
            },
            trials=paginate(
                range(summary.completed), project=project, offset=offset, limit=limit
            ),
        )

    def _resolve(
        self, manifest: ExecutionManifest, bundle: PerformanceBundle, digest: str
    ) -> tuple[SelectionEvidenceReference, OptimizationSelection]:
        search = manifest.optimization
        if search is None:
            raise PerformanceArtifactError(
                "selection_evidence_unavailable", "Run has no optimization manifest"
            )
        if bundle.manifest_hash != manifest.manifest_hash:
            raise PerformanceArtifactError(
                "selection_evidence_invalid", "Performance and manifest hashes differ"
            )
        source: Literal["trial_ledger", "verified_historical_bundle"] = "trial_ledger"
        try:
            if bundle.optimization_selection is None:
                source = "verified_historical_bundle"
                observations = self._performance.load_observations(bundle.run_id)
                PerformanceArtifacts(performance=bundle, observations=observations)
                summary = verify_research_scopes(manifest, bundle, observations)
            else:
                summary = verify_optimization_selection(manifest, bundle)
        except (TypeError, ValueError) as exc:
            raise PerformanceArtifactError(
                "selection_evidence_invalid",
                f"Search cannot be verified: {exc}",
            ) from exc
        if summary is None:
            raise PerformanceArtifactError(
                "selection_evidence_unavailable",
                "Historical search is missing recorded trial objectives",
            )
        return SelectionEvidenceReference(
            run_id=bundle.run_id,
            manifest_hash=manifest.manifest_hash,
            performance_sha256=digest,
            source=source,
        ), summary
