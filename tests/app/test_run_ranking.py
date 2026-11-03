"""Ranking saved run scopes: metric resolution, refusals, and published runs.

Two layers are covered here. The resolver and the skip reasons are unit-level, so a
regression is caught without building a whole run. The ranking itself is exercised
end to end through real workspace storage: a published execution manifest, the run
row that records its hash, and the performance bundle the value comes from.

The manifest is the part that makes these tests integration tests. A candidate's
numbers are only interpretable beside the conditions they were measured under, and
those conditions are read back from the published manifest rather than restated by
the test.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ranking_service import RankingService, resolve_metric
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    MetricDefinitionSnapshot,
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    PerformanceScope,
    ScopeObservations,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path

SHARPE = "sharpe_ratio"
DRAWDOWN = "max_drawdown"


@pytest.fixture
def storage(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    return workspace, SQLiteStore(workspace)


@pytest.fixture
def service(storage: tuple[WorkspacePaths, SQLiteStore]) -> RankingService:
    workspace, store = storage
    return RankingService(workspace=workspace, store=store)


def _manifest(
    *,
    window: int = 20,
    symbol: str = "BTC/USDT",
    fees: float = 0.0006,
    slippage: float = 0.0005,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
) -> ExecutionManifest:
    """The smallest config ``_resolve_config`` accepts, with nothing left implicit.

    ``strategy`` and ``backtest`` are the only required sections. Inside
    ``backtest`` five keys are required: symbol, timeframe, start_date, end_date,
    and init_cash -- the last only because the default mode is ``signal``. Every
    other field resolves from a default, and unknown keys are refused.
    """
    return ExecutionManifest.create(
        kind="backtest",
        config={
            "strategy": {
                "id": "fixture",
                "revision_id": "rev-1",
                "parameters": {"window": window},
            },
            "backtest": {
                "symbol": symbol,
                "timeframe": "1d",
                "start_date": start_date,
                "end_date": end_date,
                "init_cash": 1000.0,
                "fees": fees,
                "slippage": slippage,
            },
        },
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"window": window}
        ),
    )


def _scope(values: dict[str, float | None]) -> PerformanceScope:
    return PerformanceScope.model_validate(
        {
            "mode": "signal",
            "values": values,
            "metadata": {
                "execution_context": {
                    "symbol": "BTC/USDT",
                    "init_cash": 1000.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-01",
                },
            },
        }
    )


def _observations() -> ScopeObservations:
    return ScopeObservations(
        init_cash=1000.0,
        equity_curve=[1000.0, 1100.0],
        returns=[0.0, 0.1],
        timestamps=["2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z"],
        trades=[],
    )


def _publish_run(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    run_id: str,
    *,
    scopes: dict[str, dict[str, float | None] | PerformanceScope] | None = None,
    window: int = 20,
    publish_manifest: bool = True,
    record_manifest_hash: bool = True,
    manifest: ExecutionManifest | None = None,
) -> ExecutionManifest:
    """Publish one run the way a completed execution would leave it.

    The manifest hash is the same value in three places -- the run row, the
    published manifest, and the performance bundle -- because
    ``PerformanceStore.publish`` refuses a bundle whose identity differs from its
    run. Writing the hash in only one place produces a fixture that cannot exist.

    A scope may be given as plain values or as a built ``PerformanceScope``, which
    is how a fold summary (``kind: "fold_summary"``) is published.
    """
    scopes = scopes or {"full": {SHARPE: 1.0}}
    manifest = manifest or _manifest(window=window)
    recorded_hash = manifest.manifest_hash if record_manifest_hash else None
    scope_models = {
        name: values if isinstance(values, PerformanceScope) else _scope(values)
        for name, values in scopes.items()
    }
    default_scope = next(iter(scope_models))
    if publish_manifest:
        ExecutionManifestStore(workspace).publish(run_id, manifest)
    store.create_run(
        run_id=run_id,
        job_id=f"job-{run_id}",
        strategy_id="fixture",
        revision_id="rev-1",
        artifact_dir=workspace.runs / run_id,
        metrics=scope_models[default_scope].model_dump(mode="json")["values"],
        manifest_hash=recorded_hash,
    )
    names = {name for scope in scope_models.values() for name in scope.values}
    PerformanceStore(workspace, store).publish(
        PerformanceArtifacts(
            performance=PerformanceBundle(
                run_id=run_id,
                manifest_hash=recorded_hash,
                default_scope=default_scope,
                scopes=scope_models,
                definitions={
                    name: MetricDefinitionSnapshot.model_validate(
                        asdict(METRIC_CATALOG[name])
                    )
                    for name in names
                },
            ),
            observations=ObservationsBundle(
                run_id=run_id,
                manifest_hash=recorded_hash,
                # Only scalar backtest scopes carry observations; a fold summary
                # is an aggregate and has none.
                scopes={
                    name: _observations()
                    for name, scope in scope_models.items()
                    if scope.kind == "backtest"
                },
            ),
        )
    )
    return manifest


class TestMetricResolution:
    def test_a_backtest_metric_keeps_its_catalogue_direction(self) -> None:
        definition = resolve_metric("run.sharpe_ratio")
        assert definition.name == "run.sharpe_ratio"
        assert definition.direction == "maximize"
        assert definition.unit == "ratio"

    def test_a_minimized_metric_stays_minimized(self) -> None:
        assert resolve_metric("run.max_drawdown").direction == "minimize"

    @pytest.mark.parametrize(
        "metric", ["signal.ic_mean", "signal.annual_return", "sharpe_ratio"]
    )
    def test_a_signal_or_unqualified_id_is_refused(self, metric: str) -> None:
        """Signal figures are ranked through the leaderboard, not here."""
        with pytest.raises(ValueError, match="needs a 'run.' metric id"):
            resolve_metric(metric)

    def test_an_unknown_backtest_metric_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown backtest metric"):
            resolve_metric("run.not_a_metric")

    def test_a_metric_without_a_direction_cannot_rank(self) -> None:
        with pytest.raises(ValueError, match="no optimisation direction"):
            resolve_metric("run.open_trades")


class TestRefusals:
    def test_an_unknown_run_is_skipped_with_its_reason(
        self, service: RankingService
    ) -> None:
        result = service.rank_run_scopes("run.sharpe_ratio", [("missing", "full")])
        assert result["success"] is True
        assert result["ranked"] == []
        assert result["skipped"] == [
            {"candidate_id": "run:missing#full", "reason": "unknown run: missing"}
        ]

    def test_a_run_without_stored_metrics_is_skipped_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        """Nothing is recomputed; the candidate is refused with the stored error."""
        workspace = WorkspacePaths(tmp_path / "workspace")
        store = SQLiteStore(workspace)
        store.create_run(
            run_id="legacy-run",
            job_id="job-1",
            strategy_id="kd",
            artifact_dir=workspace.runs / "legacy-run",
            metrics={"sharpe_ratio": 1.0},
        )
        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("legacy-run", "full")]
        )
        assert result["ranked"] == []
        assert result["skipped"][0]["candidate_id"] == "run:legacy-run#full"
        # The stored error text is owned by the run service; pin the prefix only.
        assert result["skipped"][0]["reason"].startswith("legacy-run full: ")

    def test_an_empty_selection_returns_an_empty_ranking(
        self, service: RankingService
    ) -> None:
        result = service.rank_run_scopes("run.sharpe_ratio", [])
        assert result["success"] is True
        assert result["ranked"] == [] and result["skipped"] == []
        assert result["ranked_total"] == 0


class TestPublishedRuns:
    """The end-to-end path: stored value plus the conditions from its manifest."""

    def test_a_published_manifest_supplies_the_conditions_beside_the_value(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {SHARPE: 1.25}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("alpha", "full")]
        )

        assert result["success"] is True
        assert result["direction"] == "maximize"
        assert result["unit"] == "ratio"
        assert result["ranked_total"] == 1
        (row,) = result["ranked"]
        assert (row["rank"], row["run_id"], row["scope"], row["value"]) == (
            1,
            "alpha",
            "full",
            1.25,
        )
        # The value came from the bundle, not from a legacy summary row.
        assert row["metric_source"] == "bundle"
        assert row["parameters"] == {"window": 20}
        assert row["conditions"] == {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            # Calendar dates are resolved to midnight timestamps by the manifest.
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-03-01T00:00:00",
            "fees": 0.0006,
            "slippage": 0.0005,
        }
        assert any("only one candidate" in caveat for caveat in result["caveats"])

    def test_runs_are_ordered_by_the_value_their_bundle_recorded(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        for run_id, sharpe in (("alpha", 0.5), ("beta", 1.5), ("gamma", -0.25)):
            _publish_run(workspace, store, run_id, scopes={"full": {SHARPE: sharpe}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio",
            [("alpha", "full"), ("beta", "full"), ("gamma", "full")],
        )

        assert [
            (row["rank"], row["run_id"], row["value"]) for row in result["ranked"]
        ] == [(1, "beta", 1.5), (2, "alpha", 0.5), (3, "gamma", -0.25)]
        assert result["skipped"] == []
        assert result["ranked_total"] == 3

    def test_a_minimized_metric_ranks_the_smallest_drawdown_first(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {DRAWDOWN: 0.3}})
        _publish_run(workspace, store, "beta", scopes={"full": {DRAWDOWN: 0.1}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.max_drawdown", [("alpha", "full"), ("beta", "full")]
        )

        assert result["direction"] == "minimize"
        assert [(row["run_id"], row["value"]) for row in result["ranked"]] == [
            ("beta", 0.1),
            ("alpha", 0.3),
        ]

    def test_equal_values_share_a_rank_in_candidate_id_order(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        """The tie-break follows the candidate id, not the order of the request."""
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {SHARPE: 1.0}})
        _publish_run(workspace, store, "beta", scopes={"full": {SHARPE: 1.0}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("beta", "full"), ("alpha", "full")]
        )

        assert [
            (row["rank"], row["tied"], row["run_id"]) for row in result["ranked"]
        ] == [(1, 2, "alpha"), (1, 2, "beta")]

    def test_scopes_of_one_run_are_compared_side_by_side(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(
            workspace,
            store,
            "grid",
            scopes={"window-10": {SHARPE: 0.4}, "window-20": {SHARPE: 0.9}},
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("grid", "window-10"), ("grid", "window-20")]
        )

        assert [(row["scope"], row["value"]) for row in result["ranked"]] == [
            ("window-20", 0.9),
            ("window-10", 0.4),
        ]
        assert result["ranked_total"] == 2
        # Identity is run-level: both scopes report the manifest's submitted
        # parameters, not the parameters each scope was executed with. The scope
        # names here differ from the manifest on purpose, so a future change that
        # starts overlaying trial parameters would fail this assertion.
        assert {row["parameters"]["window"] for row in result["ranked"]} == {20}
        assert len({row["conditions"]["symbol"] for row in result["ranked"]}) == 1

    def test_near_equal_values_share_a_rank_but_keep_numeric_order(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        """A shared rank is not a promise of candidate-id order.

        ``rank_candidates`` sorts by value and uses ``math.isclose`` only to
        assign ranks, so two values inside the tolerance share a rank while the
        larger one still comes first. The ids are assigned so that numeric order
        and id order disagree: a tie-break by id would put ``alpha`` first.
        """
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {SHARPE: 1.0}})
        _publish_run(
            workspace, store, "beta", scopes={"full": {SHARPE: 1.000000000001}}
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("alpha", "full"), ("beta", "full")]
        )

        assert [
            (row["run_id"], row["rank"], row["tied"]) for row in result["ranked"]
        ] == [
            ("beta", 1, 2),
            ("alpha", 1, 2),
        ]

    def test_a_fold_summary_scope_is_skipped_instead_of_raising(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        """A walk-forward run defaults to an aggregate, which is not scalar.

        Before this refusal, ``float()`` raised ``TypeError`` inside the ranking
        core, outside the ``ValueError`` handler, so one aggregate scope aborted
        the whole call instead of producing a skipped row.
        """
        workspace, store = storage
        _publish_run(
            workspace,
            store,
            "wf",
            scopes={
                "test_summary": PerformanceScope.model_validate(
                    {
                        "kind": "fold_summary",
                        "mode": "signal",
                        "split": "test",
                        "values": {
                            SHARPE: FoldStats(
                                mean=1.2, std=0.3, min=0.9, max=1.5, valid_count=3
                            )
                        },
                        "metadata": {},
                    }
                )
            },
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("wf", "test_summary")]
        )

        assert result["success"] is True
        assert result["ranked"] == []
        assert result["skipped"] == [
            {
                "candidate_id": "run:wf#test_summary",
                "reason": (
                    "wf test_summary: fold summary cannot be ranked; "
                    "select a fold scope"
                ),
            }
        ]

    def test_top_k_limits_the_page_but_reports_the_full_total(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        for run_id, sharpe in (("alpha", 0.1), ("beta", 0.2), ("gamma", 0.3)):
            _publish_run(workspace, store, run_id, scopes={"full": {SHARPE: sharpe}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio",
            [("alpha", "full"), ("beta", "full"), ("gamma", "full")],
            top_k=2,
        )

        assert [row["run_id"] for row in result["ranked"]] == ["gamma", "beta"]
        assert result["ranked_total"] == 3


class TestManifestRefusals:
    """A missing or altered manifest is a skip with a reason, never a default."""

    def test_a_run_without_a_recorded_manifest_is_skipped(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(
            workspace,
            store,
            "legacy",
            scopes={"full": {SHARPE: 2.0}},
            publish_manifest=False,
            record_manifest_hash=False,
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("legacy", "full")]
        )

        assert result["ranked"] == []
        assert result["skipped"] == [
            {
                "candidate_id": "run:legacy#full",
                "reason": "legacy: no execution manifest recorded",
            }
        ]

    def test_a_recorded_manifest_that_was_never_published_is_skipped(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(
            workspace,
            store,
            "ghost",
            scopes={"full": {SHARPE: 2.0}},
            publish_manifest=False,
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("ghost", "full")]
        )

        assert result["ranked"] == []
        assert result["skipped"][0]["reason"].startswith(
            "ghost: Cannot read valid execution manifest"
        )

    def test_a_manifest_whose_bytes_were_replaced_is_refused(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        """The recorded hash is what makes the stored conditions trustworthy."""
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {SHARPE: 1.0}})
        replaced = _manifest(window=99)
        ExecutionManifestStore(workspace).path("alpha").write_bytes(
            replaced.model_dump_json().encode("utf-8")
        )

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("alpha", "full")]
        )

        assert result["ranked"] == []
        assert result["skipped"][0]["reason"] == (
            "alpha: Execution manifest does not match the expected hash"
        )

    def test_a_bundle_without_the_metric_is_skipped_rather_than_zero_filled(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {DRAWDOWN: 0.2}})

        result = RankingService(workspace=workspace, store=store).rank_run_scopes(
            "run.sharpe_ratio", [("alpha", "full")]
        )

        assert result["ranked"] == []
        assert result["skipped"][0]["reason"] == "no 'run.sharpe_ratio' recorded"


class TestCandidateDispatch:
    """One ranking surface, routed by the metric id's source prefix."""

    def test_a_metric_without_a_source_prefix_is_refused(
        self, service: RankingService
    ) -> None:
        with pytest.raises(ValueError, match="needs a 'signal.' or 'run.' prefix"):
            service.rank_candidates("sharpe_ratio", runs=[("alpha", "full")])

    def test_run_selections_are_refused_on_the_signal_path(
        self, service: RankingService
    ) -> None:
        """A half-filled table would look like a ranking of two things."""
        with pytest.raises(ValueError, match="not run selections"):
            service.rank_candidates("signal.ic_mean", runs=[("alpha", "full")])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dataset_id", "d1"),
            ("symbol", "BTC/USDT"),
            ("timeframe", "1d"),
            ("horizon", 1),
            ("feature_set", "core"),
            ("family", "rule"),
        ],
    )
    def test_signal_filters_are_refused_on_the_backtest_path(
        self, service: RankingService, field: str, value: Any
    ) -> None:
        with pytest.raises(ValueError, match="not signal filters"):
            service.rank_candidates(
                "run.sharpe_ratio", runs=[("alpha", "full")], **{field: value}
            )

    def test_the_backtest_path_requires_a_selection(
        self, service: RankingService
    ) -> None:
        with pytest.raises(ValueError, match="at least one run selection"):
            service.rank_candidates("run.sharpe_ratio")

    def test_the_backtest_path_returns_the_shared_shape(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        workspace, store = storage
        _publish_run(workspace, store, "alpha", scopes={"full": {SHARPE: 1.0}})
        _publish_run(workspace, store, "beta", scopes={"full": {SHARPE: 2.0}})

        result = RankingService(workspace=workspace, store=store).rank_candidates(
            "run.sharpe_ratio", runs=[("alpha", "full"), ("beta", "full")]
        )

        assert result["source"] == "run"
        assert result["metric"] == "run.sharpe_ratio"
        assert result["n_ranked"] == 2
        assert [row["run_id"] for row in result["ranked"]] == ["beta", "alpha"]
        assert {row["source"] for row in result["ranked"]} == {"run"}

    def test_the_backtest_path_pages_like_the_signal_path(
        self, storage: tuple[WorkspacePaths, SQLiteStore]
    ) -> None:
        """One shape serves both sources, so a run selection pages too."""
        workspace, store = storage
        for index, value in enumerate((1.0, 2.0, 3.0)):
            _publish_run(
                workspace, store, f"run{index}", scopes={"full": {SHARPE: value}}
            )
        service = RankingService(workspace=workspace, store=store)
        selections = [(f"run{index}", "full") for index in range(3)]

        first = service.rank_candidates("run.sharpe_ratio", runs=selections, limit=2)
        assert [row["run_id"] for row in first["ranked"]] == ["run2", "run1"]
        assert first["n_ranked"] == 3
        assert first["next_offset"] == 2

        second = service.rank_candidates(
            "run.sharpe_ratio", runs=selections, offset=2, limit=2
        )
        assert [row["run_id"] for row in second["ranked"]] == ["run0"]
        assert second["next_offset"] is None
