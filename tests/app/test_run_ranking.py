"""Ranking saved run scopes: metric resolution and refusal paths.

The end-to-end path needs a published execution manifest; these tests cover the
parts that do not, so a regression in the resolver or in the skip reasons is caught
without building a whole run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ranking_service import RankingService, resolve_metric

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def service(tmp_path: Path) -> RankingService:
    workspace = WorkspacePaths(tmp_path / "workspace")
    return RankingService(workspace=workspace, store=SQLiteStore(workspace))


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
