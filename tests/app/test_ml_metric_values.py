"""Verify the return metrics the service reports, not merely that it reports them.

The defects these tests target were not wrong formulas but a right formula fed
the wrong series, so every expectation below is rebuilt from the dataset's own
forward returns and from the documented meaning of a monotone signal. Nothing
here calls ``return_metrics`` or ``simulate_strategy_returns``: a test that
derived its expected book from the simulation under test would pass just as
happily against a mis-sliced book.

The path under test is the real one: bars on disk, a saved dataset, the saved
manifest that decides annualisation, then the leaderboard entry a report reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.registry import LeaderboardStore

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_MODEL = "c" * 16
# A continuous crypto year of hourly bars. Written out rather than imported so
# that changing the policy constant has to be a deliberate decision here too.
_HOURLY_PERIODS_PER_YEAR = 365.0 * 24.0


def _write_bars(paths: WorkspacePaths) -> None:
    index = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(_ROWS, 10.0),
        }
    )
    seed_market_cache(paths.processed_data, frame)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _dataset(service: MLService, *, horizon: int) -> str:
    built = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=horizon,
    )
    dataset_id: str = built["dataset"]["dataset_id"]
    return dataset_id


def _forward(workspace: WorkspacePaths, dataset_id: str) -> np.ndarray:
    """The saved labels themselves, not anything the service computed."""
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    return frame["forward_return"].to_numpy(dtype=float)


def _monotone(size: int) -> list[float]:
    """A strictly increasing signal, which is always its own historical maximum."""
    return [float(index) for index in range(size)]


# A position needs two prior observations, so with ``n_quantiles=2`` the first
# two bars are warm-up and no trade opens before bar index 2. Every test below
# passes ``n_quantiles=2``; changing that changes this constant too.
_WARM_UP_BARS = 2


def _book(forward: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the traded book and its benchmark on the holding grid.

    A strictly increasing signal is always the maximum of its own history, so
    once warm-up is over it exceeds the causal upper band and the book is long
    (+1) for every bar after it. Only bars ``0, horizon, 2*horizon, ...`` open
    trades, so a horizon-bar holding interval is measured on every horizon-th
    bar.
    """
    grid = np.arange(0, forward.size, horizon)
    book = np.where(grid >= _WARM_UP_BARS, forward[grid], 0.0)
    return book, forward[grid]


def _independent_metrics(forward: np.ndarray, horizon: int) -> dict[str, float]:
    """Measure the rebuilt book with primitives, not with the code under test."""
    book, market = _book(forward, horizon)
    periods = _HOURLY_PERIODS_PER_YEAR / horizon
    count = book.size

    equity = np.concatenate(([1.0], np.cumprod(1.0 + book)))
    peak = np.maximum.accumulate(equity)
    total = float(np.prod(1.0 + book) - 1.0)
    # Sample (n-1) deviations: these are observed intervals, not a population.
    spread = float(np.std(book, ddof=1))
    beta = float(np.cov(book, market, ddof=1)[0, 1] / np.var(market, ddof=1))

    return {
        "annual_return": (1.0 + total) ** (periods / count) - 1.0,
        "max_drawdown": float(np.max(1.0 - equity / peak)),
        "sharpe": float(np.mean(book)) / spread * float(np.sqrt(periods)),
        "alpha": (float(np.mean(book)) - beta * float(np.mean(market))) * periods,
        "beta": beta,
        "periods_per_year": periods,
        "n_return_periods": count,
    }


class TestReportedValues:
    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_metrics_match_an_independently_rebuilt_book(
        self, service: MLService, workspace: WorkspacePaths, horizon: int
    ) -> None:
        dataset_id = _dataset(service, horizon=horizon)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size),
            forward.tolist(),
            window=20,
            n_quantiles=2,
            dataset_id=dataset_id,
        )

        expected = _independent_metrics(forward, horizon)
        for key, value in expected.items():
            assert result["metrics"][key] == pytest.approx(value, rel=1e-12), key

    def test_a_two_bar_horizon_is_not_measured_on_every_bar(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """The shape a dense book would have is the defect this pins down.

        A one-bar horizon cannot tell the two apart, because the holding grid
        is every bar; only a multi-bar horizon makes the off-grid zero rows a
        distinct, wrong series.
        """
        dataset_id = _dataset(service, horizon=2)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size),
            forward.tolist(),
            window=20,
            n_quantiles=2,
            dataset_id=dataset_id,
        )

        on_grid = _independent_metrics(forward, 2)
        # What a book indexed by every bar would have measured: the holding
        # intervals plus a placeholder zero for each bar that opened no trade.
        dense = np.array(
            [
                forward[index] if index % 2 == 0 and index >= 2 else 0.0
                for index in range(forward.size)
            ]
        )
        dense_periods = _HOURLY_PERIODS_PER_YEAR / 2.0
        dense_sharpe = (
            float(np.mean(dense))
            / float(np.std(dense, ddof=1))
            * float(np.sqrt(dense_periods))
        )

        assert result["metrics"]["sharpe"] == pytest.approx(
            on_grid["sharpe"], rel=1e-12
        )
        # Guard the guard: if both shapes agreed, this test would prove nothing.
        assert abs(dense_sharpe - on_grid["sharpe"]) > 0.1
        assert result["metrics"]["sharpe"] != pytest.approx(dense_sharpe, rel=1e-6)
        assert result["metrics"]["n_return_periods"] == on_grid["n_return_periods"]

    @pytest.mark.parametrize(
        ("horizon", "expected"),
        [(1, 8760.0), (2, 4380.0), (3, 2920.0)],
    )
    def test_annualisation_divides_a_year_by_the_holding_horizon(
        self,
        service: MLService,
        workspace: WorkspacePaths,
        horizon: int,
        expected: float,
    ) -> None:
        dataset_id = _dataset(service, horizon=horizon)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size),
            forward.tolist(),
            window=20,
            n_quantiles=2,
            dataset_id=dataset_id,
        )

        assert result["metrics"]["periods_per_year"] == pytest.approx(expected)


class TestCalendarDependence:
    def test_without_a_dataset_only_calendar_free_metrics_are_measured(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """``None`` means "no calendar", not "nothing was measured"."""
        dataset_id = _dataset(service, horizon=1)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size), forward.tolist(), window=20, n_quantiles=2
        )
        metrics = result["metrics"]

        for key in ("annual_return", "sharpe", "alpha", "periods_per_year"):
            assert metrics[key] is None, key

        # Beta and drawdown need no calendar, so they must still be numbers.
        expected = _independent_metrics(forward, 1)
        assert metrics["beta"] == pytest.approx(expected["beta"], rel=1e-12)
        assert metrics["max_drawdown"] == pytest.approx(
            expected["max_drawdown"], rel=1e-12
        )


class TestHandComputedBooks:
    def test_drawdown_compounds_the_book_instead_of_summing_losses(
        self, service: MLService
    ) -> None:
        """A 50% gain then two 20% losses loses 36% of peak, not the 40% sum."""
        forward = [0.01, 0.01, 0.50, -0.20, -0.20, 0.00, 0.00, 0.00]

        result = service.evaluate_signal(
            _monotone(len(forward)), forward, window=2, n_quantiles=2
        )

        # Equity 1 -> 1.5 -> 1.2 -> 0.96 against a peak of 1.5.
        assert result["metrics"]["max_drawdown"] == pytest.approx(0.36)

    def test_a_book_that_holds_the_asset_has_unit_beta(
        self, service: MLService
    ) -> None:
        """Beta is a regression slope, so holding the asset outright is exactly 1."""
        forward = [0.0, 0.0, 0.10, -0.05, 0.02, -0.03, 0.04, 0.01]

        result = service.evaluate_signal(
            _monotone(len(forward)), forward, window=2, n_quantiles=2
        )

        assert result["metrics"]["beta"] == pytest.approx(1.0)


class TestSimulatedSummary:
    def test_the_summary_means_keep_inactive_rows_in_the_denominator(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """The mean averages over decision rows, so a cash bar counts as zero.

        The cumulative figure compounds only the holding intervals, so the two
        summaries deliberately disagree on what a bar is: pinning both keeps a
        future "fix" from silently redefining one into the other.
        """
        dataset_id = _dataset(service, horizon=2)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size),
            forward.tolist(),
            window=20,
            n_quantiles=2,
            dataset_id=dataset_id,
        )

        book, _ = _book(forward, 2)
        every_bar = np.array(
            [
                forward[index] if index % 2 == 0 and index >= 2 else 0.0
                for index in range(forward.size)
            ]
        )
        summary = result["simulated"]
        assert summary["long_short_mean"] == pytest.approx(
            float(np.mean(every_bar)), rel=1e-12
        )
        assert summary["long_short_cumulative"] == pytest.approx(
            float(np.prod(1.0 + book) - 1.0), rel=1e-12
        )


class TestRecordedValues:
    def test_the_stored_entry_and_the_ranking_carry_the_same_numbers(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A report reads the stored entry, so it must hold what was reported."""
        dataset_id = _dataset(service, horizon=2)
        forward = _forward(workspace, dataset_id)

        result = service.evaluate_signal(
            _monotone(forward.size),
            forward.tolist(),
            window=20,
            n_quantiles=2,
            record=True,
            model_id=_MODEL,
            dataset_id=dataset_id,
        )

        expected = _independent_metrics(forward, 2)
        entry = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        ranked = service.leaderboard("annual_return", dataset_id=dataset_id)["ranked"][
            0
        ]

        for key in ("annual_return", "max_drawdown", "sharpe", "alpha", "beta"):
            assert result["metrics"][key] == pytest.approx(expected[key], rel=1e-12)
            assert entry.metrics[key] == pytest.approx(expected[key], rel=1e-12)
        assert ranked["value"] == pytest.approx(expected["annual_return"], rel=1e-12)
