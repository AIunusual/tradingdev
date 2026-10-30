"""Rank real dataset evaluations using forward returns and their negation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import cached_market_path, seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.evaluation import simulate_strategy_returns
from tradingdev.domain.ml.leaderboard import build_entry_id, signal_fingerprint
from tradingdev.domain.ml.registry import LeaderboardStore

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_GOOD_MODEL = "a" * 16
_BAD_MODEL = "b" * 16


def _write_bars(workspace: WorkspacePaths, symbol: str) -> None:
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
    seed_market_cache(workspace.processed_data, frame, symbol=symbol)
    seed_market_cache(
        workspace.processed_data, frame, symbol=symbol, source="yahoo_finance"
    )


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths, "BTC/USDT")
    _write_bars(paths, "ETH/USDT")
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService, symbol: str = "BTC/USDT") -> str:
    result = service.build_dataset(
        symbol=symbol,
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
    )
    dataset_id: str = result["dataset"]["dataset_id"]
    return dataset_id


def _returns(workspace: WorkspacePaths, dataset_id: str) -> list[float]:
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    return frame["forward_return"].astype(float).tolist()


def _record(
    service: MLService,
    workspace: WorkspacePaths,
    *,
    dataset_id: str,
    model_id: str,
    quality: float,
) -> dict[str, Any]:
    """Record one evaluation of a signal scaled towards or against the truth."""
    realized = _returns(workspace, dataset_id)
    signal = [value * quality for value in realized]
    return service.evaluate_signal(
        signal,
        realized,
        horizon=1,
        window=20,
        n_quantiles=5,
        record=True,
        model_id=model_id,
        dataset_id=dataset_id,
    )


class TestRecording:
    def test_recording_requires_a_dataset(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        with pytest.raises(ValueError, match="requires dataset_id"):
            service.evaluate_signal(realized, realized, record=True)

    def test_recorded_response_reports_the_entry(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        result = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        recorded = result["recorded"]
        assert recorded["replaced"] is False
        assert recorded["n_candidates"] == 1
        assert "BTC/USDT" in recorded["context"]
        assert len(recorded["entry_id"]) == 16

    def test_re_recording_replaces_instead_of_inflating(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        again = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        assert again["recorded"]["replaced"] is True
        assert again["recorded"]["n_candidates"] == 1

    def test_no_non_finite_metric_reaches_storage(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A failed score is stored as None; NaN/Inf itself is never written."""
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 1
        assert entries[0].metrics
        assert all(
            value is None or np.isfinite(value) for value in entries[0].metrics.values()
        )

    def test_recording_an_unknown_dataset_is_refused(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        with pytest.raises(ValueError, match="unknown dataset_id"):
            service.evaluate_signal(
                realized,
                realized,
                record=True,
                dataset_id="ds_" + "0" * 16,
                model_id=_GOOD_MODEL,
            )

    def test_malformed_model_id_is_refused(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        with pytest.raises(ValueError, match="malformed model_id"):
            service.evaluate_signal(
                realized, realized, record=True, dataset_id=dataset_id, model_id="nope"
            )

    def test_two_different_ad_hoc_signals_stay_two_candidates(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """Without a model ID, signal content distinguishes candidates."""
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        for quality in (1.0, -1.0):
            service.evaluate_signal(
                [value * quality for value in realized],
                realized,
                window=20,
                record=True,
                dataset_id=dataset_id,
            )
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 2
        assert entries[0].label != entries[1].label

    def test_the_same_ad_hoc_signal_is_still_idempotent(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        first = service.evaluate_signal(
            realized, realized, window=20, record=True, dataset_id=dataset_id
        )
        second = service.evaluate_signal(
            realized, realized, window=20, record=True, dataset_id=dataset_id
        )
        assert first["recorded"]["replaced"] is False
        assert second["recorded"]["replaced"] is True

    def test_a_signal_that_is_not_the_dataset_is_refused(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        short = [0.1, -0.2, 0.3, -0.4] * 10
        with pytest.raises(ValueError, match="would not describe this dataset"):
            service.evaluate_signal(
                short, short, window=20, record=True, dataset_id=dataset_id
            )

    def test_the_record_keeps_the_fitted_hyperparameters(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A row that only names a model cannot be read back as a setting."""
        dataset_id = _build(service)
        trained = service.train_model(
            dataset_id=dataset_id,
            model_kind="lightgbm_direction",
            parameters={"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
        )
        model_id = str(trained["model"]["model_id"])
        service.evaluate_prediction(
            model_id=model_id, dataset_id=dataset_id, window=20, record=True
        )
        entry = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        assert entry.parameters == trained["model"]["parameters"]
        assert entry.parameters["n_estimators"] == 12
        assert entry.parameters["max_depth"] == 3

    def test_an_unknown_artifact_records_no_parameters(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A caller-declared model ID with no saved artifact stays unexplained."""
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        entry = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        assert entry.parameters == {}
        assert entry.signal_fingerprint

    def test_every_recorded_entry_carries_its_signal_fingerprint(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        # quality 1.0 evaluates exactly the realised returns as the signal.
        recorded = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        entry = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        assert entry.signal_fingerprint == recorded["signal_fingerprint"]
        assert entry.signal_fingerprint == signal_fingerprint(realized)
        assert recorded["recorded"]["signal_fingerprint"] == entry.signal_fingerprint

    def test_a_different_signal_records_a_different_fingerprint(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        scaled = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        flipped = _record(
            service,
            workspace,
            dataset_id=dataset_id,
            model_id=_GOOD_MODEL,
            quality=-1.0,
        )
        assert scaled["signal_fingerprint"] != flipped["signal_fingerprint"]

    def test_the_fingerprint_does_not_change_the_entry_id(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A new signal for the same candidate must not become a new candidate."""
        dataset_id = _build(service)
        first = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        second = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=0.5
        )
        assert first["signal_fingerprint"] != second["signal_fingerprint"]
        assert (
            first["recorded"]["entry_id"]
            == second["recorded"]["entry_id"]
            == build_entry_id(
                model_id=_GOOD_MODEL,
                dataset_id=dataset_id,
                context=LeaderboardStore(workspace.ml_leaderboard).entries()[0].context,
            )
        )
        assert second["recorded"]["replaced"] is True
        assert len(LeaderboardStore(workspace.ml_leaderboard).entries()) == 1
        assert (
            LeaderboardStore(workspace.ml_leaderboard).entries()[0].signal_fingerprint
            == second["signal_fingerprint"]
        )

    def test_a_wrong_horizon_is_refused_but_omitting_it_is_adopted(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        with pytest.raises(ValueError, match="does not match the dataset's horizon"):
            service.evaluate_signal(
                realized,
                realized,
                horizon=7,
                window=20,
                record=True,
                dataset_id=dataset_id,
            )
        adopted = service.evaluate_signal(
            realized, realized, window=20, record=True, dataset_id=dataset_id
        )
        assert adopted["horizon"] == 1


class TestRanking:
    @pytest.fixture
    def two_candidates(self, service: MLService, workspace: WorkspacePaths) -> str:
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_BAD_MODEL, quality=-1.0
        )
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        return dataset_id

    def test_the_better_signal_wins(
        self, service: MLService, two_candidates: str
    ) -> None:
        board = service.leaderboard("ic_mean")
        assert board["n_candidates"] == 2
        assert [row["model_id"] for row in board["ranked"]] == [
            _GOOD_MODEL,
            _BAD_MODEL,
        ]
        assert board["ranked"][0]["rank"] == 1
        assert board["ranked"][0]["value"] > board["ranked"][1]["value"]

    def test_the_ranked_rows_carry_headline_numbers(
        self, service: MLService, two_candidates: str
    ) -> None:
        board = service.leaderboard("ic_mean")
        summary = board["ranked"][0]["summary"]
        assert "hit_rate" in summary
        assert "rank_ic_mean" in summary

    def test_a_minimize_metric_orders_the_other_way(
        self, service: MLService, two_candidates: str
    ) -> None:
        board = service.leaderboard("ic_std")
        assert board["direction"] == "minimize"
        assert board["ranked"][0]["value"] <= board["ranked"][1]["value"]

    def test_the_winner_is_reported_with_the_selection_caveat(
        self, service: MLService, two_candidates: str
    ) -> None:
        board = service.leaderboard("ic_mean")
        assert any("best of 2 candidates" in caveat for caveat in board["caveats"])

    def test_top_k_limits_the_ranking(
        self, service: MLService, two_candidates: str
    ) -> None:
        assert len(service.leaderboard("ic_mean", top_k=1)["ranked"]) == 1

    def test_quantile_metrics_are_ranked_too(
        self, service: MLService, two_candidates: str
    ) -> None:
        board = service.leaderboard("long_short_spread")
        assert len(board["ranked"]) == 2
        assert board["ranked"][0]["model_id"] == _GOOD_MODEL

    def test_unknown_metric_is_refused(
        self, service: MLService, two_candidates: str
    ) -> None:
        with pytest.raises(ValueError, match="unknown signal metric"):
            service.leaderboard("total_return")

    def test_directionless_metric_is_refused(
        self, service: MLService, two_candidates: str
    ) -> None:
        with pytest.raises(ValueError, match="no optimisation direction"):
            service.leaderboard("n_periods")


class TestComparability:
    @pytest.mark.parametrize("changed", ["bars", "feature_code"])
    def test_dataset_filter_selects_saved_versions_and_keeps_their_models(
        self, workspace: WorkspacePaths, changed: str
    ) -> None:
        first = MLService(workspace=workspace, feature_code_fingerprint="v1")
        old_id = _build(first)
        old_record = _record(
            first, workspace, dataset_id=old_id, model_id=_GOOD_MODEL, quality=1.0
        )
        if changed == "bars":
            source = cached_market_path(workspace.processed_data)
            frame = pd.read_parquet(source)
            frame["close"] *= np.linspace(1.0, 1.2, len(frame))
            frame.to_parquet(source, index=False)
        updated = MLService(
            workspace=workspace,
            feature_code_fingerprint="v2" if changed == "feature_code" else "v1",
        )
        new_id = _build(updated)
        assert new_id != old_id
        new_record = _record(
            updated, workspace, dataset_id=new_id, model_id=_GOOD_MODEL, quality=1.0
        )
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            updated.leaderboard("ic_mean")
        assert updated.leaderboard("ic_mean", allow_mixed=True)["n_candidates"] == 2
        for dataset_id, record in ((old_id, old_record), (new_id, new_record)):
            board = updated.leaderboard("ic_mean", dataset_id=dataset_id)
            assert board["n_candidates"] == board["n_ranked"] == 1
            assert board["ranked"][0]["dataset_id"] == dataset_id
            assert board["ranked"][0]["entry_id"] == record["recorded"]["entry_id"]

        _record(
            updated, workspace, dataset_id=old_id, model_id=_BAD_MODEL, quality=-1.0
        )
        board = updated.leaderboard("ic_mean", dataset_id=old_id)
        assert board["n_candidates"] == 2
        assert [row["model_id"] for row in board["ranked"]] == [
            _GOOD_MODEL,
            _BAD_MODEL,
        ]
        assert all(row["dataset_id"] == old_id for row in board["ranked"])
        with pytest.raises(ValueError, match="no recorded evaluations"):
            updated.leaderboard("ic_mean", dataset_id=old_id, symbol="ETH/USDT")

    @pytest.mark.parametrize("dataset_id", ["", "invalid", "ds_" + "0" * 16])
    @pytest.mark.parametrize("allow_mixed", [False, True])
    def test_dataset_filter_rejects_invalid_ids_even_without_records(
        self, service: MLService, dataset_id: str, allow_mixed: bool
    ) -> None:
        with pytest.raises(ValueError, match="malformed dataset_id|unknown dataset_id"):
            service.leaderboard(
                "ic_mean", dataset_id=dataset_id, allow_mixed=allow_mixed
            )

    @pytest.mark.parametrize("allow_mixed", [False, True])
    @pytest.mark.parametrize("corrupt", ["context", "row_count", "dataset_bytes"])
    def test_dataset_filter_rejects_inconsistent_saved_evidence(
        self,
        service: MLService,
        workspace: WorkspacePaths,
        allow_mixed: bool,
        corrupt: str,
    ) -> None:
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1
        )
        store = LeaderboardStore(workspace.ml_leaderboard)
        entry = store.entries()[0]
        if corrupt == "context":
            store.record(
                replace(entry, context=replace(entry.context, data_fingerprint="other"))
            )
        elif corrupt == "row_count":
            store.record(replace(entry, n_observations=entry.n_observations - 1))
        else:
            path = workspace.ml_datasets / dataset_id / "dataset.parquet"
            path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="does not match dataset_id|content hash"):
            service.leaderboard(
                "ic_mean", dataset_id=dataset_id, allow_mixed=allow_mixed
            )

    def test_dataset_filter_preserves_evaluation_setting_comparability(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        for window in (20, 30):
            service.evaluate_signal(
                realized,
                realized,
                window=window,
                record=True,
                dataset_id=dataset_id,
                model_id=_GOOD_MODEL,
            )
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            service.leaderboard("ic_mean", dataset_id=dataset_id)
        board = service.leaderboard("ic_mean", dataset_id=dataset_id, allow_mixed=True)
        assert board["n_candidates"] == 2
        assert board["context"] == "mixed"

    def test_different_instruments_are_not_ranked_together(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        btc = _build(service, "BTC/USDT")
        eth = _build(service, "ETH/USDT")
        _record(service, workspace, dataset_id=btc, model_id=_GOOD_MODEL, quality=1.0)
        _record(service, workspace, dataset_id=eth, model_id=_BAD_MODEL, quality=-1.0)
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            service.leaderboard("ic_mean")

    def test_allow_mixed_overrides_only_when_asked(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        btc = _build(service, "BTC/USDT")
        eth = _build(service, "ETH/USDT")
        _record(service, workspace, dataset_id=btc, model_id=_GOOD_MODEL, quality=1.0)
        _record(service, workspace, dataset_id=eth, model_id=_BAD_MODEL, quality=-1.0)
        board = service.leaderboard("ic_mean", allow_mixed=True)
        assert board["context"] == "mixed"

    def test_filters_narrow_the_field_back_to_one_context(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        btc = _build(service, "BTC/USDT")
        eth = _build(service, "ETH/USDT")
        _record(service, workspace, dataset_id=btc, model_id=_GOOD_MODEL, quality=1.0)
        _record(service, workspace, dataset_id=eth, model_id=_BAD_MODEL, quality=-1.0)
        board = service.leaderboard("ic_mean", symbol="ETH/USDT")
        assert board["n_candidates"] == 1
        assert board["ranked"][0]["model_id"] == _BAD_MODEL

    def test_an_empty_leaderboard_says_so(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        with pytest.raises(ValueError, match="no recorded evaluations"):
            service.leaderboard("ic_mean")


class TestContract:
    def test_rankable_metrics_are_advertised(self, service: MLService) -> None:
        contract = service.get_contract()
        assert "rank_icir" in contract["leaderboard_metrics"]
        assert "n_periods" not in contract["leaderboard_metrics"]


class TestParameterSurface:
    """A window sweep is a surface, and must be reachable without allow_mixed."""

    @staticmethod
    def _sweep(
        service: MLService, workspace: WorkspacePaths, windows: tuple[int, ...]
    ) -> str:
        dataset_id = _build(service)
        realized = _returns(workspace, dataset_id)
        for window in windows:
            service.evaluate_signal(
                realized,
                realized,
                window=window,
                record=True,
                dataset_id=dataset_id,
                model_id=_GOOD_MODEL,
            )
        return dataset_id

    def test_the_strict_guard_still_refuses_a_sweep(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        self._sweep(service, workspace, (20, 30))
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            service.leaderboard("ic_mean")

    def test_a_sweep_is_ranked_when_the_axes_are_allowed(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = self._sweep(service, workspace, (20, 30))
        board = service.leaderboard(
            "ic_mean", dataset_id=dataset_id, allow_parameter_axes=True
        )
        assert board["n_candidates"] == board["n_ranked"] == 2
        assert board["parameter_axes"] == {
            "horizon": [1],
            "window": [20, 30],
            "n_quantiles": [5],
        }
        assert sorted(row["window"] for row in board["ranked"]) == [20, 30]
        assert all(row["horizon"] == 1 for row in board["ranked"])
        assert any(
            "parameter surface" in caveat and "window=[20, 30]" in caveat
            for caveat in board["caveats"]
        )

    def test_one_setting_carries_no_surface_caveat(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = self._sweep(service, workspace, (20,))
        board = service.leaderboard(
            "ic_mean", dataset_id=dataset_id, allow_parameter_axes=True
        )
        assert board["parameter_axes"] == {
            "horizon": [1],
            "window": [20],
            "n_quantiles": [5],
        }
        assert not any("parameter surface" in caveat for caveat in board["caveats"])

    def test_a_horizon_sweep_across_datasets_is_rankable(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A new horizon rebuilds the labels, so each point has its own dataset."""
        dataset_ids = set()
        for horizon in (1, 3):
            dataset_id = str(
                service.build_dataset(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    feature_set="returns",
                    horizon=horizon,
                )["dataset"]["dataset_id"]
            )
            dataset_ids.add(dataset_id)
            realized = _returns(workspace, dataset_id)
            service.evaluate_signal(
                realized,
                realized,
                window=20,
                record=True,
                dataset_id=dataset_id,
                model_id=_GOOD_MODEL,
            )
        assert len(dataset_ids) == 2
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            service.leaderboard("ic_mean")
        board = service.leaderboard("ic_mean", allow_parameter_axes=True)
        assert board["n_candidates"] == 2
        assert board["parameter_axes"]["horizon"] == [1, 3]
        assert any(
            "raw-data equivalence is not verified" in item for item in board["caveats"]
        )
        assert {row["dataset_id"] for row in board["ranked"]} == dataset_ids
        assert all(row["window"] == 20 for row in board["ranked"])

    def test_a_failed_point_keeps_its_coordinates(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A failed run at window=20 must not turn 10 and 30 into neighbours."""
        dataset_id = self._sweep(service, workspace, (10, 30))
        store = LeaderboardStore(workspace.ml_leaderboard)
        sample = store.entries()[0]
        failed_context = replace(sample.context, window=20)
        store.record(
            replace(
                sample,
                entry_id=build_entry_id(
                    model_id=sample.model_id,
                    dataset_id=sample.dataset_id,
                    context=failed_context,
                    signal_fingerprint="",
                ),
                context=failed_context,
                metrics={**sample.metrics, "ic_mean": None},
            )
        )

        board = service.leaderboard(
            "ic_mean", dataset_id=dataset_id, allow_parameter_axes=True
        )

        assert board["parameter_axes"]["window"] == [10, 20, 30]
        assert board["n_candidates"] == 3
        assert board["n_ranked"] == 2
        assert [row["window"] for row in board["skipped"]] == [20]
        assert "is unavailable or not finite" in board["skipped"][0]["reason"]

    def test_a_ranked_row_reports_its_own_failed_metric_as_null(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """Ranking on one metric must not hide a failure on another."""
        dataset_id = self._sweep(service, workspace, (20,))
        store = LeaderboardStore(workspace.ml_leaderboard)
        sample = store.entries()[0]
        store.record(
            replace(
                sample,
                metrics={**sample.metrics, "ic_mean": 0.5, "sharpe": None},
            )
        )

        board = service.leaderboard("ic_mean", dataset_id=dataset_id)

        summary = board["ranked"][0]["summary"]
        assert "sharpe" in summary
        assert summary["sharpe"] is None
        assert summary["ic_mean"] == 0.5

    def test_one_horizon_keeps_one_dataset_version(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """A rebuilt dataset at the same horizon must not share a grid row."""
        source = cached_market_path(workspace.processed_data)
        first = _build(service)
        _record(service, workspace, dataset_id=first, model_id=_GOOD_MODEL, quality=1.0)
        frame = pd.read_parquet(source)
        frame["close"] *= np.linspace(1.0, 1.2, len(frame))
        frame.to_parquet(source, index=False)
        second = _build(service)
        assert second != first
        _record(
            service, workspace, dataset_id=second, model_id=_GOOD_MODEL, quality=1.0
        )
        with pytest.raises(ValueError, match="different dataset versions"):
            service.leaderboard("ic_mean", allow_parameter_axes=True)

    def test_the_axis_opt_in_does_not_mix_instruments(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        btc = _build(service, "BTC/USDT")
        eth = _build(service, "ETH/USDT")
        _record(service, workspace, dataset_id=btc, model_id=_GOOD_MODEL, quality=1.0)
        _record(service, workspace, dataset_id=eth, model_id=_BAD_MODEL, quality=-1.0)
        with pytest.raises(ValueError, match="distinct identities"):
            service.leaderboard("ic_mean", allow_parameter_axes=True)

    def test_the_axis_opt_in_does_not_mix_dataset_versions(
        self, workspace: WorkspacePaths
    ) -> None:
        service = MLService(workspace=workspace, feature_code_fingerprint="v1")
        old_id = _build(service)
        _record(service, workspace, dataset_id=old_id, model_id=_GOOD_MODEL, quality=1)
        updated = MLService(workspace=workspace, feature_code_fingerprint="v2")
        new_id = _build(updated)
        _record(updated, workspace, dataset_id=new_id, model_id=_GOOD_MODEL, quality=1)
        with pytest.raises(ValueError, match="distinct identities"):
            updated.leaderboard("ic_mean", allow_parameter_axes=True)

    def test_a_ranked_row_states_its_own_settings(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = self._sweep(service, workspace, (20,))
        row = service.leaderboard("ic_mean", dataset_id=dataset_id)["ranked"][0]
        assert row["horizon"] == 1
        assert row["window"] == 20
        assert row["n_quantiles"] == 5
        assert row["parameters"] == {}
        assert row["signal_fingerprint"] == signal_fingerprint(
            _returns(workspace, dataset_id)
        )

    def test_paging_keeps_the_grid_and_the_surface_caveat(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = self._sweep(service, workspace, (20, 30, 40))
        page = service.leaderboard_page(
            "ic_mean", dataset_id=dataset_id, limit=1, allow_parameter_axes=True
        )
        assert page["n_candidates"] == 3
        assert page["parameter_axes"]["window"] == [20, 30, 40]
        assert len(page["ranked"]) == 1
        assert page["next_offset"] == 1
        assert any("parameter surface" in caveat for caveat in page["caveats"])


class TestFeatureIdentity:
    """Feature-code changes preserve distinct evaluation evidence."""

    @staticmethod
    def _service(
        workspace: WorkspacePaths, *, code_fingerprint: str | None = None
    ) -> MLService:
        return MLService(workspace=workspace, feature_code_fingerprint=code_fingerprint)

    def test_the_record_names_the_feature_set_and_code(
        self, workspace: WorkspacePaths
    ) -> None:
        service = self._service(workspace, code_fingerprint="abc123")
        dataset_id = _build(service)
        recorded = _record(
            service,
            workspace,
            dataset_id=dataset_id,
            model_id=_GOOD_MODEL,
            quality=1.0,
        )["recorded"]
        assert "features=returns" in recorded["context"]
        assert "code=abc123" in recorded["context"]

    def test_a_code_change_records_a_new_candidate_instead_of_overwriting(
        self, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(self._service(workspace, code_fingerprint="v1"))
        realized = _returns(workspace, dataset_id)
        before = self._service(workspace, code_fingerprint="v1").evaluate_signal(
            realized,
            realized,
            horizon=1,
            window=20,
            record=True,
            dataset_id=dataset_id,
            model_id=_GOOD_MODEL,
        )
        updated = self._service(workspace, code_fingerprint="v2")
        new_dataset_id = _build(updated)
        assert new_dataset_id != dataset_id
        after = updated.evaluate_signal(
            realized,
            realized,
            horizon=1,
            window=20,
            record=True,
            dataset_id=new_dataset_id,
            model_id=_GOOD_MODEL,
        )
        assert before["recorded"]["replaced"] is False
        assert after["recorded"]["replaced"] is False
        assert after["recorded"]["n_candidates"] == 2

    def test_two_code_versions_are_not_ranked_together(
        self, workspace: WorkspacePaths
    ) -> None:
        for fingerprint in ("v1", "v2"):
            version = self._service(workspace, code_fingerprint=fingerprint)
            dataset_id = _build(version)
            realized = _returns(workspace, dataset_id)
            version.evaluate_signal(
                realized,
                realized,
                horizon=1,
                window=20,
                record=True,
                dataset_id=dataset_id,
                model_id=_GOOD_MODEL,
            )
        service = self._service(workspace, code_fingerprint="v2")
        with pytest.raises(ValueError) as caught:
            service.leaderboard("ic_mean")
        assert "not measured on the same conditions" in str(caught.value)
        assert "code=v1" in str(caught.value) and "code=v2" in str(caught.value)

    def test_existing_dataset_keeps_its_build_time_code_identity(
        self, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(self._service(workspace, code_fingerprint="v1"))
        realized = _returns(workspace, dataset_id)
        response = self._service(workspace, code_fingerprint="v2").evaluate_signal(
            realized, realized, window=20, record=True, dataset_id=dataset_id
        )
        assert "code=v1" in response["recorded"]["context"]
        assert "code=v2" not in response["recorded"]["context"]

    def test_the_feature_set_is_recorded_and_can_narrow_the_field(
        self, workspace: WorkspacePaths
    ) -> None:
        service = self._service(workspace)
        returns_id = _build(service)
        technical_id = str(
            service.build_dataset(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-31",
                feature_set="technical",
            )["dataset"]["dataset_id"]
        )
        _record(
            service, workspace, dataset_id=returns_id, model_id=_GOOD_MODEL, quality=1.0
        )
        _record(
            service,
            workspace,
            dataset_id=technical_id,
            model_id=_BAD_MODEL,
            quality=-1.0,
        )
        recorded = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert {entry.context.feature_set for entry in recorded} == {
            "returns",
            "technical",
        }
        with pytest.raises(ValueError, match="not measured on the same conditions"):
            service.leaderboard("ic_mean")
        board = service.leaderboard("ic_mean", feature_set="technical")
        assert board["n_candidates"] == 1
        assert board["ranked"][0]["model_id"] == _BAD_MODEL


class TestReturnSampling:
    @pytest.mark.parametrize("horizon", [1, 3])
    def test_performance_uses_holding_intervals_including_cash(
        self, service: MLService, workspace: WorkspacePaths, horizon: int
    ) -> None:
        built = service.build_dataset(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            feature_set="returns",
            horizon=horizon,
        )
        dataset_id = built["dataset"]["dataset_id"]
        realized = _returns(workspace, dataset_id)
        signal = [1.0 if index % 2 else -1.0 for index in range(len(realized))]
        result = service.evaluate_signal(
            signal,
            realized,
            dataset_id=dataset_id,
            window=20,
            n_quantiles=2,
            record=True,
        )
        # Derive expectations from explicit non-overlapping intervals, rather
        # than calling the new performance function under test.
        simulation = simulate_strategy_returns(
            pd.Series(signal),
            pd.Series(realized),
            horizon=horizon,
            n_quantiles=2,
        )
        book = simulation.long_short.iloc[::horizon].to_numpy()
        market = np.asarray(realized)[::horizon]
        assert book[0] == 0.0  # Real warm-up cash interval must remain counted.
        factor = 8760.0 / horizon
        beta = float(np.cov(book, market, ddof=1)[0, 1] / np.var(market, ddof=1))
        expected = {
            "n_return_periods": len(book),
            "periods_per_year": factor,
            "annual_return": float(np.prod(1.0 + book) ** (factor / len(book)) - 1.0),
            "sharpe": float(np.mean(book) / np.std(book, ddof=1) * np.sqrt(factor)),
            "beta": beta,
            "alpha": float((np.mean(book) - beta * np.mean(market)) * factor),
        }
        stored = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        ranked = service.leaderboard("annual_return", dataset_id=dataset_id)["ranked"][
            0
        ]
        for key, value in expected.items():
            assert result["metrics"][key] == pytest.approx(value)
            assert stored.metrics[key] == pytest.approx(value)
            if key in ranked["summary"]:
                assert ranked["summary"][key] == pytest.approx(value)
        assert ranked["value"] == pytest.approx(expected["annual_return"])

    @pytest.mark.parametrize(
        "source,gap", [("yahoo_finance", False), ("binance_vision", True)]
    )
    def test_unknown_or_discontinuous_calendar_does_not_guess_annualization(
        self, service: MLService, workspace: WorkspacePaths, source: str, gap: bool
    ) -> None:
        if gap:
            path = cached_market_path(workspace.processed_data)
            bars = pd.read_parquet(path).drop(index=150)
            bars.to_parquet(path, index=False)
        built = service.build_dataset(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            feature_set="returns",
            source=source,
        )
        dataset_id = built["dataset"]["dataset_id"]
        realized = _returns(workspace, dataset_id)
        result = service.evaluate_signal(
            realized,
            realized,
            dataset_id=dataset_id,
            window=20,
            record=True,
        )
        stored = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        for key in ("annual_return", "sharpe", "alpha", "periods_per_year"):
            assert result["metrics"][key] is None
            assert stored.metrics[key] is None
        assert result["metrics"]["max_drawdown"] is not None
        assert result["metrics"]["beta"] is not None
        board = service.leaderboard("annual_return", dataset_id=dataset_id)
        assert board["ranked"] == []
        assert len(board["skipped"]) == 1


def test_surface_rejects_distinct_manifests_even_with_identical_data(
    service: MLService,
    workspace: WorkspacePaths,
) -> None:
    for source in ("binance_vision", "yahoo_finance"):
        built = service.build_dataset(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            feature_set="returns",
            source=source,
        )
        dataset_id = built["dataset"]["dataset_id"]
        realized = _returns(workspace, dataset_id)
        service.evaluate_signal(
            realized,
            realized,
            dataset_id=dataset_id,
            window=20,
            record=True,
        )
    entries = LeaderboardStore(workspace.ml_leaderboard).entries()
    assert len({entry.dataset_id for entry in entries}) == 2
    assert len({entry.context.data_fingerprint for entry in entries}) == 1
    with pytest.raises(ValueError, match="different dataset versions"):
        service.leaderboard("ic_mean", allow_parameter_axes=True)
