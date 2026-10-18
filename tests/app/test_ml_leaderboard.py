"""Recording evaluations and ranking them, against real built datasets.

The two signals used here are deliberately extreme: one is the realised
forward return itself (a perfect signal) and one is its negation (a perfectly
wrong signal). A leaderboard that cannot separate those two is not ranking
anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
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
    path = workspace.processed_data / (
        symbol.lower().replace("/", "") + "_1h_2024.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


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
        """The same candidate twice must not become two comparisons."""
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        again = _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        assert again["recorded"]["replaced"] is True
        assert again["recorded"]["n_candidates"] == 1

    def test_only_finite_metrics_are_stored(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record(
            service, workspace, dataset_id=dataset_id, model_id=_GOOD_MODEL, quality=1.0
        )
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 1
        assert entries[0].metrics
        assert all(np.isfinite(value) for value in entries[0].metrics.values())

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
        """With no model to identify them, the signal itself must tell them apart."""
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
        """The contract advertises them, so recording has to carry them."""
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


class TestFeatureIdentity:
    """Which feature code produced the numbers is part of the record.

    The dataset id does not depend on the feature code, so without the code in
    the entry identity a fixed indicator would simply overwrite the row written
    before the fix - and the leaderboard would present post-fix evidence as the
    same candidate it had already ranked.
    """

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
        after = self._service(workspace, code_fingerprint="v2").evaluate_signal(
            realized,
            realized,
            horizon=1,
            window=20,
            record=True,
            dataset_id=dataset_id,
            model_id=_GOOD_MODEL,
        )
        assert before["recorded"]["replaced"] is False
        assert after["recorded"]["replaced"] is False
        assert after["recorded"]["n_candidates"] == 2

    def test_two_code_versions_are_not_ranked_together(
        self, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(self._service(workspace, code_fingerprint="v1"))
        realized = _returns(workspace, dataset_id)
        for fingerprint in ("v1", "v2"):
            self._service(workspace, code_fingerprint=fingerprint).evaluate_signal(
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
