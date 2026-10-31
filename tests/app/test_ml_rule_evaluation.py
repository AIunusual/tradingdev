"""Recording a threshold rule and ranking it beside signals without a model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _write_bars(workspace: WorkspacePaths) -> None:
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
    seed_market_cache(workspace.processed_data, frame, symbol="BTC/USDT")


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService) -> str:
    result = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="technical",
    )
    dataset_id: str = result["dataset"]["dataset_id"]
    return dataset_id


def _frame(workspace: WorkspacePaths, dataset_id: str) -> pd.DataFrame:
    return pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")


def _splitting_level(workspace: WorkspacePaths, dataset_id: str) -> float:
    """A level the column really crosses, so the rule takes both legs."""
    column = _frame(workspace, dataset_id)["rsi_14"]
    return float(column.median())


def _record_rule(
    service: MLService, workspace: WorkspacePaths, dataset_id: str
) -> dict[str, Any]:
    return service.evaluate_rule(
        dataset_id=dataset_id,
        column="rsi_14",
        threshold=_splitting_level(workspace, dataset_id),
        direction="below",
        window=20,
        record=True,
    )


class TestRecordingARule:
    def test_a_rule_is_recorded_under_its_own_family(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        level = _splitting_level(workspace, dataset_id)

        result = _record_rule(service, workspace, dataset_id)

        assert result["recorded"]["method_family"] == "rule"
        assert result["recorded"]["method_name"] == f"rsi_14 < {level:g}"
        entry = LeaderboardStore(workspace.ml_leaderboard).entries()[0]
        assert entry.family == "rule"
        assert entry.label == f"rsi_14 < {level:g}"
        assert entry.parameters == {"threshold": level}
        assert entry.model_id == ""

    def test_a_rule_is_scored_on_the_dataset_rows(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)

        result = _record_rule(service, workspace, dataset_id)

        assert result["n_observations"] == int(_frame(workspace, dataset_id).shape[0])
        assert result["metrics"]["ic_mean"] is not None

    def test_re_recording_the_same_rule_replaces_it(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record_rule(service, workspace, dataset_id)

        again = _record_rule(service, workspace, dataset_id)

        assert again["recorded"]["replaced"] is True
        assert len(LeaderboardStore(workspace.ml_leaderboard).entries()) == 1

    def test_retuning_the_threshold_records_a_second_candidate(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record_rule(service, workspace, dataset_id)
        level = _splitting_level(workspace, dataset_id)

        service.evaluate_rule(
            dataset_id=dataset_id,
            column="rsi_14",
            threshold=level - 1.0,
            direction="below",
            window=20,
            record=True,
        )

        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 2
        assert {entry.method_name for entry in entries} == {
            f"rsi_14 < {level:g}",
            f"rsi_14 < {level - 1.0:g}",
        }

    def test_a_reserved_column_is_refused(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        with pytest.raises(ValueError, match="unknown rule column"):
            service.evaluate_rule(
                dataset_id=dataset_id, column="forward_return", threshold=0.0
            )

    def test_an_unknown_column_lists_what_the_dataset_supplies(
        self, service: MLService
    ) -> None:
        dataset_id = _build(service)
        with pytest.raises(ValueError, match="rsi_14"):
            service.evaluate_rule(
                dataset_id=dataset_id, column="not_a_feature", threshold=0.0
            )

    def test_a_level_the_column_never_crosses_is_refused(
        self, service: MLService
    ) -> None:
        dataset_id = _build(service)
        with pytest.raises(ValueError, match="never changes"):
            service.evaluate_rule(dataset_id=dataset_id, column="rsi_14", threshold=1e9)


class TestNamedCandidates:
    def test_nearby_thresholds_keep_distinct_candidates(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        column = _frame(workspace, dataset_id)["rsi_14"]
        middle = float(column.sort_values().iloc[len(column) // 2])
        levels = [float(np.nextafter(middle, side)) for side in (-np.inf, np.inf)]
        assert f"{levels[0]:g}" == f"{levels[1]:g}"
        results = [
            service.evaluate_rule(
                dataset_id=dataset_id,
                column="rsi_14",
                threshold=level,
                window=20,
                record=True,
            )
            for level in levels
        ]
        assert results[0]["signal_fingerprint"] != results[1]["signal_fingerprint"]
        assert all(not result["recorded"]["replaced"] for result in results)
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 2
        assert {entry.parameters["threshold"] for entry in entries} == set(levels)

    def test_family_and_parameters_are_part_of_named_candidate_identity(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        realized = _frame(workspace, dataset_id)["forward_return"].tolist()
        model_id = "a" * 16
        declarations: list[dict[str, Any]] = [{"model_id": model_id}] + [
            {
                "method_family": family,
                "method_name": model_id,
                "method_parameters": {"threshold": threshold},
            }
            for family, threshold in [("rule", 1.0), ("rule", 2.0), ("other", 2.0)]
        ]
        for declaration in declarations:
            result = service.evaluate_signal(
                realized,
                realized,
                dataset_id=dataset_id,
                window=20,
                record=True,
                **declaration,
            )
            assert not result["recorded"]["replaced"]
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 4
        assert {entry.family for entry in entries} == {"ml_model", "rule", "other"}

    def test_a_named_candidate_replaces_instead_of_inflating_the_field(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        """The name identifies the candidate; a changed signal is still one row."""
        dataset_id = _build(service)
        realized = _frame(workspace, dataset_id)["forward_return"].astype(float)

        service.evaluate_signal(
            realized.tolist(),
            realized.tolist(),
            window=20,
            record=True,
            dataset_id=dataset_id,
            method_family="other",
            method_name="blend of two signals",
        )
        again = service.evaluate_signal(
            (-realized).tolist(),
            realized.tolist(),
            window=20,
            record=True,
            dataset_id=dataset_id,
            method_family="other",
            method_name="blend of two signals",
        )

        assert again["recorded"]["replaced"] is True
        entries = LeaderboardStore(workspace.ml_leaderboard).entries()
        assert len(entries) == 1
        assert entries[0].method_name == "blend of two signals"
        assert entries[0].family == "other"


class TestRankingRulesWithOtherMethods:
    def test_a_rule_and_an_unnamed_signal_share_one_table(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record_rule(service, workspace, dataset_id)
        realized = _frame(workspace, dataset_id)["forward_return"].astype(float)
        service.evaluate_signal(
            realized.tolist(),
            realized.tolist(),
            window=20,
            record=True,
            dataset_id=dataset_id,
        )

        result = service.leaderboard("ic_mean")

        assert result["families"] == {"rule": 1, "other": 1}
        assert any("method families" in caveat for caveat in result["caveats"])
        families = {row["method_family"] for row in result["ranked"]}
        assert families == {"rule", "other"}

    def test_the_family_filter_ranks_one_kind(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)
        _record_rule(service, workspace, dataset_id)
        realized = _frame(workspace, dataset_id)["forward_return"].astype(float)
        service.evaluate_signal(
            realized.tolist(),
            realized.tolist(),
            window=20,
            record=True,
            dataset_id=dataset_id,
        )

        result = service.leaderboard("ic_mean", family="rule")

        assert result["families"] == {"rule": 1}
        assert [row["method_family"] for row in result["ranked"]] == ["rule"]
        assert not any("method families" in caveat for caveat in result["caveats"])

    def test_an_unknown_family_filter_is_refused(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="family must be one of"):
            service.leaderboard("ic_mean", family="ensemble")

    def test_the_contract_lists_the_families_a_candidate_may_declare(
        self, service: MLService
    ) -> None:
        assert service.get_contract()["method_families"] == [
            "ml_model",
            "rule",
            "other",
        ]
