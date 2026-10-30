"""Panel alignment with a 24-bar ETH gap that would bias naive concatenation."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.features.registry import get_feature_set

if TYPE_CHECKING:
    from pathlib import Path

_DATASET_ID = re.compile(r"^ds_[0-9a-f]{16}$")
_ROWS = 300
_HOLE = range(100, 124)
_SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _write_bars(
    workspace: WorkspacePaths, symbol: str, *, drop: range | None = None
) -> None:
    """Cache a successful historical query with hourly OHLCV observations."""
    skip = set(drop or ())
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
    keep = [i for i in range(_ROWS) if i not in skip]
    frame = frame.iloc[keep].reset_index(drop=True)
    seed_market_cache(workspace.processed_data, frame, symbol=symbol)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths, "BTC/USDT")
    _write_bars(paths, "ETH/USDT", drop=_HOLE)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService, **overrides: object) -> dict[str, Any]:
    kwargs: dict[str, object] = {
        "symbols": list(_SYMBOLS),
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 1,
    }
    kwargs.update(overrides)
    panel = service.build_panel(**kwargs)  # type: ignore[arg-type]
    assert isinstance(panel, dict)
    return panel


class TestBuildPanel:
    def test_rows_are_symbols_times_dates(self, service: MLService) -> None:
        panel = _build(service)["panel"]
        assert panel["rows"] == panel["n_symbols"] * panel["n_dates"]
        assert panel["n_symbols"] == 2
        assert _DATASET_ID.match(str(panel["dataset_id"]))

    def test_intersection_reaches_the_same_calendar_without_dropping(
        self, service: MLService
    ) -> None:
        """Both routes exclude the hole; intersection simply never listed it."""
        union = _build(service, calendar_mode="union")["panel"]
        intersection = _build(service, calendar_mode="intersection")["panel"]
        assert union["calendar"]["n_dropped_dates"] == len(_HOLE)
        assert intersection["calendar"]["n_dropped_dates"] == 0
        assert intersection["n_dates"] == union["n_dates"]

    def test_union_with_drop_reports_the_gap_dates(self, service: MLService) -> None:
        calendar = _build(service, calendar_mode="union")["panel"]["calendar"]
        assert calendar["n_dropped_dates"] == len(_HOLE)
        assert len(calendar["dropped_dates"]) == len(_HOLE)
        coverage = {item["symbol"]: item for item in calendar["coverage"]}
        assert coverage["BTC/USDT"]["n_missing"] == 0
        assert coverage["ETH/USDT"]["n_missing"] == len(_HOLE)
        assert coverage["ETH/USDT"]["coverage"] < 1.0

    def test_forward_fill_carries_the_hole_and_counts_it(
        self, service: MLService
    ) -> None:
        panel = _build(service, calendar_mode="union", missing_policy="forward_fill")[
            "panel"
        ]
        assert panel["calendar"]["n_dropped_dates"] == 0
        assert panel["calendar"]["forward_filled"] == {
            "BTC/USDT": 0,
            "ETH/USDT": len(_HOLE),
        }
        intersection = _build(service)["panel"]
        assert panel["n_dates"] == intersection["n_dates"] + len(_HOLE)
        assert panel["rows"] == 2 * panel["n_dates"]

    def test_max_staleness_bounds_how_far_a_bar_travels(
        self, service: MLService
    ) -> None:
        panel = _build(
            service,
            calendar_mode="union",
            missing_policy="forward_fill",
            max_staleness=5,
        )["panel"]
        # Only the first 5 dates of the 24-bar hole may be carried.
        assert panel["calendar"]["forward_filled"]["ETH/USDT"] == 5
        assert panel["calendar"]["n_dropped_dates"] == len(_HOLE) - 5

    def test_error_policy_refuses_a_panel_with_gaps(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="no bar on"):
            _build(service, calendar_mode="union", missing_policy="error")

    def test_a_carried_row_keeps_the_older_bar_values(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        panel = _build(service, calendar_mode="union", missing_policy="forward_fill")[
            "panel"
        ]
        frame = pd.read_parquet(
            workspace.ml_datasets / panel["dataset_id"] / "dataset.parquet"
        )
        eth = frame[frame["symbol"] == "ETH/USDT"].reset_index(drop=True)
        hole_start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=_HOLE[0])
        carried = eth[eth["timestamp"] == hole_start].iloc[0]
        previous = eth[eth["timestamp"] < hole_start].iloc[-1]
        assert carried["close"] == pytest.approx(previous["close"])
        # The label travels with the bar: a carried row is the same observation,
        # not a new one, so its forward return is the old bar's own.
        assert carried["forward_return"] == pytest.approx(previous["forward_return"])
        assert carried["target"] == pytest.approx(previous["target"])

    def test_panel_columns_carry_the_symbol_key(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)["panel"]["dataset_id"]
        frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
        assert set(frame["symbol"]) == set(_SYMBOLS)
        assert "timestamp" in frame.columns
        assert "forward_return" in frame.columns
        assert frame["target"].isna().sum() == 0

    def test_manifest_records_the_calendar(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)["panel"]["dataset_id"]
        manifest = json.loads(
            (workspace.ml_datasets / dataset_id / "manifest.json").read_text("utf-8")
        )
        assert manifest["kind"] == "panel"
        assert manifest["symbols"] == _SYMBOLS
        assert manifest["calendar"]["mode"] == "intersection"
        assert manifest["feature_columns"] == list(get_feature_set("returns").columns)
        assert manifest["rows"] == manifest["n_dates"] * 2

    def test_missing_policy_is_inert_under_intersection(
        self, service: MLService
    ) -> None:
        """No dates need filling, so missing policies must not fork dataset identity."""
        assert (
            _build(service, missing_policy="forward_fill", max_staleness=3)["panel"][
                "dataset_id"
            ]
            == _build(service)["panel"]["dataset_id"]
        )

    def test_dropped_date_listing_is_capped_but_the_count_is_exact(
        self, tmp_path: Path
    ) -> None:
        paths = WorkspacePaths(tmp_path / "wide")
        paths.ensure()
        _write_bars(paths, "BTC/USDT")
        _write_bars(paths, "ETH/USDT", drop=range(100, 170))
        panel = _build(MLService(workspace=paths), calendar_mode="union")["panel"]
        assert panel["calendar"]["n_dropped_dates"] == 70
        assert len(panel["calendar"]["dropped_dates"]) == 50

    def test_is_deterministic(self, service: MLService) -> None:
        assert (
            _build(service)["panel"]["dataset_id"]
            == _build(service)["panel"]["dataset_id"]
        )

    def test_calendar_mode_changes_the_identity(self, service: MLService) -> None:
        assert (
            _build(service)["panel"]["dataset_id"]
            != _build(service, calendar_mode="union")["panel"]["dataset_id"]
        )

    def test_symbol_order_does_not_change_the_identity(
        self, service: MLService
    ) -> None:
        assert (
            _build(service)["panel"]["dataset_id"]
            == _build(service, symbols=list(reversed(_SYMBOLS)))["panel"]["dataset_id"]
        )

    def test_requires_two_distinct_symbols(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="at least two distinct symbols"):
            _build(service, symbols=["BTC/USDT"])
        with pytest.raises(ValueError, match="duplicate symbols"):
            _build(service, symbols=["BTC/USDT", "BTC/USDT"])

    def test_rejects_an_unknown_calendar_option(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="unknown calendar mode"):
            _build(service, calendar_mode="weekly")
        with pytest.raises(ValueError, match="unknown missing policy"):
            _build(service, missing_policy="interpolate")

    def test_unknown_feature_set_is_rejected(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="unknown feature set"):
            _build(service, feature_set="not_a_set")

    def test_horizon_must_be_positive(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            _build(service, horizon=0)

    def test_requires_cached_data_for_every_symbol(
        self, workspace: WorkspacePaths
    ) -> None:
        with pytest.raises(ValueError, match="ensure_data"):
            _build(
                MLService(workspace=workspace),
                symbols=["BTC/USDT", "SOL/USDT"],
            )


class TestPanelTrainingLoop:
    def test_trained_model_predicts_across_the_panel(self, service: MLService) -> None:
        panel = _build(service)["panel"]
        model = service.train_model(
            dataset_id=panel["dataset_id"],
            model_kind="xgboost_direction",
            parameters={"n_estimators": 5, "max_depth": 2},
        )["model"]
        assert model["n_train_samples"] == panel["rows"]
        prediction = service.predict(
            model_id=model["model_id"], dataset_id=panel["dataset_id"]
        )["prediction"]
        assert prediction["n_rows"] == panel["rows"]
        assert len(prediction["signal"]) == panel["rows"]
        assert len(prediction["forward_returns"]) == panel["rows"]

    @pytest.mark.parametrize("record", [False, True])
    def test_panel_prediction_cannot_be_evaluated_as_one_asset(
        self, service: MLService, record: bool
    ) -> None:
        panel = _build(service)["panel"]
        model = service.train_model(
            dataset_id=panel["dataset_id"],
            model_kind="xgboost_direction",
            parameters={"n_estimators": 5, "max_depth": 2},
        )["model"]
        prediction = service.predict(
            model_id=model["model_id"], dataset_id=panel["dataset_id"]
        )["prediction"]
        with pytest.raises(ValueError, match="single-symbol dataset"):
            service.evaluate_signal(
                prediction["signal"],
                prediction["forward_returns"],
                dataset_id=panel["dataset_id"],
                record=record,
                window=20,
            )
