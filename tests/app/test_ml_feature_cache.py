"""Feature-cache hits, invalidation and eviction preserve computed datasets."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from pyarrow import parquet as pq

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.registry import FeatureCacheStore

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300


def _bars(offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01 + offset
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(_ROWS, 10.0),
        }
    )


def _write_bars(workspace: WorkspacePaths, symbol: str, offset: float = 0.0) -> Path:
    path = workspace.processed_data / (
        symbol.lower().replace("/", "") + "_1h_2024.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _bars(offset).to_parquet(path, index=False)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths, "BTC/USDT")
    _write_bars(paths, "ETH/USDT")
    return paths


def _service(
    workspace: WorkspacePaths, *, code_fingerprint: str | None = None
) -> MLService:
    return MLService(workspace=workspace, feature_code_fingerprint=code_fingerprint)


def _build(service: MLService, **overrides: object) -> dict[str, Any]:
    kwargs: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 1,
    }
    kwargs.update(overrides)
    return service.build_dataset(**kwargs)  # type: ignore[arg-type]


def _rows(workspace: WorkspacePaths, result: dict[str, Any]) -> pd.DataFrame:
    dataset_id = str(result["dataset"]["dataset_id"])
    return pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")


def _features(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(("ret_", "log_"))]


class TestReuse:
    def test_the_first_build_computes_and_the_next_one_reuses(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        assert _build(service)["dataset"]["features_from_cache"] is False
        assert _build(service)["dataset"]["features_from_cache"] is True

    def test_a_reused_matrix_is_identical_to_a_recomputed_one(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        computed = _rows(workspace, _build(service))
        reused = _rows(workspace, _build(service))
        # An empty cache forces an independent recomputation.
        empty = FeatureCacheStore(workspace.root / "elsewhere")
        assert empty.keys() == ()
        forced = _rows(
            workspace,
            _build(MLService(workspace=workspace, feature_cache=empty)),
        )
        pd.testing.assert_frame_equal(reused, computed)
        pd.testing.assert_frame_equal(forced, computed)

    def test_the_cache_outlives_the_service_instance(
        self, workspace: WorkspacePaths
    ) -> None:
        _build(_service(workspace))
        assert _build(_service(workspace))["dataset"]["features_from_cache"] is True

    def test_reuse_is_reported_per_build_not_stored(
        self, workspace: WorkspacePaths
    ) -> None:
        """The manifest is provenance; whether the work was saved is not."""
        service = _service(workspace)
        result = _build(service)
        dataset_id = str(result["dataset"]["dataset_id"])
        manifest = (workspace.ml_datasets / dataset_id / "manifest.json").read_text(
            "utf-8"
        )
        assert "features_from_cache" not in manifest
        assert result["dataset"]["features_from_cache"] is False


class TestStaleBars:
    @pytest.mark.parametrize(
        "columns", [None, [None], pytest.param("missing", id="missing-columns")]
    )
    def test_invalid_pandas_metadata_rebuilds_without_replacing_sources_or_datasets(
        self, workspace: WorkspacePaths, columns: object
    ) -> None:
        service = _service(workspace)
        first = _build(service)
        dataset_id = first["dataset"]["dataset_id"]
        dataset_root = workspace.ml_datasets / dataset_id
        retained = {
            path: path.read_bytes()
            for path in (
                *workspace.processed_data.glob("*.parquet"),
                *dataset_root.iterdir(),
            )
            if path.is_file()
        }
        store = FeatureCacheStore(workspace.ml_feature_cache)
        (key,) = store.keys()
        matrix = store.path_for(key)
        table = pq.read_table(matrix)
        metadata = dict(table.schema.metadata)
        pandas_metadata = json.loads(metadata[b"pandas"])
        if columns == "missing":
            del pandas_metadata["columns"]
        else:
            pandas_metadata["columns"] = columns
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), matrix)
        sidecar = store.meta_path_for(key)
        record = json.loads(sidecar.read_text("utf-8"))
        record["content_hash"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(record), "utf-8")

        rebuilt = _build(service)

        assert rebuilt["dataset"]["features_from_cache"] is False
        assert rebuilt["dataset"]["dataset_id"] == dataset_id
        assert rebuilt["dataset"]["content_hash"] == first["dataset"]["content_hash"]
        assert {path: path.read_bytes() for path in retained} == retained
        assert _build(service)["dataset"]["features_from_cache"] is True

    @pytest.mark.parametrize("field", ["rows", "columns", "feature_columns"])
    def test_malformed_sidecars_rebuild_without_replacing_sources_or_datasets(
        self, workspace: WorkspacePaths, field: str
    ) -> None:
        service = _service(workspace)
        first = _build(service)
        dataset_id = first["dataset"]["dataset_id"]
        dataset_root = workspace.ml_datasets / dataset_id
        retained = {
            path: path.read_bytes()
            for path in (
                *workspace.processed_data.glob("*.parquet"),
                *dataset_root.iterdir(),
            )
            if path.is_file()
        }
        store = FeatureCacheStore(workspace.ml_feature_cache)
        (key,) = store.keys()
        sidecar = store.meta_path_for(key)
        record = json.loads(sidecar.read_text("utf-8"))
        record[field] = "invalid" if field == "rows" else None
        sidecar.write_text(json.dumps(record), encoding="utf-8")

        rebuilt = _build(service)

        assert rebuilt["dataset"]["features_from_cache"] is False
        assert rebuilt["dataset"]["dataset_id"] == dataset_id
        assert rebuilt["dataset"]["content_hash"] == first["dataset"]["content_hash"]
        assert {path: path.read_bytes() for path in retained} == retained
        assert _build(service)["dataset"]["features_from_cache"] is True

    def test_edited_bars_force_a_recompute(self, workspace: WorkspacePaths) -> None:
        service = _service(workspace)
        before = _rows(workspace, _build(service))
        _write_bars(workspace, "BTC/USDT", offset=25.0)
        result = _build(service)
        assert result["dataset"]["features_from_cache"] is False
        after = _rows(workspace, result)
        assert not np.allclose(
            after["log_return"].to_numpy(), before["log_return"].to_numpy()
        )

    def test_an_edit_that_keeps_the_file_size_is_still_seen(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        before = _rows(workspace, _build(service))
        path = workspace.processed_data / "btcusdt_1h_2024.parquet"
        size_before = path.stat().st_size
        bars = pd.read_parquet(path)
        # Preserve file size but change observable values after warmup.
        bars.loc[100, "close"], bars.loc[101, "close"] = (
            bars.loc[101, "close"],
            bars.loc[100, "close"],
        )
        bars.to_parquet(path, index=False)
        assert path.stat().st_size == size_before
        result = _build(service)
        assert result["dataset"]["features_from_cache"] is False
        after = _rows(workspace, result)
        assert not np.allclose(
            after["log_return"].to_numpy(), before["log_return"].to_numpy()
        )

    def test_a_tampered_cache_entry_still_yields_the_correct_dataset(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        correct = _rows(workspace, _build(service))
        store = FeatureCacheStore(workspace.ml_feature_cache)
        (key,) = store.keys()
        tampered = pd.read_parquet(store.path_for(key))
        tampered["log_return"] = tampered["log_return"] + 5.0
        tampered.to_parquet(store.path_for(key), index=False)
        result = _build(service)
        assert result["dataset"]["features_from_cache"] is False
        pd.testing.assert_frame_equal(_rows(workspace, result), correct)


class TestIdentity:
    def test_identical_bars_under_another_symbol_are_not_reused(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        assert _build(service, symbol="BTC/USDT")["dataset"]["features_from_cache"] is (
            False
        )
        assert _build(service, symbol="BTC/USDT")["dataset"]["features_from_cache"] is (
            True
        )
        assert _build(service, symbol="ETH/USDT")["dataset"]["features_from_cache"] is (
            False
        )
        assert _build(service, symbol="ETH/USDT")["dataset"]["features_from_cache"] is (
            True
        )
        assert len(FeatureCacheStore(workspace.ml_feature_cache).keys()) == 2

    def test_a_different_feature_set_recomputes(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        assert (
            _build(service, feature_set="returns")["dataset"]["features_from_cache"]
            is False
        )
        assert (
            _build(service, feature_set="returns")["dataset"]["features_from_cache"]
            is True
        )
        assert (
            _build(service, feature_set="technical")["dataset"]["features_from_cache"]
            is False
        )
        assert (
            _build(service, feature_set="technical")["dataset"]["features_from_cache"]
            is True
        )

    def test_a_different_horizon_reuses_the_features_but_not_the_label(
        self, workspace: WorkspacePaths
    ) -> None:
        """Only the features are cached; the label is per horizon."""
        service = _service(workspace)
        first = _build(service, horizon=1)
        assert first["dataset"]["features_from_cache"] is False
        second = _build(service, horizon=5)
        assert second["dataset"]["features_from_cache"] is True
        assert second["dataset"]["dataset_id"] != first["dataset"]["dataset_id"]
        one = _rows(workspace, first)
        five = _rows(workspace, second)
        # Longer horizons drop more unlabelled tail rows; compare the shared prefix.
        assert len(five) < len(one)
        assert not np.allclose(
            five["forward_return"].to_numpy()[: len(five)],
            one["forward_return"].to_numpy()[: len(five)],
        )
        assert np.allclose(
            five["log_return"].to_numpy()[: len(five)],
            one["log_return"].to_numpy()[: len(five)],
        )


class TestStaleCode:
    def test_a_changed_code_fingerprint_recomputes(
        self, workspace: WorkspacePaths
    ) -> None:
        before = _service(workspace, code_fingerprint="v1")
        assert _build(before)["dataset"]["features_from_cache"] is False
        assert _build(before)["dataset"]["features_from_cache"] is True
        after = _service(workspace, code_fingerprint="v2")
        assert _build(after)["dataset"]["features_from_cache"] is False
        assert _build(after)["dataset"]["features_from_cache"] is True

    def test_the_fingerprint_is_the_only_thing_that_notices_a_code_change(
        self,
        workspace: WorkspacePaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Code edits reuse stale features when fingerprints stay unchanged."""
        from tradingdev.domain.ml.features import registry as feature_registry
        from tradingdev.domain.ml.features.registry import FeatureSet

        original = feature_registry.get_feature_set("returns")
        service = _service(workspace, code_fingerprint="v1")
        before = _rows(workspace, _build(service))

        def _constant_features(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {column: np.zeros(len(frame)) for column in original.columns},
                index=frame.index,
            )

        monkeypatch.setitem(
            feature_registry._REGISTRY,
            "returns",
            FeatureSet(
                name=original.name,
                description=original.description,
                columns=original.columns,
                build=_constant_features,
            ),
        )

        stale = _build(service)
        assert stale["dataset"]["features_from_cache"] is True
        pd.testing.assert_frame_equal(_rows(workspace, stale), before)

        fixed = _service(workspace, code_fingerprint="v2")
        recomputed = _build(fixed)
        assert recomputed["dataset"]["features_from_cache"] is False
        after = _rows(workspace, recomputed)
        assert np.allclose(after["log_return"].to_numpy(), 0.0)


class TestPanel:
    def _build_panel(self, service: MLService, **overrides: object) -> dict[str, Any]:
        kwargs: dict[str, object] = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "feature_set": "returns",
            "horizon": 1,
        }
        kwargs.update(overrides)
        return service.build_panel(**kwargs)  # type: ignore[arg-type]

    def test_every_symbol_is_reused_on_the_second_build(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        first = self._build_panel(service)
        assert first["panel"]["features_from_cache"] is False
        second = self._build_panel(service)
        assert second["panel"]["features_from_cache"] is True
        assert second["panel"]["dataset_id"] == first["panel"]["dataset_id"]
        assert second["panel"]["content_hash"] == first["panel"]["content_hash"]

    def test_one_changed_symbol_makes_the_panel_recompute(
        self, workspace: WorkspacePaths
    ) -> None:
        service = _service(workspace)
        self._build_panel(service)
        _write_bars(workspace, "ETH/USDT", offset=25.0)
        assert self._build_panel(service)["panel"]["features_from_cache"] is False

    def test_a_panel_reuses_the_work_a_single_symbol_dataset_already_did(
        self, workspace: WorkspacePaths
    ) -> None:
        """Panels reuse the same per-symbol feature computation."""
        service = _service(workspace)
        assert (
            _build(service, symbol="BTC/USDT")["dataset"]["features_from_cache"]
            is False
        )
        assert (
            _build(service, symbol="ETH/USDT")["dataset"]["features_from_cache"]
            is False
        )
        assert self._build_panel(service)["panel"]["features_from_cache"] is True
        assert len(FeatureCacheStore(workspace.ml_feature_cache).keys()) == 2


class TestEviction:
    """Bounded caches must produce the same datasets as recomputation."""

    @staticmethod
    def _bounded(workspace: WorkspacePaths, max_entries: int) -> MLService:
        return MLService(
            workspace=workspace,
            feature_cache=FeatureCacheStore(
                workspace.ml_feature_cache, max_entries=max_entries
            ),
        )

    @staticmethod
    def _panel(service: MLService) -> dict[str, Any]:
        return service.build_panel(
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            feature_set="returns",
            horizon=1,
        )

    def test_the_cache_stays_within_its_bound(self, workspace: WorkspacePaths) -> None:
        service = self._bounded(workspace, 2)
        _build(service)
        _build(service, symbol="ETH/USDT")
        _build(service, feature_set="technical")
        store = FeatureCacheStore(workspace.ml_feature_cache)
        assert len(store.keys()) == 2
        assert store.stats().over_limit is False

    def test_an_evicted_matrix_is_recomputed_not_substituted(
        self, workspace: WorkspacePaths
    ) -> None:
        service = self._bounded(workspace, 1)
        first = _rows(workspace, _build(service))
        _build(service, symbol="ETH/USDT")
        again = _build(service)

        assert again["dataset"]["features_from_cache"] is False
        pd.testing.assert_frame_equal(_rows(workspace, again), first)

    def test_a_bounded_cache_builds_what_an_unbounded_one_builds(
        self, workspace: WorkspacePaths, tmp_path: Path
    ) -> None:
        roomy = WorkspacePaths(tmp_path / "unbounded")
        roomy.ensure()
        _write_bars(roomy, "BTC/USDT")
        reference = _rows(roomy, _build(_service(roomy)))

        squeezed = _rows(workspace, _build(self._bounded(workspace, 1)))

        pd.testing.assert_frame_equal(squeezed, reference)

    def test_a_panel_survives_a_cache_too_small_to_hold_it(
        self, workspace: WorkspacePaths, tmp_path: Path
    ) -> None:
        """A one-entry cache must still produce the complete two-symbol panel."""
        roomy = WorkspacePaths(tmp_path / "unbounded-panel")
        roomy.ensure()
        _write_bars(roomy, "BTC/USDT")
        _write_bars(roomy, "ETH/USDT")
        reference = self._panel(_service(roomy))["panel"]

        squeezed = self._panel(self._bounded(workspace, 1))["panel"]

        assert squeezed["features_from_cache"] is False
        assert squeezed["content_hash"] == reference["content_hash"]
        assert squeezed["rows"] == reference["rows"]
        assert len(FeatureCacheStore(workspace.ml_feature_cache).keys()) == 1
