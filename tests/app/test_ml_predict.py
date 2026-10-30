"""Saved-model predictions, evaluation hand-off and artifact integrity checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.registry import ModelArtifact, ModelArtifactStore

if TYPE_CHECKING:
    from pathlib import Path


def _write_cached_bars(workspace: WorkspacePaths, rows: int = 300) -> None:
    """Write one year of oscillating OHLCV bars under the canonical name."""
    index = np.arange(rows, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )
    seed_market_cache(workspace.processed_data, frame)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_cached_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


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


def _train(service: MLService, dataset_id: str, **overrides: object) -> dict[str, Any]:
    # Avoid underfitting to a constant signal, which cannot be evaluated.
    kwargs: dict[str, object] = {
        "dataset_id": dataset_id,
        "model_kind": "xgboost_direction",
        "parameters": {"n_estimators": 60, "max_depth": 3},
    }
    kwargs.update(overrides)
    return service.train_model(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def trained(service: MLService) -> tuple[str, str]:
    """Return ``(model_id, dataset_id)`` for a small trained direction model."""
    dataset_id = str(_build(service)["dataset"]["dataset_id"])
    model_id = str(_train(service, dataset_id)["model"]["model_id"])
    return model_id, dataset_id


class TestPredict:
    def test_signal_is_aligned_with_the_realised_returns(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        model_id, dataset_id = trained
        prediction = service.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        signal = prediction["signal"]
        realized = prediction["forward_returns"]
        assert len(signal) == len(realized) == prediction["n_rows"]
        assert prediction["n_rows"] == _build(service)["dataset"]["rows"]
        assert all(value in (-1.0, 0.0, 1.0) for value in signal)

    def test_reports_the_model_identity(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        model_id, dataset_id = trained
        prediction = service.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        assert prediction["model_id"] == model_id
        assert prediction["dataset_id"] == dataset_id
        assert prediction["model_name"] == "xgboost_direction"
        assert prediction["framework"] == "xgboost"
        assert prediction["horizon"] == 1

    def test_is_deterministic(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        model_id, dataset_id = trained
        first = service.predict(model_id=model_id, dataset_id=dataset_id)
        second = service.predict(model_id=model_id, dataset_id=dataset_id)
        assert first["prediction"]["signal"] == second["prediction"]["signal"]

    def test_feeds_straight_into_evaluate_signal(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        model_id, dataset_id = trained
        prediction = service.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        verdict = service.evaluate_signal(
            prediction["signal"],
            prediction["forward_returns"],
            horizon=prediction["horizon"],
        )
        assert verdict["success"] is True
        assert verdict["n_observations"] == prediction["n_rows"]
        assert set(verdict["metrics"])

    def test_the_prediction_fingerprint_identifies_the_signal(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        """A re-run must be checkable against the stored evaluation."""
        model_id, dataset_id = trained
        prediction = service.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        verdict = service.evaluate_signal(
            prediction["signal"],
            prediction["forward_returns"],
            horizon=prediction["horizon"],
            record=True,
            model_id=model_id,
            dataset_id=dataset_id,
        )
        assert prediction["signal_fingerprint"] == verdict["signal_fingerprint"]
        assert (
            verdict["recorded"]["signal_fingerprint"] == verdict["signal_fingerprint"]
        )
        stored = service.leaderboard("ic_mean")["ranked"][0]
        assert stored["entry_id"] == verdict["recorded"]["entry_id"]
        assert stored["signal_fingerprint"] == prediction["signal_fingerprint"]

    def test_the_paged_preview_keeps_the_full_signal_fingerprint(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        """Hashing the returned page instead of the signal would break checks."""
        model_id, dataset_id = trained
        full = service.predict(model_id=model_id, dataset_id=dataset_id)["prediction"]
        page = service.predict_page(model_id=model_id, dataset_id=dataset_id, limit=10)[
            "prediction"
        ]
        assert len(page["signal"]) == 10 < page["n_rows"]
        assert page["signal_fingerprint"] == full["signal_fingerprint"]

    @pytest.mark.parametrize(
        "model_kind",
        ["xgboost_direction", "lightgbm_direction", "catboost_direction"],
    )
    def test_every_zoo_kind_can_predict(
        self, service: MLService, model_kind: str
    ) -> None:
        dataset_id = str(_build(service)["dataset"]["dataset_id"])
        model_id = str(
            _train(service, dataset_id, model_kind=model_kind)["model"]["model_id"]
        )
        prediction = service.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        assert prediction["framework"] == model_kind.split("_")[0]
        assert len(prediction["signal"]) == prediction["n_rows"]


class TestPredictRejectsBadInput:
    def test_unknown_model_id_is_rejected(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        _, dataset_id = trained
        with pytest.raises(ValueError, match="unknown model_id"):
            service.predict(model_id="0" * 16, dataset_id=dataset_id)

    def test_malformed_model_id_cannot_escape_the_store(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        _, dataset_id = trained
        for bad in (
            "../../etc/passwd",
            "0" * 15,
            "ABCDEF0123456789",
            "",
            "0" * 16 + "0",
        ):
            with pytest.raises(ValueError, match="malformed model_id"):
                service.predict(model_id=bad, dataset_id=dataset_id)

    def test_unknown_dataset_id_is_rejected(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        model_id, _ = trained
        with pytest.raises(ValueError, match="unknown dataset_id"):
            service.predict(model_id=model_id, dataset_id="ds_" + "0" * 16)

    def test_feature_contract_mismatch_is_rejected(
        self, service: MLService, trained: tuple[str, str]
    ) -> None:
        """Reject mismatched feature columns before prediction."""
        model_id, _ = trained
        other_dataset = str(
            _build(service, feature_set="technical")["dataset"]["dataset_id"]
        )
        assert other_dataset != trained[1]
        with pytest.raises(ValueError, match="feature contract mismatch"):
            service.predict(model_id=model_id, dataset_id=other_dataset)

    def test_tampered_blob_is_refused(
        self, service: MLService, trained: tuple[str, str], workspace: WorkspacePaths
    ) -> None:
        model_id, dataset_id = trained
        blob = workspace.models / f"{model_id}.bin"
        blob.write_bytes(b"not the model that was registered")
        with pytest.raises(ValueError, match="tampered"):
            service.predict(model_id=model_id, dataset_id=dataset_id)


def test_predict_uses_the_exact_verified_bytes(
    service: MLService,
    trained: tuple[str, str],
    workspace: WorkspacePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id, dataset_id = trained
    expected = service.predict(model_id=model_id, dataset_id=dataset_id)
    original_load = ModelArtifactStore.load_with_bytes

    def load_and_replace(
        store: ModelArtifactStore, identifier: str
    ) -> tuple[ModelArtifact, bytes]:
        verified = original_load(store, identifier)
        store.path_for(identifier).write_bytes(b"changed after integrity verification")
        return verified

    monkeypatch.setattr(ModelArtifactStore, "load_with_bytes", load_and_replace)
    assert service.predict(model_id=model_id, dataset_id=dataset_id) == expected
    assert (workspace.models / f"{model_id}.bin").read_bytes() == (
        b"changed after integrity verification"
    )


def test_prediction_pages_reassemble_all_verified_rows(
    service: MLService, trained: tuple[str, str]
) -> None:
    model_id, dataset_id = trained
    identity = {"model_id": model_id, "dataset_id": dataset_id}
    complete = service.predict(**identity)["prediction"]
    collected: dict[str, list[float]] = {"signal": [], "forward_returns": []}
    offset = 0
    while True:
        page = service.predict_page(**identity, offset=offset, limit=37)["prediction"]
        assert page["model_id"] == model_id and page["dataset_id"] == dataset_id
        assert page["n_rows"] == complete["n_rows"]
        assert page["offset"] == offset and page["limit"] == 37
        for key, values in collected.items():
            assert len(page[key]) <= 37
            values.extend(page[key])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + 37
        offset = page["next_offset"]
    assert collected == {key: complete[key] for key in collected}
    end = service.predict_page(
        model_id=model_id, dataset_id=dataset_id, offset=complete["n_rows"]
    )["prediction"]
    assert end["signal"] == end["forward_returns"] == []
    assert end["next_offset"] is None
    with pytest.raises(ValueError, match="exceeds prediction row count"):
        service.predict_page(
            model_id=model_id, dataset_id=dataset_id, offset=complete["n_rows"] + 1
        )


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 10), (True, 10), (0.5, 10), (0, 0), (0, 501), (0, True), (0, 1.5)],
)
def test_invalid_preview_bounds_fail_before_loading_a_model(
    service: MLService, offset: Any, limit: Any
) -> None:
    with pytest.raises(ValueError, match="offset|limit"):
        service.predict_page(
            model_id="unknown", dataset_id="unknown", offset=offset, limit=limit
        )


def test_identity_evaluation_uses_all_rows_and_the_saved_horizon(
    service: MLService,
) -> None:
    dataset = _build(service, horizon=3)["dataset"]
    model_id = str(_train(service, dataset["dataset_id"])["model"]["model_id"])
    identity = {"model_id": model_id, "dataset_id": dataset["dataset_id"]}
    preview = service.predict_page(**identity, limit=5)["prediction"]
    complete = service.predict(**identity)["prediction"]
    evaluated = service.evaluate_prediction(**identity, window=20, n_quantiles=3)
    expected = service.evaluate_signal(
        complete["signal"],
        complete["forward_returns"],
        horizon=3,
        window=20,
        n_quantiles=3,
        **identity,
    )
    assert len(preview["signal"]) == 5 < evaluated["n_observations"]
    assert evaluated == {**expected, **identity}
    assert evaluated["n_observations"] == dataset["rows"]
    assert evaluated["horizon"] == 3
