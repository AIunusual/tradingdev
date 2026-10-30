"""One captured data load supplies isolated, reproducible candidate runs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
from tests.app.test_backtest_service import (
    _raw_config,
    _service,
    _SignalStrategy,
)

from tradingdev.app.data_service import DataService
from tradingdev.app.execution_dataset import ExecutionDataset, execution_data_binding
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.randomness import get_numpy_rng, get_random, get_seed
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.signals import SignalContractError

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.app.data_service import LoadedDataset
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig


def _candidate(
    manifest: ExecutionManifest, parameters: dict[str, Any]
) -> ExecutionManifest:
    config = manifest.config_copy()
    config["strategy"]["parameters"] = parameters
    return ExecutionManifest.create(
        kind=manifest.kind,
        config=config,
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs=parameters
        ),
    )


def test_capture_contains_real_merged_features_and_loads_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, data, loader, _strategy = _service(tmp_path)
    market = data.dataset.frame
    market.to_parquet(data.dataset.processed_path)
    feature_path = tmp_path / "feature.parquet"
    features = pd.DataFrame(
        {"timestamp": market["timestamp"], "funding_rate": [0.001] * len(market)}
    )
    features.to_parquet(feature_path)
    monkeypatch.setattr(
        data, "_load_market", lambda *_args: (market, data.dataset.processed_path)
    )

    def load(raw: dict[str, Any], config: BacktestConfig) -> LoadedDataset:
        data.loads.append(config)
        return DataService.load(data, raw, config)

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(
        loader, "create_from_execution", lambda *_args: _SignalStrategy()
    )
    config = _raw_config(
        data={
            "requirements": {
                "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                "features": [
                    {
                        "type": "funding_rate",
                        "source": "local",
                        "column": "funding_rate",
                        "path": str(feature_path),
                    }
                ],
            }
        }
    )
    manifest = service.prepare_execution(config, kind="backtest")
    assert data.loads == []
    snapshot = service.load_execution_dataset(manifest)
    expected = market.merge(features, on="timestamp", how="left")
    pd.testing.assert_frame_equal(snapshot.frame_for(manifest), expected)
    pd.testing.assert_frame_equal(pd.read_parquet(BytesIO(snapshot.payload)), expected)
    assert snapshot.content_sha256 == hashlib.sha256(snapshot.payload).hexdigest()
    assert snapshot.dataset_id == f"sha256:{snapshot.content_sha256}"
    assert snapshot.source_dataset_id.startswith("query-sha256:")
    assert snapshot.payload != data.dataset.processed_path.read_bytes()
    binding = json.loads(snapshot.data_binding_json)
    assert binding["data"] == manifest.config["data"]
    assert binding["market"]["symbol"] == "BTC/USDT"

    # Subsequent runs must use the published bytes, not either source file.
    feature_path.unlink()
    data.dataset.processed_path.unlink()
    for value in (1, 2):
        run = service.run_manifest(
            _candidate(manifest, {"value": value}), loaded_dataset=snapshot
        )
        assert run.dataset_id == snapshot.dataset_id
        assert run.processed_path == snapshot.processed_path
    assert len(data.loads) == 1


def test_saved_data_binding_remains_readable_without_applying_new_defaults(
    tmp_path: Path,
) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    expected = execution_data_binding(manifest)
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    payload["config"]["backtest"].pop("periods_per_year")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    historical = ExecutionManifest.model_validate(
        {**payload, "manifest_hash": hashlib.sha256(encoded.encode()).hexdigest()}
    )
    assert execution_data_binding(historical) == expected
    with pytest.raises(ManifestError, match="current runtime schema"):
        service.load_execution_dataset(historical)
    assert data.loads == []


@pytest.mark.parametrize(
    "field", ["data", "symbol", "timeframe", "start_date", "end_date"]
)
def test_saved_data_binding_rejects_incomplete_settings(
    tmp_path: Path, field: str
) -> None:
    service, _data, _loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    if field == "data":
        payload["config"].pop(field)
    else:
        payload["config"]["backtest"].pop(field)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    historical = ExecutionManifest.model_validate(
        {**payload, "manifest_hash": hashlib.sha256(encoded.encode()).hexdigest()}
    )
    with pytest.raises(ManifestError, match="complete saved data settings"):
        execution_data_binding(historical)


def test_snapshot_preserves_index_nullable_dtypes_and_object_isolation(
    tmp_path: Path,
) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    frame = data.dataset.frame
    frame["nullable"] = pd.Series([1, None] * 12, dtype="Int64")
    frame["category"] = pd.Categorical(["up", "down"] * 12)
    frame["object_feature"] = pd.Series(
        [{"value": value} for value in range(len(frame))], dtype=object
    )
    frame.index = pd.Index(range(100, 124), name="observation")
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    snapshot = service.load_execution_dataset(manifest)
    first = snapshot.frame_for(manifest)
    pd.testing.assert_frame_equal(first, frame)
    cast("dict[str, int]", first.at[100, "object_feature"])["value"] = -1
    first.loc[:, "close"] = -100.0
    cast("dict[str, int]", frame.at[100, "object_feature"])["value"] = -2
    fresh = snapshot.frame_for(manifest)
    assert fresh.at[100, "object_feature"] == {"value": 0}
    assert fresh.at[100, "close"] == 100.0


@pytest.mark.parametrize("fail_after_mutation", [False, True])
def test_candidate_mutation_cannot_contaminate_next_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_after_mutation: bool,
) -> None:
    service, data, loader, _strategy = _service(tmp_path)
    data.dataset.frame["object_feature"] = pd.Series(
        [{"value": value} for value in range(len(data.dataset.frame))], dtype=object
    )
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    snapshot = service.load_execution_dataset(manifest)
    strategies: list[_SignalStrategy] = []
    engines: list[BaseBacktestEngine] = []
    frames: list[pd.DataFrame] = []

    class Candidate(_SignalStrategy):
        def __init__(self, mutate: bool) -> None:
            super().__init__()
            self.mutate = mutate

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            if "object_feature" not in df:
                return super().generate_signals(df)
            frames.append(df)
            assert df.at[0, "close"] == 100.0
            assert df.at[0, "object_feature"] == {"value": 0}
            if self.mutate:
                cast("dict[str, int]", df.at[0, "object_feature"])["value"] = -1
                df.loc[:, "close"] = 200.0
                if fail_after_mutation:
                    raise ValueError("candidate failed after mutation")
            return super().generate_signals(df)

    def create(
        _config: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        _parallel: ParallelConfig,
    ) -> Candidate:
        candidate = Candidate(bool(execution.constructor_kwargs["mutate"]))
        if engine is not None:
            strategies.append(candidate)
            engines.append(engine)
        return candidate

    monkeypatch.setattr(loader, "create_from_execution", create)
    mutating = _candidate(manifest, {"mutate": True})
    healthy = _candidate(manifest, {"mutate": False})
    if fail_after_mutation:
        with pytest.raises(ValueError, match="failed after mutation"):
            service.run_manifest(mutating, loaded_dataset=snapshot)
    else:
        with pytest.raises(SignalContractError, match="input_mutated"):
            service.run_manifest(mutating, loaded_dataset=snapshot)
    service.run_manifest(healthy, loaded_dataset=snapshot)
    assert len(data.loads) == 1
    assert len(strategies) == len(engines) == len(frames) == 2
    assert strategies[0] is not strategies[1]
    assert engines[0] is not engines[1]
    assert frames[0] is not frames[1]
    pd.testing.assert_frame_equal(snapshot.frame_for(manifest), data.dataset.frame)


@pytest.mark.parametrize(
    "change",
    ["period", "symbol", "timeframe", "source", "requirements", "cache_path"],
)
def test_snapshot_rejects_different_data_bindings_without_loading(
    tmp_path: Path, change: str
) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    snapshot = service.load_execution_dataset(manifest)
    config = manifest.config_copy()
    if change == "period":
        config["backtest"]["end_date"] = "2024-01-03"
    elif change in {"symbol", "timeframe"}:
        value = "ETH/USDT" if change == "symbol" else "2h"
        config["backtest"][change] = value
        config["data"]["requirements"]["market"][change] = value
    elif change == "source":
        config["data"]["requirements"]["market"]["source"] = "binance_api"
    elif change == "requirements":
        config["data"]["requirements"]["features"] = [
            {
                "type": "custom",
                "source": "local",
                "column": "new_feature",
                "path": str(tmp_path / "new.parquet"),
                "raw_path": None,
            }
        ]
    else:
        config["data"]["processed_dir"] = str(tmp_path / "other")
    other = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=manifest.strategy_execution
    )
    with pytest.raises(ManifestError, match="data settings and research period"):
        service.run_manifest(other, loaded_dataset=snapshot)
    assert len(data.loads) == 1


@pytest.mark.parametrize("failure", ["manifest", "strategy", "contract"])
def test_capture_checks_execution_before_loading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    service, data, loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    expected: type[Exception] = ManifestError
    if failure == "manifest":
        manifest.config["random_seed"] = 99
    elif failure == "strategy":
        (tmp_path / "fixture.py").write_text("# changed source\n")
        expected = StrategyNotExecutableError
    else:

        class InvalidStrategy(_SignalStrategy):
            def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
                result = super().generate_signals(df)
                result["signal"] = 99
                return result

        monkeypatch.setattr(
            loader, "create_from_execution", lambda *_args: InvalidStrategy()
        )
    with pytest.raises(expected):
        service.load_execution_dataset(manifest)
    assert data.loads == []


@pytest.mark.parametrize(
    "unsupported", ["python_object", "lossy_dtype", "list_conversion"]
)
def test_capture_rejects_unsupported_or_lossy_parquet_conversion(
    tmp_path: Path, unsupported: str
) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    if unsupported == "python_object":
        data.dataset.frame["unsupported"] = pd.Series(
            [object()] * len(data.dataset.frame), dtype=object
        )
    elif unsupported == "lossy_dtype":
        data.dataset.frame["unsupported"] = pd.Series(
            [1] * len(data.dataset.frame), dtype=object
        )
    else:
        data.dataset.frame["unsupported"] = pd.Series(
            [[1, 2]] * len(data.dataset.frame), dtype=object
        )
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    with pytest.raises((TypeError, ValueError)):
        service.load_execution_dataset(manifest)


def test_content_identity_changes_when_merged_feature_values_change(
    tmp_path: Path,
) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    data.dataset.frame["feature"] = 1.0
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    first = service.load_execution_dataset(manifest)
    data.dataset.frame["feature"] = 2.0
    second = service.load_execution_dataset(manifest)
    assert first.source_dataset_id == second.source_dataset_id
    assert first.data_binding_json == second.data_binding_json
    assert first.content_sha256 != second.content_sha256
    assert first.dataset_id != second.dataset_id


def test_shared_snapshot_replays_rng_independently_of_candidate_order_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, data, loader, _strategy = _service(tmp_path)
    data.dataset.frame["real_data"] = True
    manifest = service.prepare_execution(_raw_config(random_seed=42), kind="backtest")
    snapshot = service.load_execution_dataset(manifest)
    events: list[tuple[str, int | None, float]] = []

    class RandomCandidate(_SignalStrategy):
        def __init__(self, fail: bool) -> None:
            super().__init__()
            self.fail = fail
            events.append(("constructor", get_seed(), get_random().random()))

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            events.append(("signals", get_seed(), float(get_numpy_rng().random())))
            if "real_data" in df and self.fail:
                raise ValueError("random candidate failure")
            return super().generate_signals(df)

    monkeypatch.setattr(
        loader,
        "create_from_execution",
        lambda _config, execution, *_args: RandomCandidate(
            bool(execution.constructor_kwargs["fail"])
        ),
    )
    healthy = _candidate(manifest, {"fail": False})
    failing = _candidate(manifest, {"fail": True})
    service.run_manifest(healthy, loaded_dataset=snapshot)
    first = deepcopy(events)
    with pytest.raises(ValueError, match="random candidate failure"):
        service.run_manifest(failing, loaded_dataset=snapshot)
    config = healthy.config_copy()
    config["random_seed"] = 7
    different_seed = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=healthy.strategy_execution
    )
    events.clear()
    service.run_manifest(different_seed, loaded_dataset=snapshot)
    assert events != first
    events.clear()
    service.run_manifest(healthy, loaded_dataset=snapshot)
    assert events == first
    assert len(data.loads) == 1
    with pytest.raises(RuntimeError, match="active strategy execution"):
        get_seed()


def test_walk_forward_can_reuse_captured_data(tmp_path: Path) -> None:
    service, data, _loader, _strategy = _service(tmp_path, rows=100)
    manifest = service.prepare_execution(
        _raw_config(
            validation={"mode": "automatic", "n_splits": 2, "train_ratio": 0.5}
        ),
        kind="walk_forward",
    )
    snapshot = service.load_execution_dataset(manifest)
    run = service.run_manifest(manifest, loaded_dataset=snapshot)
    assert run.mode == "walk_forward"
    assert len(run.pipeline.fold_results) == 2
    assert run.dataset_id == snapshot.dataset_id
    assert len(data.loads) == 1


def test_payload_is_immutable_even_when_constructed_from_mutable_bytes(
    tmp_path: Path,
) -> None:
    service, _data, _loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    snapshot = service.load_execution_dataset(manifest)
    payload = bytearray(snapshot.payload)
    copied = ExecutionDataset(
        payload=payload,  # type: ignore[arg-type]
        data_binding_json=snapshot.data_binding_json,
        processed_path=snapshot.processed_path,
        source_dataset_id=snapshot.source_dataset_id,
    )
    payload[0] = 0
    assert copied.payload == snapshot.payload
    assert copied.content_sha256 == snapshot.content_sha256
