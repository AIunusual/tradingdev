"""Real service processes share complete feature-cache publications."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity, data_fingerprint
from tradingdev.domain.ml.features.registry import get_feature_set
from tradingdev.domain.ml.registry import FeatureCacheStore

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import Event

_WAIT_SECONDS = 20
_CODE = "concurrent-service-features-v1"
_Started = list[tuple["BaseProcess", int]]


@dataclass
class _Signals:
    ready: Event
    load_started: Event
    loaded: Event
    save_started: Event
    published: Event
    finished: Event

    @classmethod
    def create(cls, context: SpawnContext) -> _Signals:
        return cls(*(context.Event() for _ in range(6)))


@contextmanager
def _process_workspace() -> Iterator[tuple[Path, _Started]]:
    """Reap only registered child handles before removing their workspace."""
    # This directory must not belong to tmp_path: its fixture would delete a
    # retained workspace even when a child could not be stopped here.
    root = Path(mkdtemp(prefix="cache-processes-"))
    started: _Started = []
    try:
        yield root, started
    finally:
        failures: list[Exception] = []
        for process, pid in started:
            try:
                assert process.pid == pid, "test process identity changed"
                if process.is_alive():
                    process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                assert not process.is_alive(), "test process cleanup failed"
                process.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup(
                f"cache test process cleanup failed; workspace retained at {root}",
                failures,
            )
        shutil.rmtree(root)


def _start(process: BaseProcess, started: _Started) -> None:
    process.start()
    assert process.pid is not None
    started.append((process, process.pid))


def _join(process: BaseProcess, timeout: float = _WAIT_SECONDS) -> None:
    process.join(timeout)
    if process.is_alive():
        raise TimeoutError("cache service process did not finish")
    assert process.exitcode == 0


def _build(service: MLService, symbol: str) -> dict[str, Any]:
    return service.build_dataset(
        symbol=symbol,
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=1,
    )


def _build_in_process(
    root: Path,
    symbol: str,
    max_entries: int,
    result_path: Path,
    signals: _Signals,
    resume_load: Event | None,
    resume_publication: Event | None,
) -> None:
    workspace = WorkspacePaths(root)

    class ObservedStore(FeatureCacheStore):
        def load(
            self, identity: FeatureCacheIdentity, *, feature_columns: Sequence[str]
        ) -> pd.DataFrame | None:
            signals.load_started.set()
            frame = super().load(identity, feature_columns=feature_columns)
            signals.loaded.set()
            if resume_load is not None and not resume_load.wait(_WAIT_SECONDS):
                raise TimeoutError("test did not release the service after load")
            return frame

        def save(
            self,
            identity: FeatureCacheIdentity,
            frame: pd.DataFrame,
            *,
            feature_columns: Sequence[str],
        ) -> Path:
            signals.save_started.set()
            return super().save(identity, frame, feature_columns=feature_columns)

    original_replace = os.replace

    def observed_replace(source: str | Path, destination: str | Path) -> None:
        original_replace(source, destination)
        destination = Path(destination)
        if (
            destination.parent == workspace.ml_feature_cache
            and destination.suffix == ".parquet"
        ):
            signals.published.set()
            if resume_publication is not None and not resume_publication.wait(
                _WAIT_SECONDS
            ):
                raise TimeoutError("test did not release the cache publication")

    store = ObservedStore(workspace.ml_feature_cache, max_entries=max_entries)
    service = MLService(workspace, feature_cache=store, feature_code_fingerprint=_CODE)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", observed_replace)
        signals.ready.set()
        result = _build(service, symbol)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        signals.finished.set()


def _seed(workspace: WorkspacePaths) -> dict[str, pd.DataFrame]:
    workspace.ensure()
    frames = {}
    for number, symbol in enumerate(("BTC/USDT", "ETH/USDT", "SOL/USDT")):
        index = np.arange(300, dtype=float)
        close = 100.0 + 30 * number + 10 * np.sin(index / 5) + index * 0.01
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=300, freq="h", tz="UTC"
                ),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(300, 10.0),
            }
        )
        seed_market_cache(workspace.processed_data, frame, symbol=symbol)
        frames[symbol] = frame
    return frames


def _verify_result(
    workspace: WorkspacePaths,
    result_path: Path,
    symbol: str,
    bars: pd.DataFrame,
    store: FeatureCacheStore,
) -> str:
    result = json.loads(result_path.read_text("utf-8"))
    assert result["success"] is True
    reported = result["dataset"]
    assert reported["features_from_cache"] is False
    dataset_id = str(reported["dataset_id"])
    frame, manifest = MLDatasetStore(workspace.ml_datasets).load(dataset_id)
    parquet = workspace.ml_datasets / dataset_id / "dataset.parquet"
    assert reported["path"] == str(parquet)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert digest == reported["content_hash"] == manifest["content_hash"]
    assert manifest["dataset_id"] == dataset_id
    assert manifest["symbol"] == symbol
    assert manifest["code_fingerprint"] == _CODE
    assert len(frame) == reported["rows"] == manifest["rows"]
    close_by_time = bars.set_index("timestamp")["close"]
    expected_returns = close_by_time.shift(-1) / close_by_time - 1
    np.testing.assert_allclose(
        frame["forward_return"], expected_returns.loc[frame["timestamp"]]
    )

    identity = FeatureCacheIdentity(
        symbol, "1h", data_fingerprint(bars), "returns", _CODE
    )
    features = get_feature_set("returns")
    cached = store.load(identity, feature_columns=features.columns)
    assert cached is not None
    pd.testing.assert_frame_equal(cached, bars.join(features.build(bars)))
    sidecar = json.loads(store.meta_path_for(identity.key).read_text("utf-8"))
    assert (
        sidecar["content_hash"]
        == hashlib.sha256(store.path_for(identity.key).read_bytes()).hexdigest()
    )
    return dataset_id


@pytest.mark.parametrize("same_key", [True, False], ids=["same-key", "different-keys"])
def test_service_processes_serialize_cache_publication_and_eviction(
    same_key: bool,
) -> None:
    context = multiprocessing.get_context("spawn")
    signals = [_Signals.create(context), _Signals.create(context)]
    resume_load = [context.Event(), context.Event()]
    resume_publication = context.Event()
    symbols = ("BTC/USDT", "BTC/USDT" if same_key else "ETH/USDT")
    max_entries = len(set(symbols))
    with _process_workspace() as (root, started):
        workspace = WorkspacePaths(root)
        bars = _seed(workspace)
        store = FeatureCacheStore(workspace.ml_feature_cache, max_entries=max_entries)
        service = MLService(
            workspace, feature_cache=store, feature_code_fingerprint=_CODE
        )
        previous = _build(service, "SOL/USDT")["dataset"]
        (previous_key,) = store.keys()
        preserved = {
            path: path.read_bytes()
            for path in (
                *workspace.processed_data.rglob("*.parquet"),
                *(workspace.ml_datasets / previous["dataset_id"]).iterdir(),
            )
        }
        results = [root / "first.json", root / "second.json"]
        workers = [
            context.Process(
                target=_build_in_process,
                args=(
                    root,
                    symbol,
                    max_entries,
                    results[number],
                    signals[number],
                    resume_load[number] if same_key else None,
                    resume_publication if number == 0 else None,
                ),
            )
            for number, symbol in enumerate(symbols)
        ]
        _start(workers[0], started)
        assert signals[0].ready.wait(_WAIT_SECONDS)
        if same_key:
            # Both service calls must miss before either begins writing; this
            # exercises two writers of the same key, not a subsequent hit.
            assert signals[0].loaded.wait(_WAIT_SECONDS)
            _start(workers[1], started)
            assert signals[1].loaded.wait(_WAIT_SECONDS)
            resume_load[0].set()
        assert signals[0].published.wait(_WAIT_SECONDS)
        partial = [
            path
            for path in workspace.ml_feature_cache.glob("*.parquet")
            if not path.with_suffix(".json").exists()
        ]
        assert len(partial) == 1
        partial_bytes = partial[0].read_bytes()
        if same_key:
            resume_load[1].set()
            assert signals[1].save_started.wait(_WAIT_SECONDS)
        else:
            _start(workers[1], started)
            assert signals[1].ready.wait(_WAIT_SECONDS)
            assert signals[1].load_started.wait(_WAIT_SECONDS)
            assert not signals[1].loaded.wait(0.2)
        assert not signals[1].published.wait(0.2)
        assert not signals[1].finished.is_set()
        assert partial[0].read_bytes() == partial_bytes
        resume_publication.set()
        for worker in workers:
            _join(worker)

        dataset_ids = [
            _verify_result(workspace, result, symbol, bars[symbol], store)
            for result, symbol in zip(results, symbols, strict=True)
        ]
        assert (dataset_ids[0] == dataset_ids[1]) is same_key
        assert {path: path.read_bytes() for path in preserved} == preserved
        cached_keys = store.keys()
        assert previous_key not in cached_keys
        assert len(cached_keys) == max_entries
        stats = store.stats()
        assert stats.entries == max_entries
        assert not stats.over_limit
        assert stats.total_bytes == sum(
            path.stat().st_size
            for path in workspace.ml_feature_cache.iterdir()
            if path.suffix in {".parquet", ".json"}
        )
        assert list(workspace.ml_feature_cache.glob("*.tmp")) == []
        for symbol, dataset_id in zip(symbols, dataset_ids, strict=True):
            again = _build(service, symbol)["dataset"]
            assert again["features_from_cache"] is True
            assert again["dataset_id"] == dataset_id


def _hold_workspace(root: Path, ready: Event, release: Event) -> None:
    (root / "worker-artifact").write_text("owned by this test", encoding="utf-8")
    ready.set()
    if not release.wait(_WAIT_SECONDS):
        raise TimeoutError("test did not release the cleanup worker")


@pytest.mark.parametrize("outcome", ["normal", "failure", "timeout"])
def test_process_workspace_cleanup_on_every_exit(outcome: str) -> None:
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    expected = (
        nullcontext()
        if outcome == "normal"
        else pytest.raises(RuntimeError if outcome == "failure" else TimeoutError)
    )
    with expected, _process_workspace() as (root, started):
        worker = context.Process(target=_hold_workspace, args=(root, ready, release))
        _start(worker, started)
        pid = worker.pid
        assert ready.wait(_WAIT_SECONDS)
        assert (root / "worker-artifact").is_file()
        if outcome == "failure":
            raise RuntimeError("injected assertion failure")
        if outcome == "timeout":
            # The worker is waiting on our event, so a zero-duration join is
            # deterministically a timeout without introducing a long sleep.
            _join(worker, timeout=0)
        release.set()
        _join(worker)
    assert not root.exists()
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(ValueError, match="closed"):
        worker.is_alive()


def test_failed_process_stop_preserves_workspace_after_outer_fixture_cleanup(
    tmp_path: Path,
) -> None:
    # The handle models an unkillable child without leaving a real process
    # behind. The other tests exercise actual spawn/terminate/join handles.
    worker = Mock(spec=multiprocessing.Process)
    worker.pid = 12345
    worker.is_alive.return_value = True
    stop_error = OSError("injected process stop failure")
    worker.terminate.side_effect = stop_error
    retained: Path | None = None
    try:
        with TemporaryDirectory(prefix="outer-fixture-", dir=tmp_path) as temporary:
            outer = Path(temporary)
            with (
                pytest.raises(ExceptionGroup) as failure,
                _process_workspace() as (root, started),
            ):
                retained = root
                (root / "worker-artifact").write_text("still in use", "utf-8")
                started.append((worker, worker.pid))
            assert failure.value.exceptions == (stop_error,)
            assert str(root) in str(failure.value)
            assert not root.is_relative_to(tmp_path)
            assert (root / "worker-artifact").read_text("utf-8") == "still in use"
        # Model the enclosing fixture's teardown explicitly. A failed stop
        # must leave the child workspace intact after that teardown too.
        assert not outer.exists()
        assert (root / "worker-artifact").read_text("utf-8") == "still in use"
        worker.terminate.assert_called_once_with()
        worker.close.assert_not_called()
    finally:
        # No child was spawned in this failure-injection case; this test owns
        # the deliberately retained directory and must remove it itself.
        if retained is not None and retained.exists():
            shutil.rmtree(retained)
    assert retained is not None and not retained.exists()
