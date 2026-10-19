"""Concurrent and failed leaderboard writes preserve published evidence."""

from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING

import pytest
from filelock import FileLock, Timeout

from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.registry.leaderboard_store import LeaderboardStore

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event
    from pathlib import Path


def _entry(identifier: str) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=identifier,
        label=identifier,
        context=EvaluationContext("BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1),
        metrics={"icir": 0.5},
        n_observations=100,
    )


def _record_in_process(
    root: Path, identifier: str, ready: Event, read_started: Event, release: Event
) -> None:
    class PausedStore(LeaderboardStore):
        def _read(self) -> list[LeaderboardEntry]:
            entries = super()._read()
            read_started.set()
            if not release.wait(15):
                raise TimeoutError("test did not release the leaderboard writer")
            return entries

    store = PausedStore(root)
    ready.set()
    store.record(_entry(identifier))


def test_concurrent_process_writers_read_after_the_previous_publication(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    ready = [context.Event(), context.Event()]
    read_started = [context.Event(), context.Event()]
    root = tmp_path / "leaderboard"
    workers = [
        context.Process(
            target=_record_in_process,
            args=(root, str(index), ready[index], read_started[index], release),
        )
        for index in range(2)
    ]
    started = []
    try:
        workers[0].start()
        started.append(workers[0])
        assert ready[0].wait(15)
        assert read_started[0].wait(15)
        workers[1].start()
        started.append(workers[1])
        assert ready[1].wait(15)
        # The first writer has read and holds the lock. The second must wait
        # before reading; otherwise it would publish a stale snapshot.
        assert not read_started[1].wait(0.5)
        release.set()
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
        assert {entry.entry_id for entry in LeaderboardStore(root).entries()} == {
            "0",
            "1",
        }
        assert list(root.glob("*.tmp")) == []
    finally:
        release.set()
        # These handles identify only processes started by this test. Reap
        # them on assertion failure or timeout before tmp_path is removed.
        for worker in started:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join(5)
            assert not worker.is_alive(), "leaderboard test process cleanup failed"
            worker.close()


@pytest.mark.parametrize("operation", ["replace", "fsync"])
def test_failed_publication_cleans_staging_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = LeaderboardStore(tmp_path)
    store.record(_entry("published"))
    before = store.path.read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, operation, fail)
        with pytest.raises(OSError, match="injected disk failure"):
            store.record(_entry("failed"))
    assert store.path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
    # A fresh instance must acquire the released lock immediately.
    LeaderboardStore(tmp_path, lock_timeout=0).record(_entry("retried"))
    assert {entry.entry_id for entry in store.entries()} == {"published", "retried"}


def test_lock_timeout_preserves_published_data_and_creates_no_staging(
    tmp_path: Path,
) -> None:
    store = LeaderboardStore(tmp_path, lock_timeout=0)
    store.record(_entry("published"))
    before = store.path.read_bytes()
    with FileLock(tmp_path / ".leaderboard.lock"), pytest.raises(Timeout):
        store.record(_entry("timed-out"))
    assert store.path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
    store.record(_entry("retried"))
    assert len(store.entries()) == 2
