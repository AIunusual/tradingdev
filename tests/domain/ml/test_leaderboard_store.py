"""Leaderboard persistence must not drop or duplicate candidates:
their count determines the significance threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    build_entry_id,
)
from tradingdev.domain.ml.registry import LeaderboardStore

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = EvaluationContext("BTC/USDT", "1d", "2024-01-01", "2024-12-31", 1)


def _entry(model_id: str, value: float, *, dataset_id: str = "ds") -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=build_entry_id(
            model_id=model_id, dataset_id=dataset_id, context=_CONTEXT
        ),
        label=model_id,
        context=_CONTEXT,
        metrics={"icir": value},
        n_observations=200,
        model_id=model_id,
        dataset_id=dataset_id,
    )


@pytest.fixture
def store(tmp_path: Path) -> LeaderboardStore:
    return LeaderboardStore(tmp_path / "leaderboard")


def test_an_empty_store_has_no_entries(store: LeaderboardStore) -> None:
    assert store.entries() == ()


def test_recording_twice_updates_one_row(store: LeaderboardStore) -> None:
    entry = _entry("m", 0.5)
    assert store.record(entry) is False
    updated = LeaderboardEntry(
        entry_id=entry.entry_id,
        label="m",
        context=_CONTEXT,
        metrics={"icir": 0.9},
        n_observations=200,
        model_id="m",
        dataset_id="ds",
    )
    assert store.record(updated) is True
    entries = store.entries()
    assert len(entries) == 1
    assert entries[0].metrics["icir"] == pytest.approx(0.9)


def test_records_are_stored_in_a_stable_order(store: LeaderboardStore) -> None:
    store.record(_entry("b", 0.2))
    first = store.path.read_text("utf-8")
    fresh = LeaderboardStore(store.root)
    fresh.record(_entry("a", 0.9))
    second = fresh.path.read_text("utf-8")
    assert second.splitlines() == sorted(second.splitlines())
    assert first != second
    identifiers = [entry.entry_id for entry in fresh.entries()]
    assert identifiers == sorted(identifiers)
    assert len(identifiers) == 2


def test_filters_narrow_by_measurement_conditions(store: LeaderboardStore) -> None:
    store.record(_entry("a", 0.5))
    other = LeaderboardEntry(
        entry_id="other",
        label="other",
        context=EvaluationContext("ETH/USDT", "1d", "2024-01-01", "2024-12-31", 1),
        metrics={"icir": 0.4},
        n_observations=200,
    )
    store.record(other)
    assert len(store.entries()) == 2
    assert [entry.model_id for entry in store.entries(symbol="ETH/USDT")] == [""]
    assert len(store.entries(horizon=1)) == 2
    assert store.entries(horizon=5) == ()


def test_dataset_filter_keeps_all_models_for_exact_identity(
    store: LeaderboardStore,
) -> None:
    for model_id, dataset_id in (("a", "ds_old"), ("b", "ds_old"), ("a", "ds_new")):
        store.record(_entry(model_id, 0.5, dataset_id=dataset_id))
    entries = store.entries(dataset_id="ds_old", symbol="BTC/USDT")
    assert {entry.model_id for entry in entries} == {"a", "b"}
    assert {entry.dataset_id for entry in entries} == {"ds_old"}
    assert store.entries(dataset_id="ds_old", symbol="ETH/USDT") == ()
    assert store.entries(dataset_id="ds_missing") == ()


def test_a_corrupt_record_is_refused_not_skipped(store: LeaderboardStore) -> None:
    store.record(_entry("a", 0.5))
    store.path.write_text(store.path.read_text("utf-8") + "{not json}\n", "utf-8")
    with pytest.raises(ValueError, match="unreadable leaderboard record on line 2"):
        store.entries()


def test_a_record_without_metrics_is_refused(store: LeaderboardStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"entry_id": "a", "context": {}}\n', "utf-8")
    with pytest.raises(ValueError, match="unreadable leaderboard record"):
        store.entries()
