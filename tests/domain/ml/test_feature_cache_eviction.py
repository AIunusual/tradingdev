"""Bounded cache eviction preserves surviving values and permits recomputation.

An injected clock makes LRU order independent of execution speed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity
from tradingdev.domain.ml.registry import FeatureCacheStore

if TYPE_CHECKING:
    from pathlib import Path

_COLUMNS = ("feat_a", "feat_b")


class _Clock:
    """A clock that only moves when a test moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, step: float = 1.0) -> None:
        self.now += step


def _frame(tag: float) -> pd.DataFrame:
    """A frame whose values identify it, so a wrong hit is visible."""
    return pd.DataFrame(
        {
            "close": [100.0 + tag, 101.0 + tag, 102.0 + tag],
            "feat_a": [tag, tag + 1.0, tag + 2.0],
            "feat_b": [-tag, -tag - 1.0, -tag - 2.0],
        }
    )


def _identity(**overrides: object) -> FeatureCacheIdentity:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "data_fingerprint": "bars-1",
        "feature_set": "returns",
        "code_fingerprint": "code-v1",
    }
    values.update(overrides)
    return FeatureCacheIdentity(**values)  # type: ignore[arg-type]


def _store(
    root: Path,
    *,
    max_entries: int | None = None,
    max_bytes: int | None = None,
    clock: _Clock | None = None,
) -> FeatureCacheStore:
    return FeatureCacheStore(
        root,
        max_entries=max_entries,
        max_bytes=max_bytes,
        clock=clock,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "feature_cache"


class TestBounds:
    def test_an_unbounded_cache_keeps_everything(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, clock=clock)
        for tag in range(5):
            clock.advance()
            store.save(
                _identity(data_fingerprint=f"bars-{tag}"),
                _frame(tag),
                feature_columns=_COLUMNS,
            )
        assert len(store.keys()) == 5
        assert store.evict().empty is True

    def test_a_cache_under_its_bound_evicts_nothing(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=4, clock=clock)
        for tag in range(3):
            clock.advance()
            store.save(
                _identity(data_fingerprint=f"bars-{tag}"),
                _frame(tag),
                feature_columns=_COLUMNS,
            )
        report = store.evict()
        assert report.empty is True
        assert len(store.keys()) == 3

    @pytest.mark.parametrize("bad", [0, -1])
    @pytest.mark.parametrize("field", ["max_entries", "max_bytes"])
    def test_a_bound_of_zero_is_rejected(
        self, root: Path, field: str, bad: int
    ) -> None:
        with pytest.raises(ValueError, match="must be at least 1"):
            _store(root, **{field: bad})  # type: ignore[arg-type]

    def test_stats_reports_what_the_cache_holds(self, root: Path) -> None:
        store = _store(root, max_entries=8, max_bytes=4096, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        stats = store.stats()
        assert stats.entries == 1
        assert stats.total_bytes > 0
        assert stats.max_entries == 8
        assert stats.max_bytes == 4096
        assert stats.over_limit is False

    def test_stats_sees_a_cache_that_grew_past_its_bound(self, root: Path) -> None:
        """Reopening with a tighter bound can leave existing entries over budget."""
        clock = _Clock()
        _store(root, max_entries=8, clock=clock).save(
            _identity(), _frame(1.0), feature_columns=_COLUMNS
        )
        clock.advance()
        _store(root, max_entries=8, clock=clock).save(
            _identity(data_fingerprint="bars-2"),
            _frame(2.0),
            feature_columns=_COLUMNS,
        )
        tightened = _store(root, max_entries=1, clock=clock)
        assert tightened.stats().entries == 2
        assert tightened.stats().over_limit is True
        assert tightened.evict().remaining == 1
        assert tightened.stats().over_limit is False


class TestLeastRecentlyUsed:
    def test_the_oldest_entry_goes_first(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=1, clock=clock)
        for tag, bars in ((1.0, "bars-1"), (2.0, "bars-2"), (3.0, "bars-3")):
            clock.advance()
            store.save(
                _identity(data_fingerprint=bars), _frame(tag), feature_columns=_COLUMNS
            )
        assert store.keys() == (_identity(data_fingerprint="bars-3").key,)

    def test_reading_an_entry_makes_it_new(self, root: Path) -> None:
        """LRU tracks reads, not just publication age."""
        clock = _Clock()
        store = _store(root, max_entries=2, clock=clock)
        older = _identity(data_fingerprint="bars-1")
        newer = _identity(data_fingerprint="bars-2")
        newest = _identity(data_fingerprint="bars-3")
        clock.advance()
        store.save(older, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        store.save(newer, _frame(2.0), feature_columns=_COLUMNS)

        clock.advance()
        assert store.load(older, feature_columns=_COLUMNS) is not None

        clock.advance()
        store.save(newest, _frame(3.0), feature_columns=_COLUMNS)

        remaining = store.keys()
        assert sorted(remaining) == sorted((older.key, newest.key))
        assert newer.key not in remaining

    def test_keep_protects_an_entry_even_when_it_is_oldest(self, root: Path) -> None:
        """Explicit keep overrides recency during a post-write sweep."""
        clock = _Clock()
        keeper = _identity(data_fingerprint="bars-1")
        victim = _identity(data_fingerprint="bars-2")
        roomy = _store(root, max_entries=8, clock=clock)
        clock.advance()
        roomy.save(keeper, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        roomy.save(victim, _frame(2.0), feature_columns=_COLUMNS)

        # Without keep, the tighter bound would evict the oldest entry.
        bounded = _store(root, max_entries=1, clock=clock)
        report = bounded.evict(keep=keeper.key)

        assert report.removed == (victim.key,)
        assert bounded.keys() == (keeper.key,)

    def test_writing_protects_what_was_just_written(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=1, clock=clock)
        first = _identity(symbol="BTC/USDT")
        second = _identity(symbol="ETH/USDT")
        clock.advance()
        store.save(first, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        store.save(second, _frame(2.0), feature_columns=_COLUMNS)
        assert store.keys() == (second.key,)
        loaded = store.load(second, feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, _frame(2.0))

    def test_a_byte_bound_evicts_oldest_first(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, clock=clock)
        clock.advance()
        store.save(
            _identity(data_fingerprint="bars-1"), _frame(1.0), feature_columns=_COLUMNS
        )
        size_of_one = store.stats().total_bytes
        # Derive the byte budget from actual Parquet compression.
        bounded = _store(root, max_bytes=size_of_one, clock=clock)
        clock.advance()
        bounded.save(
            _identity(data_fingerprint="bars-2"), _frame(2.0), feature_columns=_COLUMNS
        )
        assert store.keys() == (_identity(data_fingerprint="bars-2").key,)

    def test_eviction_stops_as_soon_as_there_is_room(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=3, clock=clock)
        keys = []
        for tag in range(5):
            clock.advance()
            identity = _identity(data_fingerprint=f"bars-{tag}")
            keys.append(identity.key)
            store.save(identity, _frame(float(tag)), feature_columns=_COLUMNS)
        assert len(store.keys()) == 3
        assert set(store.keys()) == set(keys[-3:])

    def test_an_entry_without_a_recorded_use_is_not_treated_as_new(
        self, root: Path
    ) -> None:
        store = _store(root, max_entries=1)
        older = _identity(data_fingerprint="bars-1")
        store.save(older, _frame(1.0), feature_columns=_COLUMNS)
        meta = store.meta_path_for(older.key)
        payload = json.loads(meta.read_text("utf-8"))
        del payload["used_at"]
        meta.write_text(json.dumps(payload), encoding="utf-8")

        newer = _identity(data_fingerprint="bars-2")
        store.save(newer, _frame(2.0), feature_columns=_COLUMNS)

        assert store.keys() == (newer.key,)


class TestSweep:
    def test_a_matrix_without_its_sidecar_is_garbage(self, root: Path) -> None:
        store = _store(root, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        orphan = root / "orphan.parquet"
        _frame(9.0).to_parquet(orphan, index=False)

        report = store.evict()

        assert report.swept == ("orphan",)
        assert orphan.exists() is False
        assert store.keys() == (_identity().key,)

    def test_a_sidecar_without_its_matrix_is_garbage(self, root: Path) -> None:
        store = _store(root, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        lonely = root / "lonely.json"
        lonely.write_text(json.dumps({"cache_key": "lonely"}), encoding="utf-8")

        report = store.evict()

        assert report.swept == ("lonely",)
        assert lonely.exists() is False

    def test_a_write_sweeps_the_garbage_too(self, root: Path) -> None:
        store = _store(root, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        orphan = root / "orphan.parquet"
        _frame(9.0).to_parquet(orphan, index=False)

        clock = _Clock()
        clock.advance()
        store.save(_identity(symbol="ETH/USDT"), _frame(2.0), feature_columns=_COLUMNS)

        assert orphan.exists() is False
        assert len(store.keys()) == 2

    def test_a_pair_whose_sidecar_cannot_verify_anything_is_garbage(
        self, root: Path
    ) -> None:
        store = _store(root, clock=_Clock())
        identity = _identity()
        store.save(identity, _frame(1.0), feature_columns=_COLUMNS)
        # Missing verification fields make the matrix unusable.
        store.meta_path_for(identity.key).write_text(
            json.dumps({"cache_key": identity.key}), encoding="utf-8"
        )

        report = store.evict()

        assert report.swept == (identity.key,)
        assert store.keys() == ()

    def test_garbage_goes_even_when_there_is_room(self, root: Path) -> None:
        store = _store(root, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        (root / "orphan.parquet").write_bytes(b"not a parquet")

        report = store.evict()

        assert report.removed == ()
        assert report.swept == ("orphan",)
        assert report.empty is False

    def test_a_sweep_never_touches_a_loadable_entry(self, root: Path) -> None:
        store = _store(root, clock=_Clock())
        identity = _identity()
        store.save(identity, _frame(1.0), feature_columns=_COLUMNS)
        (root / "orphan.json").write_text("{}", encoding="utf-8")

        store.evict()

        loaded = store.load(identity, feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, _frame(1.0))

    def test_a_half_written_file_is_left_alone(self, root: Path) -> None:
        """A sweep must not delete another writer's staging file."""
        store = _store(root, clock=_Clock())
        store.save(_identity(), _frame(1.0), feature_columns=_COLUMNS)
        staging = root / "in-flight.parquet.tmp"
        staging.write_bytes(b"partial")

        store.evict()

        assert staging.exists() is True
        assert store.stats().entries == 1

    def test_freed_bytes_and_remaining_are_reported(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=2, clock=clock)
        victim = _identity(data_fingerprint="bars-1")
        keeper = _identity(data_fingerprint="bars-2")
        clock.advance()
        store.save(victim, _frame(1.0), feature_columns=_COLUMNS)
        size_of_both = store.stats().total_bytes
        clock.advance()
        store.save(keeper, _frame(2.0), feature_columns=_COLUMNS)

        report = _store(root, max_entries=1, clock=clock).evict(keep=keeper.key)

        assert report.removed == (victim.key,)
        assert report.remaining == 1
        assert report.freed_bytes > 0
        assert store.stats().total_bytes <= size_of_both


class TestEvictionCannotChangeAnAnswer:
    def test_a_surviving_entry_still_loads_its_own_values(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=1, clock=clock)
        first = _identity(data_fingerprint="bars-1")
        second = _identity(data_fingerprint="bars-2")
        clock.advance()
        store.save(first, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        store.save(second, _frame(2.0), feature_columns=_COLUMNS)

        loaded = store.load(second, feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, _frame(2.0))

    def test_an_evicted_entry_misses_instead_of_serving_another(
        self, root: Path
    ) -> None:
        clock = _Clock()
        store = _store(root, max_entries=1, clock=clock)
        first = _identity(data_fingerprint="bars-1")
        second = _identity(data_fingerprint="bars-2")
        clock.advance()
        store.save(first, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        store.save(second, _frame(2.0), feature_columns=_COLUMNS)

        assert store.load(first, feature_columns=_COLUMNS) is None

    def test_an_evicted_entry_can_be_rebuilt_and_comes_back_correct(
        self, root: Path
    ) -> None:
        clock = _Clock()
        store = _store(root, max_entries=1, clock=clock)
        first = _identity(data_fingerprint="bars-1")
        second = _identity(data_fingerprint="bars-2")
        clock.advance()
        store.save(first, _frame(1.0), feature_columns=_COLUMNS)
        clock.advance()
        store.save(second, _frame(2.0), feature_columns=_COLUMNS)

        clock.advance()
        store.save(first, _frame(1.0), feature_columns=_COLUMNS)

        loaded = store.load(first, feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, _frame(1.0))
        assert store.keys() == (first.key,)

    def test_evicting_does_not_disturb_the_other_entry(self, root: Path) -> None:
        clock = _Clock()
        store = _store(root, max_entries=3, clock=clock)
        identities = [
            _identity(data_fingerprint="bars-1"),
            _identity(symbol="ETH/USDT"),
            _identity(code_fingerprint="code-v2"),
            _identity(feature_set="technical"),
        ]
        for tag, identity in enumerate(identities):
            clock.advance()
            store.save(identity, _frame(float(tag)), feature_columns=_COLUMNS)

        assert set(store.keys()) == {identity.key for identity in identities[1:]}
        for index, identity in enumerate(identities[1:], start=1):
            loaded = store.load(identity, feature_columns=_COLUMNS)
            assert loaded is not None
            pd.testing.assert_frame_equal(loaded, _frame(float(index)))
