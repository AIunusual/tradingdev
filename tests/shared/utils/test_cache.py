"""Tests for the persistent disk cache utilities.

The key is a claim that a stored result was produced by this manifest, on this
data, under this code. The tests are about that claim failing loudly: a change
of the same size as the original (which an mtime/size check would not see), and
a change of working directory (which the old git-derived code fingerprint did
not survive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.shared.utils.cache import (
    cache_dir,
    clear_cache,
    compute_cache_key,
    data_fingerprint,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestDataFingerprint:
    def test_hashes_the_content(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        assert data_fingerprint(data) != "missing"

    def test_an_edit_of_the_same_size_is_seen(self, tmp_path: Path) -> None:
        """mtime and size both survive this edit; content does not."""
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        before = data_fingerprint(data)
        data.write_bytes(b"faka")
        assert data.stat().st_size == 4
        assert data_fingerprint(data) != before

    def test_a_missing_file_is_named_not_guessed(self, tmp_path: Path) -> None:
        assert data_fingerprint(tmp_path / "absent.parquet") == "missing"


class TestComputeCacheKey:
    @staticmethod
    def _key(data: Path, **overrides: str) -> str:
        values = {"manifest_hash": "1" * 64, "code_fingerprint": "code-v1"}
        values.update(overrides)
        return compute_cache_key(processed_path=data, **values)

    def test_same_inputs_same_key(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        assert self._key(data) == self._key(data)

    def test_manifest_change_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        assert self._key(data, manifest_hash="1" * 64) != self._key(
            data, manifest_hash="2" * 64
        )

    def test_data_change_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        before = self._key(data)
        data.write_bytes(b"longer data fixture")
        assert self._key(data) != before

    def test_data_change_of_the_same_size_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        before = self._key(data)
        data.write_bytes(b"faka")
        assert self._key(data) != before

    def test_code_fingerprint_change_invalidates(self, tmp_path: Path) -> None:
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        assert self._key(data, code_fingerprint="v1") != self._key(
            data, code_fingerprint="v2"
        )

    def test_missing_data_file(self, tmp_path: Path) -> None:
        """Should not raise - it just has no content to contribute."""
        key = self._key(tmp_path / "no_such_file.parquet")
        assert len(key) == 16

    def test_the_key_does_not_move_with_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key that followed the cwd could not be shared between runs.

        It used to be derived from ``git rev-parse`` of the local checkout, and
        fell back to a random value when git could not be reached - so the same
        run produced a different key depending on where it was started.
        """
        data = tmp_path / "data.parquet"
        data.write_bytes(b"fake")
        before = self._key(data)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert self._key(data) == before


class TestCacheDirectory:
    def test_default_cache_dir_uses_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
        monkeypatch.setattr("tradingdev.shared.utils.cache.CACHE_DIR", None)

        assert cache_dir() == tmp_path / "workspace" / "data" / "processed" / "cache"

    def test_data_root_overrides_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_root = tmp_path / "runtime_data"
        monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(data_root))
        monkeypatch.setattr("tradingdev.shared.utils.cache.CACHE_DIR", None)

        assert cache_dir() == data_root / "processed" / "cache"

    def test_clear_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            "tradingdev.shared.utils.cache.CACHE_DIR",
            cache_dir,
        )
        cache_dir.mkdir()
        (cache_dir / "pipeline.pkl").write_bytes(b"cached pipeline fixture")
        metadata = cache_dir / "unrelated.json"
        metadata.write_text("{}", encoding="utf-8")

        count = clear_cache()
        assert count == 1
        assert not any(cache_dir.glob("*.pkl"))
        assert metadata.exists()
