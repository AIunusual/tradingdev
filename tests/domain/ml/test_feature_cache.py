"""The key that decides whether a feature matrix may be reused.

A cache key is a claim that two computations are the same computation, so every
test here is one way that claim could be wrong: bars that were edited without
changing size, two instruments carrying identical bars, a feature set renamed,
a bug fixed in the feature code. Each has to produce a different key, because
the alternative is a dataset - and every model and leaderboard row built on it -
silently computed from the wrong inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml import feature_cache
from tradingdev.domain.ml.feature_cache import (
    FeatureCacheIdentity,
    data_fingerprint,
    feature_cache_key,
    feature_code_fingerprint,
)
from tradingdev.domain.ml.features.registry import FeatureSet, get_feature_set

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 40
_BARS = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
        "open": np.linspace(100.0, 110.0, _ROWS),
        "high": np.linspace(101.0, 111.0, _ROWS),
        "low": np.linspace(99.0, 109.0, _ROWS),
        "close": np.linspace(100.0, 110.0, _ROWS),
        "volume": np.full(_ROWS, 10.0),
    }
)


def _key(**overrides: object) -> str:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "data_fingerprint": data_fingerprint(_BARS),
        "feature_set": "returns",
        "code_fingerprint": "code-v1",
    }
    values.update(overrides)
    return feature_cache_key(**values)  # type: ignore[arg-type]


class TestDataFingerprint:
    def test_the_same_bars_agree(self) -> None:
        assert data_fingerprint(_BARS) == data_fingerprint(_BARS.copy())

    def test_one_changed_bar_disagrees(self) -> None:
        edited = _BARS.copy()
        edited.loc[5, "close"] = 999.0
        assert data_fingerprint(edited) != data_fingerprint(_BARS)

    def test_an_edit_that_keeps_the_file_size_is_still_seen(
        self, tmp_path: Path
    ) -> None:
        """Size and mtime are not evidence: swapping two bars keeps both."""
        first = tmp_path / "a.parquet"
        _BARS.to_parquet(first, index=False)
        edited = _BARS.copy()
        edited.loc[0, "close"], edited.loc[1, "close"] = (
            _BARS.loc[1, "close"],
            _BARS.loc[0, "close"],
        )
        second = tmp_path / "b.parquet"
        edited.to_parquet(second, index=False)
        assert first.stat().st_size == second.stat().st_size
        assert data_fingerprint(pd.read_parquet(second)) != data_fingerprint(
            pd.read_parquet(first)
        )

    def test_a_copy_with_a_new_mtime_agrees(self, tmp_path: Path) -> None:
        """A refreshed cache file is the same data, not new data."""
        path = tmp_path / "bars.parquet"
        _BARS.to_parquet(path, index=False)
        before = data_fingerprint(pd.read_parquet(path))
        _BARS.to_parquet(path, index=False)
        assert data_fingerprint(pd.read_parquet(path)) == before

    def test_column_names_are_part_of_it(self) -> None:
        renamed = _BARS.rename(columns={"close": "closing_price"})
        assert data_fingerprint(renamed) != data_fingerprint(_BARS)

    def test_column_order_is_part_of_it(self) -> None:
        reordered = _BARS[list(reversed(_BARS.columns))]
        assert data_fingerprint(reordered) != data_fingerprint(_BARS)

    def test_row_count_is_part_of_it(self) -> None:
        assert data_fingerprint(_BARS.iloc[:-1]) != data_fingerprint(_BARS)

    def test_an_empty_frame_is_still_fingerprinted(self) -> None:
        assert data_fingerprint(_BARS.iloc[:0]) != data_fingerprint(_BARS.iloc[:1])


class TestFeatureCacheKey:
    def test_is_a_short_hex_digest(self) -> None:
        key = _key()
        assert len(key) == 16
        assert all(character in "0123456789abcdef" for character in key)

    def test_is_stable(self) -> None:
        assert _key() == _key()

    @pytest.mark.parametrize(
        "field",
        ["symbol", "timeframe", "data_fingerprint", "feature_set", "code_fingerprint"],
    )
    def test_every_ingredient_changes_the_key(self, field: str) -> None:
        assert _key(**{field: "something-else"}) != _key()

    def test_two_instruments_with_identical_bars_do_not_share_a_key(self) -> None:
        """A synthetic feed can hand two symbols byte-identical bars."""
        assert _key(symbol="BTC/USDT") != _key(symbol="ETH/USDT")

    def test_the_same_bars_at_two_timeframes_do_not_share_a_key(self) -> None:
        assert _key(timeframe="1h") != _key(timeframe="4h")

    def test_the_identity_exposes_what_it_was_built_from(self) -> None:
        identity = FeatureCacheIdentity(
            symbol="BTC/USDT",
            timeframe="1h",
            data_fingerprint="bars",
            feature_set="returns",
            code_fingerprint="code-v1",
        )
        assert identity.key == _key(data_fingerprint="bars")


class TestFeatureCodeFingerprint:
    def test_is_stable_across_calls(self) -> None:
        assert feature_code_fingerprint() == feature_code_fingerprint()

    def test_is_a_short_hex_digest(self) -> None:
        fingerprint = feature_code_fingerprint()
        assert len(fingerprint) == 16
        assert all(character in "0123456789abcdef" for character in fingerprint)

    def test_does_not_depend_on_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A key that moved with the cwd could not be shared between runs."""
        outside = feature_code_fingerprint()
        monkeypatch.chdir(tmp_path)
        assert feature_code_fingerprint() == outside

    def test_the_package_source_is_what_it_hashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fix inside a helper must invalidate, not just a fix in a builder."""
        real = feature_code_fingerprint()
        original = feature_cache._package_sources()
        assert original  # the real sources were read, not an empty package
        tampered = [(name, source + b"\n# patched\n") for name, source in original[:1]]
        monkeypatch.setattr(
            feature_cache, "_package_sources", lambda: tampered + list(original[1:])
        )
        assert feature_code_fingerprint() != real

    def test_declared_columns_are_part_of_it(self) -> None:
        renamed = FeatureSet(
            name=get_feature_set("returns").name,
            description="same set, different contract",
            columns=("a", "b"),
            build=get_feature_set("returns").build,
        )
        assert feature_code_fingerprint([renamed]) != feature_code_fingerprint(
            [get_feature_set("returns")]
        )

    def test_an_unreadable_source_is_an_error_not_a_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silently falling back to 'unchanged' would serve features built by
        a version that is no longer there to be read."""
        from tradingdev.domain.ml import features as features_package

        monkeypatch.setattr(features_package, "__path__", [])
        with pytest.raises(ValueError, match="cannot locate the feature package"):
            feature_code_fingerprint()
