"""A changed cross-package indicator cannot reuse the previous feature matrix."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import pytest
from tooling.process_guard import ProcessCleanupError, run_captured

import tradingdev

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROBE = """
import json
import os
from pathlib import Path
import pandas as pd
from tradingdev.domain.ml.feature_cache import (
    FeatureCacheIdentity, data_fingerprint, feature_code_fingerprint,
)
from tradingdev.domain.ml.features.registry import get_feature_set
from tradingdev.domain.ml.registry import FeatureCacheStore
frame = pd.DataFrame({'close': [float(i) for i in range(1, 31)]})
entry = get_feature_set('price_ratios')
fresh = entry.build(frame)
identity = FeatureCacheIdentity('TEST', '1d', data_fingerprint(frame),
                               entry.name, feature_code_fingerprint())
cache = FeatureCacheStore(Path(os.environ['PROBE_FEATURE_CACHE']))
cached = cache.load(identity, feature_columns=entry.columns)
hit = cached is not None
if cached is None:
    cache.save(identity, fresh, feature_columns=entry.columns)
    cached = fresh
pd.testing.assert_frame_equal(cached, fresh, check_exact=True)
print(json.dumps({'fingerprint': identity.code_fingerprint,
                 'value': float(fresh.iloc[-1, 0]), 'cache_hit': hit}))
"""


@contextmanager
def _probe_workspace() -> Iterator[Path]:
    root = Path(mkdtemp(prefix="tradingdev-feature-dependency-"))
    retained = False
    try:
        yield root
    except ProcessCleanupError as error:
        retained = True
        raise ProcessCleanupError(f"{error}; retained {root}") from error
    finally:
        if not retained:
            shutil.rmtree(root)


def test_indicator_update_misses_old_cache_and_reuses_new_computation() -> None:
    source = Path(tradingdev.__file__).resolve().parent
    with _probe_workspace() as root:
        package = root / "tradingdev"
        shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
        env = {
            **os.environ,
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PROBE_FEATURE_CACHE": str(root / "cache"),
        }

        def inspect() -> dict[str, Any]:
            result = run_captured(
                [sys.executable, "-B", "-c", _PROBE],
                cwd=root,
                env=env,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            return dict(json.loads(result.stdout))

        original = inspect()
        reused = inspect()
        assert not original["cache_hit"] and reused["cache_hit"]
        assert original["fingerprint"] == reused["fingerprint"]
        indicator = package / "domain" / "indicators" / "technical.py"
        content = indicator.read_text()
        expression = "return _series(talib.SMA(values, timeperiod=length), close)"
        assert content.count(expression) == 1
        indicator.write_text(content.replace(expression, expression + " * 2"))
        changed = inspect()
        assert changed["fingerprint"] != original["fingerprint"]
        assert changed["value"] != original["value"]
        assert not changed["cache_hit"]
        assert inspect() == {**changed, "cache_hit": True}


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "unverified"])
def test_probe_workspace_cleanup_requires_verified_process_exit(outcome: str) -> None:
    root: Path | None = None
    try:
        try:
            with _probe_workspace() as root:
                (root / "evidence").write_text("retained until process exit")
                if outcome == "failure":
                    raise AssertionError("computation failed")
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired("probe", 60)
                if outcome == "unverified":
                    raise ProcessCleanupError("exit unverified")
        except (
            AssertionError,
            subprocess.TimeoutExpired,
            ProcessCleanupError,
        ) as error:
            assert outcome != "success"
            if outcome == "unverified":
                assert isinstance(error, ProcessCleanupError)
                assert f"retained {root}" in str(error)
        assert root is not None
        assert root.exists() is (outcome == "unverified")
    finally:
        # The injected failure never starts a process; this test owns safe cleanup.
        if root is not None and root.exists():
            shutil.rmtree(root)
