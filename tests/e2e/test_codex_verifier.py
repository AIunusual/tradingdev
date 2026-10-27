"""Offline checks for the live-test verifier and timeout cleanup."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.e2e.codex_harness import runtime_environment, verify_generated_strategy
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.research import NewResearchSpec

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_CODE = '''\
"""Deterministic fixture for the isolated verifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class FixtureStrategy(BaseStrategy):
    def __init__(self, *, fast_period: int, slow_period: int) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = result["close"].rolling(self.fast_period).mean()
        slow = result["close"].rolling(self.slow_period).mean()
        result["signal"] = 0
        result.loc[fast > slow, "signal"] = 1
        result.loc[fast < slow, "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}
'''


def _fixture_strategy(
    root: Path,
    code: str = _CODE,
    *,
    periods: tuple[int, int] = (5, 20),
    expected_revision_id: str | None = None,
) -> Path:
    workspace = WorkspacePaths(root / "workspace")
    saved = StrategyService(workspace).save_draft(
        code,
        yaml.safe_dump(
            {
                "strategy": {
                    "id": "codex_sma_integration",
                    "class_name": "FixtureStrategy",
                    "parameters": {
                        "fast_period": periods[0],
                        "slow_period": periods[1],
                    },
                },
                "data": {"source": "binance_vision", "market_type": "futures/um"},
            }
        ),
        NewResearchSpec.model_validate(
            {
                "strategy_outline": "Fixture SMA strategy",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "init_cash": 10000,
                "fees": 0,
                "slippage": 0,
                "annualization": 365,
            }
        ),
        strategy_id="codex_sma_integration",
        expected_revision_id=expected_revision_id,
    )
    assert saved.success
    metadata_path = Path(saved.source_path).parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "runnable"
    # Deliberately seed trusted fixture evidence to isolate verifier cleanup.
    evidence = {
        "revision_id": saved.revision_id,
        "checked_at": metadata["created_at"],
        "success": True,
        "diagnostics": [],
        "signal_analysis": {},
    }
    metadata["validation"] = evidence
    metadata["dry_run"] = evidence
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return Path(saved.source_path)


def test_verifier_checks_generated_strategy_in_a_separate_process(
    monkeypatch: MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-verifier-") as temporary:
        root = Path(temporary)
        for key, value in runtime_environment(root).items():
            monkeypatch.setenv(key, value)
        _fixture_strategy(root)
        verify_generated_strategy(root)
    assert not root.exists()


def test_novice_verifier_rejects_executed_short_revision_after_current_is_repaired(
    monkeypatch: MonkeyPatch,
) -> None:
    """A later correct draft cannot replace the revision bound to an existing run.

    Run identity and runnable evidence are trusted fixtures; no historical worker
    runs here. The old strategy is genuinely executed by the bounded verifier,
    and the independent long-only signal comparison must reject its shorts.
    """
    with TemporaryDirectory(prefix="tradingdev-novice-verifier-") as temporary:
        root = Path(temporary)
        for key, value in runtime_environment(root).items():
            monkeypatch.setenv(key, value)
        marker = root / "executed-revision.pid"
        wrong_code = (
            _CODE
            + "\nimport os\nfrom pathlib import Path\n"
            + f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        )
        executed_source = _fixture_strategy(root, wrong_code, periods=(10, 30))
        executed_revision = executed_source.parent.name
        workspace = WorkspacePaths(root / "workspace")
        store = SQLiteStore(workspace)
        store.create_run(
            run_id="approved-run",
            job_id="approved-run",
            strategy_id="codex_sma_integration",
            revision_id=executed_revision,
            artifact_dir=workspace.runs / "approved-run",
            metrics={},
        )
        correct_code = _CODE.replace(
            '        result.loc[fast < slow, "signal"] = -1\n', ""
        )
        current_source = _fixture_strategy(
            root,
            correct_code,
            periods=(10, 30),
            expected_revision_id=executed_revision,
        )
        current = StrategyService(workspace).resolve_executable("codex_sma_integration")
        assert current.revision_id == current_source.parent.name != executed_revision
        assert current_source.read_text(encoding="utf-8") == correct_code
        assert 'result.loc[fast < slow, "signal"] = -1' not in correct_code
        assert store.list_runs()[0]["revision_id"] == executed_revision

        with pytest.raises(
            AssertionError, match="Generated strategy verification failed"
        ) as error:
            verify_generated_strategy(root, scenario="novice_intake")
        assert "Series are different" in str(error.value), (
            "Failure must come from the long-only oracle, not missing fixture metrics"
        )
        assert marker.exists(), "The verifier did not load the executed source revision"
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text(encoding="utf-8")), 0)
        assert store.list_runs()[0]["revision_id"] == executed_revision
        assert executed_source.read_text(encoding="utf-8") == wrong_code
    assert not root.exists()


def test_verifier_terminates_hanging_generated_code_and_removes_artifacts(
    monkeypatch: MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-verifier-") as temporary:
        root = Path(temporary)
        for key, value in runtime_environment(root).items():
            monkeypatch.setenv(key, value)
        marker = root / "verifier.pid"
        _fixture_strategy(
            root,
            _CODE
            + "\nimport os\nfrom pathlib import Path\n"
            + f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            + "while True:\n    pass\n",
        )
        group_signals: list[int] = []

        def deny_group_signal(pid: int, sig: int) -> None:
            group_signals.append(sig)
            raise PermissionError("Simulate restricted macOS process group signals")

        monkeypatch.setattr(os, "killpg", deny_group_signal)
        with pytest.raises(AssertionError, match="verification exceeded"):
            verify_generated_strategy(
                root, timeout_seconds=3, startup_timeout_seconds=30
            )
        assert group_signals == [signal.SIGTERM]
        assert marker.exists(), "Verifier never reached the hanging strategy"
        pid = int(marker.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not root.exists()
