"""Bound generated imports, constructors and signal checks outside the caller."""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.execution.strategy_check_worker import (
    StrategyCheckRequest,
    StrategyCheckResponse,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.strategies.validator import diagnostic

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategyMetadata


def _failure(code: str, message: str) -> dict[str, Any]:
    return {
        "diagnostics": [
            diagnostic(
                code=code,
                phase="contract",
                message=message,
                fix="Inspect imports, constructor and signal logic; repair and retry.",
            )
        ],
        "effective_config": None,
        "signal_analysis": {},
    }


class StrategyContractService:
    """Return evidence only after the supervisor's owned process group stops."""

    def __init__(
        self,
        workspace: WorkspacePaths,
        *,
        timeout_seconds: float = 60,
        project_root: Path | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Strategy check timeout must be finite and positive")
        self._workspace = workspace
        self._timeout_seconds = timeout_seconds
        self._project_root = project_root

    def check(self, metadata: StrategyMetadata, *, fixture_rows: int) -> dict[str, Any]:
        request = StrategyCheckRequest(
            source_workspace=str(self._workspace.root),
            metadata=metadata,
            fixture_rows=fixture_rows,
        )
        directory = Path(mkdtemp(prefix="tradingdev-strategy-check-"))
        cleanup_verified = True
        outcome: dict[str, Any] = {}
        try:
            temporary = directory / "tmp"
            temporary.mkdir()
            request_path, result_path = (
                directory / "request.json",
                directory / "result.json",
            )
            payload = request.model_dump_json().encode("utf-8")
            request_path.write_bytes(payload)
            runner = ProcessRunner(
                self._project_root,
                workspace=WorkspacePaths(directory),
                env_overrides={
                    "TMPDIR": str(temporary),
                    "TMP": str(temporary),
                    "TEMP": str(temporary),
                    "NUMBA_CACHE_DIR": str(directory / "numba"),
                    "MPLCONFIGDIR": str(directory / "matplotlib"),
                    "MYPY_CACHE_DIR": str(directory / "mypy"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            runner.run_module(
                "tradingdev.adapters.execution.strategy_check_worker",
                str(request_path),
                str(result_path),
                timeout_seconds=self._timeout_seconds,
                operation="strategy_check",
            )
            response = StrategyCheckResponse.model_validate_json(
                result_path.read_bytes()
            )
            if response.request_hash != hashlib.sha256(payload).hexdigest():
                raise ValueError("Strategy check response does not match its request")
            if not any(item.level == "error" for item in response.diagnostics) and (
                response.effective_config is None
                or response.signal_analysis.get("rows") != fixture_rows
            ):
                raise ValueError(
                    "Successful strategy check is missing complete fixture evidence"
                )
            if response.effective_config is not None:
                BacktestRunConfig.model_validate(response.effective_config)
            outcome = {
                "diagnostics": response.diagnostics,
                "signal_analysis": response.signal_analysis,
                "effective_config": response.effective_config,
            }
        except BoundedWorkerError as error:
            cleanup_verified = error.cleanup_verified
            suffix = "" if cleanup_verified else f"; retained {directory}"
            outcome = _failure(error.code, f"{error}{suffix}")
        except (OSError, ValueError) as error:
            outcome = _failure("invalid_strategy_check_response", str(error))
        finally:
            if cleanup_verified:
                try:
                    shutil.rmtree(directory)
                except OSError as error:
                    outcome = _failure(
                        "strategy_check_cleanup_failed",
                        f"Cannot remove strategy check files {directory}: {error}",
                    )
        return outcome
