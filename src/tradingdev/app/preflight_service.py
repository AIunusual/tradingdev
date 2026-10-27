"""Prepare and check a program in a supervised, time-bounded subprocess."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.prepared_execution import (
    PreparedComparison,
    PreparedExecution,
    PreparedSubmission,
)
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.preflight import (
    ComparisonPreflightReceipt,
    PreflightPayload,
    PreflightReceipt,
    PreflightRequest,
    ProgramReceipt,
)
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionGuidance,
    StrategyExecutionModeError,
)


class PreflightError(RuntimeError):
    """A preparation failure that callers can expose as a structured rejection."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        execution_guidance: StrategyExecutionGuidance | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.execution_guidance = execution_guidance


@dataclass(frozen=True)
class PreflightResult:
    prepared: PreparedSubmission
    receipt: ProgramReceipt


class PreflightService:
    """Keep generated program checks and temporary files outside the server process."""

    def __init__(
        self,
        workspace: WorkspacePaths | None = None,
        *,
        project_root: Path | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._project_root = project_root
        self._timeout_seconds = timeout_seconds

    def prepare(self, request: PreflightRequest) -> PreflightResult:
        """Return evidence only after worker cleanup has been independently verified."""
        started = time.monotonic()
        directory = Path(mkdtemp(prefix="tradingdev-preflight-"))
        cleanup_verified = True
        try:
            temporary_directory = directory / "tmp"
            temporary_directory.mkdir()
            request_path = directory / "request.json"
            result_path = directory / "result.json"
            request_path.write_text(
                json.dumps(
                    {
                        "request": request.model_dump(mode="json"),
                        "source_workspace": str(self._workspace.root),
                    }
                ),
                encoding="utf-8",
            )
            runner = ProcessRunner(
                self._project_root,
                workspace=WorkspacePaths(directory),
                env_overrides={
                    "TMPDIR": str(temporary_directory),
                    "TMP": str(temporary_directory),
                    "TEMP": str(temporary_directory),
                    "NUMBA_CACHE_DIR": str(directory / "numba-cache"),
                    "MPLCONFIGDIR": str(directory / "matplotlib"),
                    "MYPY_CACHE_DIR": str(directory / "mypy-cache"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            try:
                runner.run_module(
                    "tradingdev.adapters.execution.preflight_worker",
                    str(request_path),
                    str(result_path),
                    timeout_seconds=self._timeout_seconds,
                )
            except BoundedWorkerError as error:
                cleanup_verified = error.cleanup_verified
                suffix = "" if cleanup_verified else f"; retained {directory}"
                raise PreflightError(error.code, f"{error}{suffix}") from error
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("Preflight response must be an object")
                if raw.get("success") is False:
                    code = str(raw["code"])
                    raw_guidance = raw.get("execution_guidance")
                    guidance = None
                    if code == StrategyExecutionModeError.code:
                        guidance = StrategyExecutionGuidance.model_validate(
                            raw_guidance
                        )
                        if guidance.strategy_id != request.arguments.get("strategy_id"):
                            raise ValueError(
                                "Execution guidance strategy must match the request"
                            )
                    elif raw_guidance is not None:
                        raise ValueError(
                            "Execution guidance requires a strategy training rejection"
                        )
                    raise PreflightError(
                        code,
                        str(raw["message"]),
                        execution_guidance=guidance,
                    )
                payload = PreflightPayload.model_validate(raw["result"])
                if payload.manifest.kind != request.kind:
                    raise ValueError("Preflight execution kind must match the request")
            except (OSError, ValueError, KeyError) as error:
                raise PreflightError(
                    "invalid_preflight_response", str(error)
                ) from error
            prepared: PreparedSubmission
            if isinstance(payload.manifest, ComparisonManifest):
                assert isinstance(payload.receipt, ComparisonPreflightReceipt)
                prepared = PreparedComparison(
                    payload.manifest,
                    Path(payload.original_config_path),
                    {
                        item.candidate_id: tuple(item.receipt.synthetic_fixture_rows)
                        for item in payload.receipt.candidates
                    },
                )
            else:
                assert isinstance(payload.receipt, PreflightReceipt)
                prepared = PreparedExecution(
                    payload.manifest,
                    Path(payload.original_config_path),
                    tuple(payload.receipt.synthetic_fixture_rows),
                )
            return PreflightResult(
                prepared,
                payload.receipt.model_copy(
                    update={"elapsed_seconds": time.monotonic() - started}
                ),
            )
        finally:
            if cleanup_verified:
                try:
                    shutil.rmtree(directory)
                except OSError as error:
                    raise PreflightError(
                        "preflight_cleanup_failed",
                        f"Cannot remove preflight files {directory}: {error}",
                    ) from error
