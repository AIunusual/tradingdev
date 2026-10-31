"""Worker replies cannot grant success without matching evidence and cleanup."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import yaml

import tradingdev
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app import strategy_contract_service as module
from tradingdev.domain.strategies.schemas import StrategyMetadata, StrategyStatus
from tradingdev.domain.strategies.templates import strategy_contract_payload


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "wrong_request",
        "missing_evidence",
        "invalid_config",
        "invalid_json",
        "worker_failed",
        "timeout",
        "unverified",
        "remove_failed",
    ],
)
def test_response_and_cleanup_failures_cannot_grant_success(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    with TemporaryDirectory(prefix="tradingdev-contract-response-test-") as name:
        root = Path(name)
        directory = root / "check"
        directory.mkdir()
        monkeypatch.setattr(module, "mkdtemp", lambda **kwargs: str(directory))
        metadata = StrategyMetadata(
            strategy_id="fixture",
            revision_id="r1",
            class_name="Fixture",
            status=StrategyStatus.DRAFT,
            created_at="2024-01-01",
            updated_at="2024-01-01",
            source_path=str(root / "strategy.py"),
            config_path=str(root / "config.yaml"),
            source_hash="a" * 64,
            config_hash="b" * 64,
        )
        example = strategy_contract_payload(Path(tradingdev.__file__).parent, "example")
        config = yaml.safe_load(example["content"]["example_yaml_config"])

        def run(
            _runner: ProcessRunner,
            _module: str,
            *arguments: str,
            timeout_seconds: float,
            operation: str,
        ) -> None:
            assert operation == "strategy_check" and timeout_seconds == 60
            if outcome in {"worker_failed", "timeout", "unverified"}:
                code = (
                    "strategy_check_timeout"
                    if outcome == "timeout"
                    else "strategy_check_failed"
                )
                raise BoundedWorkerError(
                    code, outcome, cleanup_verified=outcome != "unverified"
                )
            request, result = map(Path, arguments)
            payload: dict[str, Any] = {
                "request_hash": hashlib.sha256(request.read_bytes()).hexdigest(),
                "diagnostics": [],
                "signal_analysis": {"rows": 80},
                "effective_config": config,
            }
            if outcome == "wrong_request":
                payload["request_hash"] = "0" * 64
            elif outcome == "missing_evidence":
                payload["signal_analysis"] = {}
            elif outcome == "invalid_config":
                payload["effective_config"] = {}
            result.write_text(
                "not JSON" if outcome == "invalid_json" else json.dumps(payload)
            )

        monkeypatch.setattr(ProcessRunner, "run_module", run)
        original_remove = shutil.rmtree

        def remove(path: Path, *args: Any, **kwargs: Any) -> None:
            if path == directory:
                raise OSError("cannot remove worker files")
            original_remove(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            if outcome == "remove_failed":
                patch.setattr(shutil, "rmtree", remove)
            result = module.StrategyContractService(
                WorkspacePaths(root / "source")
            ).check(
                metadata,
                fixture_rows=80,
            )
        diagnostics = result["diagnostics"]
        if outcome == "success":
            assert diagnostics == []
            assert result["signal_analysis"]["rows"] == 80
        else:
            assert diagnostics and diagnostics[0].level == "error"
            expected = (
                "strategy_check_timeout"
                if outcome == "timeout"
                else "strategy_check_failed"
                if outcome in {"worker_failed", "unverified"}
                else "strategy_check_cleanup_failed"
                if outcome == "remove_failed"
                else "invalid_strategy_check_response"
            )
            assert diagnostics[0].code == expected
            assert result["effective_config"] is None
        assert directory.exists() is (outcome in {"unverified", "remove_failed"})
        if outcome == "unverified":
            assert str(directory) in diagnostics[0].message
        # No worker started in this protocol test; retained fixtures are safe to remove.
