"""Failure-only evidence for the reviewed novice run, before workspace cleanup."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.execution import ExecutionManifest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from tests.e2e.llm_client import ToolCall
    from tests.e2e.novice_intake import ObservedIntakeUser
    from tests.integration.mcp_harness import MCPWorkspace

_TEXT_LIMIT = 64_000
_TRADE_PAGE_LIMIT = 2


def _bounded(value: Any) -> dict[str, Any]:
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return {
        "complete": len(text) <= _TEXT_LIMIT,
        "total_characters": len(text),
        "total_utf8_bytes": len(text.encode("utf-8")),
        **(
            {"value": value}
            if len(text) <= _TEXT_LIMIT
            else {"json_prefix": text[:_TEXT_LIMIT]}
        ),
    }


def _diagnostic_failure(error: Exception) -> dict[str, Any]:
    text = f"{type(error).__name__}: {error}"
    return {
        "available": False,
        "diagnostic_error": text[:_TEXT_LIMIT],
        "complete": len(text) <= _TEXT_LIMIT,
        "total_characters": len(text),
        "total_utf8_bytes": len(text.encode("utf-8")),
    }


def _attempt(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except Exception as error:
        return _diagnostic_failure(error)


def _artifact(
    path: Path,
    paths: WorkspacePaths,
    *,
    expected_sha256: str | None = None,
    manifest_hash: str | None = None,
) -> dict[str, Any]:
    if expected_sha256 is None and manifest_hash is None:
        raise ValueError("Saved artifact verification identity is missing")
    if not path.resolve().is_relative_to(paths.root.resolve()):
        raise ValueError("Diagnostic artifact must remain inside the test workspace")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Artifact SHA256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    if manifest_hash is not None:
        ExecutionManifest.model_validate_json(content).verify(
            expected_hash=manifest_hash
        )
    text = content.decode("utf-8")
    return {
        "available": True,
        "path": str(path),
        "sha256": digest,
        "total_bytes": len(content),
        "total_characters": len(text),
        "complete": len(text) <= _TEXT_LIMIT,
        "text": text[:_TEXT_LIMIT],
    }


def _revision_checks(
    paths: WorkspacePaths, strategy_id: str, revision_id: str
) -> dict[str, Any]:
    revision = StrategyRevisionStore(paths).load(strategy_id, revision_id)
    if revision is None:
        raise LookupError(f"Reviewed revision is unavailable: {revision_id}")
    return _bounded(
        {
            "revision_id": revision.revision_id,
            "source_hash": revision.source_hash,
            "config_hash": revision.config_hash,
            "validation": revision.validation.model_dump(mode="json")
            if revision.validation is not None
            else None,
            "dry_run": revision.dry_run.model_dump(mode="json")
            if revision.dry_run is not None
            else None,
        }
    )


def _saved_trade_query(
    history: TradeHistoryService, run_id: str, direction: str | None
) -> dict[str, Any]:
    return _bounded(
        history.get_run_trades(run_id, scope="full", limit=1, direction=direction)
    )


def novice_failure_evidence(
    workspace: MCPWorkspace,
    user: ObservedIntakeUser,
    calls: list[ToolCall],
) -> dict[str, Any]:
    """Read saved JSON/text only, never import or re-execute generated strategies."""
    evidence: dict[str, Any] = {"scenario": "novice_intake"}
    plan = user.plan
    if plan is None:
        return {**evidence, "selected_plan": None, "reason": "No plan was reviewed"}
    strategy = (
        plan.manifest.common_config()
        if plan.manifest.kind == "comparison"
        else plan.manifest.config_copy()
    )["strategy"]
    evidence["selected_plan"] = {
        "plan_id": plan.plan_id,
        "manifest_hash": plan.manifest.manifest_hash,
        "strategy_id": strategy["id"],
        "revision_id": strategy["revision_id"],
        "manifest": _bounded(plan.manifest.model_dump(mode="json")),
    }
    paths = WorkspacePaths(workspace.workspace)
    evidence["selected_revision_checks"] = _attempt(
        partial(_revision_checks, paths, strategy["id"], strategy["revision_id"])
    )
    if not (paths.root / "tradingdev.sqlite").is_file():
        return {
            **evidence,
            "run_selection": {"available": False, "reason": "No saved run database"},
        }
    store = SQLiteStore(paths)
    runs = store.list_runs()
    matching = [
        run
        for run in runs
        if run["manifest_hash"] == plan.manifest.manifest_hash
        and run["strategy_id"] == strategy["id"]
        and run["revision_id"] == strategy["revision_id"]
    ]
    evidence["run_selection"] = {
        "available": len(matching) == 1,
        "total_saved_runs": len(runs),
        "matching_saved_runs": len(matching),
    }
    if len(matching) != 1:
        evidence["run_selection"]["reason"] = "No unique run matches the reviewed plan"
        return evidence
    run = matching[0]
    run_id = run["run_id"]
    evidence["run"] = {
        key: run[key]
        for key in (
            "run_id",
            "job_id",
            "strategy_id",
            "revision_id",
            "manifest_hash",
            "config_hash",
            "source_hash",
        )
    }
    directory = paths.runs / run_id
    evidence["execution_artifacts"] = {
        "manifest.json": _attempt(
            partial(
                _artifact,
                directory / "manifest.json",
                paths,
                manifest_hash=plan.manifest.manifest_hash,
            )
        ),
        **{
            filename: _attempt(
                partial(
                    _artifact,
                    directory / filename,
                    paths,
                    expected_sha256=run[hash_key],
                )
            )
            for filename, hash_key in (
                ("config.yaml", "config_hash"),
                ("strategy.py", "source_hash"),
            )
        },
    }
    pages = [
        (index, call)
        for index, call in enumerate(calls)
        if call.name == "get_run_trades" and call.arguments.get("run_id") == run_id
    ]
    evidence["recorded_trade_pages"] = {
        "origin": "completed model tool calls, including unfinished conversation turns",
        "total_recorded_pages": len(pages),
        "captured_pages": min(len(pages), _TRADE_PAGE_LIMIT),
        "complete": len(pages) <= _TRADE_PAGE_LIMIT,
        "pages": [
            {
                "call_index": index,
                "raw_arguments": _bounded(call.arguments),
                "result": _bounded(call.result),
            }
            for index, call in pages[:_TRADE_PAGE_LIMIT]
        ],
    }
    history = TradeHistoryService(workspace=paths, store=store)
    evidence["saved_trade_queries"] = {
        "origin": "failure-only read of saved history; not model tool calls",
        "queries": {
            direction or "all": _attempt(
                partial(
                    _saved_trade_query,
                    history,
                    run_id,
                    direction,
                )
            )
            for direction in (None, "long", "short")
        },
        "coverage": (
            "total/matched count the saved sequence; "
            "trades show at most one row per query"
        ),
    }
    return evidence


@contextmanager
def retain_novice_failure_evidence(
    workspace: MCPWorkspace, user: ObservedIntakeUser, calls: list[ToolCall]
) -> Iterator[None]:
    """Attach evidence to the same exception, then allow normal fixture cleanup."""
    try:
        yield
    except Exception as error:
        try:
            evidence = novice_failure_evidence(workspace, user, calls)
            note = json.dumps(evidence, ensure_ascii=False, default=str, sort_keys=True)
        except Exception as diagnostic_error:
            note = json.dumps(
                {
                    "reason": "Evidence unavailable",
                    **_diagnostic_failure(diagnostic_error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        error.add_note("Novice approved-run evidence before cleanup:\n" + note)
        raise
