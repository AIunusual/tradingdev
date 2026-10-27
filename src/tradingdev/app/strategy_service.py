"""Application service for strategy lifecycle operations."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    now_iso,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionError,
    StrategyRevisionStore,
    UnsupportedStrategyRevisionError,
)
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.strategy import (
    BundledStrategySummary,
    GeneratedStrategySummary,
    LegacyStrategySummary,
    StrategyKind,
    StrategySummary,
)
from tradingdev.app.pagination import paginate
from tradingdev.app.quality_policy import quality_config_path
from tradingdev.domain.strategies.capabilities import StrategyCapabilities
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.domain.strategies.contract import (
    DRY_RUN_FIXTURE_ROWS,
    VALIDATE_FIXTURE_ROWS,
    SignalContractChecker,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import (
    StrategyDiagnostic,
    StrategyMetadata,
    StrategySpec,
    StrategyStatus,
    ValidationResult,
)
from tradingdev.domain.strategies.validator import (
    StrategyValidator,
    diagnostic,
    has_error,
)

_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

_EXECUTABLE_STATUSES = {StrategyStatus.RUNNABLE, StrategyStatus.PROMOTED}


class StrategyNotExecutableError(RuntimeError):
    """Raised when a strategy has not reached an executable lifecycle status."""


@dataclass(frozen=True)
class StrategySaveResult:
    """Result of saving a generated strategy draft."""

    success: bool
    strategy_id: str
    source_path: str
    config_path: str
    status: str
    revision_id: str | None = None
    error: str | None = None
    code: str | None = None


class StrategyService:
    """Own strategy draft storage, validation, and listing."""

    def __init__(
        self,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._catalog = BundledStrategyCatalog()
        self._loader = StrategyLoader(
            workspace_root=self._workspace.root,
            catalog=self._catalog,
        )
        self._validator = StrategyValidator()
        self._contract_checker = SignalContractChecker(self._loader)
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)
        self._revisions = StrategyRevisionStore(self._workspace)

    def save_draft(
        self,
        strategy_id: str,
        code: str,
        yaml_config: str,
        *,
        request_summary: str = "",
    ) -> StrategySaveResult:
        """Save a generated strategy draft under workspace only."""
        if not _VALID_NAME.fullmatch(strategy_id):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="strategy_id must be lowercase snake_case",
                code="invalid_strategy_id",
            )
        if self._catalog.get(strategy_id) is not None:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="Generated strategies cannot replace a bundled strategy ID",
                code="reserved_strategy_id",
            )
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error=f"Python syntax error: {exc}",
                code="syntax_error",
            )
        try:
            parsed = yaml.safe_load(yaml_config)
        except yaml.YAMLError as exc:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error=f"YAML parse error: {exc}",
                code="invalid_yaml",
            )
        if not isinstance(parsed, dict):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML must be a mapping",
                code="invalid_strategy_config",
            )
        strategy_section = parsed.get("strategy", {})
        if not isinstance(strategy_section, dict):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML strategy section must be a mapping",
                code="invalid_strategy_config",
            )
        class_name = strategy_section.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML missing required field: strategy.class_name",
                code="invalid_strategy_config",
            )

        metadata = self._revisions.create(
            strategy_id, code, parsed, request_summary=request_summary
        )
        return StrategySaveResult(
            success=True,
            strategy_id=strategy_id,
            revision_id=metadata.revision_id,
            source_path=metadata.source_path,
            config_path=metadata.config_path,
            status="draft",
        )

    def load(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec | None:
        """Load a bundled or generated strategy spec."""
        entry = self._catalog.get(strategy_id) if revision_id is None else None
        metadata = (
            None
            if entry is not None
            else self._revisions.load(strategy_id, revision_id)
        )
        if metadata is not None:
            return StrategySpec(
                strategy_id=metadata.strategy_id,
                revision_id=metadata.revision_id,
                class_name=metadata.class_name,
                source_path=metadata.source_path,
                config_path=metadata.config_path,
                status=metadata.status,
                kind="generated",
                metadata=metadata,
            )

        if revision_id is not None:
            return None
        if entry is None:
            return None
        return StrategySpec(
            strategy_id=strategy_id,
            class_name=entry.class_name,
            source_path=str(entry.module_source_path),
            config_path=str(entry.config_path),
            status=StrategyStatus.PROMOTED,
            kind="bundled",
            execution_capabilities=entry.execution_capabilities,
            metadata={
                "version": entry.strategy_section.get("version"),
                "declared_source_path": entry.declared_source_path,
            },
        )

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        """Return the spec for a strategy allowed to execute backtests.

        Raises:
            StrategyNotExecutableError: If the strategy is unknown or has not
                reached runnable or promoted status.
        """
        try:
            spec = self.load(strategy_id, revision_id)
        except StrategyRevisionError as exc:
            raise StrategyNotExecutableError(str(exc)) from exc
        if spec is None:
            msg = f"Strategy not found: {strategy_id}, revision={revision_id}"
            raise StrategyNotExecutableError(msg)
        if spec.status not in _EXECUTABLE_STATUSES:
            msg = (
                "Strategy must be runnable or promoted before execution. "
                f"Current status: {spec.status.value}"
            )
            raise StrategyNotExecutableError(msg)
        if isinstance(spec.metadata, StrategyMetadata):
            checks = (spec.metadata.validation, spec.metadata.dry_run)
            if any(
                check is None
                or not check.success
                or check.has_error
                or check.revision_id != spec.revision_id
                for check in checks
            ):
                msg = "Strategy revision requires successful validation and dry-run"
                raise StrategyNotExecutableError(msg)
        return spec

    def record_validation_status(
        self,
        strategy_id: str,
        result: ValidationResult | dict[str, Any],
    ) -> dict[str, Any]:
        """Record external evidence against the revision it actually checked."""
        try:
            with self._revisions.lifecycle_lock(strategy_id):
                return self._record_validation_status(strategy_id, result)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)

    def _record_validation_status(
        self,
        strategy_id: str,
        result: ValidationResult | dict[str, Any],
    ) -> dict[str, Any]:
        validation = (
            result
            if isinstance(result, ValidationResult)
            else ValidationResult.model_validate(result)
        )
        try:
            metadata = self._revisions.load(strategy_id, validation.revision_id)
            if metadata is None:
                return self._not_found(strategy_id, validation.revision_id)
            if metadata.status not in {StrategyStatus.DRAFT, StrategyStatus.VALIDATED}:
                return self._state_error(
                    metadata,
                    "record_validation_status only accepts draft or validated "
                    "strategies.",
                )
            expected_metadata = metadata.model_copy(deep=True)
            if validation.has_error:
                validation = validation.model_copy(update={"success": False})
            metadata.validation = validation
            metadata.dry_run = None
            metadata.status = (
                StrategyStatus.VALIDATED if validation.success else StrategyStatus.DRAFT
            )
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)
        return self._check_response(metadata, validation)

    def validate(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Validate the selected immutable revision with static and runtime checks."""
        return self._check_revision(strategy_id, revision_id, dry_run=False)

    def dry_run(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Run the longer fixture against the revision's successful validation."""
        return self._check_revision(strategy_id, revision_id, dry_run=True)

    def _check_revision(
        self, strategy_id: str, revision_id: str | None, *, dry_run: bool
    ) -> dict[str, Any]:
        try:
            with self._revisions.lifecycle_lock(strategy_id):
                return self._check_locked_revision(
                    strategy_id, revision_id, dry_run=dry_run
                )
        except StrategyRevisionError as exc:
            return self._revision_error(exc)

    def _check_locked_revision(
        self, strategy_id: str, revision_id: str | None, *, dry_run: bool
    ) -> dict[str, Any]:
        try:
            metadata = self._revisions.load(strategy_id, revision_id)
            if metadata is None:
                return self._not_found(strategy_id, revision_id)
            if dry_run:
                if metadata.status != StrategyStatus.VALIDATED:
                    return self._state_error(
                        metadata, "dry_run_strategy requires validated strategy status"
                    )
                evidence = metadata.validation
                if (
                    evidence is None
                    or not evidence.success
                    or evidence.has_error
                    or evidence.revision_id != metadata.revision_id
                ):
                    return self._state_error(
                        metadata, "This revision requires successful validation first"
                    )
            elif metadata.status not in {
                StrategyStatus.DRAFT,
                StrategyStatus.VALIDATED,
            }:
                return self._state_error(
                    metadata,
                    "validate_strategy only accepts draft or validated strategies. "
                    "Use save_strategy to create a new draft before revalidating "
                    f"{metadata.status.value} strategies.",
                )
            expected_metadata = metadata.model_copy(deep=True)
            self._revisions.verify(metadata)
            source_path = Path(metadata.source_path)
            diagnostics: list[StrategyDiagnostic] = []
            signal_analysis: dict[str, Any] = {}
            effective_config: dict[str, Any] | None = None
            if not dry_run:
                diagnostics.extend(self._validator.syntax_diagnostics(source_path))
                if not self._has_error(diagnostics):
                    diagnostics.extend(self._validator.static_policy_scan(source_path))
                if not self._has_error(diagnostics):
                    diagnostics.extend(self._quality_gate_diagnostics(source_path))
            if not self._has_error(diagnostics):
                contract = self._contract_checker.check(
                    metadata,
                    fixture_rows=DRY_RUN_FIXTURE_ROWS
                    if dry_run
                    else VALIDATE_FIXTURE_ROWS,
                )
                diagnostics.extend(contract["diagnostics"])
                signal_analysis = contract.get("signal_analysis", {})
                effective_config = contract.get("effective_config")
            # A new current revision may have been saved while the check ran.
            # Only this revision is updated, and modified snapshot bytes are rejected.
            self._revisions.verify(metadata)
            result = ValidationResult(
                revision_id=metadata.revision_id,
                checked_at=now_iso(),
                success=not self._has_error(diagnostics),
                diagnostics=diagnostics,
                signal_analysis=signal_analysis,
                effective_config=effective_config,
            )
            if dry_run:
                metadata.dry_run = result
                if result.success:
                    metadata.status = StrategyStatus.RUNNABLE
            else:
                metadata.validation = result
                metadata.dry_run = None
                metadata.status = (
                    StrategyStatus.VALIDATED if result.success else StrategyStatus.DRAFT
                )
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)
        return self._check_response(metadata, result)

    def promote(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Promote a runnable revision without changing the current revision pointer."""
        try:
            metadata = self._revisions.load(strategy_id, revision_id)
            if metadata is None:
                return self._not_found(strategy_id, revision_id)
            if metadata.status != StrategyStatus.RUNNABLE:
                return self._state_error(
                    metadata, "Only runnable strategies can be promoted"
                )
            self.resolve_executable(strategy_id, metadata.revision_id)
            expected_metadata = metadata.model_copy(deep=True)
            metadata.status = StrategyStatus.PROMOTED
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)
        except StrategyNotExecutableError as exc:
            return {
                "success": False,
                "error": str(exc),
                "code": "strategy_not_executable",
            }
        return {
            "success": True,
            "strategy_id": strategy_id,
            "revision_id": metadata.revision_id,
            "status": "promoted",
        }

    @staticmethod
    def _check_response(
        metadata: StrategyMetadata, result: ValidationResult
    ) -> dict[str, Any]:
        return {
            "success": result.success,
            "strategy_id": metadata.strategy_id,
            "revision_id": metadata.revision_id,
            "status": metadata.status.value,
            "diagnostics": [
                item.model_dump(mode="json") for item in result.diagnostics
            ],
            "signal_analysis": result.signal_analysis,
            "effective_config": result.effective_config,
        }

    @staticmethod
    def _state_error(metadata: StrategyMetadata, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "strategy_id": metadata.strategy_id,
            "revision_id": metadata.revision_id,
            "status": metadata.status.value,
            "error": message,
            "code": "invalid_strategy_status",
        }

    @staticmethod
    def _not_found(strategy_id: str, revision_id: str | None) -> dict[str, Any]:
        return {
            "success": False,
            "error": f"Strategy not found: {strategy_id}, revision={revision_id}",
            "code": "strategy_not_found",
        }

    @staticmethod
    def _revision_error(exc: StrategyRevisionError) -> dict[str, Any]:
        return {
            "success": False,
            "error": str(exc),
            "code": (
                "strategy_revision_required"
                if isinstance(exc, UnsupportedStrategyRevisionError)
                else "strategy_revision_invalid"
            ),
        }

    def list_strategies(self) -> list[dict[str, Any]]:
        """List bundled, revisioned, and legacy strategies requiring resaving."""
        items: list[dict[str, Any]] = []
        for entry in self._catalog.entries():
            strategy = entry.strategy_section
            items.append(
                {
                    "strategy_id": entry.strategy_id,
                    "class_name": entry.class_name,
                    "description": strategy.get("description", ""),
                    "kind": "bundled",
                    "execution_capabilities": entry.execution_capabilities.model_dump(
                        mode="json"
                    ),
                    "revision_id": None,
                    "status": "promoted",
                    "config_path": str(entry.config_path),
                    "metadata": {
                        "revision_id": None,
                        "version": strategy.get("version"),
                        "source_path": str(entry.module_source_path),
                        "parameters": strategy.get("parameters", {}),
                    },
                    "data_requirements": self._data_requirements(entry.raw_config),
                    "recent_runs": self._recent_runs(entry.strategy_id),
                }
            )
        for metadata in self._revisions.list_current():
            raw = yaml.safe_load(Path(metadata.config_path).read_text(encoding="utf-8"))
            items.append(
                {
                    "strategy_id": metadata.strategy_id,
                    "revision_id": metadata.revision_id,
                    "class_name": metadata.class_name,
                    "kind": "generated",
                    "execution_capabilities": StrategyCapabilities().model_dump(
                        mode="json"
                    ),
                    "status": metadata.status.value,
                    "source_path": metadata.source_path,
                    "config_path": metadata.config_path,
                    "metadata": metadata.model_dump(mode="json"),
                    "data_requirements": self._data_requirements(raw),
                    "recent_runs": self._recent_runs(metadata.strategy_id),
                }
            )
        items.extend(
            self._legacy_summary(strategy_id)
            for strategy_id in self._revisions.list_legacy_ids()
        )
        return items

    def _legacy_summary(self, strategy_id: str) -> dict[str, Any]:
        reserved = self._catalog.get(strategy_id) is not None
        read_step = "get_strategy(legacy=true)" if reserved else "get_strategy"
        save_step = (
            "save_strategy using a different, non-reserved strategy_id"
            if reserved
            else "save_strategy"
        )
        return {
            "strategy_id": strategy_id,
            "revision_id": None,
            "kind": "legacy",
            "execution_capabilities": StrategyCapabilities().model_dump(mode="json"),
            "status": "revision_required",
            "code": "strategy_revision_required",
            "message": (
                f"Read source/config with {read_step}, then explicitly {save_step} "
                "as a new draft and validate/dry-run its returned revision_id. "
                "Legacy status and evidence do not authorize execution."
            ),
        }

    def list_strategies_page(
        self,
        *,
        kind: StrategyKind | None = None,
        strategy_id: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[StrategySummary]:
        """Discover identities without expanding configuration or run history."""
        entries: list[StrategySummary] = []
        if kind in (None, "bundled"):
            entries.extend(
                BundledStrategySummary(
                    strategy_id=entry.strategy_id,
                    revision_id=None,
                    class_name=entry.class_name,
                    description=entry.strategy_section.get("description", ""),
                    kind="bundled",
                    execution_capabilities=entry.execution_capabilities,
                    status="promoted",
                )
                for entry in self._catalog.entries()
                if strategy_id is None or entry.strategy_id == strategy_id
            )
        if kind in (None, "generated"):
            entries.extend(
                GeneratedStrategySummary(
                    strategy_id=metadata.strategy_id,
                    revision_id=metadata.revision_id,
                    class_name=metadata.class_name,
                    kind="generated",
                    execution_capabilities=StrategyCapabilities(),
                    status=metadata.status,
                )
                for metadata in self._revisions.list_current()
                if strategy_id is None or metadata.strategy_id == strategy_id
            )
        if kind in (None, "legacy"):
            entries.extend(
                LegacyStrategySummary.model_validate(self._legacy_summary(identity))
                for identity in self._revisions.list_legacy_ids()
                if strategy_id is None or identity == strategy_id
            )
        entries.sort(
            key=lambda item: (item.strategy_id, item.kind, item.revision_id or "")
        )
        return paginate(entries, project=lambda item: item, offset=offset, limit=limit)

    def get_strategy(
        self,
        strategy_id: str,
        revision_id: str | None = None,
        *,
        legacy: bool = False,
    ) -> dict[str, Any]:
        """Read bundled or generated strategy source and config."""
        if legacy:
            if revision_id is not None:
                return self._revision_error(
                    StrategyRevisionError("Legacy source has no revision_id")
                )
            if strategy_id not in self._revisions.list_legacy_ids():
                return self._not_found(strategy_id, None)
            return self._get_legacy_strategy(strategy_id)
        try:
            metadata = (
                None
                if revision_id is None and self._catalog.get(strategy_id) is not None
                else self._revisions.load(strategy_id, revision_id)
            )
        except UnsupportedStrategyRevisionError:
            return self._get_legacy_strategy(strategy_id)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)
        if metadata is not None:
            source_path = Path(metadata.source_path)
            config_path = Path(metadata.config_path)
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "generated",
                "execution_capabilities": StrategyCapabilities().model_dump(
                    mode="json"
                ),
                "revision_id": metadata.revision_id,
                "source_code": source_path.read_text(encoding="utf-8"),
                "yaml_config": config_path.read_text(encoding="utf-8"),
                "metadata": metadata.model_dump(mode="json"),
            }
        if revision_id is not None:
            return self._not_found(strategy_id, revision_id)
        entry = self._catalog.get(strategy_id)
        if entry is not None:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "bundled",
                "execution_capabilities": entry.execution_capabilities.model_dump(
                    mode="json"
                ),
                "revision_id": None,
                "source_code": entry.module_source_path.read_text(encoding="utf-8"),
                "yaml_config": entry.config_path.read_text(encoding="utf-8"),
                "metadata": {
                    "status": "promoted",
                    "revision_id": None,
                    "source_path": str(entry.module_source_path),
                    "config_path": str(entry.config_path),
                },
            }
        return {
            "success": False,
            "error": f"Strategy not found: {strategy_id}",
            "code": "strategy_not_found",
        }

    def _get_legacy_strategy(self, strategy_id: str) -> dict[str, Any]:
        try:
            source, config = self._revisions.read_legacy_source(strategy_id)
        except StrategyRevisionError as exc:
            return self._revision_error(exc)
        return {
            **self._legacy_summary(strategy_id),
            "success": True,
            "source_code": source,
            "yaml_config": config,
        }

    def _data_requirements(self, raw_config: object) -> dict[str, Any] | None:
        if not isinstance(raw_config, dict):
            return None
        data = raw_config.get("data", {})
        if not isinstance(data, dict):
            return None
        requirements = data.get("requirements")
        return requirements if isinstance(requirements, dict) else None

    def _recent_runs(self, strategy_id: str) -> list[dict[str, Any]]:
        if not strategy_id:
            return []
        recent: list[dict[str, Any]] = []
        for run in self._store.list_runs():
            if run.get("strategy_id") != strategy_id:
                continue
            recent.append(
                {
                    "run_id": run["run_id"],
                    "job_id": run["job_id"],
                    "revision_id": run.get("revision_id"),
                    "created_at": run["created_at"],
                    "dataset_id": run.get("dataset_id"),
                    "metrics": run.get("metrics", {}),
                }
            )
            if len(recent) >= 5:
                break
        return recent

    def _quality_gate_diagnostics(self, source_path: Path) -> list[StrategyDiagnostic]:
        diagnostics: list[StrategyDiagnostic] = []
        config_path = str(quality_config_path())
        for command, label, timeout in (
            (
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--config",
                    config_path,
                    str(source_path),
                ],
                "ruff",
                30,
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--config-file",
                    config_path,
                    "--follow-imports=silent",
                    str(source_path),
                ],
                "mypy",
                60,
            ),
        ):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                diagnostics.append(
                    self._diagnostic(
                        code=f"{label}_unavailable",
                        phase="quality_gate",
                        message=f"{label} quality gate could not start: {exc}",
                        fix=(
                            "Install ruff and mypy in the server's Python "
                            "environment before running "
                            "validate_strategy."
                        ),
                    )
                )
                continue
            except subprocess.TimeoutExpired as exc:
                diagnostics.append(
                    self._diagnostic(
                        code=f"{label}_timeout",
                        phase="quality_gate",
                        message=f"{label} quality gate timed out: {exc}",
                        fix=(
                            f"Make the generated strategy analyzable within "
                            f"{timeout} seconds or investigate the quality gate."
                        ),
                    )
                )
                continue
            if result.returncode != 0:
                output = (result.stdout or result.stderr).strip()
                diagnostics.append(
                    self._diagnostic(
                        code=f"{label}_failed",
                        phase="quality_gate",
                        message=f"{label} failed: {output}",
                        fix=f"Fix the {label} diagnostics and save the strategy again.",
                    )
                )
        return diagnostics

    def _diagnostic(
        self,
        *,
        code: str,
        phase: str,
        message: str,
        level: str = "error",
        line: int | None = None,
        fix: str | None = None,
    ) -> StrategyDiagnostic:
        return diagnostic(
            code=code,
            phase=phase,
            message=message,
            level=level,
            line=line,
            fix=fix,
        )

    def _has_error(self, diagnostics: list[StrategyDiagnostic]) -> bool:
        return has_error(diagnostics)
