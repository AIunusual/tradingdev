"""Static validation for generated strategy source code."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from tradingdev.domain.strategies.schemas import StrategyDiagnostic

if TYPE_CHECKING:
    from pathlib import Path

_BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "ccxt",
    "shutil",
    "pathlib",
}
_ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "math",
    "statistics",
    "typing",
    "typing_extensions",
    "numpy",
    "pandas",
    "talib",
    "tradingdev",
}
_BANNED_CALLS = {
    "open",
    "eval",
    "exec",
    "__import__",
}
_BANNED_ATTR_CALLS = {
    "chmod",
    "chown",
    "mkdir",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "unlink",
    "write_bytes",
    "write_text",
}


def static_policy_payload() -> dict[str, list[str]]:
    """Expose the exact static rules; this policy is not a Python sandbox."""
    return {
        "allowed_import_roots": sorted(_ALLOWED_IMPORT_ROOTS),
        "blocked_import_roots": sorted(_BANNED_IMPORTS),
        "blocked_call_names": sorted(_BANNED_CALLS),
        "blocked_attribute_calls": sorted(_BANNED_ATTR_CALLS),
    }


class StrategyValidator:
    """Pure strategy source validation helpers."""

    def syntax_diagnostics(self, source_path: Path) -> list[StrategyDiagnostic]:
        """Return syntax diagnostics for a source file."""
        source = source_path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return [
                diagnostic(
                    code="syntax_error",
                    phase="syntax",
                    message=str(exc),
                    line=exc.lineno,
                    fix=(
                        "Fix Python syntax before validation can run static "
                        "policy checks."
                    ),
                )
            ]
        return []

    def static_policy_scan(self, source_path: Path) -> list[StrategyDiagnostic]:
        """Return import and filesystem policy diagnostics."""
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        diagnostics: list[StrategyDiagnostic] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", maxsplit=1)[0]
                    diagnostics.extend(_import_diagnostics(root, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", maxsplit=1)[0]
                diagnostics.extend(_import_diagnostics(root, node.lineno))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _BANNED_CALLS
            ):
                diagnostics.append(
                    diagnostic(
                        code="banned_call",
                        phase="static_policy",
                        message=f"banned call: {node.func.id}",
                        line=node.lineno,
                        fix=(
                            "Remove dynamic execution or raw file access from "
                            "strategy code."
                        ),
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _BANNED_ATTR_CALLS
            ):
                diagnostics.append(
                    diagnostic(
                        code="banned_attribute_call",
                        phase="static_policy",
                        message=f"banned call: {node.func.attr}",
                        line=node.lineno,
                        fix=(
                            "Generated strategies must not create, modify, or "
                            "delete files."
                        ),
                    )
                )
        return diagnostics


def diagnostic(
    *,
    code: str,
    phase: str,
    message: str,
    level: str = "error",
    line: int | None = None,
    fix: str | None = None,
) -> StrategyDiagnostic:
    """Build a typed diagnostic."""
    return StrategyDiagnostic(
        level="warning" if level == "warning" else "error",
        code=code,
        phase=phase,
        message=message,
        line=line,
        fix=fix,
    )


def has_error(diagnostics: list[StrategyDiagnostic]) -> bool:
    """Return whether diagnostics contain an error."""
    return any(item.level == "error" for item in diagnostics)


def _import_diagnostics(root: str, line: int) -> list[StrategyDiagnostic]:
    if root in _BANNED_IMPORTS:
        return [
            diagnostic(
                code="banned_import",
                phase="static_policy",
                message=f"banned import: {root}",
                line=line,
                fix=(
                    "Remove the import; generated strategies may not use network, "
                    "subprocess, or filesystem modules."
                ),
            )
        ]
    if root not in _ALLOWED_IMPORT_ROOTS:
        return [
            diagnostic(
                code="import_not_allowed",
                phase="static_policy",
                message=f"import not allowed: {root}",
                line=line,
                fix=(
                    "Allowed import roots: "
                    f"{', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}. "
                    "Prefer tradingdev.domain.indicators for standard indicators."
                ),
            )
        ]
    return []
