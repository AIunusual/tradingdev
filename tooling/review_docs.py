"""Review documentation against immutable Git trees using an isolated Codex run."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

from tooling.codex_binary import resolve_codex_binary
from tooling.process_guard import (
    process_descendants,
    process_identity,
    terminate_tree,
)
from tooling.review_evidence import (
    MAX_EVIDENCE_BYTES as MAX_EVIDENCE_BYTES,
)
from tooling.review_evidence import (
    ReviewError as ReviewError,
)
from tooling.review_evidence import (
    build_evidence as build_evidence,
)
from tooling.review_options import add_review_arguments, validate_review_options
from tooling.review_plan import (
    MAX_PROMPT_BYTES,
    MAX_TOTAL_PROMPT_BYTES,
    ReviewPlan,
    build_review_plan,
)
from tooling.review_scope import SCOPES

# Codex transport ceiling; context policy imposes smaller independent byte budgets.
MAX_PROMPT_CHARS = 1_024_000
MAX_PARALLEL_REVIEWS = 8
# Per-channel output limits in Unicode characters, not model token budgets.
MAX_DIAGNOSTIC_CHARS = 4000
SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["file", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["passed", "findings"],
    "additionalProperties": False,
}


class _ReviewCancelledError(ReviewError):
    """A batch stopped normally after cancellation by the review coordinator."""


def _clip_diagnostic(value: str) -> str:
    marker = " [truncated]"
    if len(value) <= MAX_DIAGNOSTIC_CHARS:
        return value
    return value[: MAX_DIAGNOSTIC_CHARS - len(marker)] + marker


@dataclass
class _EventDiagnostics:
    failures: str = ""
    invalid: bool = False
    tool_attempt: bool = False


def _inspect_events(path: Path) -> _EventDiagnostics:
    """Inspect all events, retaining only bounded, recognized error fields."""
    result = _EventDiagnostics()
    with path.open("rb") as stream:
        for line in stream:
            try:
                event = json.loads(line.decode("utf-8"))
            except ValueError, RecursionError:
                result.invalid = True
                continue
            if not isinstance(event, dict):
                result.invalid = True
                continue
            kind = event.get("type")
            if not isinstance(kind, str) or not kind:
                result.invalid = True
                continue
            if kind in ("error", "turn.failed"):
                details = [str(kind)]
                error = event.get("error")
                if isinstance(error, str):
                    details.append(
                        json.dumps(_clip_diagnostic(error), ensure_ascii=False)
                    )
                for source in (event, error):
                    if isinstance(source, dict):
                        for field in ("code", "message"):
                            value = source.get(field)
                            if isinstance(value, str) and value:
                                details.append(
                                    f"{field}="
                                    + json.dumps(
                                        _clip_diagnostic(value), ensure_ascii=False
                                    )
                                )
                result.failures = _clip_diagnostic(
                    "; ".join(filter(None, (result.failures, " ".join(details))))
                )
            item = event.get("item")
            if isinstance(item, dict):
                item_kind = item.get("type")
                if not isinstance(item_kind, str) or not item_kind:
                    result.invalid = True
                elif item_kind in (
                    "command_execution",
                    "file_change",
                    "mcp_tool_call",
                    "web_search",
                ):
                    result.tool_attempt = True
    return result


def _stderr_diagnostic(path: Path) -> str:
    """Read a bounded tail; a UTF-8 character occupies at most four bytes."""
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 4 * MAX_DIAGNOSTIC_CHARS))
        tail = stream.read().decode("utf-8", errors="replace").strip()
    if size > 4 * MAX_DIAGNOSTIC_CHARS or len(tail) > MAX_DIAGNOSTIC_CHARS:
        marker = "[truncated] "
        return marker + tail[-(MAX_DIAGNOSTIC_CHARS - len(marker)) :]
    return tail


def _command(root: Path) -> list[str]:
    command = [
        resolve_codex_binary(),
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--color",
        "never",
        "--cd",
        str(root),
        "--output-schema",
        str(root / "schema.json"),
        "--output-last-message",
        str(root / "verdict.json"),
    ]
    config: dict[str, Any] = {
        "features.shell_tool": False,
        "features.shell_snapshot": False,
        "web_search": "disabled",
        "history.persistence": "none",
        "log_dir": str(root / "logs"),
        "sqlite_home": str(root / "state"),
    }
    for key, value in config.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    if model := os.environ.get("TRADINGDEV_CODEX_MODEL"):
        command.extend(["--model", model])
    return [*command, "-"]


def _run(
    root: Path,
    prompt: str,
    timeout_seconds: float,
    *,
    deadline: float | None = None,
    cancelled: Event | None = None,
) -> None:
    deadline = deadline if deadline is not None else time.monotonic() + timeout_seconds
    command = _command(root)
    if time.monotonic() >= deadline:
        raise ReviewError("Codex documentation review timed out before batch start.")
    if cancelled is not None and cancelled.is_set():
        raise _ReviewCancelledError("Codex documentation review cancelled.")
    with (
        (root / "events.jsonl").open("w", encoding="utf-8") as events,
        (root / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=stderr,
            text=True,
            cwd=root,
            start_new_session=True,
            env={**os.environ, "XDG_CACHE_HOME": str(root / "cache")},
        )
        identity = process_identity(process.pid)
        descendants = set()
        pending_input: str | None = prompt
        try:
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise _ReviewCancelledError("Codex documentation review cancelled.")
                descendants.update(process_descendants(identity))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReviewError(
                        "Codex documentation review timed out after "
                        f"{timeout_seconds:g}s."
                    )
                try:
                    process.communicate(
                        input=pending_input, timeout=min(0.2, remaining)
                    )
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
        finally:
            try:
                terminate_tree(identity, descendants)
            finally:
                process.communicate(timeout=5)
    diagnostics = _inspect_events(root / "events.jsonl")
    if process.returncode:
        parts = []
        if stderr_text := _stderr_diagnostic(root / "stderr.log"):
            parts.append(f"stderr: {stderr_text}")
        if diagnostics.failures:
            parts.append(f"events: {diagnostics.failures}")
        if not parts:
            parts.append("No stderr or recognizable error event was provided.")
        if diagnostics.invalid:
            parts.append("Codex emitted an invalid JSON event.")
        raise ReviewError(
            f"Codex documentation review exited {process.returncode}: "
            + "; ".join(parts)
        )
    if diagnostics.failures:
        raise ReviewError(f"Codex reported a failed review: {diagnostics.failures}")
    if diagnostics.invalid:
        raise ReviewError("Codex emitted an invalid JSON event.")
    if diagnostics.tool_attempt:
        raise ReviewError(
            "Documentation reviewer attempted a tool call; "
            "evidence-only review required."
        )


def _validate_verdict(raw: str) -> None:
    verdict = json.loads(raw)
    if not isinstance(verdict, dict) or set(verdict) != {"passed", "findings"}:
        raise ReviewError("Codex returned a malformed documentation verdict.")
    passed, findings = verdict["passed"], verdict["findings"]
    if not isinstance(passed, bool) or not isinstance(findings, list):
        raise ReviewError("Codex returned a malformed documentation verdict.")
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or set(finding) != {"file", "explanation"}
            or not all(
                isinstance(value, str) and value.strip() for value in finding.values()
            )
        ):
            raise ReviewError("Codex returned a malformed documentation finding.")
    if passed != (len(findings) == 0):
        raise ReviewError("Codex returned an inconsistent documentation verdict.")
    if findings:
        raise ReviewError(
            "Documentation needs updates:\n"
            + "\n".join(
                f"- {finding['file']}: {finding['explanation']}" for finding in findings
            )
        )


def prepare_review(
    base: str,
    tree: str,
    *,
    repo: Path | None = None,
    max_evidence_bytes: int = MAX_EVIDENCE_BYTES,
    review_scopes: tuple[str, ...] | None = None,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
    max_prompt_bytes: int | None = None,
    max_total_prompt_bytes: int | None = None,
) -> ReviewPlan:
    """Inspect immutable evidence and coverage without starting models or tests."""
    try:
        validate_review_options(
            review_scopes=review_scopes,
            review_context=review_context,
            review_context_for=review_context_for,
            max_prompt_bytes=max_prompt_bytes,
            max_total_prompt_bytes=max_total_prompt_bytes,
        )
    except RuntimeError as exc:
        raise ReviewError(str(exc)) from exc
    if (
        isinstance(max_evidence_bytes, bool)
        or not isinstance(max_evidence_bytes, int)
        or max_evidence_bytes <= 0
    ):
        raise ReviewError("Review evidence budget must be a positive integer.")
    evidence = build_evidence(
        repo or Path.cwd(),
        base,
        tree,
        max_bytes=max_evidence_bytes,
        review_context=review_context,
        review_context_for=review_context_for,
    )
    return build_review_plan(
        evidence,
        review_scopes=review_scopes,
        max_prompt_bytes=MAX_PROMPT_BYTES
        if max_prompt_bytes is None
        else max_prompt_bytes,
        max_total_prompt_bytes=(
            MAX_TOTAL_PROMPT_BYTES
            if max_total_prompt_bytes is None
            else max_total_prompt_bytes
        ),
        max_prompt_chars=MAX_PROMPT_CHARS,
    )


def build_review_prompts(evidence: str) -> list[str]:
    """Return bounded default prompts from the same validated planner as the CLI."""
    return list(
        build_review_plan(
            evidence,
            max_prompt_bytes=MAX_PROMPT_BYTES,
            max_total_prompt_bytes=MAX_TOTAL_PROMPT_BYTES,
            max_prompt_chars=MAX_PROMPT_CHARS,
        ).prompts
    )


def review(
    base: str,
    tree: str,
    *,
    repo: Path | None = None,
    timeout_seconds: float = 180,
    max_evidence_bytes: int = MAX_EVIDENCE_BYTES,
    review_scopes: tuple[str, ...] | None = None,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
    max_prompt_bytes: int | None = None,
    max_total_prompt_bytes: int | None = None,
) -> None:
    """Raise ReviewError for stale documentation or an incomplete review."""
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ReviewError("Review timeout must be finite and positive.")
    plan = prepare_review(
        base,
        tree,
        repo=repo,
        max_evidence_bytes=max_evidence_bytes,
        review_scopes=review_scopes,
        review_context=review_context,
        review_context_for=review_context_for,
        max_prompt_bytes=max_prompt_bytes,
        max_total_prompt_bytes=max_total_prompt_bytes,
    )
    plan.verify()
    prompts = plan.prompts
    print(
        f"Documentation review plan {plan.manifest['plan_hash']}: "
        f"{len(plan.manifest['path_owners'])} assigned paths; "
        f"scopes={','.join(plan.manifest['selected_scopes'])}; "
        f"{plan.manifest['evidence_utf8_bytes']} evidence bytes; "
        f"{len(prompts)} batch(es); "
        f"{plan.manifest['total_prompt_utf8_bytes']} total prompt bytes; "
        f"largest {max(len(p.encode('utf-8')) for p in prompts)} bytes / "
        f"{max(map(len, prompts))} characters. These are not token counts.",
        flush=True,
    )
    deadline = time.monotonic() + timeout_seconds
    cancelled = Event()

    def run_batch(prompt: str) -> None:
        if time.monotonic() >= deadline:
            raise ReviewError(
                "Codex documentation review timed out before batch start."
            )
        with TemporaryDirectory(prefix="tradingdev-doc-review-") as temporary:
            root = Path(temporary)
            (root / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
            _run(root, prompt, timeout_seconds, deadline=deadline, cancelled=cancelled)
            _validate_verdict((root / "verdict.json").read_text(encoding="utf-8"))

    try:
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REVIEWS) as executor:
            futures: list[Future[None]] = []
            try:
                for prompt in prompts:
                    futures.append(executor.submit(run_batch, prompt))
                for index, future in enumerate(futures, start=1):
                    try:
                        future.result()
                    except ReviewError as exc:
                        errors.append(f"Batch {index}/{len(prompts)}: {exc}")
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ) as exc:
                        errors.append(
                            f"Batch {index}/{len(prompts)}: Documentation review "
                            f"could not complete: {exc}"
                        )
            except BaseException as interrupted:
                cancelled.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                # Joining alone hides failures held by the running futures.
                # Preserve the interrupt and report any cleanup/execution error.
                for index, future in enumerate(futures, start=1):
                    if future.cancelled():
                        continue
                    error = future.exception()
                    if (
                        error is not None
                        and error is not interrupted
                        and not isinstance(error, _ReviewCancelledError)
                    ):
                        interrupted.add_note(
                            f"Batch {index}/{len(prompts)}: "
                            f"{type(error).__name__}: {error}"
                        )
                raise
        if errors:
            raise ReviewError("Documentation review failed:\n" + "\n".join(errors))
        plan.verify()
    except ReviewError:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise ReviewError(f"Documentation review could not complete: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base")
    parser.add_argument("--tree")
    parser.add_argument("--list-scopes", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print coverage and context sizes; no model, tests or passing receipt",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-evidence-bytes", type=int, default=MAX_EVIDENCE_BYTES)
    add_review_arguments(parser)
    args = parser.parse_args()
    if args.list_scopes:
        print(json.dumps(SCOPES, ensure_ascii=False, indent=2))
        return 0
    if args.base is None or args.tree is None:
        parser.error("--base and --tree are required unless --list-scopes is used")
    options = {
        "review_scopes": None
        if args.review_scopes is None
        else tuple(args.review_scopes),
        "review_context": tuple(args.review_context),
        "review_context_for": tuple(args.review_context_for),
        "max_prompt_bytes": args.max_prompt_bytes,
        "max_total_prompt_bytes": args.max_total_prompt_bytes,
    }
    try:
        if args.plan_only:
            plan = prepare_review(
                args.base,
                args.tree,
                repo=args.repo,
                max_evidence_bytes=args.max_evidence_bytes,
                **options,
            )
            print(json.dumps(plan.manifest, ensure_ascii=False, indent=2))
            return 0
        review(
            args.base,
            args.tree,
            repo=args.repo,
            timeout_seconds=args.timeout_seconds,
            max_evidence_bytes=args.max_evidence_bytes,
            **options,
        )
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Documentation review passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
