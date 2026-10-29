"""Select and freeze saved ML evidence without retraining or reconstructing it."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from tradingdev.domain.ml.registry import LeaderboardStore
from tradingdev.domain.ml.report import ml_report_block

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths

ML_REPORT_MAX_CANDIDATES = 50
ML_REPORT_TEMPLATE_VERSION = "4"


class MLReportError(ValueError):
    """A selection or saved-evidence failure, never permission to regenerate it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_ml_selection(
    entry_ids: list[str] | None, metric: str, allow_parameter_axes: bool
) -> None:
    """Bound the explicit selection and reject options without their evidence."""
    if (
        not isinstance(metric, str)
        or not metric.strip()
        or type(allow_parameter_axes) is not bool
        or (entry_ids is None and (metric != "ic_mean" or allow_parameter_axes))
    ):
        raise MLReportError(
            "invalid_report_ml_selection",
            "ML ranking options require selected saved ML entry IDs",
        )
    if entry_ids is not None and (
        not isinstance(entry_ids, list)
        or not 1 <= len(entry_ids) <= ML_REPORT_MAX_CANDIDATES
        or any(not isinstance(key, str) or not key.strip() for key in entry_ids)
        or len(set(entry_ids)) != len(entry_ids)
    ):
        raise MLReportError(
            "invalid_report_ml_selection",
            f"Provide one to {ML_REPORT_MAX_CANDIDATES} distinct saved ML entry IDs",
        )


def load_ml_report(
    workspace: WorkspacePaths,
    entry_ids: list[str],
    *,
    metric: str,
    allow_parameter_axes: bool,
) -> dict[str, Any]:
    """Read one complete store generation and snapshot the requested candidates.

    Re-recording an entry can change its evidence without changing its ID. The
    selected content, current definitions and ranking therefore travel inside
    the immutable report, independently of the mutable leaderboard file.
    """
    validate_ml_selection(entry_ids, metric, allow_parameter_axes)
    try:
        stored = LeaderboardStore(workspace.ml_leaderboard).entries()
        indexed = {entry.entry_id: entry for entry in stored}
        if len(indexed) != len(stored):
            raise ValueError("Saved leaderboard contains duplicate entry IDs")
        missing = [key for key in entry_ids if key not in indexed]
        if missing:
            raise MLReportError(
                "ml_entry_not_found",
                "Saved ML entries were not found: " + ", ".join(missing),
            )
        entries = [indexed[key] for key in entry_ids]
        block = ml_report_block(
            entries, metric, allow_parameter_axes=allow_parameter_axes
        )
        source = (
            json.dumps(
                [entry.to_record() for entry in entries],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except MLReportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise MLReportError("report_ml_evidence_invalid", str(exc)) from exc
    return {
        **block,
        "entry_ids": list(entry_ids),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "template_version": ML_REPORT_TEMPLATE_VERSION,
    }
