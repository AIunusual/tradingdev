"""Generate an offline report from saved runs, a comparison or ML evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.report_service import ReportService


class _ReportOptions(TypedDict, total=False):
    comparison_job_id: str
    ml_entry_ids: list[str]
    ml_ranking_metric: str
    ml_allow_parameter_axes: bool


def main(argv: list[str] | None = None) -> None:
    """Print a report artifact identity or fail with the same service error as MCP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-id", action="append", dest="run_ids")
    selection.add_argument("--comparison-job-id")
    parser.add_argument(
        "--ml-entry-id",
        action="append",
        dest="ml_entry_ids",
        help="Saved leaderboard entry ID; repeat for up to 50 distinct candidates.",
    )
    parser.add_argument("--ml-ranking-metric", default=None)
    parser.add_argument("--ml-allow-parameter-axes", action="store_true")
    parser.add_argument(
        "--sections",
        nargs="*",
        help=(
            "Section IDs in display order; omit for standard plus selected ML "
            "evidence, or ML alone without runs. Empty selects identity and prose. "
            "Comparison candidate evidence always remains visible. "
            "All loaded data remains embedded in HTML."
        ),
    )
    parser.add_argument("--commentary-json", type=Path)
    parser.add_argument(
        "--analysis-json", type=Path, help="Structured LLM research interpretation"
    )
    args = parser.parse_args(argv)
    input_field = "analysis"
    try:
        analysis = (
            json.loads(args.analysis_json.read_text(encoding="utf-8"))
            if args.analysis_json
            else None
        )
        input_field = "commentary"
        commentary = (
            json.loads(args.commentary_json.read_text(encoding="utf-8"))
            if args.commentary_json
            else None
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "code": f"invalid_report_{input_field}",
                    "error": str(exc),
                }
            )
        )
        raise SystemExit(1) from exc
    options: _ReportOptions = {}
    if args.comparison_job_id is not None:
        options["comparison_job_id"] = args.comparison_job_id
    if args.ml_entry_ids is not None:
        options["ml_entry_ids"] = args.ml_entry_ids
    if args.ml_ranking_metric is not None:
        options["ml_ranking_metric"] = args.ml_ranking_metric
    if args.ml_allow_parameter_axes:
        options["ml_allow_parameter_axes"] = True
    result = ReportService(workspace=WorkspacePaths(args.workspace)).generate_report(
        args.run_ids,
        sections=args.sections,
        commentary=commentary,
        analysis=analysis,
        **options,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
