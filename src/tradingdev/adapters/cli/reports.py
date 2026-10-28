"""Generate an offline report from saved runs or a finalized fixed comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.report_service import ReportService


def main(argv: list[str] | None = None) -> None:
    """Print a report artifact identity or fail with the same service error as MCP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", action="append", dest="run_ids")
    selection.add_argument("--comparison-job-id")
    parser.add_argument(
        "--sections",
        nargs="*",
        help=(
            "Section IDs in display order; omit for standard, empty for identity "
            "and prose. Comparison candidate evidence always remains visible. "
            "All loaded data remains embedded in HTML."
        ),
    )
    parser.add_argument("--commentary-json", type=Path)
    args = parser.parse_args(argv)
    try:
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
                    "code": "invalid_report_commentary",
                    "error": str(exc),
                }
            )
        )
        raise SystemExit(1) from exc
    result = ReportService(workspace=WorkspacePaths(args.workspace)).generate_report(
        args.run_ids,
        sections=args.sections,
        commentary=commentary,
        **(
            {"comparison_job_id": args.comparison_job_id}
            if args.comparison_job_id is not None
            else {}
        ),
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
