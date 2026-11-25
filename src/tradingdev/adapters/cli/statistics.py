"""Analyze one saved run/scope with the shared statistics service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.statistics_service import StatisticsService


def main(argv: list[str] | None = None) -> None:
    """Print bounded JSON; never execute a strategy or download market data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scope",
        default=None,
        help="Original saved scalar scope; omit for the run's default scope.",
    )
    parser.add_argument(
        "--save-artifact",
        action="store_true",
        help="Publish a derived immutable Markdown artifact in addition to JSON.",
    )
    args = parser.parse_args(argv)
    result = StatisticsService(workspace=WorkspacePaths(args.workspace)).analyze(
        args.run_id, args.scope, save_artifact=args.save_artifact
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
