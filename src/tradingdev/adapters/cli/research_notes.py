"""Save, read or export research notebooks using the same service as MCP."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import get_sqlite_store

if TYPE_CHECKING:
    from tradingdev.app.contracts.common import ContractModel
from tradingdev.app.research_note_service import ResearchNoteService
from tradingdev.domain.research_notes import ResearchNoteDraft


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    commands = parser.add_subparsers(dest="operation", required=True)
    save = commands.add_parser("save")
    save.add_argument("draft_json", type=Path)
    save.add_argument("--note-id")
    save.add_argument("--expected-version", type=int)
    read = commands.add_parser("read")
    read.add_argument("--note-id")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--limit", type=int, default=20)
    export = commands.add_parser("export")
    export.add_argument("note_id")
    export.add_argument("expected_version", type=int)
    archive = commands.add_parser("archive")
    archive.add_argument("archive_id")
    args = parser.parse_args(argv)
    workspace = WorkspacePaths(args.workspace)
    service = ResearchNoteService(workspace, get_sqlite_store(workspace))
    result: ContractModel
    try:
        if args.operation == "save":
            result = service.save(
                ResearchNoteDraft.model_validate_json(args.draft_json.read_bytes()),
                args.note_id,
                args.expected_version,
            )
        elif args.operation == "read":
            result = service.get(args.note_id, offset=args.offset, limit=args.limit)
        elif args.operation == "archive":
            result = service.get_archive(args.archive_id)
        else:
            result = service.export(args.note_id, args.expected_version)
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {"success": False, "code": "invalid_research_note", "error": str(exc)}
            )
        )
        raise SystemExit(1) from exc
    print(result.model_dump_json())
    if not result.model_dump()["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
