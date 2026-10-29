"""Research notebook snapshots, integrity, optimistic editing and bounded reads."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.app.test_diagnostics import analysis
from tests.app.test_report_service import _save

from tradingdev.adapters.cli.research_notes import main
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.research_notes import ResearchNoteStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.research_notes import (
    NoteExport,
    NoteList,
    NoteRead,
    NoteSaved,
)
from tradingdev.app.report_service import ReportService
from tradingdev.app.research_note_service import ResearchNoteService
from tradingdev.domain.research_notes import ResearchNoteDraft, ResearchNoteEntry


def draft(artifact_id: str | None = None) -> ResearchNoteDraft:
    return ResearchNoteDraft(
        title="BTC 研究 <script>",
        current_brief="後續希望評估 4h；尚未執行。",
        entries=[
            ResearchNoteEntry(
                title="回測討論",
                summary="樣本有限，不能推定波動敏感度。",
                artifact_ids=[artifact_id] if artifact_id else [],
            )
        ],
        next_steps="先確認資料與時段，再討論是否執行。",
    )


def service_at(path: Path) -> ResearchNoteService:
    workspace = WorkspacePaths(path)
    return ResearchNoteService(workspace, SQLiteStore(workspace))


def test_versions_conflicts_paging_and_pre_run_export(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    saved = service.save(draft())
    assert isinstance(saved, NoteSaved)
    first = service.export(saved.note_id, saved.version)
    assert isinstance(first, NoteExport)
    assert service.export(saved.note_id, saved.version) == first
    assert service.save(draft(), saved.note_id).success is False
    with ThreadPoolExecutor(2) as pool:
        updates = list(
            pool.map(lambda _: service.save(draft(), saved.note_id, 1), range(2))
        )
    assert sum(result.success for result in updates) == 1
    assert service.export(saved.note_id, 1).success is False
    assert (
        service.archives.download(first.archive_id, first.sha256)
        == Path(first.path).read_bytes()
    )
    loaded = service.get(saved.note_id)
    assert isinstance(loaded, NoteRead) and loaded.note.version == 2
    assert service.get(saved.note_id, offset=1).success is False
    second = service.save(draft())
    assert isinstance(second, NoteSaved)
    page = service.get(limit=1)
    assert (
        isinstance(page, NoteList)
        and page.page.total == 2
        and page.page.next_offset == 1
    )
    page2 = service.get(offset=1, limit=1)
    assert isinstance(page2, NoteList) and page2.page.next_offset is None
    assert page.page.items[0].note_id != page2.page.items[0].note_id
    assert service.get(limit=0).success is False


def test_note_size_failure_leaves_previous_revision(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    saved = service.save(draft())
    assert isinstance(saved, NoteSaved)
    too_big = draft().model_copy(
        update={
            "entries": [
                ResearchNoteEntry(title="階段", summary="大" * 2000) for _ in range(4)
            ]
        }
    )
    assert service.save(too_big, saved.note_id, 1).success is False
    assert service.notes.load(saved.note_id).version == 1


@pytest.mark.parametrize("damage", ["brief", "entry", "version"])
def test_python_updates_revalidate_without_overwriting_valid_note(
    tmp_path: Path,
    damage: str,
) -> None:
    service = service_at(tmp_path)
    proposed = draft()
    saved = service.save(proposed)
    assert isinstance(saved, NoteSaved)
    before = service.notes.load(saved.note_id)
    if damage == "brief":
        proposed.current_brief = ""
    elif damage == "entry":
        proposed.entries[0].summary = ""
    version = True if damage == "version" else 1
    assert service.save(proposed, saved.note_id, version).success is False
    assert service.notes.load(saved.note_id) == before


def test_export_report_bytes_and_snapshot_survive_note_changes(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    workspace, store = service.archives.workspace, service.notes.store
    reports = ReportService(workspace=workspace, store=store)
    _save((workspace, store, reports))
    report = reports.generate_report(["run"], sections=[], analysis=analysis())
    saved = service.save(draft(report["artifact_id"]))
    assert isinstance(saved, NoteSaved)
    result = service.export(saved.note_id, 1)
    assert isinstance(result, NoteExport), result
    assert result.html_path is not None
    with zipfile.ZipFile(result.path) as archive:
        assert set(archive.namelist()) == {
            "index.html",
            "research.json",
            "manifest.json",
            "reports/1.html",
        }
        assert archive.read("reports/1.html") == Path(report["path"]).read_bytes()
        index = archive.read("index.html").decode()
        assert "&lt;script&gt;" in index and "href='reports/1.html'" in index
        assert "尚未執行" in index and "並非逐字" in index
        assert json.loads(archive.read("research.json"))["version"] == 1
    assert service.save(draft(), saved.note_id, 1).success
    assert Path(result.path).exists()
    Path(report["path"]).unlink()
    assert service.get_archive(result.archive_id) == result
    Path(report["path"]).write_text("tampered")
    # Download reads the saved snapshot, not live source files.
    assert service.archives.download(result.archive_id, result.sha256)
    Path(result.html_path).unlink()
    retained = service.get_archive(result.archive_id)
    assert isinstance(retained, NoteExport) and retained.html_path is None
    saved2 = service.save(draft(report["artifact_id"]))
    assert isinstance(saved2, NoteSaved)
    assert service.export(saved2.note_id, 1).success is False
    Path(result.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        service.archives.download(result.archive_id, result.sha256)
    assert service.get_archive(result.archive_id).success is False


@pytest.mark.parametrize(
    "damage", ["missing", "outside", "symlink", "unallowed", "hash"]
)
def test_invalid_attachment_fails_without_partial_export(
    tmp_path: Path, damage: str
) -> None:
    service = service_at(tmp_path / "workspace")
    root = service.archives.workspace.root
    path = root / "execution_plans" / ("a" * 32) / "confirmation.html"
    path.parent.mkdir(parents=True)
    path.write_text("<html>confirmation</html>")
    artifact_id = "a" * 32 + ":confirmation_html"
    import hashlib

    service.notes.store.create_artifact(
        artifact_id=artifact_id,
        run_id=None,
        artifact_type="execution_confirmation" if damage != "unallowed" else "secret",
        path=path if damage != "outside" else tmp_path / "outside.html",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    if damage == "missing":
        path.unlink()
    elif damage == "hash":
        path.write_text("changed")
    elif damage == "symlink":
        outside = tmp_path / "outside.html"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    saved = service.save(draft(artifact_id))
    assert isinstance(saved, NoteSaved)
    assert service.export(saved.note_id, 1).success is False
    assert not (root / "research_archives").exists()


def test_publication_failure_and_retry_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = service_at(tmp_path)
    saved = service.save(draft())
    assert isinstance(saved, NoteSaved)
    record = service.notes.load(saved.note_id)

    # Fail registration after staging, forcing removal of only newly created files.
    def fail() -> sqlite3.Connection:
        raise sqlite3.OperationalError("registration failed")

    with monkeypatch.context() as patch:
        patch.setattr(service.notes.store, "connect", fail)
        with pytest.raises(sqlite3.OperationalError):
            service.archives.publish(record)
    root = tmp_path / "research_archives"
    assert all(p.name.endswith(".lock") for p in root.iterdir())
    result = service.export(saved.note_id, 1)
    assert isinstance(result, NoteExport)
    assert result.html_path is not None
    Path(result.html_path).write_text("changed")
    assert service.export(saved.note_id, 1).success is False
    assert Path(result.html_path).read_text() == "changed"


def test_cli_create_read_export_and_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "draft.json"
    path.write_text(draft().model_dump_json())
    base = ["--workspace", str(tmp_path / "workspace")]
    main([*base, "save", str(path)])
    saved = json.loads(capsys.readouterr().out)
    main([*base, "read", "--note-id", saved["note_id"]])
    assert json.loads(capsys.readouterr().out)["note"]["version"] == 1
    main([*base, "export", saved["note_id"], "1"])
    exported = json.loads(capsys.readouterr().out)
    assert Path(exported["path"]).exists()
    main([*base, "archive", exported["archive_id"]])
    assert json.loads(capsys.readouterr().out) == exported
    main([*base, "read"])
    assert json.loads(capsys.readouterr().out)["page"]["total"] == 1
    with pytest.raises(SystemExit, match="1"):
        main([*base, "export", saved["note_id"], "2"])
    assert json.loads(capsys.readouterr().out)["code"] == "research_note_conflict"
    assert isinstance(
        ResearchNoteStore(service_at(tmp_path / "workspace").notes.store)
        .load(saved["note_id"])
        .draft,
        ResearchNoteDraft,
    )


@pytest.mark.parametrize("member", ["reports/1.html", "research.json", "manifest.json"])
@pytest.mark.parametrize("damage", ["changed", "missing", "symlink"])
def test_reopened_archive_does_not_link_a_damaged_reading_tree(
    tmp_path: Path,
    member: str,
    damage: str,
) -> None:
    service = service_at(tmp_path / "workspace")
    workspace, store = service.archives.workspace, service.notes.store
    reports = ReportService(workspace=workspace, store=store)
    _save((workspace, store, reports))
    report = reports.generate_report(["run"], sections=[])
    saved = service.save(draft(report["artifact_id"]))
    assert isinstance(saved, NoteSaved)
    result = service.export(saved.note_id, 1)
    assert isinstance(result, NoteExport)
    target = Path(result.path).parent / member
    original = target.read_bytes()
    target.unlink()
    if damage == "changed":
        target.write_bytes(b"another report")
    elif damage == "symlink":
        outside = tmp_path / "replacement"
        outside.write_bytes(original)
        target.symlink_to(outside)
    reopened = service.get_archive(result.archive_id)
    assert isinstance(reopened, NoteExport) and reopened.html_path is None
    assert (
        service.archives.download(result.archive_id, result.sha256)
        == Path(result.path).read_bytes()
    )
