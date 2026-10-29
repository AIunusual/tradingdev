"""Real MCP notebook workflow and actual HTTP capability downloads."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import pytest
from tests.app.test_research_notes import draft
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.mcp.http_harness import http_plan_server


def test_stdio_note_export_has_local_paths() -> None:
    async def check() -> None:
        with temporary_mcp_workspace() as workspace:
            async with workspace.connect() as client:
                saved = await client.call(
                    "save_research_note", draft=draft().model_dump()
                )
                assert saved["success"]
                result = await client.call(
                    "export_research_note", note_id=saved["note_id"], expected_version=1
                )
                assert result["success"] and result["download_url"] is None
                assert Path(result["path"]).is_file()
                retained = await client.call(
                    "get_research_archive", archive_id=result["archive_id"]
                )
                assert retained == result
                read = await client.call("get_research_notes", note_id=saved["note_id"])
                assert read["note"]["draft"] == draft().model_dump()

    asyncio.run(check())


def test_http_download_identity_expiration_and_missing_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            saved = await client.call("save_research_note", draft=draft().model_dump())
            result = await client.call(
                "export_research_note", note_id=saved["note_id"], expected_version=1
            )
            assert result["success"] and result["download_url"]
            url = result["download_url"]
            async with httpx.AsyncClient() as http:
                response = await http.get(url)
                assert response.status_code == 200
                assert response.content == Path(result["path"]).read_bytes()
                assert response.headers["content-type"] == "application/zip"
                assert response.headers["cache-control"] == "no-store"
                assert (await http.get(url.split("?", 1)[0])).status_code == 403
                assert (
                    await http.get(url.replace("signature=", "signature=x"))
                ).status_code == 403
                assert (
                    await http.get(url.split("signature=", 1)[0] + "signature=%C3%A9")
                ).status_code == 403

                with monkeypatch.context() as patch:
                    patch.setattr(
                        "tradingdev.mcp.archive_downloads.time.time",
                        lambda: result["download_expires_at"] + 1,
                    )
                    assert (await http.get(url)).status_code == 403
                updated = await client.call(
                    "save_research_note",
                    draft=draft().model_dump(),
                    note_id=saved["note_id"],
                    expected_version=1,
                )
                assert updated["version"] == 2
                retained = await client.call(
                    "get_research_archive", archive_id=result["archive_id"]
                )
                assert retained["success"] and retained["version"] == 1
                assert (
                    await http.get(retained["download_url"])
                ).content == response.content
                from tradingdev.adapters.storage.filesystem import WorkspacePaths
                from tradingdev.adapters.storage.sqlite import SQLiteStore
                from tradingdev.app.contracts.research_notes import NoteExport
                from tradingdev.app.research_note_service import ResearchNoteService
                from tradingdev.mcp.archive_downloads import ArchiveDownloads

                # A newly started server has a new key but can sign the retained ZIP.
                workspace = WorkspacePaths(runtime.root / "workspace")
                replacement = ArchiveDownloads(
                    ResearchNoteService(workspace, SQLiteStore(workspace)).archives
                )
                fresh = replacement.attach(
                    NoteExport.model_validate(retained),
                    url.split("/research-download", 1)[0],
                )
                assert fresh.download_url is not None
                assert fresh.download_url != retained["download_url"]
                assert (await http.get(fresh.download_url)).status_code == 403
                from starlette.applications import Starlette
                from starlette.routing import Route

                app = Starlette(
                    routes=[Route("/research-download/{archive_id}", replacement.serve)]
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app)
                ) as restarted:
                    assert (
                        await restarted.get(fresh.download_url)
                    ).content == response.content
                    assert (await restarted.get(url)).status_code == 403
                Path(result["path"]).write_bytes(b"tampered")
                assert (await http.get(url)).status_code == 404

    asyncio.run(check())
