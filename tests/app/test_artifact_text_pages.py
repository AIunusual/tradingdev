"""Bounded artifact reads preserve bytes, UTF-8 characters and content identity."""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import TYPE_CHECKING, TypedDict

import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage import artifact_text
from tradingdev.adapters.storage.artifact_text import DEFAULT_TEXT_LIMIT, MAX_TEXT_LIMIT
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.contracts.research import ArtifactResponse
from tradingdev.mcp.tools import artifacts

if TYPE_CHECKING:
    from pathlib import Path


class _PageOptions(TypedDict, total=False):
    offset: int
    limit: int
    expected_sha256: str


def _service(tmp_path: Path, content: bytes) -> tuple[ArtifactService, Path]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    path = tmp_path / "content.txt"
    path.write_bytes(content)
    store.create_artifact(
        artifact_id="text", run_id=None, artifact_type="fixture", path=path
    )
    return ArtifactService(workspace=workspace, store=store), path


def test_utf8_pages_reassemble_original_text_and_bind_raw_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "首行\r\n第二行🐋e\u0301\r尾\n" * 4
    raw = text.encode("utf-8")
    service, _ = _service(tmp_path, raw)
    # Split multi-byte code points between binary reads as well as page edges.
    monkeypatch.setattr(artifact_text, "_READ_BYTES", 3)
    offset = 0
    expected_sha256 = None
    collected = []
    while True:
        response = ArtifactResponse.model_validate(
            service.get_artifact(
                "text",
                include_content=True,
                offset=offset,
                limit=5,
                expected_sha256=expected_sha256,
            )
        )
        page = response.content_page
        assert page is not None and response.content is not None
        assert page.offset == offset and page.limit == 5
        assert page.total_characters == len(text)
        assert page.content_sha256 == hashlib.sha256(raw).hexdigest()
        assert response.content == text[offset : offset + 5]
        collected.append(response.content)
        if page.next_offset is None:
            break
        assert page.next_offset == offset + len(response.content)
        offset = page.next_offset
        expected_sha256 = page.content_sha256
    assert "".join(collected) == text


def test_default_page_is_bounded_for_large_artifact(tmp_path: Path) -> None:
    text = "資料" * (MAX_TEXT_LIMIT * 4)
    service, _ = _service(tmp_path, text.encode())
    result = service.get_artifact("text", include_content=True)
    assert result["content"] == text[:DEFAULT_TEXT_LIMIT]
    assert result["content_page"]["next_offset"] == DEFAULT_TEXT_LIMIT
    assert result["content_page"]["total_characters"] == len(text)


@pytest.mark.parametrize("text,offset", [("", 0), ("abc", 3)])
def test_empty_file_and_exact_end_return_complete_empty_page(
    tmp_path: Path, text: str, offset: int
) -> None:
    service, _ = _service(tmp_path, text.encode())
    result = service.get_artifact("text", include_content=True, offset=offset)
    assert result["content"] == ""
    assert result["content_page"]["total_characters"] == len(text)
    assert result["content_page"]["next_offset"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"offset": -1},
        {"offset": 4},
        {"offset": True},
        {"limit": 0},
        {"limit": MAX_TEXT_LIMIT + 1},
        {"limit": False},
        {"expected_sha256": "bad"},
    ],
)
def test_invalid_content_requests_return_explicit_errors(
    tmp_path: Path, options: _PageOptions
) -> None:
    service, _ = _service(tmp_path, b"abc")
    result = service.get_artifact("text", include_content=True, **options)
    assert result["success"] is False
    assert result["code"] == "invalid_artifact_request"
    assert "content" not in result and "content_page" not in result


@pytest.mark.parametrize(
    "options", [{"offset": 1}, {"limit": 1}, {"expected_sha256": "a" * 64}]
)
def test_metadata_requests_do_not_silently_ignore_paging(
    tmp_path: Path, options: _PageOptions
) -> None:
    service, _ = _service(tmp_path, b"abc")
    assert service.get_artifact("text", **options)["code"] == "invalid_artifact_request"


def test_metadata_lookup_does_not_require_a_readable_file(tmp_path: Path) -> None:
    service, path = _service(tmp_path, b"abc")
    path.unlink()
    result = service.get_artifact("text")
    assert result["success"] is True and "content" not in result
    assert (
        service.get_artifact("text", include_content=True)["code"]
        == "artifact_file_missing"
    )


def test_changed_content_rejects_a_continuation_without_returning_bytes(
    tmp_path: Path,
) -> None:
    service, path = _service(tmp_path, b"first contents")
    initial = service.get_artifact("text", include_content=True, limit=5)
    path.write_text("other contents", encoding="utf-8")
    result = service.get_artifact(
        "text",
        include_content=True,
        offset=initial["content_page"]["next_offset"],
        expected_sha256=initial["content_page"]["content_sha256"],
    )
    assert result["code"] == "artifact_content_changed"
    assert "content" not in result


def test_non_utf8_beyond_requested_page_is_rejected(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, b"valid prefix" + b"\xff")
    result = service.get_artifact("text", include_content=True, limit=3)
    assert result["code"] == "artifact_not_text" and "content" not in result


def test_file_changed_during_streaming_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, path = _service(tmp_path, b"original")
    original_fstat = os.fstat
    observed = 0

    def change_after_stat(fd: int) -> os.stat_result:
        nonlocal observed
        result = original_fstat(fd)
        observed += 1
        if observed == 1:
            path.write_bytes(b"longer replacement")
        return result

    monkeypatch.setattr(os, "fstat", change_after_stat)
    result = service.get_artifact("text", include_content=True)
    assert result["code"] == "artifact_content_changed"
    assert "content" not in result


def test_file_read_failure_is_an_application_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = _service(tmp_path, b"abc")

    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("fixture denies file read")

    monkeypatch.setattr("tradingdev.app.artifact_service.read_text_page", denied)
    result = service.get_artifact("text", include_content=True)
    assert result["code"] == "artifact_file_unreadable" and "content" not in result


def test_mcp_advertises_page_bounds_and_returns_chunk_identity(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, "一二三四五".encode())
    server = MCPServer("artifact-page-test")
    artifacts.register(server, service)

    async def check() -> None:
        tool = next(
            item for item in await server.list_tools() if item.name == "get_artifact"
        )
        properties = tool.input_schema["properties"]
        assert properties["offset"]["minimum"] == 0
        assert properties["limit"]["maximum"] == MAX_TEXT_LIMIT
        result = await server.call_tool(
            "get_artifact", {"artifact_id": "text", "include_content": True, "limit": 2}
        )
        assert (
            isinstance(result, CallToolResult) and result.structured_content is not None
        )
        response = ArtifactResponse.model_validate(result.structured_content["result"])
        assert response.content == "一二"
        assert response.content_page is not None
        assert response.content_page.next_offset == 2
        assert response.content_page.total_characters == 5

    asyncio.run(check())
