"""Real HTTP MCP issues hash-bound links usable by an independent HTTP client."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
import pytest
from tests.app.test_diagnostics import analysis
from tests.app.test_report_service import _save
from tests.mcp.http_harness import http_plan_server

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.mcp.artifact_downloads import validate_public_base_url


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_confirmation_url_preserves_bytes_and_never_approves(
    monkeypatch: pytest.MonkeyPatch,
    transport: Literal["sse", "streamable-http"],
    tmp_path: Path,
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(monkeypatch, transport=transport) as runtime,
            runtime.connect() as client,
        ):
            prepared = await client.call(
                "prepare_backtest", **runtime.context.arguments
            )
            descriptor = prepared["confirmation_artifact"]
            args = {
                "artifact_id": descriptor["artifact_id"],
                "expected_sha256": descriptor["sha256"],
            }
            result = await client.call("get_artifact_download", **args)
            assert result["success"] and result["mime_type"] == "text/html"
            url = result["download_url"]
            async with httpx.AsyncClient() as reader:
                response = await reader.get(url)
                assert response.status_code == 200
                delivered = tmp_path / "client.html"
                delivered.write_bytes(response.content)
                assert (
                    hashlib.sha256(delivered.read_bytes()).hexdigest()
                    == descriptor["sha256"]
                )
                assert "<html" in response.text and response.text.endswith("</html>")
                assert response.headers["content-type"] == "text/html; charset=utf-8"
                assert response.headers["content-disposition"].startswith("inline;")
                assert response.headers["cache-control"] == "no-store"
                assert response.headers["referrer-policy"] == "no-referrer"
                csp = response.headers["content-security-policy"]
                assert "sandbox allow-scripts allow-downloads;" in csp
                assert "allow-same-origin" not in csp and "connect-src 'none'" in csp
                for invalid in (
                    url.split("?", 1)[0],
                    url.replace("expires=", "expires=x"),
                    url.replace("sha256=" + descriptor["sha256"], "sha256=" + "0" * 64),
                    url.replace(
                        quote(descriptor["artifact_id"], safe=""),
                        "b" * 32 + "%3Aconfirmation_html",
                    ),
                ):
                    assert (await reader.get(invalid)).status_code == 403
                with monkeypatch.context() as patch:
                    patch.setattr(
                        "tradingdev.mcp.artifact_downloads.time.time",
                        lambda: result["download_expires_at"],
                    )
                    assert (await reader.get(url)).status_code == 403
                    refreshed = await client.call("get_artifact_download", **args)
                    assert (
                        refreshed["download_expires_at"] > result["download_expires_at"]
                    )
                    assert (
                        await reader.get(refreshed["download_url"])
                    ).status_code == 200
                assert (
                    await client.call("get_execution_plan", plan_id=prepared["plan_id"])
                    == prepared
                )
                Path(descriptor["server_path"]).write_bytes(b"changed")
                assert (await reader.get(url)).status_code == 404
                rejected = await client.call("get_artifact_download", **args)
                assert (
                    rejected["code"] == "artifact_delivery_invalid"
                    and "download_url" not in rejected
                )
                assert "bytes changed" in rejected["error"]
            assert runtime.context.runner.calls == []
            assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())


def test_report_download_and_explicit_proxy_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(
                monkeypatch, public_base_url="https://reading.example/research/"
            ) as runtime,
            runtime.connect() as client,
        ):
            workspace = WorkspacePaths(runtime.root / "workspace")
            store = SQLiteStore(workspace)
            reports = ReportService(workspace=workspace, store=store)
            _save((workspace, store, reports))
            report = reports.generate_report(["run"], sections=[], analysis=analysis())
            assert report["success"]
            result = await client.call(
                "get_artifact_download",
                artifact_id=report["artifact_id"],
                expected_sha256=report["sha256"],
            )
            assert result["success"]
            assert result["download_url"].startswith(
                "https://reading.example/research/artifact-download/"
            )
            # Emulate a proxy stripping the prefix; no external network is used.
            host, port = runtime.address
            backend = result["download_url"].replace(
                "https://reading.example/research", f"http://{host}:{port}"
            )
            async with httpx.AsyncClient() as reader:
                response = await reader.get(backend)
                assert response.status_code == 200
                assert response.content == Path(report["path"]).read_bytes()
            assert len(store.list_runs()) == 1
            assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())


@pytest.mark.parametrize(
    "value",
    [
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://example.com?x=1",
        "https://example.com?",
        "https://example.com#",
        "https://example.com/prefix?",
        "https://example.com/prefix#",
        "https://example.com/#x",
        "https://example.com/../secret",
        "https://example.com/%2e%2e",
        "https://example.com//prefix",
        "https://example.com:99999",
        "https://ex ample.com",
    ],
)
def test_public_base_url_rejects_ambiguous_or_credentialed_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_public_base_url(value)


def test_parameter_plateau_json_over_http(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.app.test_parameter_plateau import _record

    from tradingdev.app.plateau_service import PlateauService

    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            workspace = WorkspacePaths(runtime.root / "workspace")
            sql = SQLiteStore(workspace)
            result = PlateauService(workspace=workspace, store=sql).analyze(
                metric="signal.ic_mean",
                axes={"/threshold": [1, 2, 3]},
                entry_ids=_record(workspace, [0.5, None, 0.6]),
            )
            descriptor = result["artifact"]
            response = await client.call(
                "get_artifact_download",
                artifact_id=descriptor["artifact_id"],
                expected_sha256=descriptor["sha256"],
            )
            assert response["success"] and response["mime_type"] == "application/json"
            async with httpx.AsyncClient() as reader:
                delivered = await reader.get(response["download_url"])
                assert delivered.status_code == 200
                assert delivered.headers["content-type"] == "application/json"
                assert delivered.content == Path(descriptor["path"]).read_bytes()
                assert delivered.json()["source_hash"] == result["source_hash"]
                assert delivered.json()["cells"][1]["status"] == "undefined"
                Path(descriptor["path"]).write_bytes(b"{}")
                assert (await reader.get(response["download_url"])).status_code == 404
            assert runtime.context.runner.calls == []
            assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())
