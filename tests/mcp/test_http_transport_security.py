"""Both real HTTP transports reject untrusted Host and Origin headers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import httpx
import pytest
from tests.mcp.http_harness import http_plan_server

from tradingdev.mcp.server import http_transport_security, main

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
async def test_http_transport_rejects_untrusted_headers_and_accepts_local_client(
    monkeypatch: pytest.MonkeyPatch, transport: Literal["sse", "streamable-http"]
) -> None:
    async with http_plan_server(monkeypatch, transport=transport) as server:
        host, port = server.address
        base = f"http://{host}:{port}"
        async with httpx.AsyncClient(base_url=base, timeout=5) as http:
            for headers, expected in (
                ({"Host": f"untrusted.example:{port}"}, 421),
                ({"Origin": "http://untrusted.example"}, 403),
                ({"Host": f"localhost:{port + 1}"}, 421),
                ({"Origin": "null"}, 403),
                ({"Host": ""}, 421),
            ):
                if transport == "sse":
                    async with http.stream("GET", "/sse", headers=headers) as response:
                        assert response.status_code == expected
                    response = await http.post("/messages/", headers=headers, json={})
                else:
                    response = await http.post(
                        "/mcp",
                        headers={
                            "Accept": "application/json, text/event-stream",
                            **headers,
                        },
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "security-test", "version": "1"},
                            },
                        },
                    )
                assert response.status_code == expected, response.text
        async with server.connect() as client:
            tools = await client.session.list_tools()
            assert any(tool.name == "get_strategy" for tool in tools.tools)


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
async def test_explicit_proxy_values_and_local_origins_are_allowed(
    monkeypatch: pytest.MonkeyPatch, transport: Literal["sse", "streamable-http"]
) -> None:
    async with http_plan_server(
        monkeypatch,
        transport=transport,
        allowed_hosts=("research.example:9443",),
        allowed_origins=("https://portal.example:8443",),
    ) as server:
        host, port = server.address
        async with httpx.AsyncClient(
            base_url=f"http://{host}:{port}", timeout=5
        ) as http:
            headers_to_accept = [
                {"Host": f"{local}:{port}", "Origin": f"http://{local}:{port}"}
                for local in ("127.0.0.1", "localhost", "[::1]")
            ] + [
                {
                    "Host": "research.example:9443",
                    "Origin": "https://portal.example:8443",
                }
            ]
            for headers in headers_to_accept:
                if transport == "sse":
                    async with http.stream("GET", "/sse", headers=headers) as response:
                        assert response.status_code == 200
                        assert await anext(response.aiter_lines()) == "event: endpoint"
                else:
                    response = await http.post(
                        "/mcp",
                        headers={
                            "Accept": "application/json, text/event-stream",
                            **headers,
                        },
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "security-test", "version": "1"},
                            },
                        },
                    )
                    assert response.status_code == 200 and "serverInfo" in response.text


@pytest.mark.parametrize(
    ("hosts", "origins"),
    [
        (("*.example.com",), ()),
        (("example.com:*",), ()),
        (("https://example.com",), ()),
        (("user@example.com",), ()),
        (("example.com?",), ()),
        (("example.com:99999",), ()),
        (("example.com\\foo",), ()),
        (("example.com\u0000",), ()),
        (("example.com\u007f",), ()),
        ((), ("*",)),
        ((), ("https://example.com:*",)),
        ((), ("https://example.com/path",)),
        ((), ("https://example.com#",)),
        ((), ("null",)),
        ((), ("https://example.com:abc",)),
        ((), ("https://example.com\\foo",)),
        ((), ("https://example.com\u0000",)),
        ((), ("https://example.com\u007f",)),
    ],
)
def test_allowlists_require_exact_authorities_and_origins(
    hosts: tuple[str, ...], origins: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        http_transport_security(allowed_hosts=hosts, allowed_origins=origins)


def test_cli_rejects_invalid_http_options_before_creating_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    for arguments in (
        ["--allow-host", "research.example"],
        ["--web", "--port", "0"],
        ["--web", "--allow-origin", "*"],
    ):
        with pytest.raises(SystemExit) as caught:
            main(["--workspace", str(workspace), *arguments])
        assert caught.value.code == 2
        assert not workspace.exists()
