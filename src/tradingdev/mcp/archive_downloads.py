"""Short-lived capability URLs for bytes that must not enter model context."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import TYPE_CHECKING

from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.requests import Request

    from tradingdev.adapters.storage.research_archives import ResearchArchiveStore
    from tradingdev.app.contracts.research_notes import NoteExport


class ArchiveDownloads:
    """Custom routes bypass MCP authentication: require an unguessable capability.

    This key is per server instance. Restarting invalidates URLs, not saved archives.
    A recipient with a URL can read only its selected immutable ZIP until expiry.
    """

    def __init__(self, archives: ResearchArchiveStore) -> None:
        self.archives = archives
        self._key = secrets.token_bytes(32)

    def _signature(self, archive_id: str, sha256: str, expires: int) -> str:
        message = f"{archive_id}:{sha256}:{expires}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def attach(self, result: NoteExport, base_url: str) -> NoteExport:
        expires = int(time.time()) + 900
        signature = self._signature(result.archive_id, result.sha256, expires)
        return result.model_copy(
            update={
                "download_url": (
                    f"{base_url.rstrip('/')}/research-download/{result.archive_id}"
                    f"?sha256={result.sha256}&expires={expires}&signature={signature}"
                ),
                "download_expires_at": expires,
            }
        )

    async def serve(self, request: Request) -> Response:
        archive_id = request.path_params["archive_id"]
        sha256 = request.query_params.get("sha256", "")
        signature = request.query_params.get("signature", "")
        try:
            expires = int(request.query_params.get("expires", "0"))
            if (
                re.fullmatch(r"[0-9a-f]{64}", signature) is None
                or expires <= time.time()
                or not hmac.compare_digest(
                    signature, self._signature(archive_id, sha256, expires)
                )
            ):
                return Response("Invalid or expired download link", status_code=403)
            content = await run_in_threadpool(
                self.archives.download, archive_id, sha256
            )
        except ValueError, OSError:
            return Response("Archive unavailable; export again", status_code=404)
        return Response(
            content,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="research-{archive_id[:12]}.zip"'
                ),
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )
