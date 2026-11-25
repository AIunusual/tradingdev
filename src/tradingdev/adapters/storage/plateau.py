"""Immutable, hash-verified derived collections; publication never replaces a file."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore


class PlateauStore:
    """Bind summary, paged cells and downloadable evidence to the same bytes."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.root = workspace.root / "analyses" / "plateau"
        self.store = store

    def publish(self, payload: dict[str, Any]) -> str:
        data = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()
        identity = hashlib.sha256(data).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{identity}.json"
        staging: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self.root, prefix=".publish-", delete=False
            ) as stream:
                staging = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(staging, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(
                        "Existing plateau artifact has changed; "
                        "preserve it and investigate"
                    ) from None
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
        self.store.create_artifact(
            artifact_id=f"plateau:{identity}",
            run_id=None,
            artifact_type="parameter_plateau",
            path=path,
            sha256=identity,
            metadata={"source_hash": payload["source_hash"]},
        )
        return identity

    def load(self, identity: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if re.fullmatch(r"[a-f0-9]{64}", identity) is None:
            raise ValueError("analysis_id must be the returned SHA-256 identity")
        path = self.root / f"{identity}.json"
        artifact = self.store.get_artifact(f"plateau:{identity}")
        if (
            artifact is None
            or artifact["path"] != str(path)
            or artifact["sha256"] != identity
            or artifact["artifact_type"] != "parameter_plateau"
        ):
            raise ValueError(
                "Plateau artifact registration is unavailable or inconsistent"
            )
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != identity:
            raise ValueError("Plateau artifact content hash mismatch")
        payload: dict[str, Any] = json.loads(data)
        return payload, {
            "artifact_id": f"plateau:{identity}",
            "sha256": identity,
            "path": str(path),
        }
