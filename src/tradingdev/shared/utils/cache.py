"""Storage location and provenance fingerprint for CLI pipeline artifacts.

ArtifactService writes results and retrieves them by their registered run ID.
Each saved CLI execution has its own run ID. Its descriptive cache fingerprint
is derived from:

1. Executed manifest hash (includes effective settings and strategy defaults).
2. Processed data file **content** (catches data regeneration).
3. A code fingerprint supplied by the caller (catches strategy logic changes).

Both the data and the code parts used to be weaker than they looked. The data
was hashed as ``mtime + size``, which a regenerated file can keep and an edit
can preserve; it is now hashed by content. The code fingerprint used to come
from ``git rev-parse`` of the local checkout, which made the key depend on the
working directory and on git being installed, and fell back to a *random* value
when either failed - a fallback that turns every lookup into a miss while
looking like a normal cache miss. It is now an argument, so the caller that
knows which code produced the result says so, and an unknown identity has to be
admitted rather than guessed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file

CACHE_DIR: Path | None = None


def cache_dir() -> Path:
    """Return the workspace-aligned pipeline result cache directory."""
    if CACHE_DIR is not None:
        return CACHE_DIR
    data_root = os.environ.get("TRADINGDEV_DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser().resolve() / "processed" / "cache"
    return WorkspacePaths().processed_data / "cache"


def data_fingerprint(processed_path: Path) -> str:
    """Fingerprint a processed data file by content, or say it is missing.

    Content rather than mtime and size: a file can be rewritten with the same
    length, and a copy can carry an old mtime, and either would let a stale
    result be served as if the inputs were unchanged.
    """
    if not processed_path.exists():
        return "missing"
    return sha256_file(processed_path)


def compute_cache_key(
    *,
    manifest_hash: str,
    processed_path: Path,
    code_fingerprint: str,
) -> str:
    """Fingerprint executed manifest + data content + code.

    Args:
        manifest_hash: Hash of the executed manifest.
        processed_path: The processed data file the run consumed.
        code_fingerprint: Identity of the code that produced the result.
            Required rather than derived: a key must be reproducible from a
            different working directory, and an unknown code identity has to
            be supplied by the caller who owns it, not guessed here.

    Returns:
        A 16-char hex digest; not a unique execution ID.
    """
    h = hashlib.sha256()
    h.update(manifest_hash.encode("ascii"))
    h.update(data_fingerprint(processed_path).encode("ascii"))
    h.update(code_fingerprint.encode("ascii"))
    return h.hexdigest()[:16]


def clear_cache() -> int:
    """Remove all cached results. Returns number of files removed."""
    directory = cache_dir()
    if not directory.exists():
        return 0
    count = 0
    for f in directory.glob("*.pkl"):
        f.unlink()
        count += 1
    return count
