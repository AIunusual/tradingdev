"""Read bounded UTF-8 artifact pages without materializing the entire file."""

from __future__ import annotations

import codecs
import hashlib
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_TEXT_LIMIT = 8192
MAX_TEXT_LIMIT = 32768
_READ_BYTES = 65536
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactTextChangedError(ValueError):
    """The observed file no longer matches the requested content identity."""


@dataclass(frozen=True)
class ArtifactTextPage:
    content: str
    offset: int
    limit: int
    total_characters: int
    next_offset: int | None
    content_sha256: str


def read_text_page(
    path: Path,
    *,
    offset: int = 0,
    limit: int = DEFAULT_TEXT_LIMIT,
    expected_sha256: str | None = None,
) -> ArtifactTextPage:
    """Read code-point offsets, preserving original newlines and hashing raw bytes.

    The whole file is streamed to validate UTF-8, count characters and bind the
    returned page to one content digest. Only the requested page is retained.
    """
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_TEXT_LIMIT
    ):
        raise ValueError(f"limit must be an integer between 1 and {MAX_TEXT_LIMIT}")
    if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    total = 0
    parts: list[str] = []

    def consume(text: str) -> None:
        nonlocal total
        start = max(offset - total, 0)
        end = min(offset + limit - total, len(text))
        if start < end:
            parts.append(text[start:end])
        total += len(text)

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while block := stream.read(_READ_BYTES):
            digest.update(block)
            consume(decoder.decode(block))
        consume(decoder.decode(b"", final=True))
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ArtifactTextChangedError("Artifact changed while its content was read")
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ArtifactTextChangedError(
            "Artifact content no longer matches expected_sha256"
        )
    if offset > total:
        raise ValueError(f"offset {offset} exceeds total_characters {total}")
    content = "".join(parts)
    end = offset + len(content)
    return ArtifactTextPage(
        content=content,
        offset=offset,
        limit=limit,
        total_characters=total,
        next_offset=end if end < total else None,
        content_sha256=observed_sha256,
    )
