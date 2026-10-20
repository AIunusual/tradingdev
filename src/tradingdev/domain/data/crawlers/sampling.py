"""Shared bounds and response handling for provider sampling paths."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    import httpx

SAMPLE_BATCH_ROWS = 1000
SAMPLE_RESPONSE_BYTES = 4 * 1024 * 1024


def sample_range(
    start: datetime, end: datetime, max_rows: int
) -> tuple[datetime, datetime]:
    """Validate sample bounds, treating naive datetimes as UTC."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    start_utc = (
        start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    )
    end_utc = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if start_utc > end_utc:
        raise ValueError("Sample start must not be after end")
    return start_utc, end_utc


def sample_frame(
    rows: list[list[Any]], columns: list[str], *, unit: Literal["s", "ms"]
) -> pd.DataFrame:
    """Construct a frame only from already bounded, projected rows."""
    frame = pd.DataFrame(rows, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit=unit, utc=True)
    return (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def stream_sample_json(
    client: httpx.Client, url: str, params: dict[str, str | int]
) -> dict[str, Any]:
    """Read at most a fixed number of decoded bytes before parsing JSON.

    The response context also closes the stream on HTTP, decoding, and timeout
    failures. This limit covers metadata as well as the candle arrays.
    Redirects fail closed because HTTPX reads redirect bodies before yielding
    the final response when automatic following is enabled.
    """
    with client.stream(
        "GET",
        url,
        params=params,
        headers={"Accept-Encoding": "identity"},
        follow_redirects=False,
    ) as response:
        response.raise_for_status()
        # A compressed body can expand inside HTTPX before iter_bytes yields a
        # bounded chunk, so reject it before invoking the decoder.
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            raise ValueError("Compressed sample responses are not supported")
        content = bytearray()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            if len(content) + len(chunk) > SAMPLE_RESPONSE_BYTES:
                raise ValueError("Sample response exceeds the byte limit")
            content.extend(chunk)
        payload: Any = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Sample response must be a JSON object")
    return payload
