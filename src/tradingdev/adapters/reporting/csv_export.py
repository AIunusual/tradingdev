"""Complete, precise CSV exports of the same readable report table projection."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, str):
        # CSV quoting alone does not prevent spreadsheet formula interpretation.
        # Protect control/space-prefixed formulas as well as direct prefixes.
        if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(
            ("=", "+", "-", "@")
        ):
            return "'" + value
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CSV numeric values must be finite")
        return str(value)
    if isinstance(value, dict | list):
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    raise TypeError(f"Unsupported CSV value type: {type(value).__name__}")


def table_csv(spec: dict[str, Any]) -> bytes:
    """Export all rows/columns in display order, with original numeric precision."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    columns = spec["columns"]
    writer.writerow(
        _cell_text(
            item["title"] + ("（比例，1 = 100%）" if item["kind"] == "percent" else "")
        )
        for item in columns
    )
    for row in spec["rows"]:
        writer.writerow(_cell_text(row.get(item["field"])) for item in columns)
    return output.getvalue().encode("utf-8-sig")
