"""Safe value rendering and an offline HTML shell for built-in document renderers.

Document bodies, styles and scripts must be produced by trusted built-in renderers.
All user-authored text belongs in the escaping helpers, never those raw fragments.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_BASE_STYLE = """
:root{color-scheme:light;--ink:#172b45;--muted:#5a6d83;--line:#dce5ef}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);
font:15px/1.65 system-ui,-apple-system,"PingFang TC",sans-serif}
main{max-width:1250px;margin:auto;padding:35px 25px 65px}
h1{font-size:32px;line-height:1.25;margin:10px 0}h2{font-size:22px}
h3{font-size:17px}header{margin-bottom:25px}p{margin:9px 0}
section,.panel{background:white;border:1px solid var(--line);border-radius:12px;
padding:23px;margin-top:20px}.muted,small{color:var(--muted)}
.notice{border-left:4px solid #c88a21;padding:13px 17px;background:#fff4df}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}
button,select,input{font:inherit;padding:7px 10px;border:1px solid var(--line);
border-radius:5px;background:white;color:var(--ink)}button{cursor:pointer}
.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;width:100%;
font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{white-space:nowrap;background:#edf3fa}
thead{position:sticky;top:0}td{overflow-wrap:anywhere}pre{white-space:pre-wrap;
overflow-wrap:anywhere;background:#f4f7fb;padding:12px;font-size:12px}
code{font-size:12px;overflow-wrap:anywhere}details{margin:8px 0}
summary{cursor:pointer}
.badge{background:#e8f0fa;padding:3px 8px;border-radius:15px;font-size:12px}
@media(max-width:750px){main{padding:20px 12px}}
@media print{body{background:white}.scroll{max-height:none}
.controls{display:none}section{break-inside:avoid}}
"""


def escape(value: object) -> str:
    """Escape text for either HTML content or a quoted attribute."""
    return html.escape(str(value), quote=True)


def json_text(value: object) -> str:
    """Serialize finite JSON values deterministically, preserving readable Unicode."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    )


def value_text(value: object) -> str:
    """Format an evidence value without treating missing values as zero."""
    if value is None:
        return "未知／未記錄"
    if isinstance(value, dict | list):
        return json_text(value)
    return str(value)


def value_html(value: object) -> str:
    """Render exactly the plain-text value, escaped once for HTML."""
    return escape(value_text(value))


def pairs_html(values: Mapping[str, object]) -> str:
    """Render named values; callers provide human-facing labels when available."""
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{escape(key)}</th><td>{value_html(value)}</td></tr>"
            for key, value in values.items()
        )
        + "</tbody></table>"
    )


def details_html(title: str, value: object) -> str:
    """Render escaped, inspectable JSON evidence in a collapsible element."""
    return (
        f"<details><summary>{escape(title)}</summary>"
        f"<pre>{escape(json_text(value))}</pre></details>"
    )


def _script_data(value: object) -> str:
    return (
        json_text(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_html_document(
    *,
    title: str,
    body: str,
    style: str = "",
    script: str = "",
    data: object = None,
    data_id: str = "document-data",
    allow_embedded_frames: bool = False,
) -> str:
    """Wrap built-in HTML in a self-contained document with no network sources.

    ``title`` and JSON ``data`` are escaped here. ``body``, ``style`` and
    ``script`` are trusted renderer output, never arbitrary client input.
    Documents without a built-in script disallow script execution altogether.
    """
    script_policy = "'unsafe-inline'" if script else "'none'"
    frame_policy = "frame-src 'self'; " if allow_embedded_frames else ""
    embedded = (
        f"<script id='{escape(data_id)}' type='application/json'>"
        + _script_data(data)
        + "</script>"
        if data is not None
        else ""
    )
    javascript = f"<script>{script}</script>" if script else ""
    return (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
        f"script-src {script_policy}; style-src 'unsafe-inline'; "
        f'{frame_policy}img-src data:;">'
        f"<title>{escape(title)}</title><style>{_BASE_STYLE}{style}</style>"
        f"</head><body><main>{body}</main>{embedded}{javascript}</body></html>"
    )
