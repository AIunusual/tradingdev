"""The report ships an identified, self-contained Bokeh renderer."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import bokeh

from tradingdev.adapters.reporting import bokeh_tables

if TYPE_CHECKING:
    import pytest


class _ResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.script_sources: list[str | None] = []
        self.links: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script":
            self.script_sources.append(attributes.get("src"))
        if tag == "link":
            self.links.append(attributes.get("href"))


def test_bokeh_resources_embed_tables_widgets_and_api_without_remote_assets() -> None:
    content = bokeh_tables.bokeh_resources_html()
    parser = _ResourceParser()
    parser.feed(content)
    assert len(parser.script_sources) == 5
    assert all(source is None for source in parser.script_sources)
    assert not parser.links
    for bundle in ("bokeh.min.js", "bokeh-widgets.min.js", "bokeh-tables.min.js"):
        assert f"BEGIN {bundle}" in content
    assert "BEGIN bokeh-api.min.js" in content


def test_bokeh_resources_are_repeatable_without_reusing_the_cache() -> None:
    first = bokeh_tables.bokeh_resources_html()
    bokeh_tables.bokeh_resources_html.cache_clear()
    second = bokeh_tables.bokeh_resources_html()
    assert second == first
    assert bokeh_tables.bokeh_provenance() == {
        "version": bokeh.__version__,
        "resources_sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(),
    }


def test_bokeh_provenance_tracks_resource_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bokeh_tables.bokeh_provenance()
    monkeypatch.setattr(bokeh_tables, "bokeh_resources_html", lambda: "changed bundle")
    modified = bokeh_tables.bokeh_provenance()
    assert modified["version"] == original["version"]
    assert modified["resources_sha256"] != original["resources_sha256"]
    assert modified["resources_sha256"] == hashlib.sha256(b"changed bundle").hexdigest()
