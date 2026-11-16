"""Validate the maintained SRS structure without executing its evidence links."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.token import Token

INDEX = "docs/requirements.md"
DIRECTORY = "docs/requirements"
# One registry drives both validation and immutable review-context selection.
# The first scope owns changes; the tuple identifies all affected consumers.
DOCUMENT_SCOPES = {
    "docs/requirements/strategy.md": ("strategy", "execution", "interfaces"),
    "docs/requirements/execution.md": (
        "execution",
        "strategy",
        "results",
        "interfaces",
    ),
    "docs/requirements/data.md": ("data", "execution", "ml", "interfaces"),
    "docs/requirements/results.md": ("results", "execution", "ml", "interfaces"),
    "docs/requirements/ml.md": ("ml", "strategy", "results", "interfaces"),
    "docs/requirements/interfaces.md": ("interfaces",),
    "docs/requirements/quality.md": (
        "interfaces",
        "strategy",
        "execution",
        "results",
        "data",
        "ml",
    ),
}
_ID = re.compile(r"(?:FR|NFR)-[A-Z][A-Z0-9]*-[0-9]{3}")
_FIELD = re.compile(r"\*\*(需求|驗收|契約|驗證)\*\*：[ \t]*(.*)", re.DOTALL)
_FIELDS = frozenset({"需求", "驗收", "契約", "驗證"})


class RequirementsError(RuntimeError):
    """SRS identifiers, acceptance records or trace references are invalid."""


@dataclass(frozen=True)
class RequirementRecord:
    """One requirement's unchanged source and lexically resolved evidence.

    Line bounds are one-based and inclusive. ``text`` preserves the complete
    heading section, including original line endings and nested subsections.
    Reference paths are repository-relative; existence and anchor validation
    belong to the caller's filesystem or immutable Git evidence reader.
    Manual verification may have no paths.
    """

    source: str
    identifier: str
    start_line: int
    end_line: int
    text: str
    contract_references: tuple[tuple[str, str], ...]
    verification_paths: tuple[str, ...]


@dataclass
class _RequirementFields:
    identifier: str
    start_line: int
    end_line: int
    contents: dict[str, str] = field(default_factory=dict)
    links: dict[str, tuple[str, ...]] = field(default_factory=dict)


def is_requirements_path(name: str) -> bool:
    """Include additions, deletions and both sides of renames in candidate review."""
    path = PurePosixPath(name)
    return name == INDEX or (path.is_relative_to(DIRECTORY) and path.suffix == ".md")


def markdown_lines(text: str) -> list[str]:
    """Split only CommonMark line endings, preserving every original character."""
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _links(token: Token) -> tuple[str, ...]:
    return tuple(
        href
        for child in token.children or []
        if child.type == "link_open" and isinstance(href := child.attrGet("href"), str)
    )


def _local_reference(source: str, href: str) -> tuple[str, str] | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    resolved = posixpath.normpath(
        path.lstrip("/")
        if path.startswith("/")
        else posixpath.join(posixpath.dirname(source), path)
        if path
        else source
    )
    return resolved, unquote(parsed.fragment)


def _contract_reference(source: str, href: str) -> tuple[str, str] | None:
    reference = _local_reference(source, href)
    if reference is None:
        return None
    resolved, _ = reference
    if resolved.startswith("../") or not resolved.endswith(".md"):
        raise RequirementsError(f"{source}: invalid contract reference: {href}")
    if is_requirements_path(resolved):
        raise RequirementsError(f"{source}: SRS cannot replace its contract: {href}")
    return reference


def contract_references(source: str, text: str) -> set[tuple[str, str]]:
    """Resolve SRS contract links lexically for immutable Git evidence readers."""
    references: set[tuple[str, str]] = set()
    for token in MarkdownIt("commonmark").parse(text):
        if token.type != "inline" or token.level != 3:
            continue
        match = _FIELD.fullmatch(token.content)
        if match is None or match[1] != "契約":
            continue
        for href in _links(token):
            if (reference := _contract_reference(source, href)) is not None:
                references.add(reference)
    return references


def parse_requirement_records(source: str, text: str) -> tuple[RequirementRecord, ...]:
    """Parse complete SRS records without reading or executing linked evidence.

    The chapter's required fields, headings and identifiers are checked before
    returning any records. Contract references follow ``contract_references``;
    verification fields retain all local paths, while manual-only evidence has
    an empty path tuple. Filesystem validation remains in ``check_requirements``.
    """
    contract_references(source, text)
    lines = markdown_lines(text)
    tokens = MarkdownIt("commonmark").parse(text)
    records: list[RequirementRecord] = []
    for record in _parse_fields(source, text):
        section_links: dict[str, list[str]] = {"契約": [], "驗證": []}
        for token in tokens:
            if (
                token.type != "inline"
                or token.level != 3
                or not token.map
                or not record.start_line <= token.map[0] + 1 <= record.end_line
            ):
                continue
            match = _FIELD.fullmatch(token.content)
            if match is not None and match[1] in section_links:
                section_links[match[1]].extend(_links(token))
        contracts = {
            reference
            for href in section_links["契約"]
            if (reference := _contract_reference(source, href)) is not None
        }
        paths = set()
        for href in section_links["驗證"]:
            reference = _local_reference(source, href)
            if reference is None:
                continue
            path, _ = reference
            if path == ".." or path.startswith("../"):
                raise RequirementsError(
                    f"{source}: invalid local evidence reference: {href}"
                )
            paths.add(path)
        records.append(
            RequirementRecord(
                source=source,
                identifier=record.identifier,
                start_line=record.start_line,
                end_line=record.end_line,
                text="".join(lines[record.start_line - 1 : record.end_line]),
                contract_references=tuple(sorted(contracts)),
                verification_paths=tuple(sorted(paths)),
            )
        )
    return tuple(records)


def _target(root: Path, source: Path, href: str) -> Path | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    target = (
        root / path.lstrip("/")
        if path.startswith("/")
        else source.parent / path
        if path
        else source
    ).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise RequirementsError(f"{source}: invalid local evidence reference: {href}")
    return target


def check_requirements(root: Path) -> int:
    """Check records and navigation; links are evidence targets, not test results.

    Ordinary Markdown link/anchor checks remain owned by tooling.documentation.
    An absent SRS is allowed for repositories/fixtures predating its introduction;
    a present entry or chapter requires the complete connected document set.
    """
    root = root.resolve()
    index = root / INDEX
    chapters = sorted((root / DIRECTORY).rglob("*.md"))
    if not index.exists() and not chapters:
        return 0
    if not index.is_file() or not chapters:
        raise RequirementsError(
            "SRS requires an entry document and requirement chapters"
        )
    for source in [index, *chapters]:
        if not source.resolve().is_relative_to(root):
            raise RequirementsError(f"SRS source resolves outside repository: {source}")
    for source in chapters:
        if source.relative_to(root).as_posix() not in DOCUMENT_SCOPES:
            raise RequirementsError(f"{source}: unregistered SRS review responsibility")
    parser = MarkdownIt("commonmark")
    navigation = {
        _target(root, index, href)
        for token in parser.parse(index.read_text(encoding="utf-8"))
        for child in token.children or []
        if child.type == "link_open" and isinstance(href := child.attrGet("href"), str)
    }
    missing = [
        str(path.relative_to(root)) for path in chapters if path not in navigation
    ]
    if missing:
        raise RequirementsError(
            f"SRS entry does not link chapters: {', '.join(missing)}"
        )
    identifiers: dict[str, str] = {}
    for source in chapters:
        _check_chapter(root, source, identifiers)
    return len(identifiers)


def _finish(source: str, current: _RequirementFields | None) -> None:
    if current is not None and current.contents.keys() != _FIELDS:
        raise RequirementsError(
            f"{source}: {current.identifier}: missing fields "
            f"{', '.join(sorted(_FIELDS - current.contents.keys()))}"
        )


def _parse_fields(source: str, text: str) -> tuple[_RequirementFields, ...]:
    tokens = MarkdownIt("commonmark").parse(text)
    boundaries = [
        token.map[0]
        for token in tokens
        if token.type == "heading_open"
        and token.level == 0
        and token.tag in {"h1", "h2"}
        and token.map
    ]
    boundaries.append(len(markdown_lines(text)))
    current: _RequirementFields | None = None
    records: list[_RequirementFields] = []
    identifiers: dict[str, str] = {}

    for position, token in enumerate(tokens):
        if token.type != "inline":
            continue
        if position and tokens[position - 1].type == "heading_open":
            heading = tokens[position - 1]
            if heading.level != 0:
                continue
            _finish(source, current)
            current = None
            title = token.content
            if not title.startswith(("FR-", "NFR-")):
                continue
            identifier, separator, label = title.partition(" ")
            if (
                heading.tag != "h2"
                or not _ID.fullmatch(identifier)
                or not separator
                or not label.strip()
                or not heading.map
            ):
                raise RequirementsError(
                    f"{source}: malformed requirement heading: {title}"
                )
            start = heading.map[0]
            location = f"{source}:{start + 1}"
            if identifier in identifiers:
                raise RequirementsError(
                    f"{location}: duplicate {identifier}; "
                    f"first at {identifiers[identifier]}"
                )
            identifiers[identifier] = location
            current = _RequirementFields(
                identifier=identifier,
                start_line=start + 1,
                end_line=next(line for line in boundaries if line > start),
            )
            records.append(current)
            continue
        if current is None or token.level != 3:
            continue
        match = _FIELD.fullmatch(token.content)
        if match is None:
            continue
        name, content = match.groups()
        if name in current.contents or not content.strip():
            raise RequirementsError(
                f"{source}: {current.identifier}: empty or duplicate {name}"
            )
        current.contents[name] = content
        current.links[name] = _links(token)
    _finish(source, current)
    if not records:
        raise RequirementsError(f"{source}: no requirement records")
    return tuple(records)


def _check_chapter(root: Path, source: Path, identifiers: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    name = source.relative_to(root).as_posix()
    contract_references(name, text)
    for record in _parse_fields(name, text):
        location = f"{name}:{record.start_line}"
        if record.identifier in identifiers:
            raise RequirementsError(
                f"{location}: duplicate {record.identifier}; "
                f"first at {identifiers[record.identifier]}"
            )
        identifiers[record.identifier] = location
        for field_name in ("契約", "驗證"):
            targets = [
                target
                for href in record.links[field_name]
                if (target := _target(root, source, href)) is not None
            ]
            if field_name == "契約":
                valid = any(
                    path.suffix == ".md"
                    and not is_requirements_path(path.relative_to(root).as_posix())
                    for path in targets
                )
            else:
                content = record.contents[field_name]
                valid = any(
                    path.is_relative_to(root / "tests")
                    and path.name.startswith("test_")
                    and path.suffix == ".py"
                    for path in targets
                ) or (content.startswith("人工驗收：") and bool(content[5:].strip()))
            if not valid:
                raise RequirementsError(
                    f"{source}: {record.identifier}: "
                    f"missing {field_name} evidence target"
                )
