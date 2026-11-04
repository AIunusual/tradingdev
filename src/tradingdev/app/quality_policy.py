"""Locate the canonical quality configuration in a checkout or installed wheel."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def quality_config_path() -> Path:
    """Return the same policy used by repository checks and strategy validation."""
    package = Path(__file__).resolve().parents[1]
    installed_policy = package / "_quality" / "pyproject.toml"
    if installed_policy.is_file():
        return installed_policy
    checkout_policy = package.parent.parent / "pyproject.toml"
    if package.parent.name == "src" and checkout_policy.is_file():
        return checkout_policy
    raise FileNotFoundError(
        "TradingDev quality policy is missing from the installation"
    )


@contextmanager
def quality_check_config() -> Iterator[Path]:
    """Keep the installed policy's relative Mypy path outside site-packages.

    Mypy rejects any configured search path under site-packages, even one that
    does not exist. An unchanged temporary copy lets an installed package use
    normal PEP 561 resolution; checkouts retain their explicit source-tree path.
    """
    policy = quality_config_path()
    if policy.parent.name != "_quality":
        yield policy
        return
    with TemporaryDirectory(prefix="tradingdev-quality-policy-") as directory:
        copied = Path(directory) / "pyproject.toml"
        copied.write_bytes(policy.read_bytes())
        yield copied
