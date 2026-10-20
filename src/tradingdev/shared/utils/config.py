"""YAML configuration loader."""

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
