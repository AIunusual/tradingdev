"""Discovery of git-versioned bundled strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from tradingdev.domain.strategies.capabilities import (
    StrategyCapabilities,
    TrainingRequirement,
)

_DEFAULT_BUNDLED_ROOT = Path(__file__).resolve().parent / "bundled"
_TRAINING_REQUIREMENT = TypeAdapter(TrainingRequirement)


@dataclass(frozen=True)
class BundledStrategyEntry:
    """One bundled strategy resolved from its config.yaml."""

    strategy_id: str
    class_name: str
    module_name: str
    config_path: Path
    declared_source_path: str | None
    raw_config: dict[str, Any]

    @property
    def module_source_path(self) -> Path:
        """Return the strategy.py file next to the bundled config."""
        return (self.config_path.parent / "strategy.py").resolve()

    @property
    def strategy_section(self) -> dict[str, Any]:
        section = self.raw_config.get("strategy", {})
        return section if isinstance(section, dict) else {}

    @property
    def execution_capabilities(self) -> StrategyCapabilities:
        """Read trusted bundled declarations without importing the strategy."""
        declared = _TRAINING_REQUIREMENT.validate_python(
            self.strategy_section.get("training_requirement", "unknown")
        )
        return StrategyCapabilities.from_training_requirement(declared)


class BundledStrategyCatalog:
    """Scan bundled strategy configs and expose them by strategy id."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _DEFAULT_BUNDLED_ROOT).resolve()

    @property
    def root(self) -> Path:
        """Return the bundled strategies root directory."""
        return self._root

    def entries(self) -> list[BundledStrategyEntry]:
        """Return all valid bundled strategies sorted by config path."""
        result: list[BundledStrategyEntry] = []
        for config_path in sorted(self._root.glob("*/config.yaml")):
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            strategy = raw.get("strategy", {})
            if not isinstance(strategy, dict):
                continue
            strategy_id = strategy.get("id")
            class_name = strategy.get("class_name")
            if not isinstance(strategy_id, str) or not strategy_id:
                continue
            if not isinstance(class_name, str) or not class_name:
                continue
            declared = strategy.get("source_path")
            result.append(
                BundledStrategyEntry(
                    strategy_id=strategy_id,
                    class_name=class_name,
                    module_name=(
                        "tradingdev.domain.strategies.bundled."
                        f"{config_path.parent.name}.strategy"
                    ),
                    config_path=config_path,
                    declared_source_path=(
                        declared if isinstance(declared, str) and declared else None
                    ),
                    raw_config=raw,
                )
            )
        return result

    def get(self, strategy_id: str) -> BundledStrategyEntry | None:
        """Return the bundled strategy with the given id, if any."""
        for entry in self.entries():
            if entry.strategy_id == strategy_id:
                return entry
        return None

    def ids(self) -> list[str]:
        """Return all bundled strategy ids."""
        return [entry.strategy_id for entry in self.entries()]

    def contains_source(self, source_path: Path) -> bool:
        """Return whether a source path resolves under the bundled root."""
        candidate = source_path
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve().is_relative_to(self._root)
