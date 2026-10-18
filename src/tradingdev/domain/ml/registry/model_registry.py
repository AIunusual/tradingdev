"""Model artifact registry: a stable, auditable identity for trained models.

A trained model is more than a blob on disk: it is bound to the feature
contract it was trained on, to the hyper-parameters that produced it, and to
the data it saw. Without recording that binding a later run can silently
load a model against the wrong columns and invent results.

The registry captures this as a small, deterministic record. ``model_id`` is
derived from the model name, its hyper-parameters, the feature contract and a
content hash, so identical training produces an identical id and a tampered or
substituted blob no longer matches. ``feature_importance`` is extracted where
the underlying estimator exposes it; for estimators that cannot (AutoGluon's
ensemble), it is simply absent rather than faked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    """A recorded, reproducible identity for one trained model."""

    model_id: str
    model_name: str
    schema_version: int
    feature_names: tuple[str, ...]
    parameters: dict[str, Any]
    content_hash: str
    framework: str = ""
    kind: str = "classifier"
    n_train_samples: int = 0
    feature_importance: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def n_features(self) -> int:
        """How many features the model was trained on."""
        return len(self.feature_names)

    def to_metadata(self) -> dict[str, Any]:
        """A JSON-safe projection for storage and MCP surfaces."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "schema_version": self.schema_version,
            "framework": self.framework,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "n_features": self.n_features,
            "n_train_samples": self.n_train_samples,
            "parameters": self.parameters,
            "content_hash": self.content_hash,
            "feature_importance": [
                {"feature": name, "importance": value}
                for name, value in self.feature_importance
            ],
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> ModelArtifact:
        """Rebuild a record from stored metadata, validating its shape."""
        if payload.get("schema_version") != cls._CURRENT_SCHEMA:
            raise ValueError(
                f"unsupported model schema_version: {payload.get('schema_version')}"
            )
        importance = tuple(
            (item["feature"], float(item["importance"]))
            for item in payload.get("feature_importance", [])
        )
        return cls(
            model_id=str(payload["model_id"]),
            model_name=str(payload["model_name"]),
            schema_version=int(payload["schema_version"]),
            framework=str(payload.get("framework", "")),
            kind=str(payload.get("kind", "classifier")),
            feature_names=tuple(payload.get("feature_names", [])),
            n_train_samples=int(payload.get("n_train_samples", 0)),
            parameters=dict(payload.get("parameters", {})),
            content_hash=str(payload.get("content_hash", "")),
            feature_importance=importance,
        )

    _CURRENT_SCHEMA: int = field(default=1, init=False, repr=False)


def _stable_json(value: dict[str, Any]) -> str:
    """Canonical JSON for hashing; stable across process runs."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def compute_model_id(
    *,
    model_name: str,
    parameters: dict[str, Any],
    feature_names: tuple[str, ...],
    content_hash: str,
) -> str:
    """Derive a deterministic model id from the training identity.

    The id folds in the hyper-parameters, the feature contract and the blob
    hash, so a model trained on different columns, with different settings,
    or from a different file yields a different id.
    """
    digest = hashlib.sha256()
    digest.update(
        _stable_json(
            {
                "model_name": model_name,
                "parameters": parameters,
                "feature_names": list(feature_names),
                "content_hash": content_hash,
            }
        ).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def build_model_artifact(
    *,
    model_name: str,
    framework: str,
    kind: str,
    parameters: dict[str, Any],
    feature_names: tuple[str, ...],
    content_hash: str,
    n_train_samples: int,
    feature_importance: tuple[tuple[str, float], ...] | None = None,
) -> ModelArtifact:
    """Assemble a :class:`ModelArtifact`, computing its deterministic id."""
    model_id = compute_model_id(
        model_name=model_name,
        parameters=parameters,
        feature_names=feature_names,
        content_hash=content_hash,
    )
    return ModelArtifact(
        model_id=model_id,
        model_name=model_name,
        schema_version=ModelArtifact._CURRENT_SCHEMA,
        framework=framework,
        kind=kind,
        feature_names=tuple(feature_names),
        parameters=dict(parameters),
        content_hash=content_hash,
        n_train_samples=n_train_samples,
        feature_importance=tuple(feature_importance or ()),
    )


def content_sha256(data: bytes) -> str:
    """Return the SHA-256 of model bytes for identity and integrity checks."""
    return hashlib.sha256(data).hexdigest()


def load_model_bytes(path: Path) -> bytes:
    """Read a model blob verbatim for hashing or transport."""
    return path.read_bytes()


def feature_importance_from_gain(
    feature_names: list[str],
    importance: list[float] | Any,
) -> tuple[tuple[str, float], ...]:
    """Project raw gain importance into a sorted ``(feature, value)`` tuple.

    Used by tree estimators (XGBoost, LightGBM, CatBoost) whose
    ``feature_importances_`` is a parallel array to ``feature_names``.
    """
    paired = sorted(
        zip(feature_names, [float(value) for value in importance], strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(paired)


__all__ = [
    "ModelArtifact",
    "build_model_artifact",
    "compute_model_id",
    "content_sha256",
    "feature_importance_from_gain",
    "load_model_bytes",
]
