"""Tests for the model artifact registry and store."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.ml.registry import (
    ModelArtifact,
    ModelArtifactStore,
    build_model_artifact,
    compute_model_id,
    content_sha256,
    feature_importance_from_gain,
    save_model_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_blob(payload: dict[str, object]) -> bytes:
    return pickle.dumps(payload)


def _sample_artifact(
    content_hash: str, *, feature_names: tuple[str, ...] = ("f1", "f2", "f3")
) -> ModelArtifact:
    return build_model_artifact(
        model_name="xgb_direction",
        framework="xgboost",
        kind="classifier",
        parameters={"max_depth": 3, "learning_rate": 0.1},
        feature_names=feature_names,
        content_hash=content_hash,
        n_train_samples=120,
        feature_importance=(("f2", 0.6), ("f1", 0.3), ("f3", 0.1)),
    )


def test_model_id_is_deterministic_for_same_inputs() -> None:
    params = {"max_depth": 3}
    features = ("a", "b", "c")
    digest = content_sha256(_sample_blob({"w": 1}))
    first = compute_model_id(
        model_name="m", parameters=params, feature_names=features, content_hash=digest
    )
    second = compute_model_id(
        model_name="m", parameters=params, feature_names=features, content_hash=digest
    )
    assert first == second
    assert len(first) == 16


def test_model_id_changes_with_feature_contract() -> None:
    params = {"max_depth": 3}
    digest = content_sha256(_sample_blob({"w": 1}))
    base = compute_model_id(
        model_name="m",
        parameters=params,
        feature_names=("a", "b", "c"),
        content_hash=digest,
    )
    different = compute_model_id(
        model_name="m",
        parameters=params,
        feature_names=("a", "b", "d"),
        content_hash=digest,
    )
    assert base != different


def test_model_id_changes_with_parameters() -> None:
    digest = content_sha256(_sample_blob({"w": 1}))
    features = ("a", "b", "c")
    base = compute_model_id(
        model_name="m",
        parameters={"max_depth": 3},
        feature_names=features,
        content_hash=digest,
    )
    different = compute_model_id(
        model_name="m",
        parameters={"max_depth": 6},
        feature_names=features,
        content_hash=digest,
    )
    assert base != different


def test_model_id_changes_with_content_hash() -> None:
    params = {"max_depth": 3}
    features = ("a", "b", "c")
    base = compute_model_id(
        model_name="m", parameters=params, feature_names=features, content_hash="x" * 64
    )
    different = compute_model_id(
        model_name="m", parameters=params, feature_names=features, content_hash="y" * 64
    )
    assert base != different


def test_metadata_round_trip_preserves_fields() -> None:
    artifact = _sample_artifact("a" * 64)
    restored = ModelArtifact.from_metadata(artifact.to_metadata())
    assert restored == artifact
    assert restored.feature_names == ("f1", "f2", "f3")
    assert restored.parameters == {"max_depth": 3, "learning_rate": 0.1}
    assert restored.feature_importance == (("f2", 0.6), ("f1", 0.3), ("f3", 0.1))
    assert restored.n_features == 3


def test_from_metadata_rejects_unknown_schema() -> None:
    payload = _sample_artifact("a" * 64).to_metadata()
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported model schema_version"):
        ModelArtifact.from_metadata(payload)


def test_feature_importance_from_gain_sorts_descending() -> None:
    features = ["a", "b", "c"]
    raw = [0.1, 0.5, 0.3]
    result = feature_importance_from_gain(features, raw)
    assert result == (("b", 0.5), ("c", 0.3), ("a", 0.1))


def test_feature_importance_from_gain_handles_float_coercion() -> None:
    result = feature_importance_from_gain(["x"], ["0.25"])
    assert result == (("x", 0.25),)


@pytest.mark.parametrize(
    ("blob", "expected"),
    [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ],
    ids=["empty", "abc"],
)
def test_content_sha256_matches_known_digest(blob: bytes, expected: str) -> None:
    assert content_sha256(blob) == expected


def test_store_save_and_load_round_trip(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    digest = content_sha256(blob)
    artifact = _sample_artifact(digest)
    store = ModelArtifactStore(root=tmp_path)

    assert store.exists(artifact.model_id) is False
    save_model_artifact(store=store, model_bytes=blob, artifact=artifact)
    assert store.exists(artifact.model_id) is True

    loaded = store.load(artifact.model_id)
    assert loaded == artifact
    assert store.load_with_bytes(artifact.model_id) == (artifact, blob)
    assert store.path_for(artifact.model_id).read_bytes() == blob


def test_store_rejects_mismatched_blob_hash(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    wrong_digest = content_sha256(_sample_blob({"different": True}))
    artifact = _sample_artifact(wrong_digest)
    store = ModelArtifactStore(root=tmp_path)
    with pytest.raises(ValueError, match="content_hash does not match"):
        store.save(model_bytes=blob, artifact=artifact)


def test_store_detects_tampered_blob(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    digest = content_sha256(blob)
    artifact = _sample_artifact(digest)
    store = ModelArtifactStore(root=tmp_path)
    store.save(model_bytes=blob, artifact=artifact)

    store.path_for(artifact.model_id).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model blob tampered"):
        store.load(artifact.model_id)


def test_store_load_missing_blob_raises(tmp_path: Path) -> None:
    store = ModelArtifactStore(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("does-not-exist")


def test_store_list_returns_all_metadata(tmp_path: Path) -> None:
    store = ModelArtifactStore(root=tmp_path)
    seen = []
    for idx in range(3):
        blob = _sample_blob({"idx": idx})
        digest = content_sha256(blob)
        artifact = _sample_artifact(digest, feature_names=(f"f{idx}",))
        save_model_artifact(store=store, model_bytes=blob, artifact=artifact)
        seen.append(artifact.model_id)
    listed = store.list()
    assert {item.model_id for item in listed} == set(seen)


def test_save_rejects_a_forged_identity_before_writing(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    artifact = replace(_sample_artifact(content_sha256(blob)), model_id="0" * 16)
    store = ModelArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="model_id does not match"):
        store.save(model_bytes=blob, artifact=artifact)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "0" * 16),
        ("model_name", "changed"),
        ("parameters", {"max_depth": 10}),
        ("feature_names", ["different"]),
    ],
)
def test_load_and_listing_reject_changed_identity_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    blob = _sample_blob({"model": "weights"})
    artifact = _sample_artifact(content_sha256(blob))
    store = ModelArtifactStore(tmp_path)
    store.save(model_bytes=blob, artifact=artifact)
    payload = artifact.to_metadata()
    payload[field] = value
    (tmp_path / f"{artifact.model_id}.json").write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="model_id does not match"):
        store.load_with_bytes(artifact.model_id)
    assert store.list() == []


def test_load_rejects_metadata_under_a_different_filename(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    artifact = _sample_artifact(content_sha256(blob))
    store = ModelArtifactStore(tmp_path)
    other_id = "0" * 16
    (tmp_path / f"{other_id}.json").write_text(
        json.dumps(artifact.to_metadata()), "utf-8"
    )
    store.path_for(other_id).write_bytes(blob)
    with pytest.raises(ValueError, match="model_id does not match"):
        store.load(other_id)


def test_load_rejects_a_missing_blob_for_published_metadata(tmp_path: Path) -> None:
    blob = _sample_blob({"model": "weights"})
    artifact = _sample_artifact(content_sha256(blob))
    store = ModelArtifactStore(tmp_path)
    store.save(model_bytes=blob, artifact=artifact)
    store.path_for(artifact.model_id).unlink()
    with pytest.raises(FileNotFoundError, match="model blob missing"):
        store.load_with_bytes(artifact.model_id)
