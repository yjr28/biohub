import json
from pathlib import Path

import pytest

from biohub.experiments.registry import (
    ExperimentContractError,
    ExperimentManifest,
    ExperimentResult,
    append_manifest,
    file_sha256,
    load_manifests,
    write_result,
)


GIT_SHA = "a" * 40
INVENTORY_SHA = "b" * 64


def _manifest(**overrides):
    payload = {
        "experiment_id": "exp-001",
        "hypothesis": "global motion compensation improves held-out embryo linking",
        "git_commit": GIT_SHA,
        "inventory_sha256": INVENTORY_SHA,
        "fold_name": "holdout_E2",
        "train_embryos": ("E1",),
        "validation_embryos": ("E2",),
        "train_datasets": ("E1_crop1", "E1_crop2"),
        "validation_datasets": ("E2_crop1",),
        "config": {"motion_compensation": True},
        "seeds": (7, 11),
        "leakage_controls": (
            "no threshold tuning on holdout",
            "no pseudo-labels from holdout",
        ),
    }
    payload.update(overrides)
    return ExperimentManifest(**payload)


def test_manifest_normalizes_and_validates_clean_loeo():
    manifest = _manifest(seeds=(11, 7, 11), train_datasets=("E1_crop2", "E1_crop1"))
    assert manifest.seeds == (7, 11)
    assert manifest.train_datasets == ("E1_crop1", "E1_crop2")
    assert manifest.validation_embryos == ("E2",)


def test_manifest_rejects_embryo_leakage():
    with pytest.raises(ExperimentContractError, match="held-out embryo appears"):
        _manifest(train_datasets=("E1_crop1", "E2_crop9"))


def test_manifest_rejects_multiple_holdout_embryos():
    with pytest.raises(ExperimentContractError, match="exactly one embryo"):
        _manifest(
            validation_embryos=("E2", "E3"),
            validation_datasets=("E2_crop1", "E3_crop1"),
        )


def test_manifest_rejects_short_git_sha():
    with pytest.raises(ExperimentContractError, match="full 40-character git SHA"):
        _manifest(git_commit="abc123")


def test_registry_is_append_only_and_duplicate_safe(tmp_path: Path):
    registry = tmp_path / "manifests.jsonl"
    manifest = _manifest()
    append_manifest(registry, manifest)
    loaded = load_manifests(registry)
    assert len(loaded) == 1
    assert loaded[0].experiment_id == "exp-001"

    with pytest.raises(ExperimentContractError, match="already registered"):
        append_manifest(registry, manifest)


def test_file_sha256_is_stable(tmp_path: Path):
    path = tmp_path / "inventory.json"
    path.write_text("hello\n")
    assert file_sha256(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_result_write_refuses_overwrite(tmp_path: Path):
    result = ExperimentResult(
        experiment_id="exp-001",
        status="success",
        summary={"score": 0.95},
        runtime_seconds=12.5,
    )
    path = tmp_path / "result.json"
    write_result(path, result)
    payload = json.loads(path.read_text())
    assert payload["summary"]["score"] == 0.95

    with pytest.raises(ExperimentContractError, match="Refusing to overwrite"):
        write_result(path, result)
