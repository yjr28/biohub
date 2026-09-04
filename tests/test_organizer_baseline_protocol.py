import json
from pathlib import Path

import pytest

from biohub.baselines.organizer import (
    OrganizerBaselineError,
    build_organizer_protocol,
    write_organizer_protocol,
)


def _inventory(train_datasets=None):
    return {
        "loeo_folds": [
            {
                "name": "holdout_E2",
                "train_embryos": ["E1"],
                "holdout_embryo": "E2",
                "train_datasets": train_datasets or ["E1_crop1", "E1_crop2"],
                "holdout_datasets": ["E2_crop1", "E2_crop2"],
            }
        ]
    }


def test_public_reference_monitor_never_contains_holdout_embryo():
    protocol = build_organizer_protocol(_inventory(), fold_name="holdout_E2")
    train_fold = protocol.train_splits[0]
    pred_fold = protocol.predict_splits[0]

    assert train_fold["train"] == ["E1_crop1", "E1_crop2"]
    assert train_fold["test"] == ["E1_crop1", "E1_crop2"]
    assert protocol.optimizer_datasets == ("E1_crop1", "E1_crop2")
    assert protocol.checkpoint_monitor_datasets == ("E1_crop1", "E1_crop2")
    assert protocol.monitor_fraction is None
    assert protocol.monitor_seed is None
    assert pred_fold["test"] == ["E2_crop1", "E2_crop2"]
    assert not any(name.startswith("E2_") for name in train_fold["train"] + train_fold["test"])


def test_nested_monitor_is_deterministic_disjoint_and_training_embryo_only():
    datasets = [f"E1_crop{i:02d}" for i in range(10)]
    protocol = build_organizer_protocol(
        _inventory(datasets),
        fold_name="holdout_E2",
        checkpoint_monitor_policy="train-embryo-hash-holdout",
        monitor_fraction=0.2,
        monitor_seed=17,
    )
    repeat = build_organizer_protocol(
        _inventory(datasets),
        fold_name="holdout_E2",
        checkpoint_monitor_policy="train-embryo-hash-holdout",
        monitor_fraction=0.2,
        monitor_seed=17,
    )

    assert protocol.checkpoint_monitor_datasets == repeat.checkpoint_monitor_datasets
    assert len(protocol.checkpoint_monitor_datasets) == 2
    assert len(protocol.optimizer_datasets) == 8
    assert not set(protocol.optimizer_datasets) & set(protocol.checkpoint_monitor_datasets)
    assert set(protocol.optimizer_datasets) | set(protocol.checkpoint_monitor_datasets) == set(datasets)
    assert protocol.train_splits[0]["train"] == list(protocol.optimizer_datasets)
    assert protocol.train_splits[0]["test"] == list(protocol.checkpoint_monitor_datasets)
    assert all(name.startswith("E1_") for name in protocol.train_splits[0]["train"])
    assert all(name.startswith("E1_") for name in protocol.train_splits[0]["test"])
    assert protocol.predict_splits[0]["test"] == ["E2_crop1", "E2_crop2"]
    assert protocol.monitor_fraction == pytest.approx(0.2)
    assert protocol.monitor_seed == 17


def test_nested_monitor_seed_changes_ranked_subset_for_rich_fixture():
    datasets = [f"E1_crop{i:02d}" for i in range(30)]
    first = build_organizer_protocol(
        _inventory(datasets),
        fold_name="holdout_E2",
        checkpoint_monitor_policy="train-embryo-hash-holdout",
        monitor_fraction=0.2,
        monitor_seed=1,
    )
    second = build_organizer_protocol(
        _inventory(datasets),
        fold_name="holdout_E2",
        checkpoint_monitor_policy="train-embryo-hash-holdout",
        monitor_fraction=0.2,
        monitor_seed=2,
    )
    assert first.checkpoint_monitor_datasets != second.checkpoint_monitor_datasets


def test_nested_monitor_ceil_fraction_and_leaves_optimizer_dataset():
    protocol = build_organizer_protocol(
        _inventory(["E1_a", "E1_b", "E1_c"]),
        fold_name="holdout_E2",
        checkpoint_monitor_policy="train-embryo-hash-holdout",
        monitor_fraction=0.34,
    )
    assert len(protocol.checkpoint_monitor_datasets) == 2
    assert len(protocol.optimizer_datasets) == 1


def test_nested_monitor_requires_at_least_two_training_datasets():
    with pytest.raises(OrganizerBaselineError, match="at least two datasets"):
        build_organizer_protocol(
            _inventory(["E1_only"]),
            fold_name="holdout_E2",
            checkpoint_monitor_policy="train-embryo-hash-holdout",
        )


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan")])
def test_nested_monitor_rejects_invalid_fraction(fraction):
    with pytest.raises(OrganizerBaselineError, match="monitor_fraction"):
        build_organizer_protocol(
            _inventory(),
            fold_name="holdout_E2",
            checkpoint_monitor_policy="train-embryo-hash-holdout",
            monitor_fraction=fraction,
        )


def test_wrong_holdout_dataset_prefix_fails_closed():
    inventory = _inventory()
    inventory["loeo_folds"][0]["holdout_datasets"] = ["E3_crop1"]
    with pytest.raises(OrganizerBaselineError, match="wrong embryo"):
        build_organizer_protocol(inventory, fold_name="holdout_E2")


def test_nonzero_organizer_fold_index_is_materialized():
    protocol = build_organizer_protocol(
        _inventory(), fold_name="holdout_E2", organizer_fold_index=2
    )
    assert len(protocol.train_splits) == 3
    assert protocol.train_splits[2]["split"] == 2
    assert protocol.predict_splits[2]["test"] == ["E2_crop1", "E2_crop2"]


def test_unknown_monitor_policy_is_rejected():
    with pytest.raises(OrganizerBaselineError, match="Unsupported checkpoint_monitor_policy"):
        build_organizer_protocol(
            _inventory(),
            fold_name="holdout_E2",
            checkpoint_monitor_policy="holdout",
        )


def test_protocol_writer_emits_organizer_compatible_json(tmp_path: Path):
    protocol = build_organizer_protocol(_inventory(), fold_name="holdout_E2")
    train_path = tmp_path / "train.json"
    pred_path = tmp_path / "predict.json"
    audit_path = tmp_path / "protocol.json"
    write_organizer_protocol(
        protocol,
        train_splits_path=train_path,
        predict_splits_path=pred_path,
        protocol_path=audit_path,
    )
    assert json.loads(train_path.read_text())[0]["test"] == ["E1_crop1", "E1_crop2"]
    assert json.loads(pred_path.read_text())[0]["test"] == ["E2_crop1", "E2_crop2"]
    audit = json.loads(audit_path.read_text())
    assert audit["checkpoint_monitor_policy"] == "train-embryo-all"
    assert audit["optimizer_datasets"] == ["E1_crop1", "E1_crop2"]
    assert audit["checkpoint_monitor_datasets"] == ["E1_crop1", "E1_crop2"]
