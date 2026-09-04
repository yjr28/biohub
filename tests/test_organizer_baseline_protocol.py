import json
from pathlib import Path

import pytest

from biohub.baselines.organizer import (
    OrganizerBaselineError,
    build_organizer_protocol,
    write_organizer_protocol,
)


def _inventory():
    return {
        "loeo_folds": [
            {
                "name": "holdout_E2",
                "train_embryos": ["E1"],
                "holdout_embryo": "E2",
                "train_datasets": ["E1_crop1", "E1_crop2"],
                "holdout_datasets": ["E2_crop1", "E2_crop2"],
            }
        ]
    }


def test_training_monitor_never_contains_holdout_embryo():
    protocol = build_organizer_protocol(_inventory(), fold_name="holdout_E2")
    train_fold = protocol.train_splits[0]
    pred_fold = protocol.predict_splits[0]

    assert train_fold["train"] == ["E1_crop1", "E1_crop2"]
    assert train_fold["test"] == ["E1_crop1", "E1_crop2"]
    assert pred_fold["test"] == ["E2_crop1", "E2_crop2"]
    assert not any(name.startswith("E2_") for name in train_fold["train"] + train_fold["test"])


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


def test_only_audited_monitor_policy_is_allowed():
    with pytest.raises(OrganizerBaselineError, match="Only checkpoint_monitor_policy"):
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
    assert json.loads(audit_path.read_text())["checkpoint_monitor_policy"] == "train-embryo-all"
