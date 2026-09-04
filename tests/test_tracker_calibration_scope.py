import pytest

from biohub.trackers import (
    TrackerCalibrationScopeError,
    calibration_scope_from_protocol,
)


def _protocol():
    return {
        "checkpoint_monitor_policy": "train-embryo-hash-holdout",
        "train_datasets": ["E1_a", "E1_b", "E1_c"],
        "checkpoint_monitor_datasets": ["E1_c"],
        "holdout_datasets": ["E2_a", "E2_b"],
    }


def test_training_side_monitor_is_the_only_selection_scope():
    scope = calibration_scope_from_protocol(_protocol())
    assert scope.checkpoint_monitor_policy == "train-embryo-hash-holdout"
    assert scope.calibration_datasets == ("E1_c",)
    assert scope.forbidden_loeo_holdout_datasets == ("E2_a", "E2_b")
    assert scope.train_datasets == ("E1_a", "E1_b", "E1_c")
    payload = scope.to_dict()
    assert payload["loeo_holdout_used"] is False


def test_public_reference_monitor_is_not_accepted_for_tracker_selection():
    protocol = _protocol()
    protocol["checkpoint_monitor_policy"] = "train-embryo-all"
    protocol["checkpoint_monitor_datasets"] = list(protocol["train_datasets"])
    with pytest.raises(TrackerCalibrationScopeError, match="train-embryo-hash-holdout"):
        calibration_scope_from_protocol(protocol)


def test_loeo_overlap_fails_closed_even_if_protocol_claims_nested_monitor_policy():
    protocol = _protocol()
    protocol["checkpoint_monitor_datasets"] = ["E1_c", "E2_a"]
    protocol["train_datasets"].append("E2_a")
    with pytest.raises(TrackerCalibrationScopeError, match="overlap is forbidden"):
        calibration_scope_from_protocol(protocol)


def test_monitor_must_be_subset_of_declared_training_universe():
    protocol = _protocol()
    protocol["checkpoint_monitor_datasets"] = ["E9_external"]
    with pytest.raises(TrackerCalibrationScopeError, match="not a subset"):
        calibration_scope_from_protocol(protocol)


def test_train_and_true_loeo_holdout_must_be_disjoint():
    protocol = _protocol()
    protocol["train_datasets"].append("E2_a")
    with pytest.raises(TrackerCalibrationScopeError, match="training and LOEO holdout datasets overlap"):
        calibration_scope_from_protocol(protocol)


@pytest.mark.parametrize(
    "field",
    ["train_datasets", "checkpoint_monitor_datasets", "holdout_datasets"],
)
def test_required_dataset_universes_cannot_be_empty(field):
    protocol = _protocol()
    protocol[field] = []
    with pytest.raises(TrackerCalibrationScopeError, match="no "):
        calibration_scope_from_protocol(protocol)
