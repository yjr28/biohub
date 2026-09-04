import json
from pathlib import Path

import pytest

from biohub.evaluation import (
    FrozenLOEOError,
    build_frozen_loeo_plan,
    candidate_frontier_row,
    validate_exact_holdout_prediction_names,
)
from biohub.experiments import file_sha256, path_sha256


def _artifacts(*, family="hoct"):
    monitor = ["embryoA_monitor1", "embryoA_monitor2"]
    holdout = ["embryoB_holdout1", "embryoB_holdout2"]
    candidate = {
        "selection_scope": {
            "monitor_datasets": monitor,
            "forbidden_loeo_holdout_datasets": holdout,
            "loeo_used": False,
        },
        "shortlist": {
            "allowed_config_ids": ["cand-a", "cand-b"],
            "loeo_may_expand_shortlist": False,
            "frontier_trials": [
                {
                    "config_id": "cand-a",
                    "candidate_distance_space": "physical_um",
                    "distance_threshold_um": 4.0,
                    "distance_threshold_voxels": None,
                    "n_neighbors": 5,
                    "max_delta_t": 1,
                },
                {
                    "config_id": "cand-b",
                    "candidate_distance_space": "hoct_native_voxel",
                    "distance_threshold_um": None,
                    "distance_threshold_voxels": 8.0,
                    "n_neighbors": 5,
                    "max_delta_t": 1,
                },
            ],
        },
    }
    monitor_plan = {
        "monitor_datasets": monitor,
        "forbidden_loeo_holdout_datasets": holdout,
        "predictions_dir": "/tmp/monitor",
    }
    trial = {
        "trial_id": "trial-1",
        "spec": {
            "candidate_config_id": "cand-b",
            "model_name": "general_v1",
            "window_size": 5,
            "solver": {
                "name": "solver-a",
                "appearance_weight": 0.5,
                "disappearance_weight": 0.25,
                "division_weight": 0.25,
                "node_weight": -10.0,
                "delta_t_weight": 0.5,
                "edge_bias": 0.5,
                "timeout": 600.0,
                "tracklet_solver": True,
            },
        },
        "summary": {"score": 0.9, "adj_edge_jaccard": 0.89},
        "runtime_seconds": 10.0,
    }
    winner = {"family": "organizer_control"}
    trials = [trial]
    if family == "hoct":
        winner = {"family": "hoct", "trial_id": "trial-1", "trial": trial}
    learned = {
        "selection_scope": {
            "monitor_datasets": monitor,
            "forbidden_loeo_holdout_datasets": holdout,
            "loeo_used": False,
            "loeo_may_retune_or_replace_winner": False,
        },
        "hoct_trials": trials,
        "winner": winner,
    }
    return learned, candidate, monitor_plan


def test_hoct_plan_reconstructs_exact_recorded_trial_and_candidate():
    learned, candidate, monitor_plan = _artifacts(family="hoct")
    plan = build_frozen_loeo_plan(
        learned_selection=learned,
        candidate_shortlist=candidate,
        monitor_prediction_plan=monitor_plan,
    )
    assert plan.winner_family == "hoct"
    assert plan.hoct is not None
    assert plan.hoct.trial_id == "trial-1"
    assert plan.hoct.candidate_config_id == "cand-b"
    assert plan.holdout_datasets == ("embryoB_holdout1", "embryoB_holdout2")
    assert candidate_frontier_row(candidate, "cand-b")["distance_threshold_voxels"] == 8.0


def test_organizer_control_is_a_valid_frozen_winner_without_hoct_spec():
    learned, candidate, monitor_plan = _artifacts(family="organizer_control")
    plan = build_frozen_loeo_plan(
        learned_selection=learned,
        candidate_shortlist=candidate,
        monitor_prediction_plan=monitor_plan,
    )
    assert plan.winner_family == "organizer_control"
    assert plan.hoct is None


def test_loeo_plan_refuses_any_prior_holdout_use_or_retuning_permission():
    learned, candidate, monitor_plan = _artifacts()
    learned["selection_scope"]["loeo_used"] = True
    with pytest.raises(FrozenLOEOError, match="loeo_used"):
        build_frozen_loeo_plan(
            learned_selection=learned,
            candidate_shortlist=candidate,
            monitor_prediction_plan=monitor_plan,
        )

    learned, candidate, monitor_plan = _artifacts()
    learned["selection_scope"]["loeo_may_retune_or_replace_winner"] = True
    with pytest.raises(FrozenLOEOError, match="loeo_may_retune"):
        build_frozen_loeo_plan(
            learned_selection=learned,
            candidate_shortlist=candidate,
            monitor_prediction_plan=monitor_plan,
        )


def test_hoct_winner_cannot_escape_frozen_candidate_shortlist():
    learned, candidate, monitor_plan = _artifacts()
    learned["winner"]["trial"]["spec"]["candidate_config_id"] = "cand-secret"
    learned["hoct_trials"][0]["spec"]["candidate_config_id"] = "cand-secret"
    with pytest.raises(FrozenLOEOError, match="outside Phase-2E shortlist"):
        build_frozen_loeo_plan(
            learned_selection=learned,
            candidate_shortlist=candidate,
            monitor_prediction_plan=monitor_plan,
        )


def test_phase_artifacts_must_agree_on_exact_holdout_scope():
    learned, candidate, monitor_plan = _artifacts()
    monitor_plan["forbidden_loeo_holdout_datasets"] = ["different"]
    with pytest.raises(FrozenLOEOError, match="disagree on LOEO holdout"):
        build_frozen_loeo_plan(
            learned_selection=learned,
            candidate_shortlist=candidate,
            monitor_prediction_plan=monitor_plan,
        )


def test_prediction_directory_name_contract_is_exact():
    learned, candidate, monitor_plan = _artifacts(family="organizer_control")
    plan = build_frozen_loeo_plan(
        learned_selection=learned,
        candidate_shortlist=candidate,
        monitor_prediction_plan=monitor_plan,
    )
    validate_exact_holdout_prediction_names(
        {"embryoB_holdout1", "embryoB_holdout2"}, plan
    )
    with pytest.raises(FrozenLOEOError, match="prediction set mismatch"):
        validate_exact_holdout_prediction_names({"embryoB_holdout1"}, plan)
    with pytest.raises(FrozenLOEOError, match="prediction set mismatch"):
        validate_exact_holdout_prediction_names(
            {"embryoB_holdout1", "embryoB_holdout2", "embryoA_monitor1"}, plan
        )


def test_path_sha256_handles_directory_geff_artifacts_deterministically(tmp_path: Path):
    file_path = tmp_path / "one.bin"
    file_path.write_bytes(b"abc")
    assert path_sha256(file_path) == file_sha256(file_path)

    artifact = tmp_path / "sample.geff"
    (artifact / "nested").mkdir(parents=True)
    (artifact / "z.json").write_text(json.dumps({"x": 1}))
    (artifact / "nested" / "a.bin").write_bytes(b"payload")
    first = path_sha256(artifact)
    second = path_sha256(artifact)
    assert first == second

    (artifact / "nested" / "a.bin").write_bytes(b"changed")
    assert path_sha256(artifact) != first


def test_path_sha256_detects_layout_changes(tmp_path: Path):
    artifact = tmp_path / "sample.geff"
    artifact.mkdir()
    original = artifact / "a.txt"
    original.write_text("same")
    first = path_sha256(artifact)
    original.rename(artifact / "b.txt")
    assert path_sha256(artifact) != first
