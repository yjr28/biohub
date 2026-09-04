import pytest

from biohub.calibration import (
    LearnedCalibrationError,
    expand_learned_trials,
    parse_learned_grid,
    select_training_side_winner,
)


def _grid():
    return {
        "model_names": ["general_v1", "ctc_v0"],
        "window_sizes": [5],
        "solver_configs": [
            {
                "name": "public-default",
                "appearance_weight": 0.5,
                "disappearance_weight": 0.25,
                "division_weight": 0.25,
                "node_weight": -10.0,
                "delta_t_weight": 0.5,
                "edge_bias": 0.5,
                "timeout": 600.0,
                "tracklet_solver": True,
            },
            {
                "name": "division-conservative",
                "appearance_weight": 0.5,
                "disappearance_weight": 0.25,
                "division_weight": 0.5,
                "node_weight": -10.0,
                "delta_t_weight": 0.5,
                "edge_bias": 0.5,
                "timeout": 600.0,
                "tracklet_solver": True,
            },
        ],
        "hoct_promotion_margin": 0.001,
        "allow_gap_candidates": False,
    }


def test_explicit_grid_expands_only_frozen_candidate_ids():
    grid = parse_learned_grid(_grid())
    trials = expand_learned_trials(
        allowed_candidate_config_ids=["cand-a", "cand-b"],
        grid=grid,
    )
    assert len(trials) == 2 * 2 * 1 * 2
    assert {trial.candidate_config_id for trial in trials} == {"cand-a", "cand-b"}
    assert len({trial.trial_id for trial in trials}) == len(trials)


def test_grid_refuses_unaudited_model_and_missing_quality_fields():
    payload = _grid()
    payload["model_names"] = ["mystery_model"]
    with pytest.raises(LearnedCalibrationError, match="unaudited"):
        parse_learned_grid(payload)

    payload = _grid()
    del payload["solver_configs"][0]["division_weight"]
    with pytest.raises(LearnedCalibrationError, match="keys mismatch"):
        parse_learned_grid(payload)


def test_grid_requires_predeclared_promotion_margin_and_gap_policy():
    payload = _grid()
    del payload["hoct_promotion_margin"]
    with pytest.raises(LearnedCalibrationError, match="keys mismatch"):
        parse_learned_grid(payload)

    payload = _grid()
    payload["allow_gap_candidates"] = "false"
    with pytest.raises(LearnedCalibrationError, match="boolean"):
        parse_learned_grid(payload)


def test_control_wins_when_hoct_does_not_clear_margin():
    winner = select_training_side_winner(
        organizer_control_summary={"score": 0.900, "adj_edge_jaccard": 0.895},
        hoct_trials=[
            {
                "trial_id": "h1",
                "summary": {"score": 0.9005, "adj_edge_jaccard": 0.896},
                "runtime_seconds": 10.0,
            }
        ],
        promotion_margin=0.001,
    )
    assert winner["family"] == "organizer_control"
    assert winner["best_hoct_trial_id"] == "h1"


def test_hoct_wins_only_from_training_side_score_and_predeclared_margin():
    winner = select_training_side_winner(
        organizer_control_summary={"score": 0.900, "adj_edge_jaccard": 0.895},
        hoct_trials=[
            {
                "trial_id": "h1",
                "summary": {"score": 0.902, "adj_edge_jaccard": 0.897},
                "runtime_seconds": 20.0,
            },
            {
                "trial_id": "h2",
                "summary": {"score": 0.903, "adj_edge_jaccard": 0.898},
                "runtime_seconds": 30.0,
            },
        ],
        promotion_margin=0.001,
    )
    assert winner["family"] == "hoct"
    assert winner["trial_id"] == "h2"
    assert winner["score_gain_over_control"] == pytest.approx(0.003)


def test_score_tie_prefers_higher_adjusted_edge_then_lower_runtime():
    winner = select_training_side_winner(
        organizer_control_summary={"score": 0.8, "adj_edge_jaccard": 0.79},
        hoct_trials=[
            {
                "trial_id": "slow",
                "summary": {"score": 0.9, "adj_edge_jaccard": 0.89},
                "runtime_seconds": 20.0,
            },
            {
                "trial_id": "fast",
                "summary": {"score": 0.9, "adj_edge_jaccard": 0.89},
                "runtime_seconds": 10.0,
            },
        ],
        promotion_margin=0.0,
    )
    assert winner["trial_id"] == "fast"
