import json
from pathlib import Path

import pytest

from biohub.calibration import (
    CalibrationPlanError,
    build_monitor_prediction_plan,
    frontier_shortlist,
    validate_monitor_prediction_directory,
    write_monitor_splits,
)


def _fixture(tmp_path: Path, *, policy: str = "train-embryo-hash-holdout"):
    repo = tmp_path / "repo"
    predictor = repo / "vendor" / "kaggle-cell-tracking-competition" / "scripts" / "predict_unet_transformer.py"
    predictor.parent.mkdir(parents=True)
    predictor.write_text("# fake pinned predictor\n")

    competition = tmp_path / "competition"
    (competition / "train").mkdir(parents=True)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    work = tmp_path / "calibration"

    protocol = {
        "checkpoint_monitor_policy": policy,
        "checkpoint_monitor_datasets": ["embryoA_monitor2", "embryoA_monitor1"],
        "holdout_datasets": ["embryoB_holdout"],
        "train_datasets": ["embryoA_train1", "embryoA_monitor1", "embryoA_monitor2"],
    }
    (baseline / "organizer_baseline_protocol.json").write_text(json.dumps(protocol))
    effective = {
        "predict_command": [
            "/old/python",
            "/old/repo/vendor/kaggle-cell-tracking-competition/scripts/predict_unet_transformer.py",
            "--method",
            "baseline-x",
            "--data-dir",
            "/old/data",
            "--splits",
            "/old/splits.json",
            "--split",
            "0",
            "--weights",
            "/weights/chosen.pth",
            "--unet-batch-size",
            "4",
            "--det-threshold",
            "0.99",
            "--use-ilp",
        ]
    }
    (baseline / "effective_config.json").write_text(json.dumps(effective))
    return repo, competition, baseline, work


def test_monitor_plan_rebinds_only_scope_paths_and_output_namespace(tmp_path):
    repo, competition, baseline, work = _fixture(tmp_path)
    plan = build_monitor_prediction_plan(
        repo_root=repo,
        baseline_work_dir=baseline,
        competition_root=competition,
        calibration_id="phase2e-a",
        work_dir=work,
        python_executable="/new/python",
    )

    assert plan.monitor_datasets == ("embryoA_monitor1", "embryoA_monitor2")
    assert plan.forbidden_loeo_holdout_datasets == ("embryoB_holdout",)
    assert plan.isolation_user == "biohub-cal-phase2e-a"
    assert plan.predict_command[0] == "/new/python"
    assert plan.predict_command[1] == str(
        (repo / "vendor" / "kaggle-cell-tracking-competition" / "scripts" / "predict_unet_transformer.py").resolve()
    )
    assert "--use-ilp" in plan.predict_command
    assert plan.predict_command[plan.predict_command.index("--det-threshold") + 1] == "0.99"
    assert plan.predict_command[plan.predict_command.index("--data-dir") + 1] == str((competition / "train").resolve())
    assert plan.predict_command[plan.predict_command.index("--splits") + 1] == str(
        (work / "monitor_predict_splits.json").resolve()
    )
    assert "embryoB_holdout" not in plan.predict_command

    split_path = write_monitor_splits(plan)
    payload = json.loads(split_path.read_text())
    assert payload[0]["test"] == ["embryoA_monitor1", "embryoA_monitor2"]
    assert "embryoB_holdout" not in json.dumps(payload)


def test_monitor_plan_refuses_nonindependent_train_embryo_all_monitor(tmp_path):
    repo, competition, baseline, work = _fixture(tmp_path, policy="train-embryo-all")
    with pytest.raises(CalibrationPlanError, match="train-embryo-hash-holdout"):
        build_monitor_prediction_plan(
            repo_root=repo,
            baseline_work_dir=baseline,
            competition_root=competition,
            calibration_id="phase2e-a",
            work_dir=work,
        )


def test_monitor_plan_refuses_debug_slice_or_evaluate_flags(tmp_path):
    repo, competition, baseline, work = _fixture(tmp_path)
    effective_path = baseline / "effective_config.json"
    effective = json.loads(effective_path.read_text())
    effective["predict_command"].extend(["--slice", ":1"])
    effective_path.write_text(json.dumps(effective))
    with pytest.raises(CalibrationPlanError, match="forbidden calibration flag"):
        build_monitor_prediction_plan(
            repo_root=repo,
            baseline_work_dir=baseline,
            competition_root=competition,
            calibration_id="phase2e-a",
            work_dir=work,
        )


def test_prediction_validation_requires_exact_monitor_set_and_no_holdout(tmp_path):
    repo, competition, baseline, work = _fixture(tmp_path)
    plan = build_monitor_prediction_plan(
        repo_root=repo,
        baseline_work_dir=baseline,
        competition_root=competition,
        calibration_id="phase2e-a",
        work_dir=work,
    )
    output = Path(plan.predictions_dir)
    output.mkdir(parents=True)
    for name in plan.monitor_datasets:
        (output / f"{name}.geff").write_text("placeholder")
    assert [path.stem for path in validate_monitor_prediction_directory(plan)] == list(plan.monitor_datasets)

    (output / "embryoB_holdout.geff").write_text("forbidden")
    with pytest.raises(CalibrationPlanError, match="prediction set mismatch"):
        validate_monitor_prediction_directory(plan)


def test_frontier_shortlist_freezes_only_aggregate_pareto_configs():
    report = {
        "aggregate": {
            "trials": [
                {
                    "config_id": "cheap",
                    "candidate_recall_of_detectable": 0.97,
                    "candidate_available_gt_edges": 97,
                    "candidate_edges": 1000,
                },
                {
                    "config_id": "wide",
                    "candidate_recall_of_detectable": 1.0,
                    "candidate_available_gt_edges": 100,
                    "candidate_edges": 1800,
                },
                {
                    "config_id": "dominated",
                    "candidate_recall_of_detectable": 0.95,
                    "candidate_available_gt_edges": 95,
                    "candidate_edges": 2200,
                },
                {
                    "config_id": "same-cover-more-cost",
                    "candidate_recall_of_detectable": 1.0,
                    "candidate_available_gt_edges": 100,
                    "candidate_edges": 2500,
                },
            ],
            "pareto_config_ids": ["cheap", "wide"],
        }
    }
    shortlist = frontier_shortlist(report)
    assert shortlist["allowed_config_ids"] == ["wide", "cheap"]
    assert shortlist["max_coverage_min_cost_config_id"] == "wide"
    assert shortlist["loeo_may_expand_shortlist"] is False
    assert "dominated" not in shortlist["allowed_config_ids"]
