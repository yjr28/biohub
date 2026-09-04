"""Leakage-safe organizer-monitor prediction plans for tracker calibration.

The true LOEO embryo is a validation instrument, not a tracker-hyperparameter
search set.  This module derives an isolated prediction command from an already
selected organizer baseline, replacing only paths/scope needed to infer the
nested training-embryo checkpoint-monitor datasets.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


class CalibrationPlanError(ValueError):
    """Raised when a calibration run would be ambiguous or leakage-prone."""


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class MonitorPredictionPlan:
    """Exact, isolated prediction plan for training-side tracker calibration."""

    calibration_id: str
    baseline_work_dir: str
    protocol_path: str
    effective_config_path: str
    competition_train_dir: str
    monitor_splits_path: str
    monitor_datasets: tuple[str, ...]
    forbidden_loeo_holdout_datasets: tuple[str, ...]
    train_datasets: tuple[str, ...]
    method: str
    split_index: int
    weights_path: str
    isolation_user: str
    predictions_dir: str
    predict_command: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _load_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise CalibrationPlanError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationPlanError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibrationPlanError(f"{label} root must be a JSON object: {path}")
    return payload


def _flag_value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1:
        raise CalibrationPlanError(
            f"baseline predict command must contain {flag!r} exactly once; found {len(positions)}"
        )
    index = positions[0]
    if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
        raise CalibrationPlanError(f"baseline predict command has no value for {flag}")
    return str(command[index + 1])


def _replace_flag(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def _safe_component(value: str, label: str) -> str:
    value = value.strip()
    if not value or not _SAFE_TOKEN.fullmatch(value):
        raise CalibrationPlanError(
            f"{label} must use only letters, digits, '.', '_' or '-': {value!r}"
        )
    return value


def build_monitor_prediction_plan(
    *,
    repo_root: str | Path,
    baseline_work_dir: str | Path,
    competition_root: str | Path,
    calibration_id: str,
    work_dir: str | Path,
    python_executable: str | None = None,
) -> MonitorPredictionPlan:
    """Derive a monitor-only prediction command from a selected baseline run.

    The baseline work directory must contain ``organizer_baseline_protocol.json``
    and ``effective_config.json`` produced by ``run_clean_organizer_baseline``.
    Only the deterministic nested checkpoint-monitor datasets are admitted.
    The original prediction command is reused as provenance for all quality-
    affecting prediction flags, while its executable, data path, split file and
    output namespace are safely rebound to the current environment.
    """

    repo_root = Path(repo_root).resolve()
    baseline_work_dir = Path(baseline_work_dir).resolve()
    competition_root = Path(competition_root).resolve()
    work_dir = Path(work_dir).resolve()
    calibration_id = _safe_component(calibration_id, "calibration_id")

    protocol_path = baseline_work_dir / "organizer_baseline_protocol.json"
    effective_config_path = baseline_work_dir / "effective_config.json"
    protocol = _load_object(protocol_path, "organizer baseline protocol")
    effective = _load_object(effective_config_path, "organizer effective config")

    policy = protocol.get("checkpoint_monitor_policy")
    if policy != "train-embryo-hash-holdout":
        raise CalibrationPlanError(
            "tracker calibration requires checkpoint_monitor_policy='train-embryo-hash-holdout'; "
            "train-embryo-all is not an independent calibration set"
        )

    monitor = tuple(
        sorted(str(value).strip() for value in protocol.get("checkpoint_monitor_datasets", []) if str(value).strip())
    )
    holdout = tuple(
        sorted(str(value).strip() for value in protocol.get("holdout_datasets", []) if str(value).strip())
    )
    train = tuple(
        sorted(str(value).strip() for value in protocol.get("train_datasets", []) if str(value).strip())
    )
    if not monitor or not holdout or not train:
        raise CalibrationPlanError("protocol is missing monitor, holdout, or training datasets")
    if set(monitor) & set(holdout):
        raise CalibrationPlanError("checkpoint-monitor datasets overlap the LOEO holdout")
    if not set(monitor) <= set(train):
        raise CalibrationPlanError("checkpoint-monitor datasets are not a subset of training datasets")

    raw_command = effective.get("predict_command")
    if not isinstance(raw_command, list) or len(raw_command) < 2:
        raise CalibrationPlanError("effective config has no complete predict_command list")
    command = [str(token) for token in raw_command]
    if Path(command[1]).name != "predict_unet_transformer.py":
        raise CalibrationPlanError(
            f"baseline predict command is not the pinned organizer predictor: {command[1]!r}"
        )
    for forbidden in ("--debug-video", "--slice", "--evaluate"):
        if forbidden in command:
            raise CalibrationPlanError(
                f"baseline predict command contains forbidden calibration flag {forbidden!r}"
            )

    method = _safe_component(_flag_value(command, "--method"), "baseline method")
    split_raw = _flag_value(command, "--split")
    try:
        split_index = int(split_raw)
    except ValueError as exc:
        raise CalibrationPlanError(f"baseline split must be an integer, got {split_raw!r}") from exc
    if split_index < 0:
        raise CalibrationPlanError("baseline split index must be nonnegative")

    weights = Path(_flag_value(command, "--weights")).expanduser()
    if not weights.is_absolute():
        weights = (repo_root / weights).resolve()
    else:
        weights = weights.resolve()

    train_dir = (competition_root / "train").resolve()
    if not train_dir.is_dir():
        raise CalibrationPlanError(f"competition train directory not found: {train_dir}")

    monitor_splits_path = work_dir / "monitor_predict_splits.json"
    expected_predictor = (
        repo_root / "vendor" / "kaggle-cell-tracking-competition" / "scripts" / "predict_unet_transformer.py"
    ).resolve()
    if not expected_predictor.is_file():
        raise CalibrationPlanError(f"pinned organizer predictor not found: {expected_predictor}")

    command[0] = python_executable or sys.executable
    command[1] = str(expected_predictor)
    _replace_flag(command, "--data-dir", str(train_dir))
    _replace_flag(command, "--splits", str(monitor_splits_path))
    _replace_flag(command, "--weights", str(weights))
    _replace_flag(command, "--split", str(split_index))

    isolation_user = _safe_component(f"biohub-cal-{calibration_id}", "isolation user")
    predictions_dir = (
        repo_root
        / "vendor"
        / "kaggle-cell-tracking-competition"
        / "predictions"
        / isolation_user
        / method
        / f"split_{split_index}"
    ).resolve()

    return MonitorPredictionPlan(
        calibration_id=calibration_id,
        baseline_work_dir=str(baseline_work_dir),
        protocol_path=str(protocol_path),
        effective_config_path=str(effective_config_path),
        competition_train_dir=str(train_dir),
        monitor_splits_path=str(monitor_splits_path),
        monitor_datasets=monitor,
        forbidden_loeo_holdout_datasets=holdout,
        train_datasets=train,
        method=method,
        split_index=split_index,
        weights_path=str(weights),
        isolation_user=isolation_user,
        predictions_dir=str(predictions_dir),
        predict_command=tuple(command),
    )


def write_monitor_splits(plan: MonitorPredictionPlan) -> Path:
    """Write the exact monitor-only split consumed by the organizer predictor."""

    path = Path(plan.monitor_splits_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "split": plan.split_index,
            "train": list(plan.train_datasets),
            "test": list(plan.monitor_datasets),
        }
    ]
    if plan.split_index:
        placeholder = {
            "split": -1,
            "train": list(plan.train_datasets),
            "test": list(plan.monitor_datasets),
        }
        payload = [dict(placeholder) for _ in range(plan.split_index + 1)]
        payload[plan.split_index] = {
            "split": plan.split_index,
            "train": list(plan.train_datasets),
            "test": list(plan.monitor_datasets),
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def validate_monitor_prediction_directory(plan: MonitorPredictionPlan) -> tuple[Path, ...]:
    """Require exactly the intended monitor GEFFs and no LOEO holdout outputs."""

    directory = Path(plan.predictions_dir)
    if not directory.is_dir():
        raise CalibrationPlanError(f"monitor prediction directory not found: {directory}")
    actual = {path.stem: path for path in directory.glob("*.geff")}
    expected = set(plan.monitor_datasets)
    missing = expected - set(actual)
    extras = set(actual) - expected
    if missing or extras:
        raise CalibrationPlanError(
            f"monitor prediction set mismatch: missing={sorted(missing)} extras={sorted(extras)}"
        )
    leaked = set(plan.forbidden_loeo_holdout_datasets) & set(actual)
    if leaked:
        raise CalibrationPlanError(f"LOEO holdout predictions appeared in calibration output: {sorted(leaked)}")
    return tuple(actual[name] for name in sorted(actual))


def frontier_shortlist(report: Mapping) -> dict:
    """Return a deterministic next-stage shortlist from a candidate sweep report.

    No single tracker hyperparameter is selected here.  The entire aggregate
    Pareto frontier is frozen as the allowed set for subsequent training-side
    learned-HOCT/solver calibration; LOEO evaluation must not expand this set.
    """

    aggregate = report.get("aggregate") if isinstance(report, Mapping) else None
    if not isinstance(aggregate, Mapping):
        raise CalibrationPlanError("candidate sweep report has no aggregate object")
    trials = aggregate.get("trials")
    frontier = aggregate.get("pareto_config_ids")
    if not isinstance(trials, list) or not isinstance(frontier, list) or not frontier:
        raise CalibrationPlanError("candidate sweep report has no non-empty aggregate frontier")
    by_id = {
        str(trial.get("config_id")): trial
        for trial in trials
        if isinstance(trial, Mapping) and trial.get("config_id")
    }
    if any(str(config_id) not in by_id for config_id in frontier):
        raise CalibrationPlanError("aggregate frontier references an unknown candidate config")

    frontier_trials = [dict(by_id[str(config_id)]) for config_id in frontier]
    ordered = sorted(
        frontier_trials,
        key=lambda trial: (
            -float(trial.get("candidate_recall_of_detectable", 0.0)),
            int(trial.get("candidate_edges", 0)),
            str(trial.get("config_id")),
        ),
    )
    max_available = max(int(trial.get("candidate_available_gt_edges", 0)) for trial in frontier_trials)
    max_coverage = [
        trial for trial in frontier_trials if int(trial.get("candidate_available_gt_edges", 0)) == max_available
    ]
    coverage_winner = min(
        max_coverage,
        key=lambda trial: (int(trial.get("candidate_edges", 0)), str(trial.get("config_id"))),
    )
    return {
        "policy": "freeze aggregate Pareto frontier before any LOEO evaluation",
        "allowed_config_ids": [str(trial["config_id"]) for trial in ordered],
        "priority_order": [str(trial["config_id"]) for trial in ordered],
        "max_coverage_min_cost_config_id": str(coverage_winner["config_id"]),
        "frontier_trials": ordered,
        "loeo_may_expand_shortlist": False,
    }
