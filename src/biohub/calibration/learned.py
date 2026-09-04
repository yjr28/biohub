"""Training-side learned-HOCT calibration contracts.

Phase 2F consumes the candidate frontier frozen by Phase 2E.  It may vary only
predeclared learned model / solver settings on the same training-side monitor
datasets.  The opposite-embryo LOEO set must remain untouched until one winner
(or the organizer control) is frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from biohub.trackers import HOCT_MODELS


class LearnedCalibrationError(ValueError):
    """Raised when a learned calibration request is ambiguous or leaks scope."""


@dataclass(frozen=True)
class HOCTSolverSpec:
    name: str
    appearance_weight: float
    disappearance_weight: float
    division_weight: float
    node_weight: float
    delta_t_weight: float
    edge_bias: float
    timeout: float
    tracklet_solver: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HOCTLearnedTrialSpec:
    trial_id: str
    candidate_config_id: str
    model_name: str
    window_size: int
    solver: HOCTSolverSpec

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class LearnedCalibrationGrid:
    model_names: tuple[str, ...]
    window_sizes: tuple[int, ...]
    solver_configs: tuple[HOCTSolverSpec, ...]
    hoct_promotion_margin: float
    allow_gap_candidates: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_names": list(self.model_names),
            "window_sizes": list(self.window_sizes),
            "solver_configs": [solver.to_dict() for solver in self.solver_configs],
            "hoct_promotion_margin": self.hoct_promotion_margin,
            "allow_gap_candidates": self.allow_gap_candidates,
        }


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LearnedCalibrationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise LearnedCalibrationError(f"{label} must be finite")
    return result


def parse_learned_grid(payload: Mapping[str, Any]) -> LearnedCalibrationGrid:
    """Parse a fully explicit learned-HOCT grid; no quality defaults are invented."""

    if not isinstance(payload, Mapping):
        raise LearnedCalibrationError("learned calibration grid must be a JSON object")
    required = {
        "model_names",
        "window_sizes",
        "solver_configs",
        "hoct_promotion_margin",
        "allow_gap_candidates",
    }
    unknown = set(payload) - required
    missing = required - set(payload)
    if missing or unknown:
        raise LearnedCalibrationError(
            f"learned calibration grid keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}"
        )

    raw_models = payload["model_names"]
    if not isinstance(raw_models, list) or not raw_models:
        raise LearnedCalibrationError("model_names must be a non-empty list")
    model_names = tuple(str(value).strip() for value in raw_models)
    if any(not name for name in model_names) or len(set(model_names)) != len(model_names):
        raise LearnedCalibrationError("model_names must be unique non-empty strings")
    unknown_models = set(model_names) - set(HOCT_MODELS)
    if unknown_models:
        raise LearnedCalibrationError(
            f"learned grid contains unaudited HOCT models: {sorted(unknown_models)}"
        )

    raw_windows = payload["window_sizes"]
    if not isinstance(raw_windows, list) or not raw_windows:
        raise LearnedCalibrationError("window_sizes must be a non-empty list")
    window_sizes: list[int] = []
    for value in raw_windows:
        if isinstance(value, bool):
            raise LearnedCalibrationError("window_sizes must contain positive integers")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise LearnedCalibrationError("window_sizes must contain positive integers") from exc
        if parsed <= 0 or parsed != value:
            raise LearnedCalibrationError("window_sizes must contain positive integers")
        window_sizes.append(parsed)
    if len(set(window_sizes)) != len(window_sizes):
        raise LearnedCalibrationError("window_sizes must be unique")

    raw_solvers = payload["solver_configs"]
    if not isinstance(raw_solvers, list) or not raw_solvers:
        raise LearnedCalibrationError("solver_configs must be a non-empty list")
    solver_required = {
        "name",
        "appearance_weight",
        "disappearance_weight",
        "division_weight",
        "node_weight",
        "delta_t_weight",
        "edge_bias",
        "timeout",
        "tracklet_solver",
    }
    solvers: list[HOCTSolverSpec] = []
    for index, raw in enumerate(raw_solvers):
        if not isinstance(raw, Mapping):
            raise LearnedCalibrationError(f"solver_configs[{index}] must be an object")
        missing_solver = solver_required - set(raw)
        unknown_solver = set(raw) - solver_required
        if missing_solver or unknown_solver:
            raise LearnedCalibrationError(
                f"solver_configs[{index}] keys mismatch: missing={sorted(missing_solver)} "
                f"unknown={sorted(unknown_solver)}"
            )
        name = str(raw["name"]).strip()
        if not name:
            raise LearnedCalibrationError(f"solver_configs[{index}].name cannot be empty")
        tracklet = raw["tracklet_solver"]
        if not isinstance(tracklet, bool):
            raise LearnedCalibrationError(
                f"solver_configs[{index}].tracklet_solver must be boolean"
            )
        timeout = _finite_float(raw["timeout"], f"solver_configs[{index}].timeout")
        if timeout <= 0:
            raise LearnedCalibrationError(f"solver_configs[{index}].timeout must be > 0")
        solvers.append(
            HOCTSolverSpec(
                name=name,
                appearance_weight=_finite_float(raw["appearance_weight"], f"solver_configs[{index}].appearance_weight"),
                disappearance_weight=_finite_float(raw["disappearance_weight"], f"solver_configs[{index}].disappearance_weight"),
                division_weight=_finite_float(raw["division_weight"], f"solver_configs[{index}].division_weight"),
                node_weight=_finite_float(raw["node_weight"], f"solver_configs[{index}].node_weight"),
                delta_t_weight=_finite_float(raw["delta_t_weight"], f"solver_configs[{index}].delta_t_weight"),
                edge_bias=_finite_float(raw["edge_bias"], f"solver_configs[{index}].edge_bias"),
                timeout=timeout,
                tracklet_solver=tracklet,
            )
        )
    solver_names = [solver.name for solver in solvers]
    if len(set(solver_names)) != len(solver_names):
        raise LearnedCalibrationError("solver config names must be unique")

    margin = _finite_float(payload["hoct_promotion_margin"], "hoct_promotion_margin")
    if margin < 0:
        raise LearnedCalibrationError("hoct_promotion_margin must be >= 0")
    allow_gap = payload["allow_gap_candidates"]
    if not isinstance(allow_gap, bool):
        raise LearnedCalibrationError("allow_gap_candidates must be boolean")

    return LearnedCalibrationGrid(
        model_names=model_names,
        window_sizes=tuple(window_sizes),
        solver_configs=tuple(solvers),
        hoct_promotion_margin=margin,
        allow_gap_candidates=allow_gap,
    )


def learned_trial_id(
    *,
    candidate_config_id: str,
    model_name: str,
    window_size: int,
    solver: HOCTSolverSpec,
) -> str:
    payload = {
        "candidate_config_id": str(candidate_config_id),
        "model_name": str(model_name),
        "window_size": int(window_size),
        "solver": solver.to_dict(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:12]
    return f"hoct-learned-{digest}"


def expand_learned_trials(
    *,
    allowed_candidate_config_ids: Sequence[str],
    grid: LearnedCalibrationGrid,
) -> tuple[HOCTLearnedTrialSpec, ...]:
    candidate_ids = tuple(str(value).strip() for value in allowed_candidate_config_ids)
    if not candidate_ids or any(not value for value in candidate_ids):
        raise LearnedCalibrationError("allowed candidate config IDs must be non-empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise LearnedCalibrationError("allowed candidate config IDs must be unique")

    trials: list[HOCTLearnedTrialSpec] = []
    for candidate_id in candidate_ids:
        for model_name in grid.model_names:
            for window_size in grid.window_sizes:
                for solver in grid.solver_configs:
                    trials.append(
                        HOCTLearnedTrialSpec(
                            trial_id=learned_trial_id(
                                candidate_config_id=candidate_id,
                                model_name=model_name,
                                window_size=window_size,
                                solver=solver,
                            ),
                            candidate_config_id=candidate_id,
                            model_name=model_name,
                            window_size=window_size,
                            solver=solver,
                        )
                    )
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise LearnedCalibrationError("learned calibration trial IDs collided")
    return tuple(trials)


def _score(summary: Mapping[str, Any], label: str) -> tuple[float, float]:
    try:
        score = float(summary["score"])
        adjusted = float(summary["adj_edge_jaccard"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LearnedCalibrationError(f"{label} summary lacks numeric score/adj_edge_jaccard") from exc
    if not math.isfinite(score) or not math.isfinite(adjusted):
        raise LearnedCalibrationError(f"{label} summary has non-finite score")
    return score, adjusted


def select_training_side_winner(
    *,
    organizer_control_summary: Mapping[str, Any],
    hoct_trials: Sequence[Mapping[str, Any]],
    promotion_margin: float,
) -> dict[str, Any]:
    """Freeze either organizer control or best HOCT trial from monitor evidence only."""

    margin = _finite_float(promotion_margin, "promotion_margin")
    if margin < 0:
        raise LearnedCalibrationError("promotion_margin must be >= 0")
    control_score, control_adj = _score(organizer_control_summary, "organizer control")
    if not hoct_trials:
        return {
            "family": "organizer_control",
            "reason": "no HOCT trials supplied",
            "control_score": control_score,
            "control_adj_edge_jaccard": control_adj,
            "promotion_margin": margin,
        }

    normalized = []
    seen_ids: set[str] = set()
    for row in hoct_trials:
        if not isinstance(row, Mapping) or not row.get("trial_id"):
            raise LearnedCalibrationError("each HOCT trial result must contain trial_id")
        trial_id = str(row["trial_id"])
        if trial_id in seen_ids:
            raise LearnedCalibrationError(f"duplicate HOCT trial result: {trial_id}")
        seen_ids.add(trial_id)
        summary = row.get("summary")
        if not isinstance(summary, Mapping):
            raise LearnedCalibrationError(f"HOCT trial {trial_id} has no summary")
        score, adjusted = _score(summary, f"HOCT trial {trial_id}")
        runtime = _finite_float(row.get("runtime_seconds", 0.0), f"HOCT trial {trial_id} runtime")
        normalized.append((score, adjusted, -runtime, trial_id, row))

    normalized.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
    best_score, best_adj, _, best_id, best_row = normalized[0]
    required = control_score + margin
    if best_score >= required:
        return {
            "family": "hoct",
            "trial_id": best_id,
            "trial": dict(best_row),
            "hoct_score": best_score,
            "hoct_adj_edge_jaccard": best_adj,
            "control_score": control_score,
            "control_adj_edge_jaccard": control_adj,
            "promotion_margin": margin,
            "required_score": required,
            "score_gain_over_control": best_score - control_score,
            "reason": "best HOCT monitor score cleared the predeclared promotion margin",
        }
    return {
        "family": "organizer_control",
        "best_hoct_trial_id": best_id,
        "best_hoct_score": best_score,
        "best_hoct_adj_edge_jaccard": best_adj,
        "control_score": control_score,
        "control_adj_edge_jaccard": control_adj,
        "promotion_margin": margin,
        "required_score": required,
        "score_gain_over_control": best_score - control_score,
        "reason": "HOCT did not clear the predeclared monitor-side promotion margin",
    }
