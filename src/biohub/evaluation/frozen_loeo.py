"""Contracts for one-shot evaluation of a tracker frozen before LOEO.

This module does not choose models or hyperparameters.  It validates that the
Phase-2F selection already made that choice on training-side monitor data and
extracts the single artifact family/configuration allowed to touch the opposite
embryo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class FrozenLOEOError(ValueError):
    """Raised when a supposedly frozen LOEO evaluation is not actually frozen."""


@dataclass(frozen=True)
class FrozenHOCTSpec:
    trial_id: str
    candidate_config_id: str
    model_name: str
    window_size: int
    solver: dict[str, Any]


@dataclass(frozen=True)
class FrozenLOEOPlan:
    winner_family: str
    monitor_datasets: tuple[str, ...]
    holdout_datasets: tuple[str, ...]
    hoct: FrozenHOCTSpec | None


def _names(payload: Mapping[str, Any], key: str, label: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise FrozenLOEOError(f"{label}.{key} must be a non-empty list")
    values = tuple(sorted(str(value).strip() for value in raw if str(value).strip()))
    if not values or len(values) != len(raw) or len(set(values)) != len(values):
        raise FrozenLOEOError(f"{label}.{key} must contain unique non-empty dataset names")
    return values


def _require_false(payload: Mapping[str, Any], key: str, label: str) -> None:
    if payload.get(key) is not False:
        raise FrozenLOEOError(f"{label}.{key} must be explicitly false")


def build_frozen_loeo_plan(
    *,
    learned_selection: Mapping[str, Any],
    candidate_shortlist: Mapping[str, Any],
    monitor_prediction_plan: Mapping[str, Any],
) -> FrozenLOEOPlan:
    """Validate the complete Phase-2E/2F chain and extract one frozen winner."""

    for label, payload in (
        ("learned_selection", learned_selection),
        ("candidate_shortlist", candidate_shortlist),
        ("monitor_prediction_plan", monitor_prediction_plan),
    ):
        if not isinstance(payload, Mapping):
            raise FrozenLOEOError(f"{label} must be a JSON object")

    learned_scope = learned_selection.get("selection_scope")
    candidate_scope = candidate_shortlist.get("selection_scope")
    frozen_candidates = candidate_shortlist.get("shortlist")
    if not isinstance(learned_scope, Mapping):
        raise FrozenLOEOError("learned_selection has no selection_scope object")
    if not isinstance(candidate_scope, Mapping):
        raise FrozenLOEOError("candidate_shortlist has no selection_scope object")
    if not isinstance(frozen_candidates, Mapping):
        raise FrozenLOEOError("candidate_shortlist has no shortlist object")

    _require_false(learned_scope, "loeo_used", "learned_selection.selection_scope")
    _require_false(
        learned_scope,
        "loeo_may_retune_or_replace_winner",
        "learned_selection.selection_scope",
    )
    _require_false(candidate_scope, "loeo_used", "candidate_shortlist.selection_scope")
    _require_false(
        frozen_candidates,
        "loeo_may_expand_shortlist",
        "candidate_shortlist.shortlist",
    )

    monitor = _names(learned_scope, "monitor_datasets", "learned_selection.selection_scope")
    holdout = _names(
        learned_scope,
        "forbidden_loeo_holdout_datasets",
        "learned_selection.selection_scope",
    )
    if set(monitor) & set(holdout):
        raise FrozenLOEOError("learned monitor and LOEO holdout datasets overlap")

    candidate_monitor = _names(candidate_scope, "monitor_datasets", "candidate_shortlist.selection_scope")
    candidate_holdout = _names(
        candidate_scope,
        "forbidden_loeo_holdout_datasets",
        "candidate_shortlist.selection_scope",
    )
    plan_monitor = tuple(
        sorted(str(value).strip() for value in monitor_prediction_plan.get("monitor_datasets", []))
    )
    plan_holdout = tuple(
        sorted(
            str(value).strip()
            for value in monitor_prediction_plan.get("forbidden_loeo_holdout_datasets", [])
        )
    )
    if monitor != candidate_monitor or monitor != plan_monitor:
        raise FrozenLOEOError("Phase-2E/2F artifacts disagree on monitor dataset scope")
    if holdout != candidate_holdout or holdout != plan_holdout:
        raise FrozenLOEOError("Phase-2E/2F artifacts disagree on LOEO holdout dataset scope")

    winner = learned_selection.get("winner")
    if not isinstance(winner, Mapping):
        raise FrozenLOEOError("learned_selection has no winner object")
    family = str(winner.get("family", "")).strip()
    if family not in {"organizer_control", "hoct"}:
        raise FrozenLOEOError(f"unsupported frozen winner family: {family!r}")
    if family == "organizer_control":
        return FrozenLOEOPlan(
            winner_family=family,
            monitor_datasets=monitor,
            holdout_datasets=holdout,
            hoct=None,
        )

    trial_id = str(winner.get("trial_id", "")).strip()
    trial = winner.get("trial")
    if not trial_id or not isinstance(trial, Mapping):
        raise FrozenLOEOError("HOCT winner lacks frozen trial_id/trial payload")
    if str(trial.get("trial_id", "")).strip() != trial_id:
        raise FrozenLOEOError("HOCT winner trial_id disagrees with embedded trial")
    spec = trial.get("spec")
    if not isinstance(spec, Mapping):
        raise FrozenLOEOError("HOCT winner trial has no spec object")
    candidate_id = str(spec.get("candidate_config_id", "")).strip()
    model_name = str(spec.get("model_name", "")).strip()
    solver = spec.get("solver")
    if not candidate_id or not model_name or not isinstance(solver, Mapping):
        raise FrozenLOEOError("HOCT winner spec is incomplete")
    try:
        window_size = int(spec["window_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenLOEOError("HOCT winner window_size is invalid") from exc
    if window_size <= 0:
        raise FrozenLOEOError("HOCT winner window_size must be positive")

    allowed_raw = frozen_candidates.get("allowed_config_ids")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise FrozenLOEOError("candidate shortlist has no allowed_config_ids")
    allowed = {str(value).strip() for value in allowed_raw}
    if candidate_id not in allowed:
        raise FrozenLOEOError(
            f"frozen HOCT winner uses candidate {candidate_id!r} outside Phase-2E shortlist"
        )

    trials = learned_selection.get("hoct_trials")
    if not isinstance(trials, list):
        raise FrozenLOEOError("learned_selection has no hoct_trials list")
    matches = [
        row for row in trials if isinstance(row, Mapping) and str(row.get("trial_id", "")) == trial_id
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(trial):
        raise FrozenLOEOError("frozen HOCT winner is not exactly one recorded training-side trial")

    return FrozenLOEOPlan(
        winner_family=family,
        monitor_datasets=monitor,
        holdout_datasets=holdout,
        hoct=FrozenHOCTSpec(
            trial_id=trial_id,
            candidate_config_id=candidate_id,
            model_name=model_name,
            window_size=window_size,
            solver=dict(solver),
        ),
    )


def candidate_frontier_row(
    candidate_shortlist: Mapping[str, Any],
    config_id: str,
) -> dict[str, Any]:
    """Return the exact frozen Phase-2E candidate row for one config ID."""

    shortlist = candidate_shortlist.get("shortlist")
    if not isinstance(shortlist, Mapping):
        raise FrozenLOEOError("candidate_shortlist has no shortlist object")
    rows = shortlist.get("frontier_trials")
    if not isinstance(rows, list):
        raise FrozenLOEOError("candidate shortlist has no frontier_trials")
    matches = [
        row for row in rows if isinstance(row, Mapping) and str(row.get("config_id", "")) == config_id
    ]
    if len(matches) != 1:
        raise FrozenLOEOError(
            f"candidate config {config_id!r} not uniquely found in frozen frontier"
        )
    return dict(matches[0])


def validate_exact_holdout_prediction_names(
    actual_names: set[str],
    plan: FrozenLOEOPlan,
) -> None:
    """Fail closed unless a prediction directory contains exactly the LOEO set."""

    expected = set(plan.holdout_datasets)
    if actual_names != expected:
        raise FrozenLOEOError(
            f"LOEO prediction set mismatch: missing={sorted(expected - actual_names)} "
            f"extras={sorted(actual_names - expected)}"
        )
    leaked_monitor = actual_names & set(plan.monitor_datasets)
    if leaked_monitor:
        raise FrozenLOEOError(
            f"training-side monitor outputs appeared in LOEO prediction directory: {sorted(leaked_monitor)}"
        )
