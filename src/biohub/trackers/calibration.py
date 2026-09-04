"""Leakage controls for tracker-hyperparameter calibration.

The true LOEO embryo is the scarce generalization test and must not be consumed
while choosing candidate radii, neighbour counts, solver weights, or other
tracker hyperparameters. Phase 2C already creates a deterministic dataset-level
checkpoint monitor *inside* the training embryo. Tracker calibration reuses that
training-side set and records the opposite embryo as forbidden.

This is a calibration boundary, not a claim that the same-embryo monitor is an
independent estimate of hidden-embryo performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


class TrackerCalibrationScopeError(ValueError):
    """Raised when a protocol cannot provide a leakage-safe calibration scope."""


@dataclass(frozen=True)
class TrackerCalibrationScope:
    """Training-side tracker-calibration datasets and forbidden LOEO datasets."""

    checkpoint_monitor_policy: str
    calibration_datasets: tuple[str, ...]
    forbidden_loeo_holdout_datasets: tuple[str, ...]
    train_datasets: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["loeo_holdout_used"] = False
        return payload


def _names(protocol: Mapping, key: str) -> tuple[str, ...]:
    raw = protocol.get(key)
    if not isinstance(raw, (list, tuple)):
        raise TrackerCalibrationScopeError(f"protocol field {key!r} must be a list")
    names = tuple(sorted(str(value).strip() for value in raw if str(value).strip()))
    if len(names) != len(set(names)):
        raise TrackerCalibrationScopeError(f"protocol field {key!r} contains duplicates")
    return names


def calibration_scope_from_protocol(protocol: Mapping) -> TrackerCalibrationScope:
    """Derive the only allowed tracker-selection scope from a Phase-2C protocol.

    Requirements
    ------------
    - The policy must be ``train-embryo-hash-holdout``. ``train-embryo-all`` is
      acceptable for the public-reference baseline but reuses optimizer data as
      its monitor, so it is not promoted as our tracker-selection set.
    - Calibration datasets must be a non-empty subset of declared training
      datasets.
    - The true LOEO holdout must be non-empty and disjoint from calibration.
    - Training and holdout dataset universes must themselves be disjoint.

    The returned object explicitly records the holdout as forbidden. A caller
    that wants to tune on the LOEO embryo must use a different, deliberately
    named debugging path rather than silently passing this gate.
    """

    if not isinstance(protocol, Mapping):
        raise TrackerCalibrationScopeError("protocol must be a mapping")
    policy = str(protocol.get("checkpoint_monitor_policy", "")).strip()
    if policy != "train-embryo-hash-holdout":
        raise TrackerCalibrationScopeError(
            "tracker calibration requires checkpoint_monitor_policy='train-embryo-hash-holdout'"
        )

    train = _names(protocol, "train_datasets")
    calibration = _names(protocol, "checkpoint_monitor_datasets")
    holdout = _names(protocol, "holdout_datasets")
    if not train:
        raise TrackerCalibrationScopeError("protocol has no train_datasets")
    if not calibration:
        raise TrackerCalibrationScopeError("protocol has no checkpoint_monitor_datasets")
    if not holdout:
        raise TrackerCalibrationScopeError("protocol has no holdout_datasets")

    train_set = set(train)
    calibration_set = set(calibration)
    holdout_set = set(holdout)
    if not calibration_set <= train_set:
        raise TrackerCalibrationScopeError(
            "checkpoint-monitor calibration set is not a subset of declared training datasets"
        )
    overlap = calibration_set & holdout_set
    if overlap:
        raise TrackerCalibrationScopeError(
            f"tracker calibration and LOEO holdout overlap is forbidden: {sorted(overlap)}"
        )
    train_holdout_overlap = train_set & holdout_set
    if train_holdout_overlap:
        raise TrackerCalibrationScopeError(
            f"declared training and LOEO holdout datasets overlap: {sorted(train_holdout_overlap)}"
        )

    return TrackerCalibrationScope(
        checkpoint_monitor_policy=policy,
        calibration_datasets=calibration,
        forbidden_loeo_holdout_datasets=holdout,
        train_datasets=train,
    )
