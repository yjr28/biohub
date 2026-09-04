"""Paired LOEO comparison utilities.

A mean improvement can hide a catastrophic reverse-direction regression. This
module keeps both embryo directions visible and reports the mean and worst-case
score deltas without inventing a new competition score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class ComparisonError(ValueError):
    """Raised when report summaries are missing required comparable evidence."""


PRIMARY_METRICS = (
    "score",
    "adj_edge_jaccard",
    "edge_jaccard",
    "division_jaccard",
    "node_recall",
    "total_node_ratio",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
)


def _number(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"metric {key!r} is not numeric: {value!r}") from exc
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class FoldComparison:
    fold: str
    baseline: dict[str, float | None]
    challenger: dict[str, float | None]
    delta: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedLOEOComparison:
    folds: tuple[FoldComparison, FoldComparison]
    score_delta_mean: float
    score_delta_worst: float
    score_delta_best: float
    both_score_directions_positive: bool
    both_score_directions_nonnegative: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_fold(
    fold: str,
    baseline_summary: Mapping[str, Any],
    challenger_summary: Mapping[str, Any],
) -> FoldComparison:
    """Compare organizer summary fields for one exact LOEO direction."""

    fold = str(fold).strip()
    if not fold:
        raise ComparisonError("fold cannot be empty")
    baseline: dict[str, float | None] = {}
    challenger: dict[str, float | None] = {}
    delta: dict[str, float | None] = {}
    for key in PRIMARY_METRICS:
        b = _number(baseline_summary, key)
        c = _number(challenger_summary, key)
        baseline[key] = b
        challenger[key] = c
        delta[key] = None if b is None or c is None else c - b
    if delta["score"] is None:
        raise ComparisonError(f"fold {fold!r} has no finite comparable final score")
    return FoldComparison(fold=fold, baseline=baseline, challenger=challenger, delta=delta)


def compare_two_direction_loeo(
    baseline_by_fold: Mapping[str, Mapping[str, Any]],
    challenger_by_fold: Mapping[str, Mapping[str, Any]],
) -> PairedLOEOComparison:
    """Require exactly two identical fold names and retain their tail behavior."""

    baseline_folds = set(baseline_by_fold)
    challenger_folds = set(challenger_by_fold)
    if baseline_folds != challenger_folds:
        raise ComparisonError(
            "baseline/challenger fold sets differ: "
            f"baseline={sorted(baseline_folds)}, challenger={sorted(challenger_folds)}"
        )
    if len(baseline_folds) != 2:
        raise ComparisonError(
            f"Biohub primary paired LOEO comparison requires exactly two folds; got {sorted(baseline_folds)}"
        )

    folds = tuple(
        compare_fold(name, baseline_by_fold[name], challenger_by_fold[name])
        for name in sorted(baseline_folds)
    )
    deltas = tuple(float(fold.delta["score"]) for fold in folds)
    return PairedLOEOComparison(
        folds=(folds[0], folds[1]),
        score_delta_mean=sum(deltas) / 2.0,
        score_delta_worst=min(deltas),
        score_delta_best=max(deltas),
        both_score_directions_positive=all(delta > 0 for delta in deltas),
        both_score_directions_nonnegative=all(delta >= 0 for delta in deltas),
    )
