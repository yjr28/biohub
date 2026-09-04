"""Strict orchestration around the organizer's official evaluator.

This module deliberately does not reimplement metric math. The score-producing
functions are imported from the pinned organizer submodule and called directly.
Our additions are limited to file loading, expected-set validation, provenance
constants, and fail-closed behavior for metadata required by the adjusted score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.metrics import (
    ADJUSTMENT_ALPHA,
    SCORE_DIVISION_WEIGHT,
    evaluate as _official_evaluate,
    node_recall as _official_node_recall,
    per_sample_metrics as _official_per_sample_metrics,
    summarise as _official_summarise,
)

OFFICIAL_EVALUATOR_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
TRACKSDATA_COMMIT = "39dccf3a243e44274759468cb31b2ad9e7fc1d09"
MAX_DISTANCE_UM = 7.0
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

_EXPECTED_ADJUSTMENT_ALPHA = 0.1
_EXPECTED_DIVISION_WEIGHT = 0.1


@dataclass(frozen=True)
class EvaluationRun:
    """Competition-style evaluation result for an explicit dataset set."""

    names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class EvaluationInputError(ValueError):
    """Raised when local evaluation inputs are incomplete or ambiguous."""


def assert_official_constants() -> None:
    """Fail if the imported official metric no longer matches our audited contract."""

    if ADJUSTMENT_ALPHA != _EXPECTED_ADJUSTMENT_ALPHA:
        raise RuntimeError(
            f"Official ADJUSTMENT_ALPHA drifted: {ADJUSTMENT_ALPHA!r} != "
            f"{_EXPECTED_ADJUSTMENT_ALPHA!r}"
        )
    if SCORE_DIVISION_WEIGHT != _EXPECTED_DIVISION_WEIGHT:
        raise RuntimeError(
            f"Official SCORE_DIVISION_WEIGHT drifted: {SCORE_DIVISION_WEIGHT!r} != "
            f"{_EXPECTED_DIVISION_WEIGHT!r}"
        )


def load_geff(path: str | Path) -> td.graph.BaseGraph:
    """Load one GEFF graph using the same tracksdata backend as the organizer script."""

    path = Path(path)
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def read_estimated_node_count(path: str | Path) -> float:
    """Read the GT `estimated_number_of_nodes` value required by score adjustment."""

    path = Path(path)
    metadata = GeffMetadata.read(path)
    value = (metadata.extra or {}).get("estimated_number_of_nodes")
    if value is None:
        raise EvaluationInputError(
            f"{path} has no GEFF metadata key 'estimated_number_of_nodes'; "
            "competition-style adjusted scoring would be incomplete."
        )
    value = float(value)
    if value <= 0:
        raise EvaluationInputError(
            f"{path} has invalid estimated_number_of_nodes={value!r}; expected > 0."
        )
    return value


def evaluate_graph_pair(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    *,
    estimated_total_nodes: float,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE_UM,
) -> dict[str, Any]:
    """Evaluate one in-memory pair through the exact organizer metric path.

    `pred_graph` is mutated by the official matching code, exactly as in the
    organizer implementation.
    """

    assert_official_constants()
    if estimated_total_nodes <= 0:
        raise EvaluationInputError("estimated_total_nodes must be > 0")

    result = _official_evaluate(
        pred_graph,
        gt_graph,
        scale=scale,
        max_distance=max_distance,
    )
    recall = (
        _official_node_recall(pred_graph, gt_graph)
        if pred_graph.num_nodes() > 0 and pred_graph.num_edges() > 0
        else 0.0
    )
    return _official_per_sample_metrics(result, float(estimated_total_nodes), recall)


def evaluate_geff_pair(
    pred_path: str | Path,
    gt_path: str | Path,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE_UM,
) -> dict[str, Any]:
    """Load and evaluate one predicted/GT GEFF pair."""

    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    return evaluate_graph_pair(
        load_geff(pred_path),
        load_geff(gt_path),
        estimated_total_nodes=read_estimated_node_count(gt_path),
        scale=scale,
        max_distance=max_distance,
    )


def _stem_set(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.geff")}


def _resolve_names(
    pred_dir: Path,
    gt_dir: Path,
    expected_names: Iterable[str] | None,
    strict: bool,
) -> tuple[str, ...]:
    pred_names = _stem_set(pred_dir)
    gt_names = _stem_set(gt_dir)

    if expected_names is not None:
        requested = set(expected_names)
        if not requested:
            raise EvaluationInputError("expected_names cannot be empty")
        missing_pred = requested - pred_names
        missing_gt = requested - gt_names
        if missing_pred or missing_gt:
            raise EvaluationInputError(
                "Requested evaluation set is incomplete: "
                f"missing predictions={sorted(missing_pred)}, missing GT={sorted(missing_gt)}"
            )
        return tuple(sorted(requested))

    if strict:
        missing_pred = gt_names - pred_names
        extra_pred = pred_names - gt_names
        if missing_pred or extra_pred:
            raise EvaluationInputError(
                "Prediction/GT GEFF sets differ under strict evaluation: "
                f"missing predictions={sorted(missing_pred)}, extra predictions={sorted(extra_pred)}"
            )
        if not gt_names:
            raise EvaluationInputError("No GEFF files found for evaluation")
        return tuple(sorted(gt_names))

    intersection = pred_names & gt_names
    if not intersection:
        raise EvaluationInputError("Prediction and GT directories have no GEFF names in common")
    return tuple(sorted(intersection))


def evaluate_directory(
    pred_dir: str | Path,
    gt_dir: str | Path,
    *,
    expected_names: Iterable[str] | None = None,
    strict: bool = True,
    default_scale: tuple[float, float, float] = DEFAULT_SCALE,
    scale_by_name: Mapping[str, tuple[float, float, float]] | None = None,
    max_distance: float = MAX_DISTANCE_UM,
) -> EvaluationRun:
    """Evaluate an explicit set of GEFFs and aggregate with official `summarise`.

    Parameters
    ----------
    expected_names
        If provided, evaluate exactly these dataset stems and require every one
        to exist in both directories. Extra files are ignored. This is the
        preferred mode for embryo/fold validation.
    strict
        Used only when `expected_names` is omitted. If True (default), the two
        directories must contain exactly the same GEFF stems. If False, use the
        intersection, matching the organizer development CLI's convenience
        behavior.
    scale_by_name
        Optional per-dataset physical `(z, y, x)` scale. Otherwise
        `default_scale` is used.
    """

    assert_official_constants()
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if not pred_dir.is_dir():
        raise EvaluationInputError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.is_dir():
        raise EvaluationInputError(f"GT directory does not exist: {gt_dir}")

    names = _resolve_names(pred_dir, gt_dir, expected_names, strict)
    rows: list[dict[str, Any]] = []
    for name in names:
        scale = scale_by_name[name] if scale_by_name and name in scale_by_name else default_scale
        rows.append(
            evaluate_geff_pair(
                pred_dir / f"{name}.geff",
                gt_dir / f"{name}.geff",
                scale=scale,
                max_distance=max_distance,
            )
        )

    summary = _official_summarise(rows)
    return EvaluationRun(names=names, rows=tuple(rows), summary=summary)
