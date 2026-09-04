"""Oracle-first candidate sweeps for fixed-detection HOCT experiments.

The learned HOCT model should not spend GPU/ILP time on a proposal graph that
cannot even contain the recoverable GT edges. This module evaluates candidate
generation *before* learned scoring, on the exact same frozen detections and the
official node matching prepared by :mod:`biohub.analysis.oracles`.

A sweep point changes only candidate geometry/radius/neighbour count. It does
not run HOCT weights, tune the detector, or solve a lineage graph. The Pareto
frontier maximizes GT-edge proposal coverage while minimizing candidate-edge
count; it is an engineering shortlist, not a leaderboard score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import polars as pl

from biohub.analysis import FixedDetectionOracleContext

from .hoct_analysis import candidate_edges_in_source_detection_space
from .hoct_compat import HOCTPointGraphConfig, build_hoct_point_graph


class HOCTCandidateSweepError(ValueError):
    """Raised when a candidate sweep cannot be interpreted causally."""


@dataclass(frozen=True)
class HOCTCandidateTrial:
    """One candidate-generation configuration evaluated at fixed detections."""

    config_id: str
    candidate_distance_space: str
    candidate_distance_threshold: float
    distance_threshold_um: float | None
    distance_threshold_voxels: float | None
    n_neighbors: int
    max_delta_t: int
    detections: int
    candidate_edges: int
    candidate_edges_per_detection: float
    gt_edges: int
    detectable_gt_edges: int
    candidate_available_gt_edges: int
    candidate_generation_gap: int
    candidate_recall_of_detectable: float
    candidate_recall_all_gt: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HOCTCandidateSweepReport:
    """Results and cost/coverage Pareto frontiers for a candidate sweep."""

    trials: tuple[HOCTCandidateTrial, ...]
    pareto_config_ids: tuple[str, ...]
    pareto_config_ids_by_space: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict:
        return {
            "trials": [trial.to_dict() for trial in self.trials],
            "pareto_config_ids": list(self.pareto_config_ids),
            "pareto_config_ids_by_space": {
                key: list(values) for key, values in sorted(self.pareto_config_ids_by_space.items())
            },
        }


def _canonical_config_payload(config: HOCTPointGraphConfig) -> dict:
    return {
        "candidate_distance_space": config.candidate_distance_space,
        "candidate_distance_threshold": config.candidate_distance_threshold,
        "distance_threshold_um": config.distance_threshold_um,
        "distance_threshold_voxels": config.distance_threshold_voxels,
        "n_neighbors": config.n_neighbors,
        "max_delta_t": config.max_delta_t,
        "scale_zyx_um": list(config.scale_zyx_um),
    }


def candidate_config_id(config: HOCTPointGraphConfig) -> str:
    """Return a deterministic short ID for one complete candidate config."""

    payload = json.dumps(
        _canonical_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"hoct-cand-{digest}"


def _dominates(left: HOCTCandidateTrial, right: HOCTCandidateTrial) -> bool:
    """Return True when left is no worse in recall/cost and strictly better in one."""

    no_worse = (
        left.candidate_available_gt_edges >= right.candidate_available_gt_edges
        and left.candidate_edges <= right.candidate_edges
    )
    strictly_better = (
        left.candidate_available_gt_edges > right.candidate_available_gt_edges
        or left.candidate_edges < right.candidate_edges
    )
    return no_worse and strictly_better


def pareto_frontier(trials: Sequence[HOCTCandidateTrial]) -> tuple[str, ...]:
    """Return deterministic non-dominated config IDs in increasing graph-cost order."""

    unique_ids = {trial.config_id for trial in trials}
    if len(unique_ids) != len(trials):
        raise HOCTCandidateSweepError("candidate sweep contains duplicate config IDs")
    frontier = [
        trial
        for trial in trials
        if not any(_dominates(other, trial) for other in trials if other.config_id != trial.config_id)
    ]
    frontier.sort(
        key=lambda trial: (
            trial.candidate_edges,
            -trial.candidate_available_gt_edges,
            trial.candidate_distance_space,
            trial.candidate_distance_threshold,
            trial.n_neighbors,
            trial.config_id,
        )
    )
    return tuple(trial.config_id for trial in frontier)


def _validate_configs(configs: Iterable[HOCTPointGraphConfig]) -> tuple[HOCTPointGraphConfig, ...]:
    configs = tuple(configs)
    if not configs:
        raise HOCTCandidateSweepError("candidate sweep requires at least one configuration")
    if any(not isinstance(config, HOCTPointGraphConfig) for config in configs):
        raise HOCTCandidateSweepError("all candidate sweep entries must be HOCTPointGraphConfig")
    ids = [candidate_config_id(config) for config in configs]
    if len(ids) != len(set(ids)):
        raise HOCTCandidateSweepError("candidate sweep contains duplicate configurations")
    return configs


def evaluate_hoct_candidate_configs(
    detections: pl.DataFrame,
    oracle: FixedDetectionOracleContext,
    configs: Iterable[HOCTPointGraphConfig],
    *,
    shape_tzyx: tuple[int, int, int, int] | None = None,
) -> HOCTCandidateSweepReport:
    """Evaluate candidate proposal coverage for one dataset.

    ``detections`` must be the canonical fixed-detection table whose
    ``detection_id`` values are the node IDs in ``oracle``'s scored baseline
    prediction. A mismatch fails closed through the oracle's invalid-reference
    check instead of quietly comparing different cells.
    """

    configs = _validate_configs(configs)
    trials: list[HOCTCandidateTrial] = []
    for config in configs:
        graph = build_hoct_point_graph(detections, config, shape_tzyx=shape_tzyx)
        source_edges = candidate_edges_in_source_detection_space(graph)
        coverage = oracle.measure_candidate_coverage(source_edges)
        if coverage.candidate_invalid_node_refs:
            raise HOCTCandidateSweepError(
                f"candidate config {candidate_config_id(config)} references "
                f"{coverage.candidate_invalid_node_refs} nodes outside the fixed baseline; "
                "detection provenance does not match"
            )
        detections_count = int(graph.num_nodes())
        candidate_count = int(graph.num_edges())
        if candidate_count != coverage.candidate_edges_supplied:
            raise HOCTCandidateSweepError(
                "candidate graph contains duplicate/reconciled edges; graph edge count differs from "
                "unique source-detection candidate count"
            )
        trials.append(
            HOCTCandidateTrial(
                config_id=candidate_config_id(config),
                candidate_distance_space=config.candidate_distance_space,
                candidate_distance_threshold=config.candidate_distance_threshold,
                distance_threshold_um=config.distance_threshold_um,
                distance_threshold_voxels=config.distance_threshold_voxels,
                n_neighbors=config.n_neighbors,
                max_delta_t=config.max_delta_t,
                detections=detections_count,
                candidate_edges=candidate_count,
                candidate_edges_per_detection=(candidate_count / detections_count),
                gt_edges=oracle.gt_edge_count,
                detectable_gt_edges=oracle.gt_edges_both_endpoints_available,
                candidate_available_gt_edges=coverage.gt_edges_candidate_available,
                candidate_generation_gap=coverage.candidate_generation_gap,
                candidate_recall_of_detectable=coverage.candidate_recall_of_detectable,
                candidate_recall_all_gt=coverage.candidate_recall_all_gt,
            )
        )

    trials.sort(
        key=lambda trial: (
            trial.candidate_distance_space,
            trial.candidate_distance_threshold,
            trial.n_neighbors,
            trial.max_delta_t,
            trial.config_id,
        )
    )
    by_space: dict[str, tuple[str, ...]] = {}
    for space in sorted({trial.candidate_distance_space for trial in trials}):
        by_space[space] = pareto_frontier(
            [trial for trial in trials if trial.candidate_distance_space == space]
        )
    return HOCTCandidateSweepReport(
        trials=tuple(trials),
        pareto_config_ids=pareto_frontier(trials),
        pareto_config_ids_by_space=by_space,
    )


def expand_candidate_grid(
    payload: Mapping,
    *,
    scale_zyx_um: tuple[float, float, float],
) -> tuple[HOCTPointGraphConfig, ...]:
    """Expand an explicit JSON-style candidate grid into complete configurations.

    Expected schema::

        {
          "n_neighbors": [3, 5],
          "max_delta_t": [1],
          "physical_um": [2.0, 4.0],
          "hoct_native_voxel": [4.0, 8.0]
        }

    No distance or neighbour defaults are invented. At least one distance-space
    list and a non-empty ``n_neighbors`` list are required. ``max_delta_t`` may
    be omitted only to use the competition-focused value ``[1]``.
    """

    if not isinstance(payload, Mapping):
        raise HOCTCandidateSweepError("candidate grid must be a JSON object")
    allowed = {"n_neighbors", "max_delta_t", "physical_um", "hoct_native_voxel"}
    unknown = set(payload) - allowed
    if unknown:
        raise HOCTCandidateSweepError(f"unknown candidate-grid keys: {sorted(unknown)}")

    neighbors = payload.get("n_neighbors")
    if not isinstance(neighbors, list) or not neighbors:
        raise HOCTCandidateSweepError("candidate grid requires a non-empty n_neighbors list")
    delta_ts = payload.get("max_delta_t", [1])
    if not isinstance(delta_ts, list) or not delta_ts:
        raise HOCTCandidateSweepError("max_delta_t must be a non-empty list when supplied")

    spaces: list[tuple[str, list]] = []
    for key in ("physical_um", "hoct_native_voxel"):
        values = payload.get(key, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise HOCTCandidateSweepError(f"{key} must be a list")
        if values:
            spaces.append((key, values))
    if not spaces:
        raise HOCTCandidateSweepError(
            "candidate grid requires physical_um and/or hoct_native_voxel thresholds"
        )

    configs: list[HOCTPointGraphConfig] = []
    for space, thresholds in spaces:
        for threshold in thresholds:
            for n_neighbors in neighbors:
                for max_delta_t in delta_ts:
                    kwargs = {
                        "n_neighbors": n_neighbors,
                        "max_delta_t": max_delta_t,
                        "scale_zyx_um": scale_zyx_um,
                    }
                    if space == "physical_um":
                        kwargs["distance_threshold_um"] = threshold
                    else:
                        kwargs["distance_threshold_voxels"] = threshold
                    configs.append(HOCTPointGraphConfig(**kwargs))
    return _validate_configs(configs)
