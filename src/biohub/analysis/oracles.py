"""Fixed-detection bottleneck decomposition for the Biohub edge metric.

This module does not invent a replacement score. It first runs the pinned
organizer evaluator so predicted nodes receive the same matching attributes used
by official scoring. It then asks counterfactual *coverage* questions:

* how many GT edges have both endpoint cells represented by matched detections?
* if a pre-solver candidate edge set is supplied, how many of those detectable
  GT edges are present in the candidate graph?
* how many GT edges were actually recovered by the official scorer?

The resulting gaps are opportunity counts, not alternative leaderboard scores.
They are intended to decide whether the next unit of work belongs in detection,
candidate generation, or association/global selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import tracksdata as td

from biohub.evaluation.official import (
    DEFAULT_SCALE,
    MAX_DISTANCE_UM,
    evaluate_graph_pair,
)


class OracleAnalysisError(ValueError):
    """Raised when a decomposition request is ambiguous or internally invalid."""


@dataclass(frozen=True)
class BottleneckDecomposition:
    """Opportunity counts at fixed detections under official node matching."""

    gt_edges: int
    official_edge_tp: int
    official_edge_fp: int
    official_edge_fn: int
    gt_nodes: int
    pred_nodes: int
    matched_gt_nodes: int
    gt_edges_both_endpoints_available: int
    gt_edges_source_only_available: int
    gt_edges_target_only_available: int
    gt_edges_neither_endpoint_available: int
    detection_unavailable_edges: int
    fixed_detection_edge_ceiling_recall: float
    fixed_detection_recoverable_edges: int
    candidate_edges_supplied: int | None
    candidate_invalid_node_refs: int | None
    gt_edges_candidate_available: int | None
    candidate_generation_gap: int | None
    candidate_to_selected_gap: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _node_match_sets(pred_graph: td.graph.BaseGraph) -> tuple[dict[int, set[int]], set[int]]:
    """Return GT→pred match sets and the set of valid predicted node IDs."""

    node_id = td.DEFAULT_ATTR_KEYS.NODE_ID
    matched_id = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    if matched_id not in pred_graph.node_attr_keys():
        raise OracleAnalysisError(
            "Predicted graph has no official matching attributes after evaluation. "
            "The decomposition currently requires at least one predicted edge so "
            "the pinned evaluator executes node matching."
        )

    attrs = pred_graph.node_attrs(attr_keys=[node_id, matched_id])
    valid_pred_ids = {int(value) for value in attrs[node_id].to_list()}
    gt_to_pred: dict[int, set[int]] = {}
    for pred, gt in zip(attrs[node_id].to_list(), attrs[matched_id].to_list()):
        if gt is None or int(gt) == -1:
            continue
        gt_to_pred.setdefault(int(gt), set()).add(int(pred))
    return gt_to_pred, valid_pred_ids


def _gt_edges(gt_graph: td.graph.BaseGraph) -> tuple[tuple[int, int], ...]:
    source = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    target = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    attrs = gt_graph.edge_attrs(attr_keys=[source, target])
    return tuple((int(s), int(t)) for s, t in zip(attrs[source].to_list(), attrs[target].to_list()))


def _candidate_gt_coverage(
    candidate_edges: Iterable[tuple[int, int]],
    *,
    gt_to_pred: dict[int, set[int]],
    gt_edges: tuple[tuple[int, int], ...],
    valid_pred_ids: set[int],
) -> tuple[int, int, int]:
    """Return (supplied unique candidates, invalid refs, GT edges covered)."""

    candidates = {(int(source), int(target)) for source, target in candidate_edges}
    invalid = sum(
        source not in valid_pred_ids or target not in valid_pred_ids
        for source, target in candidates
    )
    valid_candidates = {
        (source, target)
        for source, target in candidates
        if source in valid_pred_ids and target in valid_pred_ids
    }

    covered = 0
    for gt_source, gt_target in gt_edges:
        pred_sources = gt_to_pred.get(gt_source, set())
        pred_targets = gt_to_pred.get(gt_target, set())
        if not pred_sources or not pred_targets:
            continue
        if any((ps, pt) in valid_candidates for ps in pred_sources for pt in pred_targets):
            covered += 1
    return len(candidates), invalid, covered


def decompose_fixed_detections(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    *,
    estimated_total_nodes: float,
    candidate_edges: Iterable[tuple[int, int]] | None = None,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE_UM,
) -> BottleneckDecomposition:
    """Decompose recoverable edge opportunity without changing detections.

    The official evaluator is executed first and mutates ``pred_graph`` by
    writing official node/edge matching attributes. ``candidate_edges`` should
    represent the pre-solver/pre-threshold edge proposal set using *predicted
    graph node IDs*. If omitted, only the detection-vs-selection gap is measured.

    ``fixed_detection_edge_ceiling_recall`` is simply the fraction of GT edges
    whose two endpoints are represented by official-matched detections. It is a
    coverage ceiling on edge recall at these detections; it is **not** an
    adjusted-Jaccard or final-score oracle because FP, count adjustment,
    topology, and division terms still matter.
    """

    if gt_graph.num_edges() <= 0:
        raise OracleAnalysisError("GT graph has no edges; edge bottleneck decomposition is undefined")
    if pred_graph.num_edges() <= 0:
        raise OracleAnalysisError(
            "Predicted graph has no edges. The pinned evaluator skips node matching in this case; "
            "provide a scored prediction graph with at least one edge."
        )

    official_row = evaluate_graph_pair(
        pred_graph,
        gt_graph,
        estimated_total_nodes=estimated_total_nodes,
        scale=scale,
        max_distance=max_distance,
    )
    gt_to_pred, valid_pred_ids = _node_match_sets(pred_graph)
    edges = _gt_edges(gt_graph)

    both = source_only = target_only = neither = 0
    for source, target in edges:
        source_available = bool(gt_to_pred.get(source))
        target_available = bool(gt_to_pred.get(target))
        if source_available and target_available:
            both += 1
        elif source_available:
            source_only += 1
        elif target_available:
            target_only += 1
        else:
            neither += 1

    gt_edge_count = len(edges)
    unavailable = gt_edge_count - both
    edge_tp = int(official_row["edge_tp"])
    fixed_detection_gap = max(0, both - edge_tp)

    candidate_count = invalid_refs = candidate_available = None
    candidate_generation_gap = candidate_to_selected_gap = None
    if candidate_edges is not None:
        candidate_count, invalid_refs, candidate_available = _candidate_gt_coverage(
            candidate_edges,
            gt_to_pred=gt_to_pred,
            gt_edges=edges,
            valid_pred_ids=valid_pred_ids,
        )
        if candidate_available > both:
            raise OracleAnalysisError("candidate coverage exceeded fixed-detection endpoint coverage")
        candidate_generation_gap = both - candidate_available
        # If selected prediction edges recover more GT edges than the supplied
        # candidate set, those candidates cannot be the actual pre-selection set.
        if edge_tp > candidate_available:
            raise OracleAnalysisError(
                "Official TP count exceeds GT-edge coverage of candidate_edges; "
                "the supplied candidate set cannot contain the selected solution."
            )
        candidate_to_selected_gap = candidate_available - edge_tp

    return BottleneckDecomposition(
        gt_edges=gt_edge_count,
        official_edge_tp=edge_tp,
        official_edge_fp=int(official_row["edge_fp"]),
        official_edge_fn=int(official_row["edge_fn"]),
        gt_nodes=int(gt_graph.num_nodes()),
        pred_nodes=int(pred_graph.num_nodes()),
        matched_gt_nodes=len(gt_to_pred),
        gt_edges_both_endpoints_available=both,
        gt_edges_source_only_available=source_only,
        gt_edges_target_only_available=target_only,
        gt_edges_neither_endpoint_available=neither,
        detection_unavailable_edges=unavailable,
        fixed_detection_edge_ceiling_recall=both / gt_edge_count,
        fixed_detection_recoverable_edges=fixed_detection_gap,
        candidate_edges_supplied=candidate_count,
        candidate_invalid_node_refs=invalid_refs,
        gt_edges_candidate_available=candidate_available,
        candidate_generation_gap=candidate_generation_gap,
        candidate_to_selected_gap=candidate_to_selected_gap,
    )
