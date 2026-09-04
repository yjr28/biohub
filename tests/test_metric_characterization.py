"""Phase 0C: executable characterization of competition metric behavior.

These tests call the pinned organizer evaluator. They are not a second metric
implementation. Each test isolates one behavior that later experiments are
allowed to rely on.
"""

from __future__ import annotations

import copy

import polars as pl
import pytest
import tracksdata as td
from tracking_cellmot.metrics import summarise

from biohub.evaluation.official import evaluate_graph_pair

UNIT_SCALE = (1.0, 1.0, 1.0)


def graph(nodes: dict[str, dict], edges: list[tuple[str, str]]) -> td.graph.InMemoryGraph:
    """Construct a small tracking graph while preserving edge insertion order."""

    result = td.graph.InMemoryGraph()
    result.add_node_attr_key("z", pl.Float64, 0.0)
    result.add_node_attr_key("y", pl.Float64, 0.0)
    result.add_node_attr_key("x", pl.Float64, 0.0)
    ids: dict[str, int] = {}
    for name, attrs in nodes.items():
        ids[name] = result.add_node(attrs=copy.deepcopy(attrs))
    for source, target in edges:
        result.add_edge(ids[source], ids[target], {})
    return result


def linear_nodes(*, z: float = 0.0) -> dict[str, dict]:
    return {
        "A": {"t": 0, "z": z, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": z, "y": 0.0, "x": 0.0},
    }


def score_pair(
    pred: td.graph.BaseGraph,
    gt: td.graph.BaseGraph,
    *,
    estimated_total_nodes: float,
    scale: tuple[float, float, float] = UNIT_SCALE,
    max_distance: float = 7.0,
) -> dict:
    return evaluate_graph_pair(
        pred,
        gt,
        estimated_total_nodes=estimated_total_nodes,
        scale=scale,
        max_distance=max_distance,
    )


def test_perfect_linear_track_scores_one() -> None:
    nodes = linear_nodes()
    row = score_pair(
        graph(nodes, [("A", "B")]),
        graph(nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )
    assert row["edge_tp"] == 1
    assert row["edge_fp"] == 0
    assert row["edge_fn"] == 0
    assert row["edge_jaccard"] == pytest.approx(1.0)
    assert row["adj_edge_jaccard"] == pytest.approx(1.0)


def test_distance_threshold_is_inclusive_at_exactly_seven_micrometers() -> None:
    gt_nodes = linear_nodes(z=0.0)
    at_boundary = linear_nodes(z=7.0)
    outside = linear_nodes(z=7.000001)

    boundary_row = score_pair(
        graph(at_boundary, [("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )
    outside_row = score_pair(
        graph(outside, [("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )

    assert boundary_row["edge_jaccard"] == pytest.approx(1.0)
    assert outside_row["edge_jaccard"] == pytest.approx(0.0)


def test_anisotropic_scale_is_applied_before_distance_threshold() -> None:
    gt_nodes = linear_nodes(z=0.0)
    within = linear_nodes(z=4.0)  # 4 * 1.625 = 6.5 um
    outside = linear_nodes(z=5.0)  # 5 * 1.625 = 8.125 um
    physical_scale = (1.625, 0.40625, 0.40625)

    within_row = score_pair(
        graph(within, [("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=2,
        scale=physical_scale,
    )
    outside_row = score_pair(
        graph(outside, [("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=2,
        scale=physical_scale,
    )

    assert within_row["edge_jaccard"] == pytest.approx(1.0)
    assert outside_row["edge_jaccard"] == pytest.approx(0.0)


def test_node_matching_is_timepoint_aware() -> None:
    gt_nodes = linear_nodes()
    pred_nodes = {
        "A": {"t": 5, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 6, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    row = score_pair(
        graph(pred_nodes, [("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )
    assert row["edge_tp"] == 0
    assert row["edge_fn"] == 1
    assert row["edge_jaccard"] == pytest.approx(0.0)


def test_skip_edge_is_not_a_substitute_for_two_consecutive_gt_edges() -> None:
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    row = score_pair(
        graph(nodes, [("A", "C")]),
        graph(nodes, [("A", "B"), ("B", "C")]),
        estimated_total_nodes=3,
    )
    assert row["edge_tp"] == 0
    assert row["edge_fp"] == 0
    assert row["edge_fn"] == 2
    assert row["edge_jaccard"] == pytest.approx(0.0)


def test_backward_edge_is_dropped_from_edge_metric() -> None:
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    row = score_pair(
        graph(nodes, [("A", "B"), ("B", "C"), ("C", "B")]),
        graph(nodes, [("A", "B"), ("B", "C")]),
        estimated_total_nodes=3,
    )
    assert row["edge_tp"] == 2
    assert row["edge_fp"] == 0
    assert row["edge_fn"] == 0
    assert row["edge_jaccard"] == pytest.approx(1.0)


def test_sparse_gt_extra_edge_after_track_end_is_edge_invisible() -> None:
    gt_nodes = linear_nodes()
    pred_nodes = {
        **gt_nodes,
        "C": {"t": 2, "z": 100.0, "y": 100.0, "x": 100.0},
    }
    row = score_pair(
        graph(pred_nodes, [("A", "B"), ("B", "C")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=3,
    )
    assert row["edge_tp"] == 1
    assert row["edge_fp"] == 0
    assert row["edge_fn"] == 0
    assert row["edge_jaccard"] == pytest.approx(1.0)


def test_wrong_edge_into_annotated_interior_node_is_penalized() -> None:
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    pred_nodes = {
        **gt_nodes,
        "D": {"t": 0, "z": 100.0, "y": 100.0, "x": 100.0},
    }
    row = score_pair(
        graph(pred_nodes, [("A", "B"), ("B", "C"), ("D", "B")]),
        graph(gt_nodes, [("A", "B"), ("B", "C")]),
        estimated_total_nodes=4,
    )
    assert row["edge_tp"] == 2
    assert row["edge_fp"] == 1
    assert row["edge_fn"] == 0
    assert row["edge_jaccard"] == pytest.approx(2 / 3)


def test_duplicate_prediction_edge_cannot_inflate_score() -> None:
    nodes = linear_nodes()
    row = score_pair(
        graph(nodes, [("A", "B"), ("A", "B")]),
        graph(nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )
    assert row["edge_tp"] == 1
    assert row["edge_fp"] == 0
    assert row["edge_jaccard"] == pytest.approx(1.0)


def test_outdegree_cap_keeps_two_lowest_edge_ids_and_is_order_sensitive() -> None:
    gt_nodes = linear_nodes()
    pred_nodes = {
        **gt_nodes,
        "X": {"t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
        "Y": {"t": 1, "z": 0.0, "y": 200.0, "x": 0.0},
    }

    correct_first = score_pair(
        graph(pred_nodes, [("A", "B"), ("A", "X"), ("A", "Y")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=4,
    )
    correct_third = score_pair(
        graph(pred_nodes, [("A", "X"), ("A", "Y"), ("A", "B")]),
        graph(gt_nodes, [("A", "B")]),
        estimated_total_nodes=4,
    )

    assert correct_first["edge_tp"] == 1
    assert correct_first["edge_fp"] == 1
    assert correct_first["edge_jaccard"] == pytest.approx(1 / 2)
    assert correct_third["edge_tp"] == 0
    assert correct_third["edge_fp"] == 2
    assert correct_third["edge_fn"] == 1
    assert correct_third["edge_jaccard"] == pytest.approx(0.0)


def test_node_count_adjustment_rewards_underestimate_and_penalizes_overestimate() -> None:
    nodes = linear_nodes()

    equal = score_pair(
        graph(nodes, [("A", "B")]),
        graph(nodes, [("A", "B")]),
        estimated_total_nodes=2,
    )
    overpredicted = score_pair(
        graph(nodes, [("A", "B")]),
        graph(nodes, [("A", "B")]),
        estimated_total_nodes=1,
    )
    underpredicted = score_pair(
        graph(nodes, [("A", "B")]),
        graph(nodes, [("A", "B")]),
        estimated_total_nodes=4,
    )

    assert equal["edge_jaccard"] == pytest.approx(1.0)
    assert equal["adj_edge_jaccard"] == pytest.approx(1.0)
    assert overpredicted["adj_edge_jaccard"] == pytest.approx(0.9)
    assert underpredicted["adj_edge_jaccard"] == pytest.approx(1.05)
    assert underpredicted["adj_edge_jaccard"] > underpredicted["edge_jaccard"]


def test_exact_division_is_recovered() -> None:
    nodes = {
        "P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "D": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
        "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
        "G1": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
        "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
    }
    edges = [("P", "D"), ("D", "C1"), ("D", "C2"), ("C1", "G1"), ("C2", "G2")]
    row = score_pair(
        graph(nodes, edges),
        graph(nodes, edges),
        estimated_total_nodes=6,
        max_distance=1.0,
    )
    assert row["division_tp"] == 1
    assert row["division_fp"] == 0
    assert row["division_fn"] == 0
    assert row["edge_jaccard"] == pytest.approx(1.0)


def test_run_adjusted_edge_is_sample_weighted_not_plain_averaged() -> None:
    rows = [
        {
            "edge_tp": 9,
            "edge_fp": 1,
            "edge_fn": 0,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 10,
            "node_recall": 0.9,
            "total_node_ratio": 0.0,
            "edge_jaccard": 0.9,
            "adj_edge_jaccard": 0.9,
        },
        {
            "edge_tp": 1,
            "edge_fp": 0,
            "edge_fn": 0,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 1,
            "node_recall": 1.0,
            "total_node_ratio": 0.0,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 0.5,
        },
    ]
    with pytest.warns(UserWarning, match="No divisions"):
        summary = summarise(rows)

    expected = (10 * 0.9 + 1 * 0.5) / 11
    assert summary["adj_edge_jaccard"] == pytest.approx(expected)
    assert summary["adj_edge_jaccard"] != pytest.approx((0.9 + 0.5) / 2)
    assert summary["score"] == pytest.approx(expected)
