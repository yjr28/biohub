import polars as pl
import pytest
import tracksdata as td

from biohub.analysis.oracles import (
    OracleAnalysisError,
    decompose_fixed_detections,
    prepare_fixed_detection_oracle,
)


def _graph(nodes, edges):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    ids = {}
    for name, attrs in nodes.items():
        ids[name] = graph.add_node(attrs=attrs)
    for source, target in edges:
        graph.add_edge(ids[source], ids[target], {})
    return graph, ids


def _gt():
    return _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
            "C": {"t": 2, "z": 0.0, "y": 2.0, "x": 0.0},
        },
        [("A", "B"), ("B", "C")],
    )


def test_fixed_detections_separate_detection_from_selection_gap():
    gt, _ = _gt()
    pred, pred_ids = _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
            "C": {"t": 2, "z": 0.0, "y": 2.0, "x": 0.0},
        },
        [("A", "B")],
    )

    result = decompose_fixed_detections(
        pred,
        gt,
        estimated_total_nodes=3,
        candidate_edges=[
            (pred_ids["A"], pred_ids["B"]),
            (pred_ids["B"], pred_ids["C"]),
        ],
    )

    assert result.gt_edges == 2
    assert result.official_edge_tp == 1
    assert result.official_edge_fn == 1
    assert result.gt_edges_both_endpoints_available == 2
    assert result.detection_unavailable_edges == 0
    assert result.fixed_detection_edge_ceiling_recall == 1.0
    assert result.fixed_detection_recoverable_edges == 1
    assert result.gt_edges_candidate_available == 2
    assert result.candidate_generation_gap == 0
    assert result.candidate_to_selected_gap == 1


def test_missing_detection_is_counted_upstream_of_association():
    gt, _ = _gt()
    pred, _ = _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
        },
        [("A", "B")],
    )

    result = decompose_fixed_detections(pred, gt, estimated_total_nodes=3)

    assert result.matched_gt_nodes == 2
    assert result.gt_edges_both_endpoints_available == 1
    assert result.gt_edges_source_only_available == 1
    assert result.detection_unavailable_edges == 1
    assert result.fixed_detection_edge_ceiling_recall == 0.5
    assert result.fixed_detection_recoverable_edges == 0


def test_candidate_generation_gap_is_visible_before_solver_selection():
    gt, _ = _gt()
    pred, pred_ids = _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
            "C": {"t": 2, "z": 0.0, "y": 2.0, "x": 0.0},
        },
        [("A", "B")],
    )

    result = decompose_fixed_detections(
        pred,
        gt,
        estimated_total_nodes=3,
        candidate_edges=[(pred_ids["A"], pred_ids["B"])],
    )

    assert result.gt_edges_candidate_available == 1
    assert result.candidate_generation_gap == 1
    assert result.candidate_to_selected_gap == 0


def test_alternative_candidate_coverage_does_not_require_baseline_selected_edges():
    gt, _ = _gt()
    pred, pred_ids = _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
            "C": {"t": 2, "z": 0.0, "y": 2.0, "x": 0.0},
        },
        [("A", "B")],
    )

    context = prepare_fixed_detection_oracle(pred, gt, estimated_total_nodes=3)

    # This alternative candidate graph intentionally omits the baseline TP A→B.
    # Candidate proposal analysis should still be valid even though the legacy
    # strict decomposition quite correctly rejects it as a possible superset of
    # the baseline solution.
    coverage = context.measure_candidate_coverage([(pred_ids["B"], pred_ids["C"])])
    assert coverage.candidate_edges_supplied == 1
    assert coverage.gt_edges_candidate_available == 1
    assert coverage.candidate_generation_gap == 1
    assert coverage.candidate_recall_of_detectable == pytest.approx(0.5)
    assert coverage.candidate_recall_all_gt == pytest.approx(0.5)

    with pytest.raises(OracleAnalysisError, match="Official TP count exceeds"):
        context.decompose_strict([(pred_ids["B"], pred_ids["C"])])


def test_candidate_coverage_reports_invalid_fixed_detection_ids():
    gt, _ = _gt()
    pred, pred_ids = _graph(
        {
            "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
            "C": {"t": 2, "z": 0.0, "y": 2.0, "x": 0.0},
        },
        [("A", "B")],
    )
    context = prepare_fixed_detection_oracle(pred, gt, estimated_total_nodes=3)
    coverage = context.measure_candidate_coverage(
        [
            (pred_ids["A"], pred_ids["B"]),
            (999_999, pred_ids["C"]),
        ]
    )
    assert coverage.candidate_edges_supplied == 2
    assert coverage.candidate_invalid_node_refs == 1
    assert coverage.gt_edges_candidate_available == 1
