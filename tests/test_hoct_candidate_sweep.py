import polars as pl
import pytest
import tracksdata as td

from biohub.analysis import prepare_fixed_detection_oracle
from biohub.trackers import (
    HOCTCandidateSweepError,
    HOCTPointGraphConfig,
    aggregate_candidate_sweep_reports,
    candidate_config_id,
    evaluate_hoct_candidate_configs,
    expand_candidate_grid,
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


def _fixture():
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 4.0, "x": 0.0},
    }
    gt, _ = _graph(nodes, [("A", "B"), ("B", "C")])
    pred, pred_ids = _graph(nodes, [("A", "B")])
    detections = pl.DataFrame(
        [
            (pred_ids["A"], 0, 0.0, 0.0, 0.0),
            (pred_ids["B"], 1, 0.0, 1.0, 0.0),
            (pred_ids["C"], 2, 0.0, 4.0, 0.0),
        ],
        schema=["detection_id", "t", "z", "y", "x"],
        orient="row",
    )
    oracle = prepare_fixed_detection_oracle(pred, gt, estimated_total_nodes=3)
    return detections, oracle


def test_sweep_separates_candidate_recall_from_graph_cost():
    detections, oracle = _fixture()
    narrow = HOCTPointGraphConfig(distance_threshold_voxels=1.1, n_neighbors=1)
    wide = HOCTPointGraphConfig(distance_threshold_voxels=3.1, n_neighbors=1)

    report = evaluate_hoct_candidate_configs(detections, oracle, [narrow, wide])
    by_id = {trial.config_id: trial for trial in report.trials}

    narrow_trial = by_id[candidate_config_id(narrow)]
    wide_trial = by_id[candidate_config_id(wide)]
    assert narrow_trial.candidate_available_gt_edges == 1
    assert narrow_trial.candidate_recall_of_detectable == pytest.approx(0.5)
    assert narrow_trial.candidate_edges == 1

    assert wide_trial.candidate_available_gt_edges == 2
    assert wide_trial.candidate_recall_of_detectable == pytest.approx(1.0)
    assert wide_trial.candidate_edges == 2

    # Neither dominates: narrow is cheaper; wide has higher recoverable coverage.
    assert set(report.pareto_config_ids) == {
        candidate_config_id(narrow),
        candidate_config_id(wide),
    }


def test_candidate_sweep_fails_closed_when_detection_identity_mismatches_baseline():
    detections, oracle = _fixture()
    detections = detections.with_columns((pl.col("detection_id") + 10_000).alias("detection_id"))
    config = HOCTPointGraphConfig(distance_threshold_voxels=4.0, n_neighbors=1)
    with pytest.raises(HOCTCandidateSweepError, match="detection provenance does not match"):
        evaluate_hoct_candidate_configs(detections, oracle, [config])


def test_expand_grid_is_explicit_and_deterministic_across_both_distance_spaces():
    payload = {
        "n_neighbors": [3, 5],
        "max_delta_t": [1],
        "physical_um": [2.0],
        "hoct_native_voxel": [4.0, 8.0],
    }
    configs = expand_candidate_grid(payload, scale_zyx_um=(1.625, 0.40625, 0.40625))
    assert len(configs) == 6
    assert len({candidate_config_id(config) for config in configs}) == 6
    assert {config.candidate_distance_space for config in configs} == {
        "physical_um",
        "hoct_native_voxel",
    }
    again = expand_candidate_grid(payload, scale_zyx_um=(1.625, 0.40625, 0.40625))
    assert [candidate_config_id(config) for config in configs] == [
        candidate_config_id(config) for config in again
    ]


def test_config_id_is_stable_across_dataset_spatial_scale_metadata():
    first = HOCTPointGraphConfig(
        distance_threshold_um=3.0,
        n_neighbors=5,
        scale_zyx_um=(1.625, 0.40625, 0.40625),
    )
    second = HOCTPointGraphConfig(
        distance_threshold_um=3.0,
        n_neighbors=5,
        scale_zyx_um=(2.0, 0.5, 0.5),
    )
    assert candidate_config_id(first) == candidate_config_id(second)


def test_expand_grid_refuses_to_invent_radius_or_neighbor_values():
    with pytest.raises(HOCTCandidateSweepError, match="n_neighbors"):
        expand_candidate_grid({"physical_um": [2.0]}, scale_zyx_um=(1.0, 1.0, 1.0))
    with pytest.raises(HOCTCandidateSweepError, match="physical_um and/or"):
        expand_candidate_grid({"n_neighbors": [5]}, scale_zyx_um=(1.0, 1.0, 1.0))


def test_dominated_configuration_is_removed_from_frontier():
    detections, oracle = _fixture()
    # k=2 adds a distractor from A directly to C only when max_delta_t=2; both
    # configs still cover the same two consecutive GT edges. That makes the
    # denser config strictly dominated by the cheaper max_delta_t=1 proposal.
    efficient = HOCTPointGraphConfig(
        distance_threshold_voxels=5.0,
        n_neighbors=2,
        max_delta_t=1,
    )
    dense = HOCTPointGraphConfig(
        distance_threshold_voxels=5.0,
        n_neighbors=2,
        max_delta_t=2,
    )
    report = evaluate_hoct_candidate_configs(detections, oracle, [efficient, dense])
    trials = {trial.config_id: trial for trial in report.trials}
    assert trials[candidate_config_id(efficient)].candidate_available_gt_edges == 2
    assert trials[candidate_config_id(dense)].candidate_available_gt_edges == 2
    assert trials[candidate_config_id(dense)].candidate_edges > trials[candidate_config_id(efficient)].candidate_edges
    assert report.pareto_config_ids == (candidate_config_id(efficient),)


def test_multi_dataset_aggregation_sums_counts_before_recomputing_recall():
    detections, oracle = _fixture()
    configs = [
        HOCTPointGraphConfig(distance_threshold_voxels=1.1, n_neighbors=1),
        HOCTPointGraphConfig(distance_threshold_voxels=3.1, n_neighbors=1),
    ]
    one = evaluate_hoct_candidate_configs(detections, oracle, configs)
    aggregate = aggregate_candidate_sweep_reports({"dataset_a": one, "dataset_b": one})
    one_by_id = {trial.config_id: trial for trial in one.trials}
    agg_by_id = {trial.config_id: trial for trial in aggregate.trials}

    for config in configs:
        config_id = candidate_config_id(config)
        assert agg_by_id[config_id].detections == 2 * one_by_id[config_id].detections
        assert agg_by_id[config_id].candidate_edges == 2 * one_by_id[config_id].candidate_edges
        assert agg_by_id[config_id].gt_edges == 2 * one_by_id[config_id].gt_edges
        assert agg_by_id[config_id].detectable_gt_edges == 2 * one_by_id[config_id].detectable_gt_edges
        assert (
            agg_by_id[config_id].candidate_available_gt_edges
            == 2 * one_by_id[config_id].candidate_available_gt_edges
        )
        assert agg_by_id[config_id].candidate_recall_of_detectable == pytest.approx(
            one_by_id[config_id].candidate_recall_of_detectable
        )


def test_aggregation_refuses_different_config_sets_across_datasets():
    detections, oracle = _fixture()
    a = evaluate_hoct_candidate_configs(
        detections,
        oracle,
        [HOCTPointGraphConfig(distance_threshold_voxels=1.1, n_neighbors=1)],
    )
    b = evaluate_hoct_candidate_configs(
        detections,
        oracle,
        [HOCTPointGraphConfig(distance_threshold_voxels=3.1, n_neighbors=1)],
    )
    with pytest.raises(HOCTCandidateSweepError, match="different candidate-config set"):
        aggregate_candidate_sweep_reports({"a": a, "b": b})
