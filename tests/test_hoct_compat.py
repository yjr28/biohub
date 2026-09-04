import numpy as np
import polars as pl
import pytest
import tracksdata as td

from biohub.trackers.hoct_compat import (
    HOCTCompatibilityError,
    HOCT_POINT_API_IMPLEMENTED,
    HOCT_REVISION,
    HOCTPointGraphConfig,
    build_hoct_point_graph,
)


def _points(rows):
    return pl.DataFrame(rows, schema=["t", "z", "y", "x"], orient="row")


def _edge_times(graph):
    node_df = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T])
    times = dict(zip(node_df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), node_df[td.DEFAULT_ATTR_KEYS.T].to_list()))
    edge_df = graph.edge_attrs(attr_keys=["edge_dist", "delta_t"])
    return [
        (
            times[row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]],
            times[row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]],
            row["edge_dist"],
            row["delta_t"],
        )
        for row in edge_df.iter_rows(named=True)
    ]


def test_pins_upstream_state_and_does_not_claim_point_api_exists():
    assert HOCT_REVISION == "2ccc5040823bc944ab67790abd1f56eea7cd4f05"
    assert HOCT_POINT_API_IMPLEMENTED is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"distance_threshold_um": 0}, "distance_threshold_um"),
        ({"distance_threshold_um": float("nan")}, "distance_threshold_um"),
        ({"distance_threshold_um": 5, "n_neighbors": 0}, "n_neighbors"),
        ({"distance_threshold_um": 5, "max_delta_t": 0}, "max_delta_t"),
        ({"distance_threshold_um": 5, "scale_zyx_um": (1.0, -1.0, 1.0)}, "scale_zyx_um"),
    ],
)
def test_config_rejects_invalid_candidate_settings(kwargs, message):
    with pytest.raises(HOCTCompatibilityError, match=message):
        HOCTPointGraphConfig(**kwargs)


def test_graph_contains_hoct_feature_contract_with_neutral_missing_features():
    graph = build_hoct_point_graph(
        _points([(0, 2.0, 8.0, 9.0), (1, 2.0, 8.5, 9.5)]),
        HOCTPointGraphConfig(distance_threshold_um=2.0),
        shape_tzyx=(2, 20, 20, 20),
    )
    attrs = graph.node_attrs(
        attr_keys=[
            "z",
            "y",
            "x",
            "equivalent_diameter_area",
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_std",
            "inertia_tensor",
            "border_dist",
        ]
    )
    assert graph.num_nodes() == 2
    assert attrs["equivalent_diameter_area"].to_list() == pytest.approx([11.521, 11.521])
    assert attrs["intensity_mean"].to_list() == pytest.approx([0.574, 0.574])
    expected_inertia = np.asarray(
        [[167.81, -0.027, 0.05], [-0.027, 87.012, -1.401], [0.05, -1.401, 83.695]],
        dtype=np.float32,
    )
    assert np.allclose(np.asarray(attrs["inertia_tensor"][0]), expected_inertia)
    assert graph.metadata["neutral_missing_regionprops"] is True
    assert graph.metadata["hoct_upstream_point_api_implemented"] is False


def test_candidate_radius_uses_anisotropic_physical_microns_not_voxel_distance():
    # t=1 node A is only one z voxel from t=0 source: 1.625 um, so a 1 um
    # radius must reject it. Node B is two x voxels away: 0.8125 um, so it must
    # be accepted even though its raw voxel displacement is larger.
    points = _points(
        [
            (0, 0.0, 0.0, 0.0),
            (1, 1.0, 0.0, 0.0),
            (1, 0.0, 0.0, 2.0),
        ]
    )
    graph = build_hoct_point_graph(
        points,
        HOCTPointGraphConfig(distance_threshold_um=1.0, n_neighbors=2),
    )
    edge_times = _edge_times(graph)
    assert graph.num_edges() == 1
    assert edge_times[0][0:2] == (0, 1)
    assert edge_times[0][2] == pytest.approx(0.8125)

    wider = build_hoct_point_graph(
        points,
        HOCTPointGraphConfig(distance_threshold_um=2.0, n_neighbors=2),
    )
    distances = sorted(row[2] for row in _edge_times(wider))
    assert distances == pytest.approx([0.8125, 1.625])


def test_max_delta_t_defaults_to_scored_consecutive_edges_only():
    points = _points([(0, 1.0, 1.0, 1.0), (2, 1.0, 1.0, 1.0)])
    consecutive = build_hoct_point_graph(
        points,
        HOCTPointGraphConfig(distance_threshold_um=1.0),
    )
    assert consecutive.num_edges() == 0

    gaps = build_hoct_point_graph(
        points,
        HOCTPointGraphConfig(distance_threshold_um=1.0, max_delta_t=2),
    )
    assert gaps.num_edges() == 1
    assert _edge_times(gaps)[0][0:2] == (0, 2)
    assert _edge_times(gaps)[0][3] == pytest.approx(2.0)


def test_all_candidate_edges_are_forward_in_time_and_have_consistent_delta_t():
    graph = build_hoct_point_graph(
        _points(
            [
                (0, 1.0, 1.0, 1.0),
                (1, 1.0, 1.0, 1.0),
                (2, 1.0, 1.0, 1.0),
            ]
        ),
        HOCTPointGraphConfig(distance_threshold_um=1.0, max_delta_t=2),
    )
    for source_t, target_t, _, delta_t in _edge_times(graph):
        assert target_t > source_t
        assert delta_t == pytest.approx(target_t - source_t)


def test_border_distance_matches_hoct_cutoff_semantics():
    graph = build_hoct_point_graph(
        _points([(0, 0.0, 0.0, 0.0), (0, 10.0, 10.0, 10.0)]),
        HOCTPointGraphConfig(distance_threshold_um=1.0),
        shape_tzyx=(1, 20, 20, 20),
    )
    border = graph.node_attrs(attr_keys=["border_dist"])["border_dist"].to_list()
    assert border == pytest.approx([1.0, 0.0])


def test_rejects_fractional_time_and_missing_coordinate_columns():
    with pytest.raises(HOCTCompatibilityError, match="integer-valued"):
        build_hoct_point_graph(
            pl.DataFrame({"t": [0.5], "z": [0.0], "y": [0.0], "x": [0.0]}),
            HOCTPointGraphConfig(distance_threshold_um=1.0),
        )
    with pytest.raises(HOCTCompatibilityError, match="missing required"):
        build_hoct_point_graph(
            pl.DataFrame({"t": [0], "y": [0.0], "x": [0.0]}),
            HOCTPointGraphConfig(distance_threshold_um=1.0),
        )
