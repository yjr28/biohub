import polars as pl
import pytest
import tracksdata as td

from biohub.trackers import (
    HOCTPointGraphConfig,
    build_hoct_point_graph,
    candidate_edges_in_source_detection_space,
)
from biohub.trackers.hoct_compat import HOCTCompatibilityError


def test_candidate_edges_map_back_to_original_fixed_detection_ids():
    points = pl.DataFrame(
        {
            "detection_id": [1001, 2002],
            "t": [0, 1],
            "z": [1.0, 1.0],
            "y": [2.0, 2.0],
            "x": [3.0, 3.0],
        }
    )
    graph = build_hoct_point_graph(
        points,
        HOCTPointGraphConfig(distance_threshold_um=1.0),
    )
    assert candidate_edges_in_source_detection_space(graph) == ((1001, 2002),)


def test_mapping_rejects_non_adapter_graph_without_source_identity():
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    graph.bulk_add_nodes([{"t": 0, "z": 0.0, "y": 0.0, "x": 0.0}])
    with pytest.raises(HOCTCompatibilityError, match="source_detection_id"):
        candidate_edges_in_source_detection_space(graph)
