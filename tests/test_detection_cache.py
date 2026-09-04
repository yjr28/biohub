from pathlib import Path

import polars as pl
import pytest
import tracksdata as td

from biohub.detections import (
    DetectionCacheError,
    detections_from_graph,
    load_detection_cache,
    validate_detection_cache,
    write_detection_cache,
)


def _graph():
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    ids = graph.bulk_add_nodes(
        [
            {"t": 1, "z": 2.0, "y": 3.0, "x": 4.0},
            {"t": 0, "z": 1.0, "y": 2.0, "x": 3.0},
        ]
    )
    graph.bulk_add_edges([{"source_id": ids[1], "target_id": ids[0]}])
    return graph


def test_extracts_nodes_only_and_sorts_deterministically():
    frame = detections_from_graph(_graph())
    assert frame.columns == ["detection_id", "t", "z", "y", "x"]
    assert frame["t"].to_list() == [0, 1]
    assert frame["z"].to_list() == pytest.approx([1.0, 2.0])
    assert frame.height == 2


def test_parquet_round_trip_preserves_canonical_table(tmp_path: Path):
    source = detections_from_graph(_graph())
    path = write_detection_cache(source, tmp_path / "fixed.parquet")
    loaded = load_detection_cache(path)
    assert loaded.equals(source)


def test_rejects_duplicate_detection_ids():
    frame = pl.DataFrame(
        {
            "detection_id": [1, 1],
            "t": [0, 1],
            "z": [0.0, 1.0],
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        }
    )
    with pytest.raises(DetectionCacheError, match="unique"):
        validate_detection_cache(frame)


def test_rejects_fractional_time_and_nonfinite_coordinate():
    fractional = pl.DataFrame(
        {"detection_id": [1], "t": [0.5], "z": [0.0], "y": [0.0], "x": [0.0]}
    )
    with pytest.raises(DetectionCacheError, match="integer-valued"):
        validate_detection_cache(fractional)

    nonfinite = pl.DataFrame(
        {"detection_id": [1], "t": [0], "z": [float("nan")], "y": [0.0], "x": [0.0]}
    )
    with pytest.raises(DetectionCacheError, match="z"):
        validate_detection_cache(nonfinite)
