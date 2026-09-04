"""Integration contract against the pinned public HOCT package.

The ordinary evaluator-contract job skips this file when the optional HOCT
runtime is not installed. The dedicated ``hoct-integration`` workflow installs
the exact audited public revision and executes these tests. No competition data
or model weights are required.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import tracksdata as td

torch = pytest.importorskip("torch")
hoct = pytest.importorskip("hoct")

from biohub.trackers import HOCT_REVISION, HOCTPointGraphConfig, build_hoct_point_graph
from hoct.data import DataKeys
from hoct.features import create_graph as public_create_graph


def _fixed_points() -> pl.DataFrame:
    rows = []
    detection_id = 1000
    for t in range(6):
        rows.append((detection_id, t, 4.0, 5.0 + 0.25 * t, 6.0))
        detection_id += 1
        rows.append((detection_id, t, 4.0, 12.0 + 0.20 * t, 12.0))
        detection_id += 1
    return pl.DataFrame(
        rows,
        schema=["detection_id", "t", "z", "y", "x"],
        orient="row",
    )


def test_public_predict_consumes_centroid_adapter_graph(monkeypatch):
    """Exercise the public ``predict(graph=...)`` path up to model inference."""

    graph = build_hoct_point_graph(
        _fixed_points(),
        HOCTPointGraphConfig(distance_threshold_voxels=3.0, n_neighbors=2),
        shape_tzyx=(6, 20, 24, 24),
    )
    assert graph.num_edges() >= 10
    assert graph.metadata["hoct_revision"] == HOCT_REVISION
    assert graph.metadata["candidate_distance_space"] == "hoct_native_voxel"

    # Patch only the expensive model/ILP boundary. The real public predict()
    # still constructs HOCT's FrameDataset, applies its published
    # Standardize(_MEAN, _STD), and hands that dataset to this function.
    import hoct._api as hoct_api

    captured = {}
    sentinel_model = object()

    def fake_model_predict(model, dataset, solver_config=None, return_solution=True):
        assert model is sentinel_model
        assert return_solution is False
        item = dataset[0]
        assert item is not None

        node_feats = item[DataKeys.NODE_FEATS]
        edges = item[DataKeys.EDGE_BATCH_ID]
        node_pos = item[DataKeys.NODE_POS]
        edge_pos = item[DataKeys.EDGE_POS]

        assert node_feats.ndim == 2
        assert node_feats.shape[1] == 19
        assert edges.ndim == 2 and edges.shape[1] == 2
        assert node_pos.ndim == 2 and node_pos.shape[1] == 3
        assert edge_pos.ndim == 2 and edge_pos.shape[1] == 3
        assert torch.isfinite(node_feats).all()
        assert torch.isfinite(node_pos).all()
        assert torch.isfinite(edge_pos).all()

        # equivalent_diameter + 4 intensity moments + 9 inertia values were
        # filled with HOCT's own training means. After the real Standardize
        # transform those unknown feature dimensions must be approximately zero.
        assert torch.allclose(
            node_feats[:, 4:18],
            torch.zeros_like(node_feats[:, 4:18]),
            atol=2e-4,
            rtol=0,
        )
        captured["dataset_len"] = len(dataset)
        captured["feature_width"] = node_feats.shape[1]
        return None

    monkeypatch.setattr(hoct_api, "model_predict", fake_model_predict)
    result = hoct.predict(
        sentinel_model,
        graph=graph,
        window_size=5,
        return_solution=False,
    )
    assert result is None
    assert captured == {"dataset_len": 2, "feature_width": 19}


def test_public_create_graph_scale_does_not_change_candidate_distance_space():
    """Characterize a subtle pinned-HOCT behavior that affects our search grid.

    The public function accepts a physical scale, but at the audited revision it
    creates ``DistanceEdges`` on raw ``z/y/x`` columns. A one-z-voxel movement
    therefore remains a distance of one even when z spacing is set to 10 um.
    This is a characterization test, not an endorsement of that behavior.
    """

    labels = np.zeros((2, 3, 7, 7), dtype=np.uint16)
    labels[0, 0, 3, 3] = 1
    labels[1, 1, 3, 3] = 1

    graph = public_create_graph(
        labels,
        distance_threshold=1.1,
        n_neighbors=1,
        delta_t=1,
        scale=(1.0, 10.0, 1.0, 1.0),
        images=None,
    )
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 1
    distance_key = td.DEFAULT_ATTR_KEYS.EDGE_DIST
    edge_dist = graph.edge_attrs(attr_keys=[distance_key])[distance_key].to_list()
    assert edge_dist == pytest.approx([1.0])
    assert tuple(graph.metadata["scale"]) == pytest.approx((1.0, 10.0, 1.0, 1.0))
