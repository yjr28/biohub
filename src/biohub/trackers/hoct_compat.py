"""Centroid-only compatibility graph for evaluating HOCT on Biohub detections.

This module is intentionally *not* a copy of HOCT's public point API.  At the
pinned HOCT revision below, ``hoct.create_graph_from_points`` is declared but
its body is still a TODO/``pass``.  We therefore build the minimal tracksdata
graph ourselves, preserving the feature contract expected by HOCT's
``FrameDataset`` and pretrained standardization.

The non-coordinate morphology/intensity features below are filled with the
training means published in HOCT's own inference API.  After HOCT's fixed
standardization those dimensions become zero, making this a clean
"centroids-only" ablation rather than inventing pseudo-segmentation features.
If this smoke test is useful, later experiments can replace these neutral values
with image-derived features without changing candidate-node identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree


HOCT_REVISION = "2ccc5040823bc944ab67790abd1f56eea7cd4f05"
HOCT_POINT_API_IMPLEMENTED = False

# Audited from hoct._api._MEAN at HOCT_REVISION.  Feature order in FrameDataset
# is [t, z, y, x, *REGIONPROPS], with inertia_tensor unpacked to 9 scalars.
_NEUTRAL_EQUIVALENT_DIAMETER = 11.521
_NEUTRAL_INTENSITY_MIN = 0.276
_NEUTRAL_INTENSITY_MAX = 0.966
_NEUTRAL_INTENSITY_MEAN = 0.574
_NEUTRAL_INTENSITY_STD = 0.162
_NEUTRAL_INERTIA = np.asarray(
    [
        [167.81, -0.027, 0.05],
        [-0.027, 87.012, -1.401],
        [0.05, -1.401, 83.695],
    ],
    dtype=np.float32,
)
_NEUTRAL_BORDER_DIST = 0.009

_REQUIRED_COLUMNS = ("t", "z", "y", "x")


class HOCTCompatibilityError(ValueError):
    """Raised when fixed detections cannot form an unambiguous HOCT graph."""


@dataclass(frozen=True)
class HOCTPointGraphConfig:
    """Candidate-generation settings for the centroid-only HOCT ablation.

    Distances are always computed in physical microns.  ``max_delta_t`` defaults
    to one because the Biohub competition metric directly scores only edges
    between consecutive frames.
    """

    distance_threshold_um: float
    n_neighbors: int = 5
    max_delta_t: int = 1
    scale_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    def __post_init__(self) -> None:
        radius = float(self.distance_threshold_um)
        if not isfinite(radius) or radius <= 0:
            raise HOCTCompatibilityError("distance_threshold_um must be finite and > 0")
        if int(self.n_neighbors) != self.n_neighbors or self.n_neighbors < 1:
            raise HOCTCompatibilityError("n_neighbors must be an integer >= 1")
        if int(self.max_delta_t) != self.max_delta_t or self.max_delta_t < 1:
            raise HOCTCompatibilityError("max_delta_t must be an integer >= 1")
        scale = tuple(float(value) for value in self.scale_zyx_um)
        if len(scale) != 3 or any(not isfinite(value) or value <= 0 for value in scale):
            raise HOCTCompatibilityError("scale_zyx_um must contain three finite positive values")
        object.__setattr__(self, "distance_threshold_um", radius)
        object.__setattr__(self, "n_neighbors", int(self.n_neighbors))
        object.__setattr__(self, "max_delta_t", int(self.max_delta_t))
        object.__setattr__(self, "scale_zyx_um", scale)


def _validated_points(points: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(points, pl.DataFrame):
        raise HOCTCompatibilityError("points must be a polars.DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in points.columns]
    if missing:
        raise HOCTCompatibilityError(f"points are missing required columns: {missing}")
    if points.height == 0:
        raise HOCTCompatibilityError("points cannot be empty")

    # Cast explicitly so graph construction cannot depend on upstream dataframe
    # integer/float widths.  Preserve input order within a frame for stable node IDs.
    data = points.with_row_index("_input_order").select(
        pl.col("_input_order").cast(pl.Int64),
        pl.col("t").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
    )
    values = data.select("t", "z", "y", "x").to_numpy()
    if not np.isfinite(values).all():
        raise HOCTCompatibilityError("t/z/y/x must all be finite")
    times = values[:, 0]
    if np.any(times < 0) or not np.allclose(times, np.rint(times)):
        raise HOCTCompatibilityError("t must contain nonnegative integer-valued frame indices")
    return data.with_columns(pl.col("t").cast(pl.Int64)).sort("t", "_input_order")


def _hoct_border_dist(coords_zyx: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Match HOCT's 3-D border-distance feature (cutoff=5 voxels)."""

    shape = np.asarray(shape_zyx, dtype=np.float64)[None, :]
    distance = np.minimum(coords_zyx, shape - coords_zyx).min(axis=1)
    return 1.0 - np.minimum(1.0, distance / 5.0)


def _node_rows(points: pl.DataFrame, shape_tzyx: tuple[int, int, int, int] | None) -> list[dict]:
    coords = points.select("z", "y", "x").to_numpy()
    if shape_tzyx is None:
        border = np.full(points.height, _NEUTRAL_BORDER_DIST, dtype=np.float32)
    else:
        if len(shape_tzyx) != 4 or any(int(dim) <= 0 for dim in shape_tzyx):
            raise HOCTCompatibilityError("shape_tzyx must contain four positive dimensions")
        shape = tuple(int(dim) for dim in shape_tzyx)
        if int(points["t"].max()) >= shape[0]:
            raise HOCTCompatibilityError(
                f"point frame {int(points['t'].max())} lies outside shape_tzyx T={shape[0]}"
            )
        border = _hoct_border_dist(coords, shape[1:]).astype(np.float32)

    rows: list[dict] = []
    for i, row in enumerate(points.iter_rows(named=True)):
        rows.append(
            {
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
                "equivalent_diameter_area": _NEUTRAL_EQUIVALENT_DIAMETER,
                "intensity_min": _NEUTRAL_INTENSITY_MIN,
                "intensity_max": _NEUTRAL_INTENSITY_MAX,
                "intensity_mean": _NEUTRAL_INTENSITY_MEAN,
                "intensity_std": _NEUTRAL_INTENSITY_STD,
                "inertia_tensor": _NEUTRAL_INERTIA.copy(),
                "border_dist": float(border[i]),
            }
        )
    return rows


def _add_node_schema(graph: td.graph.InMemoryGraph) -> None:
    for key in ("z", "y", "x", "equivalent_diameter_area", "intensity_min", "intensity_max", "intensity_mean", "intensity_std", "border_dist"):
        graph.add_node_attr_key(key, pl.Float32, 0.0)
    graph.add_node_attr_key(
        "inertia_tensor",
        pl.Array(pl.Float32, (3, 3)),
        np.zeros((3, 3), dtype=np.float32),
    )


def _candidate_edges(
    points: pl.DataFrame,
    node_ids: list[int],
    config: HOCTPointGraphConfig,
) -> list[dict]:
    times = points["t"].to_numpy()
    coords = points.select("z", "y", "x").to_numpy().astype(np.float64)
    physical = coords * np.asarray(config.scale_zyx_um, dtype=np.float64)[None, :]
    node_ids_arr = np.asarray(node_ids, dtype=np.int64)

    by_time: dict[int, np.ndarray] = {}
    for t in np.unique(times):
        by_time[int(t)] = np.flatnonzero(times == t)

    edges: list[dict] = []
    for target_t in sorted(by_time):
        target_idx = by_time[target_t]
        if target_idx.size == 0:
            continue
        for delta_t in range(1, config.max_delta_t + 1):
            source_idx = by_time.get(target_t - delta_t)
            if source_idx is None or source_idx.size == 0:
                continue
            tree = cKDTree(physical[source_idx])
            k = min(config.n_neighbors, int(source_idx.size))
            distances, neighbor_local = tree.query(
                physical[target_idx],
                k=k,
                distance_upper_bound=config.distance_threshold_um,
            )
            distances = np.asarray(distances)
            neighbor_local = np.asarray(neighbor_local)
            if k == 1:
                distances = distances[:, None]
                neighbor_local = neighbor_local[:, None]

            for target_pos, target_global in enumerate(target_idx):
                pairs = []
                for dist, local in zip(distances[target_pos], neighbor_local[target_pos], strict=True):
                    if not np.isfinite(dist) or int(local) >= source_idx.size:
                        continue
                    source_global = int(source_idx[int(local)])
                    pairs.append((float(dist), source_global))
                # KDTree is normally sorted already, but make tie behavior explicit.
                pairs.sort(key=lambda item: (item[0], int(node_ids_arr[item[1]])))
                for dist, source_global in pairs:
                    edges.append(
                        {
                            "source_id": int(node_ids_arr[source_global]),
                            "target_id": int(node_ids_arr[int(target_global)]),
                            "edge_dist": dist,
                            "delta_t": float(delta_t),
                        }
                    )
    return edges


def build_hoct_point_graph(
    points: pl.DataFrame,
    config: HOCTPointGraphConfig,
    *,
    shape_tzyx: tuple[int, int, int, int] | None = None,
) -> td.graph.InMemoryGraph:
    """Build a HOCT-compatible candidate graph from fixed Biohub centroids.

    Candidate search uses anisotropic **physical** distance, not raw voxel
    distance.  All missing segmentation/intensity features are set to HOCT's
    pretrained training means so HOCT standardization maps them to zero.

    This function does not run HOCT, solve an ILP, or claim an official HOCT
    point-cloud inference path.  It supplies a deterministic graph for the
    fixed-detection association experiment while the upstream point API remains
    unimplemented at :data:`HOCT_REVISION`.
    """

    if not isinstance(config, HOCTPointGraphConfig):
        raise HOCTCompatibilityError("config must be an HOCTPointGraphConfig")
    clean = _validated_points(points)

    graph = td.graph.InMemoryGraph()
    _add_node_schema(graph)
    node_ids = graph.bulk_add_nodes(_node_rows(clean, shape_tzyx))

    graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    graph.add_edge_attr_key("delta_t", pl.Float32, 1.0)
    edges = _candidate_edges(clean, list(node_ids), config)
    if edges:
        graph.bulk_add_edges(edges)

    graph.metadata.update(
        biohub_adapter="centroid_only_hoct_compat",
        hoct_revision=HOCT_REVISION,
        hoct_upstream_point_api_implemented=False,
        distance_threshold_um=config.distance_threshold_um,
        n_neighbors=config.n_neighbors,
        max_delta_t=config.max_delta_t,
        scale=(1.0, *config.scale_zyx_um),
        shape=shape_tzyx,
        neutral_missing_regionprops=True,
    )
    return graph
