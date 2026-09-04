"""Centroid-only compatibility graph for evaluating HOCT on Biohub detections.

This module is intentionally *not* a copy of HOCT's public point API. At the
pinned HOCT revision below, ``hoct.create_graph_from_points`` is declared but
its body is still a TODO/``pass``. We therefore build the minimal tracksdata
graph ourselves, preserving the feature contract expected by HOCT's
``FrameDataset`` and pretrained standardization.

The non-coordinate morphology/intensity features below are filled with the
training means published in HOCT's own inference API. After HOCT's fixed
standardization those dimensions become zero, making this a clean
"centroids-only" ablation rather than inventing pseudo-segmentation features.
If this smoke test is useful, later experiments can replace these neutral values
with image-derived features without changing candidate-node identity.

Candidate generation deliberately supports two audited distance spaces:

``physical_um``
    Applies the Biohub OME-Zarr anisotropic scale before KD-tree search. This is
    the biologically meaningful distance space and the one used by our original
    adapter.

``hoct_native_voxel``
    Reproduces the public HOCT ``create_graph`` candidate-search geometry at the
    pinned revision. Although that function accepts ``scale``, it constructs
    ``DistanceEdges`` without passing scaled coordinate columns, and the pinned
    tracksdata operator defaults to raw ``z/y/x``. This mode lets us test the
    pretrained model under the candidate distribution closest to its public
    implementation instead of silently assuming the scale argument affects edge
    proposals.
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

# Audited from hoct._api._MEAN at HOCT_REVISION. Feature order in FrameDataset
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

    Exactly one distance threshold must be supplied:

    * ``distance_threshold_um`` searches in anisotropically scaled physical
      microns.
    * ``distance_threshold_voxels`` searches in raw ``z/y/x`` coordinates and
      reproduces the public HOCT candidate-distance convention at
      :data:`HOCT_REVISION`.

    ``max_delta_t`` defaults to one because the Biohub competition directly
    scores consecutive-frame edges. Gap candidates remain an explicit ablation.
    """

    distance_threshold_um: float | None = None
    distance_threshold_voxels: float | None = None
    n_neighbors: int = 5
    max_delta_t: int = 1
    scale_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    def __post_init__(self) -> None:
        supplied = [
            self.distance_threshold_um is not None,
            self.distance_threshold_voxels is not None,
        ]
        if sum(supplied) != 1:
            raise HOCTCompatibilityError(
                "supply exactly one of distance_threshold_um or distance_threshold_voxels"
            )

        if self.distance_threshold_um is not None:
            radius = float(self.distance_threshold_um)
            if not isfinite(radius) or radius <= 0:
                raise HOCTCompatibilityError("distance_threshold_um must be finite and > 0")
            object.__setattr__(self, "distance_threshold_um", radius)
        else:
            radius = float(self.distance_threshold_voxels)
            if not isfinite(radius) or radius <= 0:
                raise HOCTCompatibilityError("distance_threshold_voxels must be finite and > 0")
            object.__setattr__(self, "distance_threshold_voxels", radius)

        if int(self.n_neighbors) != self.n_neighbors or self.n_neighbors < 1:
            raise HOCTCompatibilityError("n_neighbors must be an integer >= 1")
        if int(self.max_delta_t) != self.max_delta_t or self.max_delta_t < 1:
            raise HOCTCompatibilityError("max_delta_t must be an integer >= 1")
        scale = tuple(float(value) for value in self.scale_zyx_um)
        if len(scale) != 3 or any(not isfinite(value) or value <= 0 for value in scale):
            raise HOCTCompatibilityError("scale_zyx_um must contain three finite positive values")
        object.__setattr__(self, "n_neighbors", int(self.n_neighbors))
        object.__setattr__(self, "max_delta_t", int(self.max_delta_t))
        object.__setattr__(self, "scale_zyx_um", scale)

    @property
    def candidate_distance_space(self) -> str:
        """Return the audited coordinate space used for candidate generation."""

        return "physical_um" if self.distance_threshold_um is not None else "hoct_native_voxel"

    @property
    def candidate_distance_threshold(self) -> float:
        """Return the active radius in the units of ``candidate_distance_space``."""

        if self.distance_threshold_um is not None:
            return self.distance_threshold_um
        assert self.distance_threshold_voxels is not None
        return self.distance_threshold_voxels


def _validated_points(points: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(points, pl.DataFrame):
        raise HOCTCompatibilityError("points must be a polars.DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in points.columns]
    if missing:
        raise HOCTCompatibilityError(f"points are missing required columns: {missing}")
    if points.height == 0:
        raise HOCTCompatibilityError("points cannot be empty")

    expressions = [
        pl.col("_input_order").cast(pl.Int64),
        pl.col("t").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
    ]
    if "detection_id" in points.columns:
        expressions.append(pl.col("detection_id").cast(pl.Int64))

    # Cast explicitly so graph construction cannot depend on upstream dataframe
    # integer/float widths. Preserve input order within a frame for stable node IDs.
    data = points.with_row_index("_input_order").select(*expressions)
    values = data.select("t", "z", "y", "x").to_numpy()
    if not np.isfinite(values).all():
        raise HOCTCompatibilityError("t/z/y/x must all be finite")
    times = values[:, 0]
    if np.any(times < 0) or not np.allclose(times, np.rint(times)):
        raise HOCTCompatibilityError("t must contain nonnegative integer-valued frame indices")
    if "detection_id" in data.columns and data["detection_id"].n_unique() != data.height:
        raise HOCTCompatibilityError("detection_id values must be unique when supplied")
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

    preserve_detection_id = "detection_id" in points.columns
    rows: list[dict] = []
    for i, row in enumerate(points.iter_rows(named=True)):
        node = {
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
        if preserve_detection_id:
            node["source_detection_id"] = int(row["detection_id"])
        rows.append(node)
    return rows


def _add_node_schema(graph: td.graph.InMemoryGraph, *, preserve_detection_id: bool) -> None:
    for key in (
        "z",
        "y",
        "x",
        "equivalent_diameter_area",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "intensity_std",
        "border_dist",
    ):
        graph.add_node_attr_key(key, pl.Float32, 0.0)
    graph.add_node_attr_key(
        "inertia_tensor",
        pl.Array(pl.Float32, (3, 3)),
        np.zeros((3, 3), dtype=np.float32),
    )
    if preserve_detection_id:
        graph.add_node_attr_key("source_detection_id", pl.Int64, -1)


def _candidate_coordinates(points: pl.DataFrame, config: HOCTPointGraphConfig) -> np.ndarray:
    coords = points.select("z", "y", "x").to_numpy().astype(np.float64)
    if config.candidate_distance_space == "physical_um":
        return coords * np.asarray(config.scale_zyx_um, dtype=np.float64)[None, :]
    return coords


def _candidate_edges(
    points: pl.DataFrame,
    node_ids: list[int],
    config: HOCTPointGraphConfig,
) -> list[dict]:
    times = points["t"].to_numpy()
    candidate_coords = _candidate_coordinates(points, config)
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
            tree = cKDTree(candidate_coords[source_idx])
            k = min(config.n_neighbors, int(source_idx.size))
            distances, neighbor_local = tree.query(
                candidate_coords[target_idx],
                k=k,
                distance_upper_bound=config.candidate_distance_threshold,
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

    Candidate search is explicit about its coordinate system: either anisotropic
    physical microns or the public HOCT raw-voxel convention. All missing
    segmentation/intensity features are set to HOCT's pretrained training means
    so HOCT standardization maps them to zero. If the canonical ``detection_id``
    column is supplied, it is preserved on each graph node as
    ``source_detection_id`` so tracker outputs can be reconciled against the
    exact same fixed detections.

    This function does not run HOCT, solve an ILP, or claim an official HOCT
    point-cloud inference path. It supplies a deterministic graph for the
    fixed-detection association experiment while the upstream point API remains
    unimplemented at :data:`HOCT_REVISION`.
    """

    if not isinstance(config, HOCTPointGraphConfig):
        raise HOCTCompatibilityError("config must be an HOCTPointGraphConfig")
    clean = _validated_points(points)
    preserve_detection_id = "detection_id" in clean.columns

    graph = td.graph.InMemoryGraph()
    _add_node_schema(graph, preserve_detection_id=preserve_detection_id)
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
        candidate_distance_space=config.candidate_distance_space,
        candidate_distance_threshold=config.candidate_distance_threshold,
        distance_threshold_um=config.distance_threshold_um,
        distance_threshold_voxels=config.distance_threshold_voxels,
        n_neighbors=config.n_neighbors,
        max_delta_t=config.max_delta_t,
        scale=(1.0, *config.scale_zyx_um),
        shape=shape_tzyx,
        neutral_missing_regionprops=True,
        preserves_source_detection_id=preserve_detection_id,
    )
    return graph
