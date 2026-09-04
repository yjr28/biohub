"""Canonical fixed-detection cache for tracker-to-tracker comparisons.

Association experiments must not quietly change the cells being tracked.  This
module extracts only node identity and centroids from a prediction GEFF and
stores them in a deterministic Parquet table.  Predicted edges are deliberately
ignored.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import polars as pl
import tracksdata as td


_REQUIRED_COLUMNS = ("detection_id", "t", "z", "y", "x")


class DetectionCacheError(ValueError):
    """Raised when a fixed-detection table is incomplete or ambiguous."""


def validate_detection_cache(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a deterministic, strongly typed fixed-detection table."""

    if not isinstance(frame, pl.DataFrame):
        raise DetectionCacheError("detection cache must be a polars.DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DetectionCacheError(f"detection cache is missing required columns: {missing}")
    if frame.height == 0:
        raise DetectionCacheError("detection cache cannot be empty")

    typed = frame.select(
        pl.col("detection_id").cast(pl.Int64),
        pl.col("t").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
    )
    if typed["detection_id"].n_unique() != typed.height:
        raise DetectionCacheError("detection_id values must be unique")

    times = typed["t"].to_list()
    if any(not isfinite(value) or value < 0 or value != round(value) for value in times):
        raise DetectionCacheError("t must contain finite nonnegative integer-valued frame indices")
    for column in ("z", "y", "x"):
        if any(not isfinite(value) for value in typed[column].to_list()):
            raise DetectionCacheError(f"{column} must contain only finite values")

    return typed.with_columns(pl.col("t").cast(pl.Int64)).sort("t", "detection_id")


def detections_from_graph(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Extract the canonical node table from a tracksdata prediction graph."""

    required_attrs = [td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T, "z", "y", "x"]
    missing = [key for key in ("z", "y", "x") if key not in graph.node_attr_keys()]
    if missing:
        raise DetectionCacheError(f"prediction graph is missing centroid attributes: {missing}")
    attrs = graph.node_attrs(attr_keys=required_attrs).rename(
        {
            td.DEFAULT_ATTR_KEYS.NODE_ID: "detection_id",
            td.DEFAULT_ATTR_KEYS.T: "t",
        }
    )
    return validate_detection_cache(attrs)


def load_detections_from_geff(path: str | Path) -> pl.DataFrame:
    """Load a prediction GEFF and discard all association edges."""

    path = Path(path)
    if not path.exists():
        raise DetectionCacheError(f"prediction GEFF does not exist: {path}")
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    return detections_from_graph(graph)


def write_detection_cache(frame: pl.DataFrame, path: str | Path) -> Path:
    """Validate and write deterministic Parquet without mutating source graphs."""

    path = Path(path)
    clean = validate_detection_cache(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.write_parquet(path, compression="zstd", statistics=True)
    return path


def load_detection_cache(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DetectionCacheError(f"detection cache does not exist: {path}")
    return validate_detection_cache(pl.read_parquet(path))
