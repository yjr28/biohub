#!/usr/bin/env python3
"""Build one centroid-only HOCT candidate GEFF from a frozen detection cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.detections import load_detection_cache
from biohub.trackers import HOCTPointGraphConfig, build_hoct_point_graph
from tracking_cellmot.io import save_graph


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a centroid-only HOCT candidate graph from fixed detections. "
            "Choose either physical-micron geometry or the public HOCT raw-voxel candidate geometry."
        )
    )
    parser.add_argument("--detections", required=True, type=Path, help="Canonical fixed-detection Parquet")
    parser.add_argument("--inventory", required=True, type=Path, help="Phase-2A inventory JSON")
    parser.add_argument("--dataset", required=True, help="Dataset stem represented by the cache")
    parser.add_argument("--out", required=True, type=Path, help="Output candidate .geff")
    distance = parser.add_mutually_exclusive_group(required=True)
    distance.add_argument(
        "--distance-threshold-um",
        type=float,
        help="Candidate radius after applying OME-Zarr anisotropic spatial scale.",
    )
    distance.add_argument(
        "--distance-threshold-voxels",
        type=float,
        help=(
            "Candidate radius in raw z/y/x coordinates, reproducing public HOCT's "
            "pinned create_graph/DistanceEdges geometry."
        ),
    )
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument(
        "--max-delta-t",
        type=int,
        default=1,
        help="Default 1 because Biohub directly scores only consecutive-frame edges.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata(inventory_path: Path, dataset: str) -> tuple[tuple[int, int, int, int], tuple[float, float, float]]:
    payload = json.loads(inventory_path.read_text())
    matches = [
        row
        for split in ("train", "visible_test")
        for row in payload.get(split, [])
        if str(row.get("dataset")) == dataset
    ]
    if len(matches) != 1:
        raise SystemExit(f"inventory must contain dataset {dataset!r} exactly once; found {len(matches)}")
    row = matches[0]
    shape_raw = row.get("image_shape_tzyx")
    scale_raw = row.get("scale_zyx_um")
    if not isinstance(shape_raw, list) or len(shape_raw) != 4:
        raise SystemExit(f"invalid image_shape_tzyx for {dataset}: {shape_raw!r}")
    if not isinstance(scale_raw, list) or len(scale_raw) != 3:
        raise SystemExit(f"invalid scale_zyx_um for {dataset}: {scale_raw!r}")
    return tuple(int(v) for v in shape_raw), tuple(float(v) for v in scale_raw)


def main() -> None:
    args = _args()
    detections = args.detections.resolve()
    inventory = args.inventory.resolve()
    output = args.out.resolve()
    if not inventory.is_file():
        raise SystemExit(f"inventory not found: {inventory}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite candidate graph: {output}")

    shape, scale = _metadata(inventory, args.dataset)
    frame = load_detection_cache(detections)
    config = HOCTPointGraphConfig(
        distance_threshold_um=args.distance_threshold_um,
        distance_threshold_voxels=args.distance_threshold_voxels,
        n_neighbors=args.n_neighbors,
        max_delta_t=args.max_delta_t,
        scale_zyx_um=scale,
    )
    graph = build_hoct_point_graph(frame, config, shape_tzyx=shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, output, overwrite=args.overwrite)

    print(f"dataset={args.dataset}")
    print(f"detections={graph.num_nodes()}")
    print(f"candidate_edges={graph.num_edges()}")
    print(f"candidate_distance_space={config.candidate_distance_space}")
    print(f"candidate_distance_threshold={config.candidate_distance_threshold}")
    print(f"n_neighbors={config.n_neighbors}")
    print(f"max_delta_t={config.max_delta_t}")
    print(f"scale_zyx_um={config.scale_zyx_um}")
    print(f"candidate_geff={output}")


if __name__ == "__main__":
    main()
