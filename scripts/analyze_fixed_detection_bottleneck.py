#!/usr/bin/env python3
"""Decompose one scored prediction into detection/candidate/selection opportunity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from biohub.analysis import decompose_fixed_detections
from biohub.evaluation.official import load_geff, read_estimated_node_count
from biohub.trackers import candidate_edges_in_source_detection_space


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official matching, then measure fixed-detection GT-edge coverage."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="Dataset stem present in inventory train records")
    parser.add_argument("--pred-geff", required=True, type=Path)
    parser.add_argument("--gt-geff", required=True, type=Path)
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="CSV with baseline predicted-graph source_id,target_id columns before solver/selection",
    )
    candidates.add_argument(
        "--candidate-geff",
        type=Path,
        default=None,
        help=(
            "Candidate GEFF whose nodes carry source_detection_id. This is the direct path for "
            "Biohub's centroid-only HOCT adapter."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _scale_from_inventory(path: Path, dataset: str) -> tuple[float, float, float]:
    if not path.is_file():
        raise SystemExit(f"inventory not found: {path}")
    inventory = json.loads(path.read_text())
    records = inventory.get("train") if isinstance(inventory, dict) else None
    if not isinstance(records, list):
        raise SystemExit("inventory has no train record list")
    matches = [row for row in records if row.get("dataset") == dataset]
    if len(matches) != 1:
        raise SystemExit(f"dataset {dataset!r} not uniquely present in inventory train records")
    raw = matches[0].get("scale_zyx_um")
    if not isinstance(raw, list) or len(raw) != 3:
        raise SystemExit(f"invalid scale_zyx_um for {dataset}: {raw!r}")
    scale = tuple(float(value) for value in raw)
    if any(value <= 0 for value in scale):
        raise SystemExit(f"non-positive scale for {dataset}: {scale!r}")
    return scale


def _candidate_edges_csv(path: Path | None) -> list[tuple[int, int]] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"candidate CSV not found: {path}")
    edges: list[tuple[int, int]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"source_id", "target_id"}
        if not required <= fields:
            raise SystemExit(
                f"candidate CSV requires columns {sorted(required)}; found {sorted(fields)}"
            )
        for line, row in enumerate(reader, start=2):
            try:
                edges.append((int(row["source_id"]), int(row["target_id"])))
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"invalid candidate node ID at {path}:{line}: {row}") from exc
    return edges


def _candidate_edges_geff(path: Path | None) -> tuple[tuple[int, int], ...] | None:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"candidate GEFF not found: {path}")
    return candidate_edges_in_source_detection_space(load_geff(path))


def main() -> None:
    args = _args()
    for label, path in (("prediction", args.pred_geff), ("ground truth", args.gt_geff)):
        if not path.exists():
            raise SystemExit(f"{label} GEFF not found: {path}")

    scale = _scale_from_inventory(args.inventory, args.dataset)
    pred_graph = load_geff(args.pred_geff)
    gt_graph = load_geff(args.gt_geff)
    candidate_edges = (
        _candidate_edges_geff(args.candidate_geff)
        if args.candidate_geff is not None
        else _candidate_edges_csv(args.candidate_csv)
    )
    result = decompose_fixed_detections(
        pred_graph,
        gt_graph,
        estimated_total_nodes=read_estimated_node_count(args.gt_geff),
        candidate_edges=candidate_edges,
        scale=scale,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"dataset={args.dataset}")
    print(f"edge_tp={result.official_edge_tp}/{result.gt_edges}")
    print(f"fixed_detection_edge_ceiling_recall={result.fixed_detection_edge_ceiling_recall:.6f}")
    print(f"detection_unavailable_edges={result.detection_unavailable_edges}")
    print(f"fixed_detection_recoverable_edges={result.fixed_detection_recoverable_edges}")
    if result.gt_edges_candidate_available is not None:
        print(f"candidate_available_edges={result.gt_edges_candidate_available}")
        print(f"candidate_generation_gap={result.candidate_generation_gap}")
        print(f"candidate_to_selected_gap={result.candidate_to_selected_gap}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
