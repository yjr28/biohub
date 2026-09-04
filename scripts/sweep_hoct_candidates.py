#!/usr/bin/env python3
"""Calibrate HOCT candidate generation on training-side monitor datasets only.

This command intentionally refuses to sweep the true LOEO embryo. It consumes a
Phase-2C ``train-embryo-hash-holdout`` organizer protocol and automatically uses
only ``checkpoint_monitor_datasets`` as the tracker-calibration set. The output
is a candidate coverage/cost report, not a model score.

Required prediction GEFFs should be produced by the already-selected organizer
checkpoint on those monitor datasets. Frozen detection Parquets must come from
those exact prediction GEFFs so the oracle can fail closed on node-provenance
mismatches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.analysis import prepare_fixed_detection_oracle
from biohub.detections import load_detection_cache
from biohub.evaluation.official import load_geff, read_estimated_node_count
from biohub.experiments import file_sha256
from biohub.trackers import (
    aggregate_candidate_sweep_reports,
    evaluate_hoct_candidate_configs,
    expand_candidate_grid,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep HOCT candidate geometry on Phase-2C training-side monitor datasets; "
            "the true LOEO holdout is forbidden."
        )
    )
    parser.add_argument("--protocol", required=True, type=Path, help="organizer_baseline_protocol.json")
    parser.add_argument("--inventory", required=True, type=Path, help="Phase-2A data_inventory.json")
    parser.add_argument("--grid", required=True, type=Path, help="Explicit candidate-grid JSON")
    parser.add_argument("--pred-dir", required=True, type=Path, help="Baseline GEFFs on checkpoint-monitor datasets")
    parser.add_argument("--detections-dir", required=True, type=Path, help="Frozen detection Parquets for the same GEFFs")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Competition train directory containing GT GEFFs")
    parser.add_argument("--out", required=True, type=Path, help="Private JSON report path")
    return parser.parse_args()


def _load_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be a JSON object: {path}")
    return payload


def _calibration_scope(protocol: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    policy = protocol.get("checkpoint_monitor_policy")
    if policy != "train-embryo-hash-holdout":
        raise SystemExit(
            "candidate calibration requires checkpoint_monitor_policy='train-embryo-hash-holdout'; "
            "the train-embryo-all public-reference monitor is not an independent dataset set"
        )
    calibration = tuple(sorted(str(value) for value in protocol.get("checkpoint_monitor_datasets", [])))
    holdout = tuple(sorted(str(value) for value in protocol.get("holdout_datasets", [])))
    if not calibration:
        raise SystemExit("protocol has no checkpoint_monitor_datasets")
    if not holdout:
        raise SystemExit("protocol has no holdout_datasets")
    overlap = set(calibration) & set(holdout)
    if overlap:
        raise SystemExit(f"calibration/LOEO holdout overlap is forbidden: {sorted(overlap)}")
    train = set(str(value) for value in protocol.get("train_datasets", []))
    if not set(calibration) <= train:
        raise SystemExit("checkpoint-monitor calibration set is not a subset of declared training datasets")
    return calibration, holdout


def _inventory_rows(inventory: dict) -> dict[str, dict]:
    rows = inventory.get("train")
    if not isinstance(rows, list):
        raise SystemExit("inventory has no train record list")
    by_name: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("dataset"):
            raise SystemExit(f"invalid inventory train record: {row!r}")
        name = str(row["dataset"])
        if name in by_name:
            raise SystemExit(f"duplicate inventory train dataset: {name}")
        by_name[name] = row
    return by_name


def _metadata(row: dict, dataset: str) -> tuple[tuple[int, int, int, int], tuple[float, float, float]]:
    shape_raw = row.get("image_shape_tzyx")
    scale_raw = row.get("scale_zyx_um")
    if not isinstance(shape_raw, list) or len(shape_raw) != 4:
        raise SystemExit(f"invalid image_shape_tzyx for {dataset}: {shape_raw!r}")
    if not isinstance(scale_raw, list) or len(scale_raw) != 3:
        raise SystemExit(f"invalid scale_zyx_um for {dataset}: {scale_raw!r}")
    shape = tuple(int(value) for value in shape_raw)
    scale = tuple(float(value) for value in scale_raw)
    if any(value <= 0 for value in shape) or any(value <= 0 for value in scale):
        raise SystemExit(f"non-positive shape/scale metadata for {dataset}: shape={shape}, scale={scale}")
    return shape, scale


def _require_dataset_paths(
    dataset: str,
    *,
    pred_dir: Path,
    detections_dir: Path,
    gt_dir: Path,
) -> tuple[Path, Path, Path]:
    pred = pred_dir / f"{dataset}.geff"
    detections = detections_dir / f"{dataset}.parquet"
    gt = gt_dir / f"{dataset}.geff"
    for label, path in (("prediction", pred), ("fixed detections", detections), ("GT", gt)):
        if not path.exists():
            raise SystemExit(f"{label} missing for calibration dataset {dataset}: {path}")
    return pred, detections, gt


def main() -> None:
    args = _args()
    protocol_path = args.protocol.resolve()
    inventory_path = args.inventory.resolve()
    grid_path = args.grid.resolve()
    protocol = _load_object(protocol_path, "protocol")
    inventory = _load_object(inventory_path, "inventory")
    grid = _load_object(grid_path, "candidate grid")
    calibration_datasets, holdout_datasets = _calibration_scope(protocol)
    inventory_by_name = _inventory_rows(inventory)

    pred_dir = args.pred_dir.resolve()
    detections_dir = args.detections_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    for label, directory in (
        ("prediction directory", pred_dir),
        ("fixed-detection directory", detections_dir),
        ("GT directory", gt_dir),
    ):
        if not directory.is_dir():
            raise SystemExit(f"{label} not found: {directory}")

    per_dataset = {}
    for dataset in calibration_datasets:
        if dataset not in inventory_by_name:
            raise SystemExit(f"calibration dataset absent from inventory: {dataset}")
        shape, scale = _metadata(inventory_by_name[dataset], dataset)
        pred_path, detections_path, gt_path = _require_dataset_paths(
            dataset,
            pred_dir=pred_dir,
            detections_dir=detections_dir,
            gt_dir=gt_dir,
        )
        pred_graph = load_geff(pred_path)
        gt_graph = load_geff(gt_path)
        oracle = prepare_fixed_detection_oracle(
            pred_graph,
            gt_graph,
            estimated_total_nodes=read_estimated_node_count(gt_path),
            scale=scale,
        )
        configs = expand_candidate_grid(grid, scale_zyx_um=scale)
        detections = load_detection_cache(detections_path)
        report = evaluate_hoct_candidate_configs(
            detections,
            oracle,
            configs,
            shape_tzyx=shape,
        )
        per_dataset[dataset] = report
        print(
            f"dataset={dataset} detections={oracle.pred_nodes} "
            f"detectable_gt_edges={oracle.gt_edges_both_endpoints_available}/{oracle.gt_edge_count} "
            f"configs={len(report.trials)} frontier={len(report.pareto_config_ids)}"
        )

    aggregate = aggregate_candidate_sweep_reports(per_dataset)
    payload = {
        "purpose": "training-side HOCT candidate calibration; not LOEO validation",
        "selection_scope": {
            "checkpoint_monitor_policy": protocol.get("checkpoint_monitor_policy"),
            "calibration_datasets": list(calibration_datasets),
            "forbidden_loeo_holdout_datasets": list(holdout_datasets),
            "loeo_holdout_used": False,
            "selection_from_this_report_allowed": True,
        },
        "provenance": {
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": file_sha256(inventory_path),
            "grid_path": str(grid_path),
            "grid_sha256": file_sha256(grid_path),
        },
        "grid": grid,
        "aggregate": aggregate.to_dict(),
        "per_dataset": {
            dataset: report.to_dict() for dataset, report in sorted(per_dataset.items())
        },
        "interpretation": [
            "Pareto frontiers trade fixed-detection GT-edge proposal coverage against candidate-edge count.",
            "They are not official competition scores and do not measure HOCT transformer or ILP quality.",
            "Candidate hyperparameters selected here must be frozen before evaluating the opposite-embryo LOEO holdout.",
        ],
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\nAGGREGATE CANDIDATE FRONTIER")
    by_id = {trial.config_id: trial for trial in aggregate.trials}
    for config_id in aggregate.pareto_config_ids:
        trial = by_id[config_id]
        print(
            f"{config_id} space={trial.candidate_distance_space} "
            f"threshold={trial.candidate_distance_threshold:g} k={trial.n_neighbors} dt={trial.max_delta_t} "
            f"candidate_recall_detectable={trial.candidate_recall_of_detectable:.6f} "
            f"candidate_edges={trial.candidate_edges} edges_per_detection={trial.candidate_edges_per_detection:.3f}"
        )
    print(f"report={out}")
    print("FREEZE a candidate hypothesis from training-side evidence before LOEO holdout evaluation.")


if __name__ == "__main__":
    main()
