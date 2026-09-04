#!/usr/bin/env python3
"""Evaluate one declared LOEO fold through the pinned organizer scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.evaluation import build_report, evaluate_directory, write_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score exactly one inventory-declared leave-one-embryo-out fold."
    )
    parser.add_argument("--inventory", required=True, type=Path, help="Phase 1A data_inventory.json")
    parser.add_argument("--fold", required=True, help="Fold name, e.g. holdout_<embryo>")
    parser.add_argument("--pred-dir", required=True, type=Path, help="Directory containing predicted .geff files")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Training directory containing GT .geff files")
    parser.add_argument("--out", required=True, type=Path, help="Strict JSON evaluation report")
    parser.add_argument(
        "--group-map-json",
        type=Path,
        default=None,
        help="Optional JSON object mapping every holdout dataset to a predeclared diagnostic group",
    )
    return parser.parse_args()


def _load_inventory(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"inventory not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("inventory root must be a JSON object")
    return payload


def _resolve_fold(inventory: dict, fold_name: str) -> dict:
    folds = inventory.get("loeo_folds")
    if not isinstance(folds, list):
        raise SystemExit("inventory has no loeo_folds list")
    matches = [fold for fold in folds if fold.get("name") == fold_name]
    if len(matches) != 1:
        available = sorted(str(fold.get("name")) for fold in folds)
        raise SystemExit(f"fold {fold_name!r} not uniquely found; available={available}")
    return matches[0]


def _scale_map(inventory: dict, names: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    records = inventory.get("train")
    if not isinstance(records, list):
        raise SystemExit("inventory has no train record list")
    by_name = {record.get("dataset"): record for record in records}
    missing = set(names) - set(by_name)
    if missing:
        raise SystemExit(f"holdout datasets missing from inventory train records: {sorted(missing)}")

    result: dict[str, tuple[float, float, float]] = {}
    for name in names:
        raw = by_name[name].get("scale_zyx_um")
        if not isinstance(raw, list) or len(raw) != 3:
            raise SystemExit(f"invalid scale_zyx_um for {name}: {raw!r}")
        scale = tuple(float(value) for value in raw)
        if any(value <= 0 for value in scale):
            raise SystemExit(f"non-positive physical scale for {name}: {scale!r}")
        result[name] = scale
    return result


def main() -> None:
    args = _parse_args()
    inventory = _load_inventory(args.inventory)
    fold = _resolve_fold(inventory, args.fold)
    names = tuple(sorted(str(name) for name in fold.get("holdout_datasets", [])))
    if not names:
        raise SystemExit(f"fold {args.fold!r} has no holdout_datasets")

    scale_by_name = _scale_map(inventory, names)
    run = evaluate_directory(
        args.pred_dir,
        args.gt_dir,
        expected_names=names,
        scale_by_name=scale_by_name,
    )

    group_map = None
    if args.group_map_json is not None:
        group_map = json.loads(args.group_map_json.read_text())
        if not isinstance(group_map, dict):
            raise SystemExit("group map must be a JSON object")

    report = build_report(run, group_by_dataset=group_map)
    write_report(report, args.out)

    summary = report.overall
    print(f"fold={args.fold}")
    print(f"datasets={len(names)}")
    print(f"adj_edge_jaccard={summary.get('adj_edge_jaccard')}")
    print(f"division_jaccard={summary.get('division_jaccard')}")
    print(f"score={summary.get('score')}")
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
