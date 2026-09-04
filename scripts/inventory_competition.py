#!/usr/bin/env python
"""Inventory the mounted Biohub competition dataset without loading image voxels."""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub.data import inventory_competition, write_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        help="Directory containing train/ and test/.",
    )
    parser.add_argument("--json", type=Path, default=Path("artifacts/data_inventory.json"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/data_inventory.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inventory_competition(args.competition_root)
    write_inventory(report, json_path=args.json, csv_path=args.csv)

    print(f"train datasets: {len(report.train)}")
    print(f"train embryos:  {report.train_embryos}")
    print(f"visible test:   {len(report.visible_test)} datasets")
    print(f"visible overlap:{report.train_visible_test_name_overlap}")
    for fold in report.loeo_folds:
        print(
            f"{fold['name']}: train={len(fold['train_datasets'])} "
            f"holdout={len(fold['holdout_datasets'])}"
        )
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
