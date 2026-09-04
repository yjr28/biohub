#!/usr/bin/env python3
"""Inventory the accepted Kaggle mount and open the Phase-2A experiment gate.

This command reads only metadata/GT graph information through the Phase-1A
inventory path; it does not export microscopy voxels. On success it writes the
inventory + acceptance summary and prints dry-run commands for both clean LOEO
baseline directions.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from biohub.data import inventory_competition, validate_real_inventory, write_inventory


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-data gate before any baseline compute.")
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/private/phase2a"),
        help="Generated metadata only; default path is gitignored",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="organizer-clean-3ep",
        help="Prefix used only for the printed next-step experiment IDs",
    )
    return parser.parse_args()


def _baseline_command(
    *,
    inventory: Path,
    competition_root: Path,
    fold: str,
    experiment_id: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_clean_organizer_baseline.py"),
        "--inventory",
        str(inventory),
        "--competition-root",
        str(competition_root),
        "--fold",
        fold,
        "--experiment-id",
        experiment_id,
    )


def main() -> None:
    args = _args()
    competition_root = args.competition_root.resolve()
    if not competition_root.is_dir():
        raise SystemExit(
            f"competition root not found: {competition_root}\n"
            "Attach/accept the Kaggle competition data before running this gate."
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_json = out_dir / "data_inventory.json"
    inventory_csv = out_dir / "data_inventory.csv"
    gate_json = out_dir / "real_data_gate.json"

    report = inventory_competition(competition_root)
    write_inventory(report, json_path=inventory_json, csv_path=inventory_csv)
    gate = validate_real_inventory(report)
    gate_json.write_text(json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n")

    print("PHASE 2A REAL-DATA GATE: ACCEPTED")
    print(f"competition_root={competition_root}")
    print(f"train_embryos={','.join(gate.train_embryos)}")
    print(f"train_datasets={gate.train_dataset_count}")
    print(f"visible_test_placeholders={gate.visible_test_dataset_count}")
    for embryo in gate.train_embryos:
        print(
            f"embryo={embryo} "
            f"datasets={gate.dataset_count_by_embryo[embryo]} "
            f"gt_nodes={gate.gt_nodes_by_embryo[embryo]} "
            f"gt_edges={gate.gt_edges_by_embryo[embryo]} "
            f"gt_divisions={gate.gt_divisions_by_embryo[embryo]} "
            f"image_voxels={gate.image_voxels_by_embryo[embryo]}"
        )
    for warning in gate.warnings:
        print(f"WARNING: {warning}")
    print(f"inventory_json={inventory_json}")
    print(f"inventory_csv={inventory_csv}")
    print(f"gate_json={gate_json}")

    print("\nNEXT: dry-run both clean organizer baseline directions")
    for index, fold in enumerate(gate.loeo_fold_names, start=1):
        experiment_id = f"{args.experiment_prefix}-{index:02d}"
        command = _baseline_command(
            inventory=inventory_json,
            competition_root=competition_root,
            fold=fold,
            experiment_id=experiment_id,
        )
        print(shlex.join(command))
    print(
        "\nOnly add --execute after both dry-run plans resolve the intended embryo memberships. "
        "The executable runner re-checks git/submodule provenance before spending GPU compute."
    )


if __name__ == "__main__":
    main()
