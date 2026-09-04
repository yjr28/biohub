#!/usr/bin/env python3
"""Create holdout-safe split files for the pinned organizer baseline scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.baselines import build_organizer_protocol, write_organizer_protocol


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit separate train/predict split files for one clean LOEO direction."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--organizer-fold-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.inventory.is_file():
        raise SystemExit(f"inventory not found: {args.inventory}")
    inventory = json.loads(args.inventory.read_text())
    if not isinstance(inventory, dict):
        raise SystemExit("inventory root must be a JSON object")

    protocol = build_organizer_protocol(
        inventory,
        fold_name=args.fold,
        organizer_fold_index=args.organizer_fold_index,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "organizer_train_splits.json"
    predict_path = args.out_dir / "organizer_predict_splits.json"
    audit_path = args.out_dir / "organizer_baseline_protocol.json"
    write_organizer_protocol(
        protocol,
        train_splits_path=train_path,
        predict_splits_path=predict_path,
        protocol_path=audit_path,
    )

    print(f"fold={protocol.fold_name}")
    print(f"train_embryos={','.join(protocol.train_embryos)}")
    print(f"holdout_embryo={protocol.holdout_embryo}")
    print(f"train_splits={train_path}")
    print(f"predict_splits={predict_path}")
    print(f"protocol={audit_path}")
    for warning in protocol.warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
