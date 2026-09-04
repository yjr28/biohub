#!/usr/bin/env python3
"""Register one immutable clean-LOEO experiment manifest before execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from biohub.experiments import ExperimentManifest, append_manifest, file_sha256


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a fail-closed experiment manifest to JSONL.")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--config", required=True, type=Path, help="JSON object containing run config")
    parser.add_argument("--seed", required=True, type=int, action="append", dest="seeds")
    parser.add_argument(
        "--leakage-control",
        required=True,
        action="append",
        dest="leakage_controls",
        help="Repeat for every concrete holdout-protection rule used by this run",
    )
    parser.add_argument("--registry", default=Path("experiments/manifests.jsonl"), type=Path)
    parser.add_argument("--git-commit", default=None, help="Full SHA; defaults to git rev-parse HEAD")
    parser.add_argument("--parent-experiment-id", default=None)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def _git_sha(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        raise SystemExit(f"Cannot determine git commit; pass --git-commit explicitly: {exc}") from exc


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return payload


def _fold(inventory: dict, name: str) -> dict:
    folds = inventory.get("loeo_folds")
    if not isinstance(folds, list):
        raise SystemExit("inventory has no loeo_folds list")
    matches = [fold for fold in folds if fold.get("name") == name]
    if len(matches) != 1:
        available = sorted(str(item.get("name")) for item in folds)
        raise SystemExit(f"fold {name!r} not uniquely found; available={available}")
    return matches[0]


def main() -> None:
    args = _args()
    inventory = _load_json_object(args.inventory, "inventory")
    config = _load_json_object(args.config, "config")
    fold = _fold(inventory, args.fold)

    holdout_embryo = str(fold.get("holdout_embryo", "")).strip()
    train_embryos = tuple(str(value) for value in fold.get("train_embryos", []))
    train_datasets = tuple(str(value) for value in fold.get("train_datasets", []))
    holdout_datasets = tuple(str(value) for value in fold.get("holdout_datasets", []))
    if not holdout_embryo:
        raise SystemExit(f"fold {args.fold!r} has no holdout_embryo")

    manifest = ExperimentManifest(
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        git_commit=_git_sha(args.git_commit),
        inventory_sha256=file_sha256(args.inventory),
        fold_name=args.fold,
        train_embryos=train_embryos,
        validation_embryos=(holdout_embryo,),
        train_datasets=train_datasets,
        validation_datasets=holdout_datasets,
        config=config,
        seeds=tuple(args.seeds),
        leakage_controls=tuple(args.leakage_controls),
        parent_experiment_id=args.parent_experiment_id,
        notes=args.notes,
    )
    append_manifest(args.registry, manifest)
    print(f"registered={manifest.experiment_id}")
    print(f"fold={manifest.fold_name}")
    print(f"git_commit={manifest.git_commit}")
    print(f"inventory_sha256={manifest.inventory_sha256}")
    print(f"registry={args.registry}")


if __name__ == "__main__":
    main()
