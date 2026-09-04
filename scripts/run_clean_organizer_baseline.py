#!/usr/bin/env python3
"""Plan or execute one holdout-safe organizer baseline LOEO direction.

Default mode is a dry run: it writes the exact split/config/command plan but
spends no GPU compute. ``--execute`` registers the immutable experiment before
training, runs the pinned organizer train/predict scripts, scores the exact
holdout set through our pinned official evaluator, and records a private local
result under ignored paths.

For the public 3-epoch reference, the default checkpoint monitor remains
``train-embryo-all``. Longer/converged runs should explicitly use
``train-embryo-hash-holdout`` so checkpoint selection happens on a deterministic
nested dataset subset from the training embryo while the true LOEO embryo stays
completely unseen.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from biohub.baselines import (
    OrganizerRunSettings,
    build_organizer_commands,
    build_organizer_protocol,
    write_organizer_protocol,
)
from biohub.evaluation import build_report, evaluate_directory, write_report
from biohub.evaluation.official import OFFICIAL_EVALUATOR_COMMIT, TRACKSDATA_COMMIT
from biohub.experiments import (
    ExperimentManifest,
    ExperimentResult,
    append_manifest,
    file_sha256,
    write_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA_VENDOR = REPO_ROOT / "vendor" / "tracksdata"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan/execute a clean LOEO adaptation of the pinned organizer baseline."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--competition-root", required=True, type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--method", default=None, help="Organizer output method token; defaults to experiment ID")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Spend compute; default only prepares/prints plan")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument(
        "--checkpoint-monitor-policy",
        choices=("train-embryo-all", "train-embryo-hash-holdout"),
        default="train-embryo-all",
        help=(
            "Checkpoint-selection split inside the training embryo. Preserve train-embryo-all "
            "for the public 3-epoch reference; prefer train-embryo-hash-holdout for longer runs."
        ),
    )
    parser.add_argument(
        "--monitor-fraction",
        type=float,
        default=0.1,
        help="Nested training-embryo dataset fraction reserved for checkpoint monitoring.",
    )
    parser.add_argument(
        "--monitor-seed",
        type=int,
        default=0,
        help="Deterministic SHA-256 ranking seed for the nested checkpoint monitor.",
    )
    return parser.parse_args()


def _load_inventory(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"inventory not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("inventory root must be a JSON object")
    return payload


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _require_clean_tracked_tree() -> str:
    try:
        head = _git_output("rev-parse", "HEAD")
        dirty = _git_output(
            "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=untracked"
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot verify git provenance: {exc.output}") from exc
    if dirty:
        raise SystemExit(
            "tracked working tree is not clean; commit/stash code changes before an executable experiment:\n"
            + dirty
        )
    return head


def _require_submodule_pin(path: Path, expected: str, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} submodule missing: {path}")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot inspect {label} submodule: {exc.output}") from exc
    if actual != expected:
        raise SystemExit(f"{label} revision mismatch: {actual} != {expected}")


def _scale_map(inventory: dict, names: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    records = inventory.get("train")
    if not isinstance(records, list):
        raise SystemExit("inventory has no train record list")
    by_name = {str(row.get("dataset")): row for row in records}
    missing = set(names) - set(by_name)
    if missing:
        raise SystemExit(f"holdout datasets missing from inventory: {sorted(missing)}")
    result: dict[str, tuple[float, float, float]] = {}
    for name in names:
        raw = by_name[name].get("scale_zyx_um")
        if not isinstance(raw, list) or len(raw) != 3:
            raise SystemExit(f"invalid scale_zyx_um for {name}: {raw!r}")
        result[name] = tuple(float(value) for value in raw)
    return result


def _run(command: tuple[str, ...], *, label: str) -> float:
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    elapsed = time.monotonic() - started
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def main() -> None:
    args = _args()
    inventory_path = args.inventory.resolve()
    inventory = _load_inventory(inventory_path)
    competition_root = args.competition_root.resolve()
    train_dir = competition_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"competition train directory not found: {train_dir}")

    protocol = build_organizer_protocol(
        inventory,
        fold_name=args.fold,
        organizer_fold_index=0,
        checkpoint_monitor_policy=args.checkpoint_monitor_policy,
        monitor_fraction=args.monitor_fraction,
        monitor_seed=args.monitor_seed,
    )
    method = (args.method or args.experiment_id).strip()
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else (REPO_ROOT / "artifacts" / "private" / "baselines" / args.experiment_id).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    train_splits = work_dir / "organizer_train_splits.json"
    predict_splits = work_dir / "organizer_predict_splits.json"
    protocol_path = work_dir / "organizer_baseline_protocol.json"
    write_organizer_protocol(
        protocol,
        train_splits_path=train_splits,
        predict_splits_path=predict_splits,
        protocol_path=protocol_path,
    )

    settings = OrganizerRunSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_parallel=not args.single_gpu,
    )
    username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    commands = build_organizer_commands(
        repo_root=REPO_ROOT,
        data_dir=train_dir,
        train_splits_path=train_splits,
        predict_splits_path=predict_splits,
        method=method,
        username=username,
        fold_index=0,
        settings=settings,
    )

    effective_config = {
        "baseline": "pinned organizer TemporalUNet3D + SimpleNodeTransformer",
        "organizer_revision": OFFICIAL_EVALUATOR_COMMIT,
        "tracksdata_revision": TRACKSDATA_COMMIT,
        "fold": args.fold,
        "holdout_embryo": protocol.holdout_embryo,
        "checkpoint_monitor_policy": protocol.checkpoint_monitor_policy,
        "optimizer_datasets": list(protocol.optimizer_datasets),
        "checkpoint_monitor_datasets": list(protocol.checkpoint_monitor_datasets),
        "monitor_fraction": protocol.monitor_fraction,
        "monitor_seed": protocol.monitor_seed,
        "settings": settings.to_dict(),
        "prediction_implicit_defaults_at_pinned_revision": {
            "det_tta": True,
            "pool_kernel_um": 3.0,
            "edge_activation": "softmax",
            "edge_threshold": 0.5,
            "max_parents_per_node": 1,
            "max_children_per_node": 2,
        },
        "stochastic_control": "uncontrolled",
        "known_reproducibility_caveat": (
            "pinned trainer CLI does not pass train(seed=...), and FrameWindowDataset augmentation "
            "uses numpy.default_rng() without an explicit seed"
        ),
        "train_command": list(commands.train),
        "predict_command": list(commands.predict),
    }
    config_path = work_dir / "effective_config.json"
    config_path.write_text(json.dumps(effective_config, indent=2, sort_keys=True) + "\n")
    commands_path = work_dir / "commands.txt"
    commands_path.write_text(
        "TRAIN\n" + shlex.join(commands.train) + "\n\nPREDICT\n" + shlex.join(commands.predict) + "\n"
    )

    print(f"experiment_id={args.experiment_id}")
    print(f"fold={args.fold}")
    print(f"train_embryos={','.join(protocol.train_embryos)}")
    print(f"holdout_embryo={protocol.holdout_embryo}")
    print(f"checkpoint_monitor_policy={protocol.checkpoint_monitor_policy}")
    print(f"optimizer_datasets={','.join(protocol.optimizer_datasets)}")
    print(f"checkpoint_monitor_datasets={','.join(protocol.checkpoint_monitor_datasets)}")
    if protocol.monitor_fraction is not None:
        print(f"monitor_fraction={protocol.monitor_fraction}")
        print(f"monitor_seed={protocol.monitor_seed}")
    print(f"work_dir={work_dir}")
    print(f"train_command={shlex.join(commands.train)}")
    print(f"predict_command={shlex.join(commands.predict)}")
    for warning in protocol.warnings:
        print(f"WARNING: {warning}")

    if not args.execute:
        print("DRY_RUN: plan prepared; rerun with --execute in a GPU environment to spend compute.")
        return

    git_commit = _require_clean_tracked_tree()
    _require_submodule_pin(VENDOR, OFFICIAL_EVALUATOR_COMMIT, "organizer evaluator")
    _require_submodule_pin(TRACKSDATA_VENDOR, TRACKSDATA_COMMIT, "tracksdata")

    leakage_controls = (
        "held-out embryo absent from organizer optimizer/train list",
        "held-out embryo absent from organizer checkpoint-monitor/test list",
        "checkpoint monitor contains training-embryo datasets only",
        "held-out embryo not used for early stopping or threshold tuning before this run",
        "prediction split introduced only after checkpoint selection is complete",
    )
    if protocol.checkpoint_monitor_policy == "train-embryo-hash-holdout":
        leakage_controls += (
            "nested checkpoint-monitor datasets are disjoint from optimizer datasets",
            "nested checkpoint-monitor selection is deterministic from recorded fraction and seed",
        )

    hypothesis = (
        "The holdout-safe organizer baseline establishes a clean cross-embryo reference error budget "
        "without held-out-embryo checkpoint selection."
    )
    if args.epochs > 3 or protocol.checkpoint_monitor_policy == "train-embryo-hash-holdout":
        hypothesis = (
            "A longer-trained organizer baseline selected only on a nested training-embryo monitor "
            "provides a stronger cross-embryo reference without exposing the LOEO embryo to checkpoint selection."
        )

    manifest = ExperimentManifest(
        experiment_id=args.experiment_id,
        hypothesis=hypothesis,
        git_commit=git_commit,
        inventory_sha256=file_sha256(inventory_path),
        fold_name=args.fold,
        train_embryos=protocol.train_embryos,
        validation_embryos=(protocol.holdout_embryo,),
        train_datasets=protocol.train_datasets,
        validation_datasets=protocol.holdout_datasets,
        config=effective_config,
        seeds=(),
        stochastic_control="uncontrolled",
        leakage_controls=leakage_controls,
        notes=(
            "Pinned organizer training still has uncontrolled augmentation stochasticity; repeat before "
            "interpreting small score differences. The nested monitor seed controls dataset partitioning "
            "only, not PyTorch/augmentation randomness."
        ),
    )
    registry = REPO_ROOT / "experiments" / "manifests.jsonl"
    append_manifest(registry, manifest)
    print(f"registered_manifest={registry}")

    total_started = time.monotonic()
    train_seconds = _run(commands.train, label="train")
    weights = Path(commands.weights_path)
    if not weights.is_file():
        raise SystemExit(f"organizer training completed but expected weights are missing: {weights}")

    predict_seconds = _run(commands.predict, label="predict")
    predictions = Path(commands.predictions_dir)
    holdout_names = tuple(protocol.holdout_datasets)
    run = evaluate_directory(
        predictions,
        train_dir,
        expected_names=holdout_names,
        scale_by_name=_scale_map(inventory, holdout_names),
    )
    report = build_report(run)
    report_path = work_dir / "evaluation_report.json"
    write_report(report, report_path)

    total_seconds = time.monotonic() - total_started
    result = ExperimentResult(
        experiment_id=args.experiment_id,
        status="success",
        summary=report.overall,
        report_path=str(report_path),
        runtime_seconds=total_seconds,
        notes=f"train_seconds={train_seconds:.3f}; predict_seconds={predict_seconds:.3f}",
    )
    result_path = REPO_ROOT / "experiments" / "results" / f"{args.experiment_id}.json"
    write_result(result_path, result)

    print("\nEXACT HOLDOUT RESULT")
    print(f"adj_edge_jaccard={report.overall.get('adj_edge_jaccard')}")
    print(f"division_jaccard={report.overall.get('division_jaccard')}")
    print(f"score={report.overall.get('score')}")
    print(f"evaluation_report={report_path}")
    print(f"private_result={result_path}")


if __name__ == "__main__":
    main()
