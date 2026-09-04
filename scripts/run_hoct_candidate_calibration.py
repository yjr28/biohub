#!/usr/bin/env python3
"""Run leakage-safe training-side HOCT candidate calibration in one command.

The runner consumes an already-selected organizer baseline checkpoint.  It:

1. derives a prediction split containing only the deterministic nested
   training-embryo checkpoint-monitor datasets;
2. runs the pinned organizer predictor in an isolated output namespace;
3. freezes those exact detector nodes into tracker-neutral Parquet caches;
4. executes the oracle-first HOCT candidate sweep; and
5. freezes the aggregate Pareto frontier as the only candidate-config set
   allowed to enter the next learned-HOCT/solver calibration stage.

The opposite-embryo LOEO holdout is never predicted or scored here.
Default mode is a dry run; pass ``--execute`` to spend compute.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from biohub.calibration import (
    CalibrationPlanError,
    build_monitor_prediction_plan,
    frontier_shortlist,
    validate_monitor_prediction_directory,
    write_monitor_splits,
)
from biohub.detections import load_detections_from_geff, write_detection_cache
from biohub.evaluation.official import OFFICIAL_EVALUATOR_COMMIT, TRACKSDATA_COMMIT
from biohub.experiments import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA_VENDOR = REPO_ROOT / "vendor" / "tracksdata"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict nested monitor datasets, freeze detections, and calibrate HOCT candidates."
    )
    parser.add_argument("--baseline-work-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--competition-root", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
            "tracked working tree is not clean; commit/stash code changes before calibration:\n" + dirty
        )
    return head


def _require_submodule_pin(path: Path, expected: str, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} submodule missing: {path}")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot inspect {label} submodule: {exc.output}") from exc
    if actual != expected:
        raise SystemExit(f"{label} revision mismatch: {actual} != {expected}")


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def _run(command: list[str] | tuple[str, ...], *, label: str, env: dict[str, str] | None = None) -> float:
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)
    elapsed = time.monotonic() - started
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def _prepare_empty_dir(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists():
        if any(path.iterdir()) and not overwrite:
            raise SystemExit(f"{label} is not empty; pass --overwrite to replace it: {path}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _freeze_detections(
    *,
    prediction_paths: tuple[Path, ...],
    out_dir: Path,
    overwrite: bool,
) -> Path:
    _prepare_empty_dir(out_dir, overwrite=overwrite, label="fixed-detection directory")
    index = []
    for source in prediction_paths:
        frame = load_detections_from_geff(source)
        output = out_dir / f"{source.stem}.parquet"
        write_detection_cache(frame, output)
        index.append(
            {
                "dataset": source.stem,
                "prediction_geff": str(source.resolve()),
                "prediction_geff_sha256": file_sha256(source),
                "cache_file": output.name,
                "cache_sha256": file_sha256(output),
                "num_detections": frame.height,
                "t_min": int(frame["t"].min()),
                "t_max": int(frame["t"].max()),
            }
        )
        print(
            f"freeze dataset={source.stem} detections={frame.height} "
            f"frames={index[-1]['t_min']}..{index[-1]['t_max']}"
        )
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index_path


def main() -> None:
    args = _args()
    inventory = _require_file(args.inventory, "inventory")
    grid = _require_file(args.grid, "candidate grid")
    competition_root = args.competition_root.resolve()
    train_dir = competition_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"competition train directory not found: {train_dir}")

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else (REPO_ROOT / "artifacts" / "private" / "hoct_calibration" / args.calibration_id).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        plan = build_monitor_prediction_plan(
            repo_root=REPO_ROOT,
            baseline_work_dir=args.baseline_work_dir,
            competition_root=competition_root,
            calibration_id=args.calibration_id,
            work_dir=work_dir,
        )
    except CalibrationPlanError as exc:
        raise SystemExit(str(exc)) from exc

    monitor_splits = write_monitor_splits(plan)
    plan_path = work_dir / "monitor_prediction_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n")

    print(f"calibration_id={plan.calibration_id}")
    print(f"baseline_work_dir={plan.baseline_work_dir}")
    print(f"monitor_datasets={','.join(plan.monitor_datasets)}")
    print(f"forbidden_loeo_holdout={','.join(plan.forbidden_loeo_holdout_datasets)}")
    print(f"weights={plan.weights_path}")
    print(f"isolated_predictions={plan.predictions_dir}")
    print(f"predict_command={shlex.join(plan.predict_command)}")
    print(f"work_dir={work_dir}")

    if not args.execute:
        print("DRY_RUN: monitor-only plan prepared; rerun with --execute in the checkpoint environment.")
        return

    git_commit = _require_clean_tracked_tree()
    _require_submodule_pin(VENDOR, OFFICIAL_EVALUATOR_COMMIT, "organizer evaluator")
    _require_submodule_pin(TRACKSDATA_VENDOR, TRACKSDATA_COMMIT, "tracksdata")
    weights = _require_file(Path(plan.weights_path), "selected organizer checkpoint")

    prediction_dir = Path(plan.predictions_dir)
    _prepare_empty_dir(prediction_dir, overwrite=args.overwrite, label="isolated prediction directory")
    # The pinned predictor derives its output namespace from USER/USERNAME.
    # Rebinding both variables gives this calibration a collision-free path while
    # leaving model/data/threshold flags identical to the selected baseline.
    env = os.environ.copy()
    env["USER"] = plan.isolation_user
    env["USERNAME"] = plan.isolation_user
    predict_seconds = _run(plan.predict_command, label="monitor-predict", env=env)

    try:
        prediction_paths = validate_monitor_prediction_directory(plan)
    except CalibrationPlanError as exc:
        raise SystemExit(str(exc)) from exc

    detections_dir = work_dir / "fixed_detections"
    detection_index = _freeze_detections(
        prediction_paths=prediction_paths,
        out_dir=detections_dir,
        overwrite=args.overwrite,
    )

    sweep_report = work_dir / "candidate_sweep_report.json"
    if sweep_report.exists() and not args.overwrite:
        raise SystemExit(f"candidate sweep report exists; pass --overwrite: {sweep_report}")
    sweep_command = [
        sys.executable,
        str((REPO_ROOT / "scripts" / "sweep_hoct_candidates.py").resolve()),
        "--protocol",
        plan.protocol_path,
        "--inventory",
        str(inventory),
        "--grid",
        str(grid),
        "--pred-dir",
        str(prediction_dir),
        "--detections-dir",
        str(detections_dir),
        "--gt-dir",
        str(train_dir),
        "--out",
        str(sweep_report),
    ]
    sweep_seconds = _run(sweep_command, label="candidate-sweep")

    report = json.loads(sweep_report.read_text())
    try:
        shortlist = frontier_shortlist(report)
    except CalibrationPlanError as exc:
        raise SystemExit(str(exc)) from exc
    shortlist_payload = {
        "calibration_id": plan.calibration_id,
        "purpose": "training-side candidate shortlist; not LOEO model selection",
        "selection_scope": {
            "monitor_datasets": list(plan.monitor_datasets),
            "forbidden_loeo_holdout_datasets": list(plan.forbidden_loeo_holdout_datasets),
            "loeo_used": False,
        },
        "shortlist": shortlist,
        "provenance": {
            "git_commit": git_commit,
            "organizer_revision": OFFICIAL_EVALUATOR_COMMIT,
            "tracksdata_revision": TRACKSDATA_COMMIT,
            "checkpoint_path": str(weights),
            "checkpoint_sha256": file_sha256(weights),
            "protocol_path": plan.protocol_path,
            "protocol_sha256": file_sha256(Path(plan.protocol_path)),
            "effective_config_path": plan.effective_config_path,
            "effective_config_sha256": file_sha256(Path(plan.effective_config_path)),
            "monitor_splits_path": str(monitor_splits),
            "monitor_splits_sha256": file_sha256(monitor_splits),
            "inventory_path": str(inventory),
            "inventory_sha256": file_sha256(inventory),
            "grid_path": str(grid),
            "grid_sha256": file_sha256(grid),
            "detection_index_path": str(detection_index),
            "detection_index_sha256": file_sha256(detection_index),
            "candidate_sweep_report": str(sweep_report),
            "candidate_sweep_report_sha256": file_sha256(sweep_report),
        },
        "runtime_seconds": {
            "monitor_prediction": predict_seconds,
            "candidate_sweep": sweep_seconds,
        },
        "next_gate": (
            "Only allowed_config_ids may enter training-side learned-HOCT/solver calibration. "
            "Freeze the winning learned configuration before any opposite-embryo LOEO evaluation."
        ),
    }
    shortlist_path = work_dir / "candidate_shortlist.json"
    shortlist_path.write_text(json.dumps(shortlist_payload, indent=2, sort_keys=True) + "\n")

    print("\nCANDIDATE SHORTLIST FROZEN")
    print(f"allowed_config_ids={','.join(shortlist['allowed_config_ids'])}")
    print(f"max_coverage_min_cost={shortlist['max_coverage_min_cost_config_id']}")
    print(f"candidate_sweep_report={sweep_report}")
    print(f"candidate_shortlist={shortlist_path}")
    print("LOEO HOLDOUT REMAINS UNTOUCHED.")


if __name__ == "__main__":
    main()
