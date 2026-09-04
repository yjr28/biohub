#!/usr/bin/env python3
"""Evaluate exactly one Phase-2F-frozen winner on the opposite embryo.

This is the clean cross-embryo evidence boundary.  The command consumes a
``learned_selection.json`` that was already frozen using training-side monitor
data.  It refuses to choose or tune anything from the LOEO set.

If the frozen winner is ``organizer_control``, the command scores the already
produced organizer holdout predictions.  If the winner is ``hoct``, it extracts
only detector nodes from those same organizer holdout predictions, reconstructs
the exact Phase-2E candidate geometry, runs the exact frozen Phase-2F HOCT
model/solver spec, and only then opens the GT through the pinned evaluator.

Default mode prepares a provenance-hashed dry-run plan. ``--execute`` consumes
the LOEO evidence and writes an immutable result. Existing result directories
are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from biohub.detections import load_detections_from_geff, write_detection_cache
from biohub.evaluation import (
    FrozenLOEOError,
    build_frozen_loeo_plan,
    build_report,
    candidate_frontier_row,
    evaluate_directory,
    validate_exact_holdout_prediction_names,
    write_report,
)
from biohub.evaluation.official import OFFICIAL_EVALUATOR_COMMIT, TRACKSDATA_COMMIT
from biohub.experiments import file_sha256, path_sha256
from biohub.trackers import (
    HOCTPointGraphConfig,
    build_hoct_point_graph,
    candidate_config_id,
    verify_hoct_checkpoint,
)
from tracking_cellmot.io import save_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA_VENDOR = REPO_ROOT / "vendor" / "tracksdata"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen tracker on the opposite-embryo LOEO set without retuning."
    )
    parser.add_argument("--learned-selection", required=True, type=Path)
    parser.add_argument("--candidate-shortlist", required=True, type=Path)
    parser.add_argument("--monitor-prediction-plan", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--competition-root", required=True, type=Path)
    parser.add_argument(
        "--organizer-holdout-pred-dir",
        required=True,
        type=Path,
        help="Holdout predictions from the already-selected organizer checkpoint/detector settings.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Required only when the frozen winner is HOCT; expects <model_name>.pt.",
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _load_object(path: Path, label: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be a JSON object: {path}")
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
            "tracked working tree is not clean; commit/stash changes before LOEO evaluation:\n" + dirty
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


def _prediction_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SystemExit(f"organizer holdout prediction directory not found: {directory}")
    return {path.stem: path for path in directory.glob("*.geff")}


def _inventory_metadata(
    inventory: dict,
    names: tuple[str, ...],
) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, tuple[float, float, float]]]:
    rows = inventory.get("train")
    if not isinstance(rows, list):
        raise SystemExit("inventory has no train record list")
    by_name = {str(row.get("dataset")): row for row in rows if isinstance(row, dict)}
    missing = set(names) - set(by_name)
    if missing:
        raise SystemExit(f"LOEO datasets missing from inventory: {sorted(missing)}")
    shapes = {}
    scales = {}
    for name in names:
        row = by_name[name]
        shape_raw = row.get("image_shape_tzyx")
        scale_raw = row.get("scale_zyx_um")
        if not isinstance(shape_raw, list) or len(shape_raw) != 4:
            raise SystemExit(f"invalid image_shape_tzyx for {name}: {shape_raw!r}")
        if not isinstance(scale_raw, list) or len(scale_raw) != 3:
            raise SystemExit(f"invalid scale_zyx_um for {name}: {scale_raw!r}")
        shape = tuple(int(value) for value in shape_raw)
        scale = tuple(float(value) for value in scale_raw)
        if any(value <= 0 for value in shape) or any(value <= 0 for value in scale):
            raise SystemExit(f"non-positive image metadata for {name}: shape={shape} scale={scale}")
        shapes[name] = shape
        scales[name] = scale
    return shapes, scales


def _candidate_config(row: dict[str, Any], scale: tuple[float, float, float]) -> HOCTPointGraphConfig:
    kwargs: dict[str, Any] = {
        "n_neighbors": int(row["n_neighbors"]),
        "max_delta_t": int(row["max_delta_t"]),
        "scale_zyx_um": scale,
    }
    space = str(row.get("candidate_distance_space", ""))
    if space == "physical_um":
        kwargs["distance_threshold_um"] = float(row["distance_threshold_um"])
    elif space == "hoct_native_voxel":
        kwargs["distance_threshold_voxels"] = float(row["distance_threshold_voxels"])
    else:
        raise SystemExit(f"unknown frozen candidate distance space: {space!r}")
    config = HOCTPointGraphConfig(**kwargs)
    expected = str(row["config_id"])
    actual = candidate_config_id(config)
    if actual != expected:
        raise SystemExit(f"frozen candidate reconstruction mismatch: {actual} != {expected}")
    return config


def _solver_value(solver: dict[str, Any], key: str, cast):
    if key not in solver:
        raise SystemExit(f"frozen HOCT solver spec lacks {key}")
    try:
        return cast(solver[key])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid frozen HOCT solver value {key}={solver[key]!r}") from exc


def _hoct_command(
    *,
    candidate: Path,
    checkpoint: Path,
    out: Path,
    plan,
    device: str,
) -> list[str]:
    assert plan.hoct is not None
    solver = plan.hoct.solver
    tracklet = solver.get("tracklet_solver")
    if not isinstance(tracklet, bool):
        raise SystemExit("frozen HOCT solver tracklet_solver must be boolean")
    command = [
        sys.executable,
        str((REPO_ROOT / "scripts" / "run_hoct_candidate.py").resolve()),
        "--candidate", str(candidate),
        "--checkpoint", str(checkpoint),
        "--model-name", plan.hoct.model_name,
        "--out", str(out),
        "--device", device,
        "--window-size", str(plan.hoct.window_size),
        "--timeout", repr(_solver_value(solver, "timeout", float)),
        "--appearance-weight", repr(_solver_value(solver, "appearance_weight", float)),
        "--disappearance-weight", repr(_solver_value(solver, "disappearance_weight", float)),
        "--division-weight", repr(_solver_value(solver, "division_weight", float)),
        "--node-weight", repr(_solver_value(solver, "node_weight", float)),
        "--delta-t-weight", repr(_solver_value(solver, "delta_t_weight", float)),
        "--edge-bias", repr(_solver_value(solver, "edge_bias", float)),
        "--tracklet-solver" if tracklet else "--no-tracklet-solver",
    ]
    return command


def _run(command: list[str], *, label: str) -> float:
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    elapsed = time.monotonic() - started
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def _strict_new_directory(path: Path) -> None:
    if path.exists():
        raise SystemExit(
            f"LOEO output directory already exists; this command never overwrites clean evidence: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = _args()
    learned_path = args.learned_selection.resolve()
    candidate_path = args.candidate_shortlist.resolve()
    monitor_plan_path = args.monitor_prediction_plan.resolve()
    inventory_path = args.inventory.resolve()

    learned = _load_object(learned_path, "learned selection")
    candidate = _load_object(candidate_path, "candidate shortlist")
    monitor_plan = _load_object(monitor_plan_path, "monitor prediction plan")
    inventory = _load_object(inventory_path, "inventory")
    try:
        plan = build_frozen_loeo_plan(
            learned_selection=learned,
            candidate_shortlist=candidate,
            monitor_prediction_plan=monitor_plan,
        )
    except FrozenLOEOError as exc:
        raise SystemExit(str(exc)) from exc

    competition_root = args.competition_root.resolve()
    train_dir = competition_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"competition train directory not found: {train_dir}")
    shapes, scales = _inventory_metadata(inventory, plan.holdout_datasets)

    organizer_dir = args.organizer_holdout_pred_dir.resolve()
    monitor_dir = Path(str(monitor_plan.get("predictions_dir", ""))).resolve()
    if organizer_dir == monitor_dir:
        raise SystemExit("organizer LOEO prediction directory cannot be the training-side monitor directory")
    organizer_paths = _prediction_paths(organizer_dir)
    try:
        validate_exact_holdout_prediction_names(set(organizer_paths), plan)
    except FrozenLOEOError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (REPO_ROOT / "artifacts" / "private" / "loeo" / args.evaluation_id).resolve()
    )
    _strict_new_directory(out_dir)

    holdout_prediction_hashes = {
        name: path_sha256(path) for name, path in sorted(organizer_paths.items())
    }
    checkpoint = None
    checkpoint_hash = None
    frontier_row = None
    if plan.winner_family == "hoct":
        assert plan.hoct is not None
        if args.checkpoint_dir is None:
            raise SystemExit("--checkpoint-dir is required because the frozen winner is HOCT")
        checkpoint = (args.checkpoint_dir.resolve() / f"{plan.hoct.model_name}.pt")
        if not checkpoint.is_file():
            raise SystemExit(f"frozen HOCT checkpoint not found: {checkpoint}")
        verify_hoct_checkpoint(checkpoint, plan.hoct.model_name)
        checkpoint_hash = file_sha256(checkpoint)
        try:
            frontier_row = candidate_frontier_row(candidate, plan.hoct.candidate_config_id)
        except FrozenLOEOError as exc:
            raise SystemExit(str(exc)) from exc

    dry_plan = {
        "evaluation_id": args.evaluation_id,
        "purpose": "one-shot evaluation of a tracker frozen before opposite-embryo LOEO",
        "winner_family": plan.winner_family,
        "frozen_hoct": (
            {
                "trial_id": plan.hoct.trial_id,
                "candidate_config_id": plan.hoct.candidate_config_id,
                "model_name": plan.hoct.model_name,
                "window_size": plan.hoct.window_size,
                "solver": plan.hoct.solver,
            }
            if plan.hoct is not None
            else None
        ),
        "holdout_datasets": list(plan.holdout_datasets),
        "training_side_monitor_datasets": list(plan.monitor_datasets),
        "organizer_holdout_prediction_dir": str(organizer_dir),
        "organizer_holdout_prediction_sha256": holdout_prediction_hashes,
        "provenance": {
            "learned_selection_path": str(learned_path),
            "learned_selection_sha256": file_sha256(learned_path),
            "candidate_shortlist_path": str(candidate_path),
            "candidate_shortlist_sha256": file_sha256(candidate_path),
            "monitor_prediction_plan_path": str(monitor_plan_path),
            "monitor_prediction_plan_sha256": file_sha256(monitor_plan_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": file_sha256(inventory_path),
            "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": checkpoint_hash,
            "organizer_revision": OFFICIAL_EVALUATOR_COMMIT,
            "tracksdata_revision": TRACKSDATA_COMMIT,
        },
        "evidence_policy": {
            "configuration_frozen_before_gt_scoring": True,
            "retuning_from_this_result_allowed": False,
            "overwrite_allowed": False,
        },
    }
    plan_path = out_dir / "frozen_loeo_plan.json"
    plan_path.write_text(json.dumps(dry_plan, indent=2, sort_keys=True) + "\n")

    print(f"evaluation_id={args.evaluation_id}")
    print(f"winner_family={plan.winner_family}")
    if plan.hoct is not None:
        print(f"winner_trial_id={plan.hoct.trial_id}")
        print(f"candidate_config_id={plan.hoct.candidate_config_id}")
        print(f"model_name={plan.hoct.model_name}")
    print(f"holdout_datasets={','.join(plan.holdout_datasets)}")
    print(f"plan={plan_path}")

    if not args.execute:
        print(
            "DRY_RUN: frozen plan and input fingerprints recorded. "
            "Delete this dry-run directory, then rerun with --execute when ready to consume LOEO evidence."
        )
        return

    git_commit = _require_clean_tracked_tree()
    _require_submodule_pin(VENDOR, OFFICIAL_EVALUATOR_COMMIT, "organizer evaluator")
    _require_submodule_pin(TRACKSDATA_VENDOR, TRACKSDATA_COMMIT, "tracksdata")

    # Re-fingerprint the frozen organizer detections immediately before any GT
    # score is computed. If anything changed after planning, abort without score.
    current_hashes = {name: path_sha256(path) for name, path in sorted(organizer_paths.items())}
    if current_hashes != holdout_prediction_hashes:
        raise SystemExit("organizer holdout predictions changed after frozen LOEO plan was written")

    score_dir = organizer_dir
    inference_seconds = 0.0
    generated_hashes = None
    if plan.winner_family == "hoct":
        assert plan.hoct is not None and checkpoint is not None and frontier_row is not None
        detections_dir = out_dir / "frozen_holdout_detections"
        candidates_dir = out_dir / "candidate_graphs"
        solutions_dir = out_dir / "winner_predictions"
        detections_dir.mkdir()
        candidates_dir.mkdir()
        solutions_dir.mkdir()

        # Construct every frozen prediction before calling evaluate_directory.
        # No GT graph is opened in this loop.
        for name in plan.holdout_datasets:
            frame = load_detections_from_geff(organizer_paths[name])
            detection_path = detections_dir / f"{name}.parquet"
            write_detection_cache(frame, detection_path)

            config = _candidate_config(frontier_row, scales[name])
            graph = build_hoct_point_graph(frame, config, shape_tzyx=shapes[name])
            candidate_geff = candidates_dir / f"{name}.geff"
            save_graph(graph, candidate_geff, overwrite=False)

            output_geff = solutions_dir / f"{name}.geff"
            command = _hoct_command(
                candidate=candidate_geff,
                checkpoint=checkpoint,
                out=output_geff,
                plan=plan,
                device=args.device,
            )
            inference_seconds += _run(command, label=f"frozen-hoct:{name}")

        generated_names = {path.stem for path in solutions_dir.glob("*.geff")}
        try:
            validate_exact_holdout_prediction_names(generated_names, plan)
        except FrozenLOEOError as exc:
            raise SystemExit(str(exc)) from exc
        generated_hashes = {
            path.stem: path_sha256(path) for path in sorted(solutions_dir.glob("*.geff"))
        }
        score_dir = solutions_dir

    # Evidence is opened here, after winner, detections, candidate geometry,
    # checkpoint, solver and all predictions are already fixed.
    run = evaluate_directory(
        score_dir,
        train_dir,
        expected_names=plan.holdout_datasets,
        scale_by_name=scales,
    )
    report = build_report(run)
    report_path = out_dir / "loeo_evaluation_report.json"
    write_report(report, report_path)

    result = {
        "evaluation_id": args.evaluation_id,
        "status": "success",
        "winner_family": plan.winner_family,
        "winner_trial_id": plan.hoct.trial_id if plan.hoct is not None else None,
        "holdout_datasets": list(plan.holdout_datasets),
        "summary": report.overall,
        "inference_seconds": inference_seconds,
        "git_commit": git_commit,
        "frozen_plan_sha256": file_sha256(plan_path),
        "scored_prediction_dir": str(score_dir),
        "scored_prediction_sha256": (
            generated_hashes if generated_hashes is not None else holdout_prediction_hashes
        ),
        "evaluation_report": str(report_path),
        "evaluation_report_sha256": file_sha256(report_path),
        "interpretation_policy": (
            "This is clean LOEO evidence for the pre-frozen configuration. "
            "Do not tune this direction from this result and reuse it as unbiased validation."
        ),
    }
    result_path = out_dir / "loeo_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\nFROZEN LOEO RESULT")
    print(f"adj_edge_jaccard={report.overall.get('adj_edge_jaccard')}")
    print(f"division_jaccard={report.overall.get('division_jaccard')}")
    print(f"score={report.overall.get('score')}")
    print(f"report={report_path}")
    print(f"result={result_path}")
    print("CONFIGURATION REMAINS FROZEN; use this score as evidence, not as a tuning oracle.")


if __name__ == "__main__":
    main()
