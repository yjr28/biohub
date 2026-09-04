#!/usr/bin/env python3
"""Calibrate learned HOCT/solver variants without touching the LOEO embryo.

Consumes the Phase-2E candidate calibration work directory.  Only candidate
configurations frozen in ``candidate_shortlist.json`` may run.  All learned
model/solver settings must be predeclared in an explicit JSON grid.

The runner evaluates the organizer NodeTransformer prediction as a fixed-
detection control on the same nested training-side monitor datasets, evaluates
all allowed HOCT trials with the pinned official metric, and freezes either the
best HOCT trial or the organizer control according to a predeclared promotion
margin.  The opposite-embryo LOEO holdout is never predicted or scored here.

Default mode is a dry run.  Pass ``--execute`` in an environment containing the
audited HOCT package/checkpoints to spend GPU/ILP compute.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from biohub.calibration import (
    LearnedCalibrationError,
    expand_learned_trials,
    parse_learned_grid,
    select_training_side_winner,
)
from biohub.detections import load_detection_cache
from biohub.evaluation import build_report, evaluate_directory, write_report
from biohub.evaluation.official import OFFICIAL_EVALUATOR_COMMIT, TRACKSDATA_COMMIT
from biohub.experiments import file_sha256
from biohub.trackers import (
    HOCTPointGraphConfig,
    candidate_config_id,
    build_hoct_point_graph,
    verify_hoct_checkpoint,
)
from tracking_cellmot.io import save_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor" / "kaggle-cell-tracking-competition"
TRACKSDATA_VENDOR = REPO_ROOT / "vendor" / "tracksdata"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a learned HOCT/solver trial on training-side monitor data only."
    )
    parser.add_argument("--candidate-calibration-work-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--competition-root", required=True, type=Path)
    parser.add_argument("--learned-grid", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


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
            "tracked working tree is not clean; commit/stash changes before calibration:\n" + dirty
        )
    return head


def _require_submodule_pin(path: Path, expected: str, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} submodule missing: {path}")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual != expected:
        raise SystemExit(f"{label} revision mismatch: {actual} != {expected}")


def _prepare_dir(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists():
        if any(path.iterdir()) and not overwrite:
            raise SystemExit(f"{label} is not empty; pass --overwrite to replace it: {path}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run(command: list[str], *, label: str) -> float:
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    elapsed = time.monotonic() - started
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def _upstream_scope(candidate_work: Path) -> tuple[dict, dict, tuple[str, ...], tuple[str, ...]]:
    shortlist = _load_object(candidate_work / "candidate_shortlist.json", "candidate shortlist")
    plan = _load_object(candidate_work / "monitor_prediction_plan.json", "monitor prediction plan")

    selection = shortlist.get("selection_scope")
    frozen = shortlist.get("shortlist")
    if not isinstance(selection, dict) or not isinstance(frozen, dict):
        raise SystemExit("candidate shortlist lacks selection_scope/shortlist objects")
    if selection.get("loeo_used") is not False:
        raise SystemExit("candidate shortlist is not certified training-side only")
    if frozen.get("loeo_may_expand_shortlist") is not False:
        raise SystemExit("candidate shortlist does not forbid LOEO expansion")

    monitor = tuple(sorted(str(value) for value in selection.get("monitor_datasets", [])))
    holdout = tuple(sorted(str(value) for value in selection.get("forbidden_loeo_holdout_datasets", [])))
    plan_monitor = tuple(sorted(str(value) for value in plan.get("monitor_datasets", [])))
    plan_holdout = tuple(sorted(str(value) for value in plan.get("forbidden_loeo_holdout_datasets", [])))
    if not monitor or not holdout:
        raise SystemExit("candidate shortlist has empty monitor or forbidden LOEO set")
    if monitor != plan_monitor or holdout != plan_holdout:
        raise SystemExit("candidate shortlist and monitor prediction plan disagree on dataset scope")
    if set(monitor) & set(holdout):
        raise SystemExit("monitor and LOEO holdout datasets overlap")
    return shortlist, plan, monitor, holdout


def _frontier_map(shortlist: dict) -> tuple[tuple[str, ...], dict[str, dict]]:
    frozen = shortlist["shortlist"]
    allowed = tuple(str(value) for value in frozen.get("allowed_config_ids", []))
    trials = frozen.get("frontier_trials")
    if not allowed or not isinstance(trials, list):
        raise SystemExit("candidate shortlist has no allowed configs/frontier trials")
    by_id = {
        str(row.get("config_id")): dict(row)
        for row in trials
        if isinstance(row, dict) and row.get("config_id")
    }
    if len(by_id) != len(trials) or set(by_id) != set(allowed):
        raise SystemExit("candidate frontier trial set does not exactly match allowed_config_ids")
    return allowed, by_id


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
        raise SystemExit(f"monitor datasets missing from inventory: {sorted(missing)}")
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
        shapes[name] = tuple(int(value) for value in shape_raw)
        scales[name] = tuple(float(value) for value in scale_raw)
    return shapes, scales


def _candidate_config(row: dict, scale: tuple[float, float, float]) -> HOCTPointGraphConfig:
    space = row.get("candidate_distance_space")
    kwargs = {
        "n_neighbors": int(row["n_neighbors"]),
        "max_delta_t": int(row["max_delta_t"]),
        "scale_zyx_um": scale,
    }
    if space == "physical_um":
        kwargs["distance_threshold_um"] = float(row["distance_threshold_um"])
    elif space == "hoct_native_voxel":
        kwargs["distance_threshold_voxels"] = float(row["distance_threshold_voxels"])
    else:
        raise SystemExit(f"unknown candidate distance space in shortlist: {space!r}")
    config = HOCTPointGraphConfig(**kwargs)
    expected = str(row["config_id"])
    actual = candidate_config_id(config)
    if actual != expected:
        raise SystemExit(f"candidate config provenance mismatch: reconstructed {actual} != frozen {expected}")
    return config


def _verify_prediction_set(directory: Path, monitor: tuple[str, ...], holdout: tuple[str, ...]) -> None:
    if not directory.is_dir():
        raise SystemExit(f"organizer monitor prediction directory missing: {directory}")
    actual = {path.stem for path in directory.glob("*.geff")}
    if actual != set(monitor):
        raise SystemExit(
            f"organizer control prediction set mismatch: expected={sorted(monitor)} actual={sorted(actual)}"
        )
    leaked = actual & set(holdout)
    if leaked:
        raise SystemExit(f"LOEO holdout present in training-side prediction directory: {sorted(leaked)}")


def _hoct_command(
    *,
    candidate: Path,
    checkpoint: Path,
    model_name: str,
    out: Path,
    device: str,
    window_size: int,
    solver,
    allow_gap: bool,
) -> list[str]:
    command = [
        sys.executable,
        str((REPO_ROOT / "scripts" / "run_hoct_candidate.py").resolve()),
        "--candidate", str(candidate),
        "--checkpoint", str(checkpoint),
        "--model-name", model_name,
        "--out", str(out),
        "--device", device,
        "--window-size", str(window_size),
        "--timeout", repr(solver.timeout),
        "--appearance-weight", repr(solver.appearance_weight),
        "--disappearance-weight", repr(solver.disappearance_weight),
        "--division-weight", repr(solver.division_weight),
        "--node-weight", repr(solver.node_weight),
        "--delta-t-weight", repr(solver.delta_t_weight),
        "--edge-bias", repr(solver.edge_bias),
        "--tracklet-solver" if solver.tracklet_solver else "--no-tracklet-solver",
    ]
    if allow_gap:
        command.append("--allow-gap-candidates")
    return command


def main() -> None:
    args = _args()
    candidate_work = args.candidate_calibration_work_dir.resolve()
    if not candidate_work.is_dir():
        raise SystemExit(f"candidate calibration work directory not found: {candidate_work}")
    shortlist, monitor_plan, monitor, holdout = _upstream_scope(candidate_work)
    allowed_ids, frontier = _frontier_map(shortlist)

    inventory_path = _require_file(args.inventory, "inventory")
    inventory = _load_object(inventory_path, "inventory")
    learned_grid_path = _require_file(args.learned_grid, "learned calibration grid")
    try:
        grid = parse_learned_grid(_load_object(learned_grid_path, "learned calibration grid"))
        learned_trials = expand_learned_trials(
            allowed_candidate_config_ids=allowed_ids,
            grid=grid,
        )
    except LearnedCalibrationError as exc:
        raise SystemExit(str(exc)) from exc

    if not grid.allow_gap_candidates:
        gaps = [config_id for config_id, row in frontier.items() if int(row["max_delta_t"]) != 1]
        if gaps:
            raise SystemExit(
                f"frozen shortlist contains gap candidates but learned grid forbids them: {sorted(gaps)}"
            )

    competition_root = args.competition_root.resolve()
    train_dir = competition_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"competition train directory not found: {train_dir}")
    shapes, scales = _inventory_metadata(inventory, monitor)

    prediction_dir = Path(str(monitor_plan.get("predictions_dir", ""))).resolve()
    _verify_prediction_set(prediction_dir, monitor, holdout)
    detections_dir = candidate_work / "fixed_detections"
    if not detections_dir.is_dir():
        raise SystemExit(f"frozen detection directory missing: {detections_dir}")
    for name in monitor:
        _require_file(detections_dir / f"{name}.parquet", f"frozen detections for {name}")

    checkpoint_dir = args.checkpoint_dir.resolve()
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else (REPO_ROOT / "artifacts" / "private" / "hoct_learned" / args.calibration_id).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = {
        "calibration_id": args.calibration_id,
        "purpose": "training-side learned HOCT/solver selection before LOEO",
        "monitor_datasets": list(monitor),
        "forbidden_loeo_holdout_datasets": list(holdout),
        "loeo_used": False,
        "allowed_candidate_config_ids": list(allowed_ids),
        "grid": grid.to_dict(),
        "trials": [trial.to_dict() for trial in learned_trials],
        "organizer_control_predictions": str(prediction_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }
    plan_path = work_dir / "learned_calibration_plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n")

    print(f"calibration_id={args.calibration_id}")
    print(f"monitor_datasets={','.join(monitor)}")
    print(f"forbidden_loeo_holdout={','.join(holdout)}")
    print(f"candidate_configs={len(allowed_ids)}")
    print(f"learned_trials={len(learned_trials)}")
    print(f"hoct_promotion_margin={grid.hoct_promotion_margin}")
    print(f"plan={plan_path}")

    if not args.execute:
        print("DRY_RUN: learned calibration plan frozen; rerun with --execute to spend HOCT GPU/ILP compute.")
        return

    git_commit = _require_clean_tracked_tree()
    _require_submodule_pin(VENDOR, OFFICIAL_EVALUATOR_COMMIT, "organizer evaluator")
    _require_submodule_pin(TRACKSDATA_VENDOR, TRACKSDATA_COMMIT, "tracksdata")

    checkpoint_paths = {}
    for model_name in grid.model_names:
        checkpoint = _require_file(checkpoint_dir / f"{model_name}.pt", f"HOCT checkpoint {model_name}")
        verify_hoct_checkpoint(checkpoint, model_name)
        checkpoint_paths[model_name] = checkpoint

    # Score the fixed-detection organizer NodeTransformer control on exactly the
    # same training-side monitor movies used for HOCT selection.
    control_run = evaluate_directory(
        prediction_dir,
        train_dir,
        expected_names=monitor,
        scale_by_name=scales,
    )
    control_report = build_report(control_run)
    control_report_path = work_dir / "organizer_control_report.json"
    write_report(control_report, control_report_path)

    candidates_root = work_dir / "candidate_graphs"
    solutions_root = work_dir / "hoct_solutions"
    reports_root = work_dir / "trial_reports"
    _prepare_dir(candidates_root, overwrite=args.overwrite, label="candidate graph directory")
    _prepare_dir(solutions_root, overwrite=args.overwrite, label="HOCT solution directory")
    _prepare_dir(reports_root, overwrite=args.overwrite, label="HOCT report directory")

    # Candidate graphs are shared across model/solver trials.
    candidate_paths: dict[tuple[str, str], Path] = {}
    for config_id in allowed_ids:
        row = frontier[config_id]
        for name in monitor:
            config = _candidate_config(row, scales[name])
            detections = load_detection_cache(detections_dir / f"{name}.parquet")
            graph = build_hoct_point_graph(detections, config, shape_tzyx=shapes[name])
            path = candidates_root / config_id / f"{name}.geff"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_graph(graph, path, overwrite=args.overwrite)
            candidate_paths[(config_id, name)] = path

    trial_results = []
    for index, trial in enumerate(learned_trials, start=1):
        print(
            f"\nTRIAL {index}/{len(learned_trials)} {trial.trial_id} "
            f"candidate={trial.candidate_config_id} model={trial.model_name} "
            f"window={trial.window_size} solver={trial.solver.name}"
        )
        trial_dir = solutions_root / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        runtime = 0.0
        for name in monitor:
            output = trial_dir / f"{name}.geff"
            command = _hoct_command(
                candidate=candidate_paths[(trial.candidate_config_id, name)],
                checkpoint=checkpoint_paths[trial.model_name],
                model_name=trial.model_name,
                out=output,
                device=args.device,
                window_size=trial.window_size,
                solver=trial.solver,
                allow_gap=grid.allow_gap_candidates,
            )
            runtime += _run(command, label=f"{trial.trial_id}:{name}")

        run = evaluate_directory(
            trial_dir,
            train_dir,
            expected_names=monitor,
            scale_by_name=scales,
        )
        report = build_report(run)
        report_path = reports_root / f"{trial.trial_id}.json"
        write_report(report, report_path)
        row = {
            "trial_id": trial.trial_id,
            "spec": trial.to_dict(),
            "summary": report.overall,
            "runtime_seconds": runtime,
            "report_path": str(report_path),
            "report_sha256": file_sha256(report_path),
        }
        trial_results.append(row)
        print(
            f"trial_score={report.overall.get('score')} "
            f"adj_edge_jaccard={report.overall.get('adj_edge_jaccard')} "
            f"division_jaccard={report.overall.get('division_jaccard')}"
        )

    try:
        winner = select_training_side_winner(
            organizer_control_summary=control_report.overall,
            hoct_trials=trial_results,
            promotion_margin=grid.hoct_promotion_margin,
        )
    except LearnedCalibrationError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint_hashes = {
        name: file_sha256(path) for name, path in sorted(checkpoint_paths.items())
    }
    selection_payload = {
        "calibration_id": args.calibration_id,
        "purpose": "freeze one learned tracking configuration before LOEO",
        "selection_scope": {
            "monitor_datasets": list(monitor),
            "forbidden_loeo_holdout_datasets": list(holdout),
            "loeo_used": False,
            "loeo_may_retune_or_replace_winner": False,
        },
        "organizer_control": {
            "summary": control_report.overall,
            "report_path": str(control_report_path),
            "report_sha256": file_sha256(control_report_path),
        },
        "hoct_trials": trial_results,
        "winner": winner,
        "provenance": {
            "git_commit": git_commit,
            "organizer_revision": OFFICIAL_EVALUATOR_COMMIT,
            "tracksdata_revision": TRACKSDATA_COMMIT,
            "candidate_shortlist_path": str((candidate_work / "candidate_shortlist.json").resolve()),
            "candidate_shortlist_sha256": file_sha256(candidate_work / "candidate_shortlist.json"),
            "monitor_prediction_plan_path": str((candidate_work / "monitor_prediction_plan.json").resolve()),
            "monitor_prediction_plan_sha256": file_sha256(candidate_work / "monitor_prediction_plan.json"),
            "fixed_detection_index_sha256": file_sha256(candidate_work / "fixed_detections" / "index.json"),
            "inventory_path": str(inventory_path),
            "inventory_sha256": file_sha256(inventory_path),
            "learned_grid_path": str(learned_grid_path),
            "learned_grid_sha256": file_sha256(learned_grid_path),
            "checkpoint_sha256": checkpoint_hashes,
        },
        "next_gate": (
            "Evaluate only this frozen winner on the opposite-embryo LOEO set. "
            "Do not change candidate geometry, model, solver, window, detector settings, or promotion rule from that result."
        ),
    }
    selection_path = work_dir / "learned_selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n")

    print("\nLEARNED CONFIGURATION FROZEN")
    print(f"winner_family={winner['family']}")
    if winner["family"] == "hoct":
        print(f"winner_trial_id={winner['trial_id']}")
        print(f"score_gain_over_control={winner['score_gain_over_control']}")
    else:
        print(f"best_hoct_trial_id={winner.get('best_hoct_trial_id')}")
        print(f"score_gain_over_control={winner.get('score_gain_over_control')}")
    print(f"selection={selection_path}")
    print("LOEO HOLDOUT REMAINS UNTOUCHED.")


if __name__ == "__main__":
    main()
