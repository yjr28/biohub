#!/usr/bin/env python3
"""Run audited HOCT inference on a prebuilt fixed-detection candidate graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import tracksdata as td

from biohub.trackers import HOCT_REVISION, verify_hoct_checkpoint
from tracking_cellmot.io import save_graph


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score/solve a Biohub centroid-only candidate graph with an audited local HOCT checkpoint."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-name", required=True, choices=("general_v1", "ctc_v0", "general_v0"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--appearance-weight", type=float, default=0.5)
    parser.add_argument("--disappearance-weight", type=float, default=0.25)
    parser.add_argument("--division-weight", type=float, default=0.25)
    parser.add_argument("--node-weight", type=float, default=-10.0)
    parser.add_argument("--delta-t-weight", type=float, default=0.5)
    parser.add_argument("--edge-bias", type=float, default=0.5)
    parser.add_argument("--tracklet-solver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-gap-candidates",
        action="store_true",
        help="Permit candidate graphs with max_delta_t > 1. Primary Biohub experiment keeps this off.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_graph(path: Path) -> td.graph.BaseGraph:
    if not path.exists():
        raise SystemExit(f"candidate GEFF not found: {path}")
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = _args()
    output = args.out.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite HOCT output: {output}")

    spec = verify_hoct_checkpoint(args.checkpoint.resolve(), args.model_name)
    graph = _load_graph(args.candidate.resolve())
    if graph.metadata.get("biohub_adapter") != "centroid_only_hoct_compat":
        raise SystemExit("candidate graph was not produced by the Biohub centroid-only HOCT adapter")
    if graph.metadata.get("hoct_revision") != HOCT_REVISION:
        raise SystemExit(
            f"candidate HOCT revision mismatch: {graph.metadata.get('hoct_revision')!r} != {HOCT_REVISION}"
        )
    max_delta_t = int(graph.metadata.get("max_delta_t", 1))
    if max_delta_t != 1 and not args.allow_gap_candidates:
        raise SystemExit(
            f"candidate max_delta_t={max_delta_t}; primary competition experiment requires 1. "
            "Pass --allow-gap-candidates only for an explicit gap ablation."
        )

    try:
        import hoct
        from hoct.tracking import ILPSolverConfig
    except ImportError as exc:
        raise SystemExit(
            "HOCT is not installed in this runtime. Install/package the audited HOCT revision "
            f"{HOCT_REVISION} before running this command."
        ) from exc

    solver = ILPSolverConfig(
        appearance_weight=args.appearance_weight,
        disappearance_weight=args.disappearance_weight,
        division_weight=args.division_weight,
        node_weight=args.node_weight,
        delta_t_weight=args.delta_t_weight,
        edge_bias=args.edge_bias,
        timeout=args.timeout,
        tracklet_solver=args.tracklet_solver,
    )
    model = hoct.load_model(str(args.checkpoint.resolve()), device=args.device)
    solution = hoct.predict(
        model,
        graph=graph,
        solver_config=solver,
        window_size=args.window_size,
        return_solution=True,
    )
    if solution is None:
        raise SystemExit("HOCT returned no solution graph")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_graph(solution, output, overwrite=args.overwrite)
    print(f"hoct_revision={HOCT_REVISION}")
    print(f"checkpoint_model={spec.name}")
    print(f"checkpoint_sha256={spec.sha256}")
    print(f"input_nodes={graph.num_nodes()}")
    print(f"input_candidate_edges={graph.num_edges()}")
    print(f"solution_nodes={solution.num_nodes()}")
    print(f"solution_edges={solution.num_edges()}")
    print(f"solver_timeout={args.timeout}")
    print(f"tracklet_solver={args.tracklet_solver}")
    print(f"output_geff={output}")


if __name__ == "__main__":
    main()
