#!/usr/bin/env python3
"""Attach one evaluation report to a previously registered experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.experiments import ExperimentResult, load_manifests, write_result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an immutable experiment result JSON.")
    parser.add_argument("--registry", default=Path("experiments/manifests.jsonl"), type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runtime-seconds", type=float, default=None)
    parser.add_argument("--status", choices=("success", "failed", "aborted"), default="success")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = _args()
    manifests = load_manifests(args.registry)
    matches = [item for item in manifests if item.experiment_id == args.experiment_id]
    if len(matches) != 1:
        raise SystemExit(
            f"experiment {args.experiment_id!r} not uniquely registered in {args.registry}; "
            f"matches={len(matches)}"
        )
    if not args.report.is_file():
        raise SystemExit(f"evaluation report not found: {args.report}")
    report = json.loads(args.report.read_text())
    if not isinstance(report, dict) or not isinstance(report.get("overall"), dict):
        raise SystemExit("evaluation report must contain an 'overall' JSON object")

    result = ExperimentResult(
        experiment_id=args.experiment_id,
        status=args.status,
        summary=report["overall"],
        report_path=str(args.report),
        runtime_seconds=args.runtime_seconds,
        notes=args.notes,
    )
    write_result(args.out, result)
    print(f"experiment={args.experiment_id}")
    print(f"status={args.status}")
    print(f"score={result.summary.get('score')}")
    print(f"result={args.out}")


if __name__ == "__main__":
    main()
