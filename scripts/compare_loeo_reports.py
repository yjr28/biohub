#!/usr/bin/env python3
"""Compare a baseline and challenger across both clean LOEO directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.analysis import compare_two_direction_loeo


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact organizer summaries across the two embryo-held-out directions."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        action="append",
        metavar="FOLD=REPORT.json",
        help="Repeat exactly twice",
    )
    parser.add_argument(
        "--challenger",
        required=True,
        action="append",
        metavar="FOLD=REPORT.json",
        help="Repeat exactly twice",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _pairs(values: list[str], label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"{label} entry must be FOLD=REPORT.json: {item!r}")
        fold, raw_path = item.split("=", 1)
        fold = fold.strip()
        path = Path(raw_path).expanduser()
        if not fold or fold in result:
            raise SystemExit(f"{label} fold is empty or duplicated: {fold!r}")
        if not path.is_file():
            raise SystemExit(f"{label} report not found for {fold}: {path}")
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not isinstance(report.get("overall"), dict):
            raise SystemExit(f"{path} has no 'overall' summary object")
        result[fold] = report["overall"]
    return result


def main() -> None:
    args = _args()
    comparison = compare_two_direction_loeo(
        _pairs(args.baseline, "baseline"),
        _pairs(args.challenger, "challenger"),
    )
    payload = comparison.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
