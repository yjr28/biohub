#!/usr/bin/env python3
"""Freeze detector nodes from prediction GEFFs into tracker-neutral Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub.detections import load_detections_from_geff, write_detection_cache
from biohub.experiments import file_sha256


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract centroid nodes from prediction GEFFs and discard all predicted edges."
    )
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--names",
        default=None,
        help="Optional comma-separated GEFF stems to freeze; default uses every *.geff in pred-dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not pred_dir.is_dir():
        raise SystemExit(f"prediction directory not found: {pred_dir}")

    if args.names:
        names = tuple(sorted({value.strip() for value in args.names.split(",") if value.strip()}))
    else:
        names = tuple(sorted(path.stem for path in pred_dir.glob("*.geff")))
    if not names:
        raise SystemExit(f"no prediction GEFFs selected under {pred_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name in names:
        source = pred_dir / f"{name}.geff"
        if not source.exists():
            raise SystemExit(f"selected prediction GEFF is missing: {source}")
        output = out_dir / f"{name}.parquet"
        if output.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite fixed-detection cache: {output}")
        frame = load_detections_from_geff(source)
        write_detection_cache(frame, output)
        index.append(
            {
                "dataset": name,
                "cache_file": output.name,
                "cache_sha256": file_sha256(output),
                "num_detections": frame.height,
                "t_min": int(frame["t"].min()),
                "t_max": int(frame["t"].max()),
            }
        )
        print(f"{name}: detections={frame.height} frames={index[-1]['t_min']}..{index[-1]['t_max']}")

    index_path = out_dir / "index.json"
    if index_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite cache index: {index_path}")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"fixed_detection_index={index_path}")


if __name__ == "__main__":
    main()
