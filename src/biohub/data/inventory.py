"""Metadata-only inventory of the Biohub competition data.

The inventory intentionally avoids loading image volumes into memory. It reads
OME-Zarr shape/metadata and, where present, GEFF graph metadata/counts. This is
safe to run on Kaggle before any training and gives us the exact dataset set
that later validation manifests must reference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tracksdata as td
import zarr
from geff import GeffMetadata

from .validation import build_loeo_folds, embryo_id_from_dataset

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


class DataInventoryError(ValueError):
    """Raised when the downloaded competition layout is ambiguous or incomplete."""


@dataclass(frozen=True)
class DatasetRecord:
    """Metadata for one image dataset/crop."""

    split: str
    dataset: str
    embryo_id: str
    image_shape_tzyx: tuple[int, int, int, int]
    scale_zyx_um: tuple[float, float, float]
    has_geff: bool
    gt_nodes: int | None = None
    gt_edges: int | None = None
    gt_divisions: int | None = None
    gt_t_min: int | None = None
    gt_t_max: int | None = None
    estimated_total_nodes: float | None = None


@dataclass(frozen=True)
class InventoryReport:
    """Inventory for train/test folders plus validation-contract diagnostics."""

    competition_root: str
    train: tuple[DatasetRecord, ...]
    visible_test: tuple[DatasetRecord, ...]
    train_embryos: tuple[str, ...]
    visible_test_embryos: tuple[str, ...]
    train_visible_test_name_overlap: tuple[str, ...]
    loeo_folds: tuple[dict[str, Any], ...]


def _parse_scale(attrs: dict[str, Any]) -> tuple[float, float, float]:
    """Read OME-NGFF spatial scale using the organizer's documented convention."""

    multiscales = attrs.get("multiscales")
    if not multiscales:
        return DEFAULT_SCALE
    try:
        transform = multiscales[0]["datasets"][0]["coordinateTransformations"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise DataInventoryError(f"Malformed OME-Zarr multiscales metadata: {multiscales!r}") from exc

    if transform.get("type") != "scale":
        raise DataInventoryError(f"Unsupported coordinate transform: {transform!r}")

    try:
        values = tuple(float(value) for value in transform["scale"][-3:])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataInventoryError(f"Malformed OME-Zarr scale transform: {transform!r}") from exc

    if len(values) != 3 or any(value <= 0 for value in values):
        raise DataInventoryError(f"Invalid spatial scale: {values!r}")
    return values


def _inspect_image(path: Path) -> tuple[tuple[int, int, int, int], tuple[float, float, float]]:
    group = zarr.open_group(path, mode="r")
    if "0" not in group:
        raise DataInventoryError(f"Expected OME-Zarr array '0' in {path}")
    shape = tuple(int(value) for value in group["0"].shape)
    if len(shape) != 4:
        raise DataInventoryError(f"Expected (T,Z,Y,X) image shape in {path}, found {shape!r}")
    return shape, _parse_scale(dict(group.attrs))


def _load_graph(path: Path) -> td.graph.BaseGraph:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _inspect_geff(path: Path) -> dict[str, Any]:
    graph = _load_graph(path)
    attrs = graph.node_attrs()
    if graph.num_nodes() > 0 and "t" not in attrs.columns:
        raise DataInventoryError(f"GT graph {path} has nodes but no 't' attribute")

    degrees = graph.out_degree()
    if isinstance(degrees, int):
        degrees = [degrees]
    t_min = int(attrs["t"].min()) if graph.num_nodes() else None
    t_max = int(attrs["t"].max()) if graph.num_nodes() else None

    metadata = GeffMetadata.read(path)
    raw_estimate = (metadata.extra or {}).get("estimated_number_of_nodes")
    estimate = float(raw_estimate) if raw_estimate is not None else None

    return {
        "gt_nodes": int(graph.num_nodes()),
        "gt_edges": int(graph.num_edges()),
        "gt_divisions": int(sum(int(degree) >= 2 for degree in degrees)),
        "gt_t_min": t_min,
        "gt_t_max": t_max,
        "estimated_total_nodes": estimate,
    }


def _dataset_names(split_dir: Path) -> tuple[str, ...]:
    names = tuple(sorted(path.stem for path in split_dir.glob("*.zarr")))
    if not names:
        raise DataInventoryError(f"No .zarr datasets found in {split_dir}")
    if len(names) != len(set(names)):
        raise DataInventoryError(f"Duplicate dataset stems found in {split_dir}")
    return names


def inventory_split(split_dir: str | Path, *, split: str, require_geff: bool) -> tuple[DatasetRecord, ...]:
    """Inventory one competition split without loading image voxels."""

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise DataInventoryError(f"Split directory does not exist: {split_dir}")

    records: list[DatasetRecord] = []
    for name in _dataset_names(split_dir):
        image_path = split_dir / f"{name}.zarr"
        geff_path = split_dir / f"{name}.geff"
        has_geff = geff_path.exists()
        if require_geff and not has_geff:
            raise DataInventoryError(f"Training image {image_path} has no paired GT {geff_path.name}")

        shape, scale = _inspect_image(image_path)
        graph_fields = _inspect_geff(geff_path) if has_geff else {}
        records.append(
            DatasetRecord(
                split=split,
                dataset=name,
                embryo_id=embryo_id_from_dataset(name),
                image_shape_tzyx=shape,
                scale_zyx_um=scale,
                has_geff=has_geff,
                **graph_fields,
            )
        )
    return tuple(records)


def inventory_competition(root: str | Path) -> InventoryReport:
    """Inspect `<root>/train` and the visible `<root>/test` placeholder split."""

    root = Path(root).resolve()
    train = inventory_split(root / "train", split="train", require_geff=True)
    visible_test = inventory_split(root / "test", split="visible_test", require_geff=False)

    train_names = tuple(record.dataset for record in train)
    test_names = tuple(record.dataset for record in visible_test)
    folds = build_loeo_folds(train_names)

    return InventoryReport(
        competition_root=str(root),
        train=train,
        visible_test=visible_test,
        train_embryos=tuple(sorted({record.embryo_id for record in train})),
        visible_test_embryos=tuple(sorted({record.embryo_id for record in visible_test})),
        train_visible_test_name_overlap=tuple(sorted(set(train_names) & set(test_names))),
        loeo_folds=tuple(asdict(fold) for fold in folds),
    )


def _flatten_record(record: DatasetRecord) -> dict[str, Any]:
    row = asdict(record)
    row["image_shape_tzyx"] = "x".join(str(value) for value in record.image_shape_tzyx)
    row["scale_zyx_um"] = ",".join(f"{value:.8g}" for value in record.scale_zyx_um)
    return row


def write_inventory(
    report: InventoryReport,
    *,
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Persist the inventory in machine-readable JSON and optional flat CSV."""

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")

    if csv_path is None:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_record(record) for record in (*report.train, *report.visible_test)]
    if not rows:
        raise DataInventoryError("refusing to write an empty inventory CSV")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
