"""Acceptance gate for the *actual* downloaded competition inventory.

Phase 1A deliberately separated metadata discovery from model work.  This
module is the gate between them: it refuses to start a clean baseline if the
real Kaggle mount violates the host/evaluator assumptions our validation system
relies on.

The gate consumes only :class:`InventoryReport`; it never loads image voxels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import prod
from typing import Any

from .inventory import InventoryReport


class RealDataGateError(ValueError):
    """Raised when the downloaded competition data cannot support clean LOEO."""


@dataclass(frozen=True)
class RealDataGate:
    accepted: bool
    train_embryos: tuple[str, str]
    train_dataset_count: int
    visible_test_dataset_count: int
    dataset_count_by_embryo: dict[str, int]
    gt_nodes_by_embryo: dict[str, int]
    gt_edges_by_embryo: dict[str, int]
    gt_divisions_by_embryo: dict[str, int]
    image_voxels_by_embryo: dict[str, int]
    loeo_fold_names: tuple[str, str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_real_inventory(report: InventoryReport) -> RealDataGate:
    """Validate the real metadata inventory before any baseline training.

    Hard failures protect assumptions that would make a score incomparable or a
    fold contaminated.  Plausible-but-surprising metadata is surfaced as a
    warning rather than silently normalized.
    """

    train = tuple(report.train)
    if not train:
        raise RealDataGateError("training inventory is empty")

    embryos = tuple(sorted(report.train_embryos))
    if len(embryos) != 2:
        raise RealDataGateError(
            "host-verified validation contract expects exactly two training embryo IDs; "
            f"found {embryos}"
        )

    train_names = {record.dataset for record in train}
    if len(train_names) != len(train):
        raise RealDataGateError("training inventory contains duplicate dataset IDs")
    if report.train_visible_test_name_overlap:
        raise RealDataGateError(
            "train/visible-test dataset-name overlap is ambiguous: "
            f"{report.train_visible_test_name_overlap}"
        )

    warnings: list[str] = []
    dataset_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    division_counts: Counter[str] = Counter()
    voxel_counts: Counter[str] = Counter()

    for record in train:
        if record.embryo_id not in embryos:
            raise RealDataGateError(
                f"dataset {record.dataset} has embryo_id={record.embryo_id!r} outside {embryos}"
            )
        if not record.has_geff:
            raise RealDataGateError(f"training dataset {record.dataset} has no GT GEFF")
        if any(int(dim) <= 0 for dim in record.image_shape_tzyx):
            raise RealDataGateError(
                f"training dataset {record.dataset} has invalid image shape {record.image_shape_tzyx}"
            )
        if any(float(scale) <= 0 for scale in record.scale_zyx_um):
            raise RealDataGateError(
                f"training dataset {record.dataset} has invalid physical scale {record.scale_zyx_um}"
            )
        if record.gt_nodes is None or record.gt_nodes <= 0:
            raise RealDataGateError(f"training dataset {record.dataset} has no GT nodes")
        if record.gt_edges is None or record.gt_edges <= 0:
            raise RealDataGateError(f"training dataset {record.dataset} has no GT edges")
        if record.gt_divisions is None or record.gt_divisions < 0:
            raise RealDataGateError(f"training dataset {record.dataset} has invalid division count")
        if record.estimated_total_nodes is None or record.estimated_total_nodes <= 0:
            raise RealDataGateError(
                f"training dataset {record.dataset} lacks positive estimated_number_of_nodes; "
                "adjusted edge scoring would be incomplete"
            )
        if record.gt_t_min is None or record.gt_t_max is None or record.gt_t_min > record.gt_t_max:
            raise RealDataGateError(
                f"training dataset {record.dataset} has invalid annotated time range "
                f"{record.gt_t_min}..{record.gt_t_max}"
            )
        if record.estimated_total_nodes < record.gt_nodes:
            warnings.append(
                f"{record.dataset}: estimated total nodes ({record.estimated_total_nodes:g}) "
                f"is below annotated GT nodes ({record.gt_nodes})"
            )

        dataset_counts[record.embryo_id] += 1
        node_counts[record.embryo_id] += int(record.gt_nodes)
        edge_counts[record.embryo_id] += int(record.gt_edges)
        division_counts[record.embryo_id] += int(record.gt_divisions)
        voxel_counts[record.embryo_id] += int(prod(record.image_shape_tzyx))

    if any(dataset_counts[embryo] == 0 for embryo in embryos):
        raise RealDataGateError(f"at least one training embryo has zero datasets: {dict(dataset_counts)}")

    folds = tuple(report.loeo_folds)
    if len(folds) != 2:
        raise RealDataGateError(
            f"clean validation contract requires exactly two LOEO folds; found {len(folds)}"
        )
    fold_names = tuple(sorted(str(fold.get("name", "")) for fold in folds))
    if any(not name for name in fold_names) or len(set(fold_names)) != 2:
        raise RealDataGateError(f"LOEO fold names are missing or duplicated: {fold_names}")

    held_out: set[str] = set()
    for fold in folds:
        holdout = str(fold.get("holdout_embryo", "")).strip()
        train_embryos = tuple(sorted(str(value) for value in fold.get("train_embryos", ())))
        train_datasets = set(str(value) for value in fold.get("train_datasets", ()))
        holdout_datasets = set(str(value) for value in fold.get("holdout_datasets", ()))
        if holdout not in embryos:
            raise RealDataGateError(f"LOEO fold has unknown holdout embryo: {holdout!r}")
        expected_train_embryos = tuple(embryo for embryo in embryos if embryo != holdout)
        if train_embryos != expected_train_embryos:
            raise RealDataGateError(
                f"fold {fold.get('name')} train_embryos={train_embryos} != {expected_train_embryos}"
            )
        expected_holdout = {record.dataset for record in train if record.embryo_id == holdout}
        expected_train = train_names - expected_holdout
        if holdout_datasets != expected_holdout:
            raise RealDataGateError(
                f"fold {fold.get('name')} does not cover the held-out embryo exactly: "
                f"missing={sorted(expected_holdout - holdout_datasets)}, "
                f"extra={sorted(holdout_datasets - expected_holdout)}"
            )
        if train_datasets != expected_train:
            raise RealDataGateError(
                f"fold {fold.get('name')} does not cover the training embryo exactly: "
                f"missing={sorted(expected_train - train_datasets)}, "
                f"extra={sorted(train_datasets - expected_train)}"
            )
        if train_datasets & holdout_datasets:
            raise RealDataGateError(f"fold {fold.get('name')} has train/holdout dataset leakage")
        held_out.add(holdout)

    if held_out != set(embryos):
        raise RealDataGateError(
            f"the two LOEO folds must each hold out one distinct training embryo; found {sorted(held_out)}"
        )

    if not report.visible_test:
        warnings.append("visible test placeholder inventory is empty")

    scales = {tuple(float(x) for x in record.scale_zyx_um) for record in train}
    if len(scales) > 1:
        warnings.append(
            "training datasets use multiple physical scales; per-dataset scoring already preserves them"
        )

    return RealDataGate(
        accepted=True,
        train_embryos=(embryos[0], embryos[1]),
        train_dataset_count=len(train),
        visible_test_dataset_count=len(report.visible_test),
        dataset_count_by_embryo={embryo: dataset_counts[embryo] for embryo in embryos},
        gt_nodes_by_embryo={embryo: node_counts[embryo] for embryo in embryos},
        gt_edges_by_embryo={embryo: edge_counts[embryo] for embryo in embryos},
        gt_divisions_by_embryo={embryo: division_counts[embryo] for embryo in embryos},
        image_voxels_by_embryo={embryo: voxel_counts[embryo] for embryo in embryos},
        loeo_fold_names=(fold_names[0], fold_names[1]),
        warnings=tuple(sorted(set(warnings))),
    )
