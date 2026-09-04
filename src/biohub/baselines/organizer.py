"""Holdout-safe split files for the pinned organizer baseline scripts.

The pinned organizer training script uses its ``test`` split every epoch and
selects ``edge_predictor_best.pth`` by ``test_acc * test_recall``. Therefore,
putting the LOEO holdout embryo in that script's test split would contaminate
our model-selection evidence even though the model does not backprop through the
holdout.  This module emits separate training and prediction split files:

* training split: train embryo only; its monitor set also contains train-embryo
  data, never the LOEO holdout;
* prediction split: the same trained fold index but test points at the true
  held-out embryo so the organizer prediction script produces those GEFFs.

This preserves the organizer architecture/training implementation while making
the cross-embryo evaluation boundary explicit. It is not claimed to reproduce
the organizer's original random 90/10 checkpoint-selection protocol exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from biohub.data.validation import embryo_id_from_dataset


class OrganizerBaselineError(ValueError):
    """Raised when a split would violate the clean-LOEO baseline protocol."""


@dataclass(frozen=True)
class OrganizerBaselineProtocol:
    fold_name: str
    holdout_embryo: str
    train_embryos: tuple[str, ...]
    train_datasets: tuple[str, ...]
    holdout_datasets: tuple[str, ...]
    organizer_fold_index: int
    train_splits: tuple[dict[str, Any], ...]
    predict_splits: tuple[dict[str, Any], ...]
    checkpoint_monitor_policy: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_fold(inventory: Mapping[str, Any], fold_name: str) -> Mapping[str, Any]:
    folds = inventory.get("loeo_folds")
    if not isinstance(folds, list):
        raise OrganizerBaselineError("inventory has no loeo_folds list")
    matches = [fold for fold in folds if isinstance(fold, Mapping) and fold.get("name") == fold_name]
    if len(matches) != 1:
        available = sorted(str(fold.get("name")) for fold in folds if isinstance(fold, Mapping))
        raise OrganizerBaselineError(
            f"fold {fold_name!r} not uniquely found; available={available}"
        )
    return matches[0]


def build_organizer_protocol(
    inventory: Mapping[str, Any],
    *,
    fold_name: str,
    organizer_fold_index: int = 0,
    checkpoint_monitor_policy: str = "train-embryo-all",
) -> OrganizerBaselineProtocol:
    """Create separate train/predict split payloads for a clean LOEO direction.

    ``train-embryo-all`` intentionally uses the complete training-embryo set as
    both the organizer script's train and test lists. The test list is only a
    checkpoint monitor in the pinned training implementation; reusing training
    embryo data keeps every held-out-embryo voxel/GT out of checkpoint selection
    while preserving all available training-embryo samples for optimization.

    This monitor is not an independent validation set. The *only* clean model
    selection result remains the later external LOEO score.
    """

    if organizer_fold_index < 0:
        raise OrganizerBaselineError("organizer_fold_index must be >= 0")
    if checkpoint_monitor_policy != "train-embryo-all":
        raise OrganizerBaselineError(
            "Only checkpoint_monitor_policy='train-embryo-all' is implemented; "
            "add and test any new policy explicitly before use."
        )

    fold = _resolve_fold(inventory, fold_name)
    holdout_embryo = str(fold.get("holdout_embryo", "")).strip()
    train_embryos = tuple(sorted(str(value).strip() for value in fold.get("train_embryos", []) if str(value).strip()))
    train_datasets = tuple(sorted(str(value).strip() for value in fold.get("train_datasets", []) if str(value).strip()))
    holdout_datasets = tuple(sorted(str(value).strip() for value in fold.get("holdout_datasets", []) if str(value).strip()))

    if not holdout_embryo:
        raise OrganizerBaselineError(f"fold {fold_name!r} has no holdout_embryo")
    if not train_embryos or not train_datasets or not holdout_datasets:
        raise OrganizerBaselineError(f"fold {fold_name!r} is missing train/holdout members")
    if holdout_embryo in train_embryos:
        raise OrganizerBaselineError("holdout embryo appears in train_embryos")

    bad_train = [name for name in train_datasets if embryo_id_from_dataset(name) not in train_embryos]
    bad_holdout = [name for name in holdout_datasets if embryo_id_from_dataset(name) != holdout_embryo]
    if bad_train:
        raise OrganizerBaselineError(f"train_datasets contain undeclared embryos: {bad_train}")
    if bad_holdout:
        raise OrganizerBaselineError(f"holdout_datasets contain wrong embryo IDs: {bad_holdout}")
    overlap = set(train_datasets) & set(holdout_datasets)
    if overlap:
        raise OrganizerBaselineError(f"train/holdout dataset overlap: {sorted(overlap)}")

    # The organizer's training code indexes ``folds[fold]`` rather than looking
    # at a 'split' field, so materialize placeholder entries if a nonzero fold
    # index is ever requested.
    placeholder = {"split": -1, "train": list(train_datasets), "test": list(train_datasets)}
    train_payload: list[dict[str, Any]] = [dict(placeholder) for _ in range(organizer_fold_index + 1)]
    predict_payload: list[dict[str, Any]] = [dict(placeholder) for _ in range(organizer_fold_index + 1)]

    train_payload[organizer_fold_index] = {
        "split": organizer_fold_index,
        "train": list(train_datasets),
        "test": list(train_datasets),
    }
    predict_payload[organizer_fold_index] = {
        "split": organizer_fold_index,
        "train": list(train_datasets),
        "test": list(holdout_datasets),
    }

    # Final invariant: the training script's train+test universe contains no
    # held-out dataset or held-out embryo.
    training_visible = set(train_payload[organizer_fold_index]["train"]) | set(
        train_payload[organizer_fold_index]["test"]
    )
    if training_visible & set(holdout_datasets):
        raise OrganizerBaselineError("holdout dataset leaked into organizer training/monitor split")
    if any(embryo_id_from_dataset(name) == holdout_embryo for name in training_visible):
        raise OrganizerBaselineError("holdout embryo leaked into organizer training/monitor split")

    warnings = (
        "The pinned organizer trainer chooses its best checkpoint from the supplied test split.",
        "train-embryo-all uses training-embryo data for that monitor to avoid LOEO holdout leakage; it is not independent validation.",
        "The pinned augmentation path calls numpy.default_rng() without an explicit seed, so exact bitwise training reproducibility is not guaranteed.",
    )

    return OrganizerBaselineProtocol(
        fold_name=fold_name,
        holdout_embryo=holdout_embryo,
        train_embryos=train_embryos,
        train_datasets=train_datasets,
        holdout_datasets=holdout_datasets,
        organizer_fold_index=organizer_fold_index,
        train_splits=tuple(train_payload),
        predict_splits=tuple(predict_payload),
        checkpoint_monitor_policy=checkpoint_monitor_policy,
        warnings=warnings,
    )


def write_organizer_protocol(
    protocol: OrganizerBaselineProtocol,
    *,
    train_splits_path: str | Path,
    predict_splits_path: str | Path,
    protocol_path: str | Path | None = None,
) -> None:
    """Write organizer-compatible split JSON plus an optional audit record."""

    train_splits_path = Path(train_splits_path)
    predict_splits_path = Path(predict_splits_path)
    train_splits_path.parent.mkdir(parents=True, exist_ok=True)
    predict_splits_path.parent.mkdir(parents=True, exist_ok=True)
    train_splits_path.write_text(json.dumps(list(protocol.train_splits), indent=2) + "\n")
    predict_splits_path.write_text(json.dumps(list(protocol.predict_splits), indent=2) + "\n")

    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol.to_dict(), indent=2, sort_keys=True) + "\n")
