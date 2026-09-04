"""Holdout-safe split files for the pinned organizer baseline scripts.

The pinned organizer training script uses its ``test`` split every epoch and
selects ``edge_predictor_best.pth`` by ``test_acc * test_recall``. Therefore,
putting the LOEO holdout embryo in that script's test split would contaminate
our model-selection evidence even though the model does not backprop through the
holdout. This module emits separate training and prediction split files:

* training split: optimization and checkpoint-monitor datasets come only from
  the training embryo, never the LOEO holdout;
* prediction split: the same trained fold index but test points at the true
  held-out embryo so the organizer prediction script produces those GEFFs.

For the public 3-epoch reference we preserve ``train-embryo-all``. For longer
training, ``train-embryo-hash-holdout`` deterministically withholds a nested
monitor subset from the training embryo, giving checkpoint selection an
independent dataset set without exposing the opposite embryo.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    optimizer_datasets: tuple[str, ...]
    checkpoint_monitor_datasets: tuple[str, ...]
    holdout_datasets: tuple[str, ...]
    organizer_fold_index: int
    train_splits: tuple[dict[str, Any], ...]
    predict_splits: tuple[dict[str, Any], ...]
    checkpoint_monitor_policy: str
    monitor_fraction: float | None
    monitor_seed: int | None
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


def _hash_rank(dataset: str, seed: int) -> tuple[str, str]:
    """Stable cross-platform ordering for a nested monitor split."""

    digest = hashlib.sha256(f"{seed}:{dataset}".encode("utf-8")).hexdigest()
    return digest, dataset


def _nested_monitor_split(
    train_datasets: tuple[str, ...],
    *,
    monitor_fraction: float,
    monitor_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (optimizer, monitor) with deterministic dataset-level separation."""

    fraction = float(monitor_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise OrganizerBaselineError("monitor_fraction must be finite and strictly between 0 and 1")
    if isinstance(monitor_seed, bool) or int(monitor_seed) != monitor_seed:
        raise OrganizerBaselineError("monitor_seed must be an integer")
    seed = int(monitor_seed)
    if len(train_datasets) < 2:
        raise OrganizerBaselineError(
            "train-embryo-hash-holdout requires at least two datasets from the training embryo"
        )

    # ceil means the monitor never falls below the requested fraction, while
    # min(..., n-1) guarantees optimization still receives at least one dataset.
    n_monitor = min(len(train_datasets) - 1, max(1, math.ceil(len(train_datasets) * fraction)))
    ranked = sorted(train_datasets, key=lambda name: _hash_rank(name, seed))
    monitor = tuple(sorted(ranked[:n_monitor]))
    optimizer = tuple(sorted(set(train_datasets) - set(monitor)))
    if not optimizer or not monitor:
        raise OrganizerBaselineError("nested checkpoint split produced an empty optimizer or monitor set")
    if set(optimizer) & set(monitor):
        raise OrganizerBaselineError("nested optimizer and checkpoint-monitor sets overlap")
    if set(optimizer) | set(monitor) != set(train_datasets):
        raise OrganizerBaselineError("nested optimizer/monitor sets do not cover the training embryo")
    return optimizer, monitor


def build_organizer_protocol(
    inventory: Mapping[str, Any],
    *,
    fold_name: str,
    organizer_fold_index: int = 0,
    checkpoint_monitor_policy: str = "train-embryo-all",
    monitor_fraction: float = 0.1,
    monitor_seed: int = 0,
) -> OrganizerBaselineProtocol:
    """Create separate train/predict split payloads for a clean LOEO direction.

    Policies
    --------
    ``train-embryo-all``
        Public-reference behavior: all training-embryo datasets are used for
        both optimization and the organizer's checkpoint monitor. This is safe
        from LOEO leakage but the monitor is not independent.

    ``train-embryo-hash-holdout``
        Deterministically withhold a dataset-level monitor subset *inside the
        training embryo*. The monitor never enters optimization, and the true
        LOEO embryo remains completely invisible until external prediction and
        scoring. This is the preferred policy for longer/converged training.

    Neither monitor is the primary model-selection target. The clean evidence
    remains the later two-direction LOEO score.
    """

    if organizer_fold_index < 0:
        raise OrganizerBaselineError("organizer_fold_index must be >= 0")
    allowed_policies = {"train-embryo-all", "train-embryo-hash-holdout"}
    if checkpoint_monitor_policy not in allowed_policies:
        raise OrganizerBaselineError(
            f"Unsupported checkpoint_monitor_policy={checkpoint_monitor_policy!r}; "
            f"allowed={sorted(allowed_policies)}"
        )

    fold = _resolve_fold(inventory, fold_name)
    holdout_embryo = str(fold.get("holdout_embryo", "")).strip()
    train_embryos = tuple(
        sorted(str(value).strip() for value in fold.get("train_embryos", []) if str(value).strip())
    )
    train_datasets = tuple(
        sorted(str(value).strip() for value in fold.get("train_datasets", []) if str(value).strip())
    )
    holdout_datasets = tuple(
        sorted(str(value).strip() for value in fold.get("holdout_datasets", []) if str(value).strip())
    )

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

    if checkpoint_monitor_policy == "train-embryo-all":
        optimizer_datasets = train_datasets
        checkpoint_monitor_datasets = train_datasets
        effective_fraction = None
        effective_seed = None
        policy_warning = (
            "train-embryo-all reuses optimization datasets for checkpoint monitoring; "
            "use it for the public 3-epoch reference, not as independent validation."
        )
    else:
        optimizer_datasets, checkpoint_monitor_datasets = _nested_monitor_split(
            train_datasets,
            monitor_fraction=monitor_fraction,
            monitor_seed=monitor_seed,
        )
        effective_fraction = float(monitor_fraction)
        effective_seed = int(monitor_seed)
        policy_warning = (
            "train-embryo-hash-holdout is independent at the dataset level but still comes from the "
            "same embryo; use it only for checkpoint selection, never as the cross-embryo score."
        )

    # The organizer's training code indexes ``folds[fold]`` rather than looking
    # at a 'split' field, so materialize placeholder entries if a nonzero fold
    # index is ever requested.
    placeholder = {
        "split": -1,
        "train": list(optimizer_datasets),
        "test": list(checkpoint_monitor_datasets),
    }
    train_payload: list[dict[str, Any]] = [dict(placeholder) for _ in range(organizer_fold_index + 1)]
    predict_payload: list[dict[str, Any]] = [dict(placeholder) for _ in range(organizer_fold_index + 1)]

    train_payload[organizer_fold_index] = {
        "split": organizer_fold_index,
        "train": list(optimizer_datasets),
        "test": list(checkpoint_monitor_datasets),
    }
    predict_payload[organizer_fold_index] = {
        "split": organizer_fold_index,
        # Prediction code only consumes test, but retain the complete
        # training-embryo universe here for provenance rather than implying the
        # nested optimizer subset is the whole training fold.
        "train": list(train_datasets),
        "test": list(holdout_datasets),
    }

    training_visible = set(optimizer_datasets) | set(checkpoint_monitor_datasets)
    if training_visible & set(holdout_datasets):
        raise OrganizerBaselineError("holdout dataset leaked into organizer training/monitor split")
    if any(embryo_id_from_dataset(name) == holdout_embryo for name in training_visible):
        raise OrganizerBaselineError("holdout embryo leaked into organizer training/monitor split")
    if training_visible != set(train_datasets):
        raise OrganizerBaselineError("organizer optimizer/monitor universe differs from declared training embryo")

    warnings = (
        "The pinned organizer trainer chooses its best checkpoint from the supplied test split.",
        policy_warning,
        "The pinned augmentation path calls numpy.default_rng() without an explicit seed, so exact bitwise training reproducibility is not guaranteed.",
    )

    return OrganizerBaselineProtocol(
        fold_name=fold_name,
        holdout_embryo=holdout_embryo,
        train_embryos=train_embryos,
        train_datasets=train_datasets,
        optimizer_datasets=optimizer_datasets,
        checkpoint_monitor_datasets=checkpoint_monitor_datasets,
        holdout_datasets=holdout_datasets,
        organizer_fold_index=organizer_fold_index,
        train_splits=tuple(train_payload),
        predict_splits=tuple(predict_payload),
        checkpoint_monitor_policy=checkpoint_monitor_policy,
        monitor_fraction=effective_fraction,
        monitor_seed=effective_seed,
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
