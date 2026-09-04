"""Validation split construction for the Biohub competition.

The competition host has explicitly stated that the public training set has
exactly two embryo IDs and that hidden-test embryo IDs do not overlap training.
Dataset names use `<embryo_id>_<crop_id>`; the host clarified that only the
prefix before the first underscore is the embryo ID.

Primary Kaggle discussions (host comments):
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723694
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class ValidationContractError(ValueError):
    """Raised when dataset IDs violate the host-verified validation contract."""


@dataclass(frozen=True)
class LOEOFold:
    """One leave-one-embryo-out fold."""

    name: str
    train_embryos: tuple[str, ...]
    holdout_embryo: str
    train_datasets: tuple[str, ...]
    holdout_datasets: tuple[str, ...]


def embryo_id_from_dataset(dataset_name: str) -> str:
    """Return the host-defined embryo prefix from `<embryo>_<crop>` dataset ID."""

    name = dataset_name.strip()
    if not name:
        raise ValidationContractError("dataset name cannot be empty")
    if "_" not in name:
        raise ValidationContractError(
            f"Dataset {dataset_name!r} has no '_' separator; cannot derive embryo ID "
            "without violating the host-defined naming contract."
        )
    embryo, crop = name.split("_", 1)
    if not embryo or not crop:
        raise ValidationContractError(f"Malformed dataset ID: {dataset_name!r}")
    return embryo


def group_by_embryo(dataset_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Group unique dataset IDs by host-defined embryo ID."""

    unique = sorted(set(dataset_names))
    if not unique:
        raise ValidationContractError("cannot build validation folds from an empty dataset set")

    grouped: dict[str, list[str]] = {}
    for name in unique:
        grouped.setdefault(embryo_id_from_dataset(name), []).append(name)
    return {embryo: tuple(names) for embryo, names in sorted(grouped.items())}


def build_loeo_folds(
    dataset_names: Iterable[str],
    *,
    expected_embryo_count: int = 2,
) -> tuple[LOEOFold, ...]:
    """Build leave-one-embryo-out folds and fail closed on unexpected grouping.

    The default expectation of two embryos is not an inferred property: it is
    pinned to the competition host's public clarification. If the downloaded
    competition data changes, this function intentionally fails rather than
    silently constructing a different validation regime.
    """

    grouped = group_by_embryo(dataset_names)
    embryos = tuple(grouped)
    if len(embryos) != expected_embryo_count:
        raise ValidationContractError(
            f"Expected {expected_embryo_count} training embryos from the host clarification, "
            f"but found {len(embryos)}: {embryos}. Re-audit competition data/source before proceeding."
        )

    folds: list[LOEOFold] = []
    all_names = set().union(*(set(names) for names in grouped.values()))
    for holdout in embryos:
        holdout_names = grouped[holdout]
        train_embryos = tuple(embryo for embryo in embryos if embryo != holdout)
        train_names = tuple(sorted(all_names - set(holdout_names)))
        folds.append(
            LOEOFold(
                name=f"holdout_{holdout}",
                train_embryos=train_embryos,
                holdout_embryo=holdout,
                train_datasets=train_names,
                holdout_datasets=holdout_names,
            )
        )

    return tuple(folds)
