"""Fail-closed experiment provenance for private-leaderboard model selection.

The registry is intentionally boring: JSON/JSONL plus strict validation. The
purpose is to make it impossible to later confuse a contaminated validation run
with a clean leave-one-embryo-out result or to lose the exact code/evaluator/data
state that produced a score.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from biohub.evaluation.official import OFFICIAL_EVALUATOR_COMMIT, TRACKSDATA_COMMIT


class ExperimentContractError(ValueError):
    """Raised when an experiment cannot satisfy the reproducibility contract."""


def _nonempty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ExperimentContractError(f"{field_name} cannot be empty")
    return value


def _clean_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(sorted({_nonempty(str(v), field_name) for v in values}))
    if not result:
        raise ExperimentContractError(f"{field_name} cannot be empty")
    return result


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest for a file without loading it at once."""

    path = Path(path)
    if not path.is_file():
        raise ExperimentContractError(f"Cannot fingerprint missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    value = value.strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ExperimentContractError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return value


def _validate_git_sha(value: str, field_name: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ExperimentContractError(f"{field_name} must be a full 40-character git SHA")
    return value


@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable description of one train/evaluate attempt before its result exists."""

    experiment_id: str
    hypothesis: str
    git_commit: str
    inventory_sha256: str
    fold_name: str
    train_embryos: tuple[str, ...]
    validation_embryos: tuple[str, ...]
    train_datasets: tuple[str, ...]
    validation_datasets: tuple[str, ...]
    config: Mapping[str, Any]
    seeds: tuple[int, ...]
    leakage_controls: tuple[str, ...]
    stochastic_control: str = "controlled"
    evaluator_commit: str = OFFICIAL_EVALUATOR_COMMIT
    tracksdata_commit: str = TRACKSDATA_COMMIT
    parent_experiment_id: str | None = None
    notes: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _nonempty(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "hypothesis", _nonempty(self.hypothesis, "hypothesis"))
        object.__setattr__(self, "fold_name", _nonempty(self.fold_name, "fold_name"))
        object.__setattr__(self, "git_commit", _validate_git_sha(self.git_commit, "git_commit"))
        object.__setattr__(self, "inventory_sha256", _validate_sha256(self.inventory_sha256, "inventory_sha256"))
        object.__setattr__(self, "evaluator_commit", _validate_git_sha(self.evaluator_commit, "evaluator_commit"))
        object.__setattr__(self, "tracksdata_commit", _validate_git_sha(self.tracksdata_commit, "tracksdata_commit"))

        train_embryos = _clean_tuple(self.train_embryos, "train_embryos")
        validation_embryos = _clean_tuple(self.validation_embryos, "validation_embryos")
        train_datasets = _clean_tuple(self.train_datasets, "train_datasets")
        validation_datasets = _clean_tuple(self.validation_datasets, "validation_datasets")
        leakage_controls = _clean_tuple(self.leakage_controls, "leakage_controls")
        seeds = tuple(sorted({int(seed) for seed in self.seeds}))
        stochastic_control = self.stochastic_control.strip().lower()
        if stochastic_control not in {"controlled", "partial", "uncontrolled"}:
            raise ExperimentContractError(
                "stochastic_control must be one of: controlled, partial, uncontrolled"
            )
        if stochastic_control == "controlled" and not seeds:
            raise ExperimentContractError(
                "controlled stochastic runs must record at least one seed"
            )

        if set(train_embryos) & set(validation_embryos):
            raise ExperimentContractError("train_embryos and validation_embryos must be disjoint")
        if set(train_datasets) & set(validation_datasets):
            raise ExperimentContractError("train_datasets and validation_datasets must be disjoint")
        if len(validation_embryos) != 1:
            raise ExperimentContractError(
                "A clean Biohub LOEO manifest must hold out exactly one embryo; "
                f"got {validation_embryos}"
            )
        if set(validation_embryos) & {name.split("_", 1)[0] for name in train_datasets}:
            raise ExperimentContractError("held-out embryo appears in train_datasets")
        if any(name.split("_", 1)[0] not in validation_embryos for name in validation_datasets):
            raise ExperimentContractError("validation_datasets contain a non-held-out embryo")
        if any(name.split("_", 1)[0] not in train_embryos for name in train_datasets):
            raise ExperimentContractError("train_datasets contain an embryo not declared in train_embryos")
        if self.evaluator_commit != OFFICIAL_EVALUATOR_COMMIT:
            raise ExperimentContractError(
                "evaluator_commit differs from the repository's pinned official evaluator; "
                "re-audit before registering this run"
            )
        if self.tracksdata_commit != TRACKSDATA_COMMIT:
            raise ExperimentContractError(
                "tracksdata_commit differs from the repository compatibility pin; "
                "re-audit before registering this run"
            )
        if not isinstance(self.config, Mapping):
            raise ExperimentContractError("config must be a mapping")

        object.__setattr__(self, "train_embryos", train_embryos)
        object.__setattr__(self, "validation_embryos", validation_embryos)
        object.__setattr__(self, "train_datasets", train_datasets)
        object.__setattr__(self, "validation_datasets", validation_datasets)
        object.__setattr__(self, "leakage_controls", leakage_controls)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "stochastic_control", stochastic_control)
        object.__setattr__(self, "config", dict(self.config))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """Result record linked to one immutable manifest."""

    experiment_id: str
    status: str
    summary: Mapping[str, Any]
    report_path: str | None = None
    runtime_seconds: float | None = None
    notes: str = ""
    completed_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _nonempty(self.experiment_id, "experiment_id"))
        status = self.status.strip().lower()
        if status not in {"success", "failed", "aborted"}:
            raise ExperimentContractError("status must be one of: success, failed, aborted")
        object.__setattr__(self, "status", status)
        if not isinstance(self.summary, Mapping):
            raise ExperimentContractError("summary must be a mapping")
        object.__setattr__(self, "summary", dict(self.summary))
        if self.runtime_seconds is not None:
            runtime = float(self.runtime_seconds)
            if not math.isfinite(runtime) or runtime < 0:
                raise ExperimentContractError("runtime_seconds must be finite and >= 0")
            object.__setattr__(self, "runtime_seconds", runtime)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifests(path: str | Path) -> tuple[ExperimentManifest, ...]:
    """Load an append-only JSONL manifest registry."""

    path = Path(path)
    if not path.exists():
        return ()
    manifests: list[ExperimentManifest] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                manifests.append(ExperimentManifest(**payload))
            except Exception as exc:
                raise ExperimentContractError(f"Invalid manifest on {path}:{line_number}: {exc}") from exc
    ids = [manifest.experiment_id for manifest in manifests]
    if len(ids) != len(set(ids)):
        raise ExperimentContractError(f"Duplicate experiment_id found in registry {path}")
    return tuple(manifests)


def append_manifest(path: str | Path, manifest: ExperimentManifest) -> None:
    """Append a manifest iff its experiment ID has never been registered."""

    path = Path(path)
    existing = load_manifests(path)
    if manifest.experiment_id in {item.experiment_id for item in existing}:
        raise ExperimentContractError(f"experiment_id already registered: {manifest.experiment_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(manifest.to_dict(), sort_keys=True, allow_nan=False) + "\n")


def write_result(path: str | Path, result: ExperimentResult) -> None:
    """Write one result atomically enough for notebook/local workflows."""

    path = Path(path)
    if path.exists():
        raise ExperimentContractError(f"Refusing to overwrite existing experiment result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload)
    temp.replace(path)
