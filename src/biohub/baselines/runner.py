"""Command planning for the pinned organizer baseline implementation.

The planner makes every CLI-exposed quality-affecting baseline setting explicit
and computes the exact weight/prediction paths used by the organizer scripts.
It does not execute training itself, which keeps command construction unit
-testable and lets Kaggle/local launchers decide when compute is spent.
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


class OrganizerCommandError(ValueError):
    """Raised when a baseline command cannot be constructed unambiguously."""


_SAFE_METHOD = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class OrganizerRunSettings:
    """Pinned public-baseline CLI settings made explicit for provenance."""

    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 16
    num_workers: int = 8
    unet_out_channels: int = 32
    unet_layers: tuple[int, ...] = (32, 64, 128)
    downsample_zyx: tuple[int, int, int] = (1, 4, 4)
    det_loss_weight: float = 1.0
    det_neg_weight: float = 1e-2
    window_size: int = 2
    train_pool_kernel_um: float = 5.0
    data_parallel: bool = True
    pred_det_threshold: float = 0.99
    pred_unet_batch_size: int = 4
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise OrganizerCommandError("epochs/batch_size must be positive and num_workers >= 0")
        if self.lr <= 0 or self.det_loss_weight < 0 or self.det_neg_weight < 0:
            raise OrganizerCommandError("learning/loss weights must be non-negative with lr > 0")
        if len(self.downsample_zyx) != 3 or any(value <= 0 for value in self.downsample_zyx):
            raise OrganizerCommandError("downsample_zyx must contain three positive integers")
        if not self.unet_layers or any(value <= 0 for value in self.unet_layers):
            raise OrganizerCommandError("unet_layers must contain positive widths")
        if not (0.0 <= self.pred_det_threshold <= 1.0):
            raise OrganizerCommandError("pred_det_threshold must lie in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OrganizerCommands:
    train: tuple[str, ...]
    predict: tuple[str, ...]
    weights_path: str
    predictions_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_organizer_commands(
    *,
    repo_root: str | Path,
    data_dir: str | Path,
    train_splits_path: str | Path,
    predict_splits_path: str | Path,
    method: str,
    username: str,
    fold_index: int = 0,
    python_executable: str | None = None,
    settings: OrganizerRunSettings | None = None,
) -> OrganizerCommands:
    """Build explicit train/predict commands for the vendored organizer code."""

    repo_root = Path(repo_root).resolve()
    data_dir = Path(data_dir).resolve()
    train_splits_path = Path(train_splits_path).resolve()
    predict_splits_path = Path(predict_splits_path).resolve()
    method = method.strip()
    username = username.strip()
    if not method or not _SAFE_METHOD.fullmatch(method):
        raise OrganizerCommandError(
            "method must be a non-empty filesystem-safe token using only letters, digits, '.', '_' or '-'"
        )
    if not username or "/" in username or "\\" in username:
        raise OrganizerCommandError("username must be a non-empty path component")
    if fold_index < 0:
        raise OrganizerCommandError("fold_index must be >= 0")

    vendor = repo_root / "vendor" / "kaggle-cell-tracking-competition"
    train_script = vendor / "scripts" / "train_unet_transformer.py"
    predict_script = vendor / "scripts" / "predict_unet_transformer.py"
    for label, path in (("organizer train script", train_script), ("organizer predict script", predict_script)):
        if not path.is_file():
            raise OrganizerCommandError(f"{label} not found: {path}; initialize pinned submodules first")
    if not data_dir.is_dir():
        raise OrganizerCommandError(f"data directory not found: {data_dir}")
    for label, path in (("training split file", train_splits_path), ("prediction split file", predict_splits_path)):
        if not path.is_file():
            raise OrganizerCommandError(f"{label} not found: {path}")

    cfg = settings or OrganizerRunSettings()
    python = python_executable or sys.executable
    fold = str(fold_index)
    layers = ",".join(str(value) for value in cfg.unet_layers)
    downsample = ",".join(str(value) for value in cfg.downsample_zyx)

    train = [
        python,
        str(train_script),
        "--method", method,
        "--data-dir", str(data_dir),
        "--splits", str(train_splits_path),
        "--split", fold,
        "--epochs", str(cfg.epochs),
        "--lr", repr(cfg.lr),
        "--batch-size", str(cfg.batch_size),
        "--num-workers", str(cfg.num_workers),
        "--unet-out-channels", str(cfg.unet_out_channels),
        "--unet-layers", layers,
        "--downsample", downsample,
        "--det-loss-weight", repr(cfg.det_loss_weight),
        "--det-neg-weight", repr(cfg.det_neg_weight),
        "--window-size", str(cfg.window_size),
        "--pool-kernel-um", repr(cfg.train_pool_kernel_um),
    ]
    train.append("--data-parallel" if cfg.data_parallel else "--single-gpu")

    weights_path = vendor / "weights" / method / f"split_{fold_index}" / "edge_predictor_best.pth"
    predictions_dir = vendor / "predictions" / username / method / f"split_{fold_index}"

    predict = [
        python,
        str(predict_script),
        "--method", method,
        "--data-dir", str(data_dir),
        "--splits", str(predict_splits_path),
        "--split", fold,
        "--weights", str(weights_path),
        "--unet-batch-size", str(cfg.pred_unet_batch_size),
        "--det-threshold", repr(cfg.pred_det_threshold),
    ]
    if cfg.use_ilp:
        predict.extend(
            [
                "--use-ilp",
                "--ilp-edge-weight", repr(cfg.ilp_edge_weight),
                "--ilp-appearance-weight", repr(cfg.ilp_appearance_weight),
                "--ilp-disappearance-weight", repr(cfg.ilp_disappearance_weight),
                "--ilp-division-weight", repr(cfg.ilp_division_weight),
            ]
        )

    return OrganizerCommands(
        train=tuple(train),
        predict=tuple(predict),
        weights_path=str(weights_path),
        predictions_dir=str(predictions_dir),
    )
