"""Audited HOCT checkpoint registry for offline/reproducible experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .hoct_compat import HOCT_REVISION


class HOCTCheckpointError(ValueError):
    """Raised when a checkpoint does not match the audited public artifact."""


@dataclass(frozen=True)
class HOCTModelSpec:
    name: str
    release_url: str
    sha256: str
    hoct_revision: str = HOCT_REVISION


HOCT_MODELS: dict[str, HOCTModelSpec] = {
    "general_v1": HOCTModelSpec(
        name="general_v1",
        release_url="https://github.com/royerlab/hoct/releases/download/weights-v1/general_v1.pt",
        sha256="5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33",
    ),
    "ctc_v0": HOCTModelSpec(
        name="ctc_v0",
        release_url="https://github.com/royerlab/hoct/releases/download/weights-v0/ctc_v0.pt",
        sha256="b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000",
    ),
    "general_v0": HOCTModelSpec(
        name="general_v0",
        release_url="https://github.com/royerlab/hoct/releases/download/weights-v0/general_v0.pt",
        sha256="024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a",
    ),
}


def checkpoint_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    if not path.is_file():
        raise HOCTCheckpointError(f"HOCT checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hoct_checkpoint(path: str | Path, model_name: str) -> HOCTModelSpec:
    """Fail closed unless a local checkpoint exactly matches HOCT's registry."""

    if model_name not in HOCT_MODELS:
        raise HOCTCheckpointError(
            f"unknown audited HOCT model {model_name!r}; choose one of {sorted(HOCT_MODELS)}"
        )
    spec = HOCT_MODELS[model_name]
    actual = checkpoint_sha256(path)
    if actual != spec.sha256:
        raise HOCTCheckpointError(
            f"checkpoint SHA256 mismatch for {model_name}: {actual} != {spec.sha256}"
        )
    return spec
