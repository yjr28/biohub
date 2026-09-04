from pathlib import Path

import pytest

from biohub.trackers import (
    HOCT_MODELS,
    HOCT_REVISION,
    HOCTCheckpointError,
    checkpoint_sha256,
    verify_hoct_checkpoint,
)


def test_public_checkpoint_registry_is_pinned_to_audited_hoct_revision():
    assert set(HOCT_MODELS) == {"general_v1", "ctc_v0", "general_v0"}
    assert all(spec.hoct_revision == HOCT_REVISION for spec in HOCT_MODELS.values())
    assert HOCT_MODELS["general_v1"].sha256 == "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"
    assert HOCT_MODELS["ctc_v0"].sha256 == "b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000"


def test_checkpoint_sha256_is_deterministic(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"hello\n")
    assert checkpoint_sha256(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_verification_fails_closed_on_unknown_or_modified_checkpoint(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not a real HOCT checkpoint")
    with pytest.raises(HOCTCheckpointError, match="unknown audited"):
        verify_hoct_checkpoint(path, "mystery")
    with pytest.raises(HOCTCheckpointError, match="SHA256 mismatch"):
        verify_hoct_checkpoint(path, "general_v1")


def test_checksum_requires_local_file(tmp_path: Path):
    with pytest.raises(HOCTCheckpointError, match="does not exist"):
        checkpoint_sha256(tmp_path / "missing.pt")
