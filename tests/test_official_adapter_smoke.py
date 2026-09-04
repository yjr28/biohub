"""Phase 0B smoke tests: prove the adapter is wired to the pinned official code."""

import pytest

from biohub.evaluation.official import (
    MAX_DISTANCE_UM,
    OFFICIAL_EVALUATOR_COMMIT,
    TRACKSDATA_COMMIT,
    EvaluationInputError,
    _resolve_names,
    assert_official_constants,
)


def test_audited_upstream_pins_and_constants() -> None:
    assert OFFICIAL_EVALUATOR_COMMIT == "075fc5f5a52d11077f9dc2b074644618f26939e2"
    assert TRACKSDATA_COMMIT == "39dccf3a243e44274759468cb31b2ad9e7fc1d09"
    assert MAX_DISTANCE_UM == 7.0
    assert_official_constants()


def test_strict_directory_resolution_fails_closed(tmp_path) -> None:
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    (pred / "sample_a.geff").touch()
    (gt / "sample_a.geff").touch()
    (gt / "sample_b.geff").touch()

    with pytest.raises(EvaluationInputError, match="sets differ"):
        _resolve_names(pred, gt, expected_names=None, strict=True)

    assert _resolve_names(
        pred,
        gt,
        expected_names=["sample_a"],
        strict=True,
    ) == ("sample_a",)
