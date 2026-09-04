"""Tests for the host-verified embryo grouping and LOEO split contract."""

import pytest

from biohub.data.validation import (
    ValidationContractError,
    build_loeo_folds,
    embryo_id_from_dataset,
)


def test_embryo_id_is_prefix_before_first_underscore() -> None:
    assert embryo_id_from_dataset("44b6_0113de3b") == "44b6"
    assert embryo_id_from_dataset("6bba_05db0fb1") == "6bba"


def test_malformed_dataset_id_fails_closed() -> None:
    with pytest.raises(ValidationContractError):
        embryo_id_from_dataset("44b6")
    with pytest.raises(ValidationContractError):
        embryo_id_from_dataset("")


def test_two_embryos_produce_exact_leave_one_embryo_out_folds() -> None:
    names = ["aaaa_crop1", "aaaa_crop2", "bbbb_crop1", "bbbb_crop2", "bbbb_crop3"]
    folds = build_loeo_folds(names)
    assert len(folds) == 2

    all_names = set(names)
    holdouts = {fold.holdout_embryo for fold in folds}
    assert holdouts == {"aaaa", "bbbb"}

    for fold in folds:
        train = set(fold.train_datasets)
        holdout = set(fold.holdout_datasets)
        assert train.isdisjoint(holdout)
        assert train | holdout == all_names
        assert {embryo_id_from_dataset(name) for name in holdout} == {fold.holdout_embryo}
        assert fold.holdout_embryo not in fold.train_embryos


def test_unexpected_embryo_count_requires_reaudit() -> None:
    with pytest.raises(ValidationContractError, match="Expected 2 training embryos"):
        build_loeo_folds(["aaaa_crop1"])
    with pytest.raises(ValidationContractError, match="Expected 2 training embryos"):
        build_loeo_folds(["aaaa_crop1", "bbbb_crop1", "cccc_crop1"])
