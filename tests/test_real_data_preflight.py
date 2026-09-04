from dataclasses import asdict

import pytest

from biohub.data.inventory import DatasetRecord, InventoryReport
from biohub.data.preflight import RealDataGateError, validate_real_inventory
from biohub.data.validation import build_loeo_folds


def _record(dataset: str, embryo: str, *, scale=(1.625, 0.40625, 0.40625), estimate=1000.0):
    return DatasetRecord(
        split="train",
        dataset=dataset,
        embryo_id=embryo,
        image_shape_tzyx=(10, 8, 16, 16),
        scale_zyx_um=scale,
        has_geff=True,
        gt_nodes=100,
        gt_edges=90,
        gt_divisions=3,
        gt_t_min=0,
        gt_t_max=9,
        estimated_total_nodes=estimate,
    )


def _test_record(dataset: str):
    return DatasetRecord(
        split="visible_test",
        dataset=dataset,
        embryo_id=dataset.split("_", 1)[0],
        image_shape_tzyx=(10, 8, 16, 16),
        scale_zyx_um=(1.625, 0.40625, 0.40625),
        has_geff=False,
    )


def _report(*, train=None, overlap=()):
    train = tuple(train or (_record("E1_a", "E1"), _record("E1_b", "E1"), _record("E2_a", "E2")))
    names = tuple(record.dataset for record in train)
    embryos = tuple(sorted({record.embryo_id for record in train}))
    # For deliberately-invalid inventory fixtures (e.g. three embryos), do not
    # ask the stricter LOEO constructor to fail before the real-data gate itself
    # is exercised.  The gate checks embryo cardinality before inspecting folds.
    folds = tuple(asdict(fold) for fold in build_loeo_folds(names)) if len(embryos) == 2 else ()
    return InventoryReport(
        competition_root="/kaggle/input/competitions/biohub-cell-tracking-during-development",
        train=train,
        visible_test=(_test_record("TEST_0"),),
        train_embryos=embryos,
        visible_test_embryos=("TEST",),
        train_visible_test_name_overlap=tuple(overlap),
        loeo_folds=folds,
    )


def test_accepts_exact_two_embryo_inventory_and_reports_totals():
    gate = validate_real_inventory(_report())
    assert gate.accepted
    assert gate.train_embryos == ("E1", "E2")
    assert gate.dataset_count_by_embryo == {"E1": 2, "E2": 1}
    assert gate.gt_nodes_by_embryo == {"E1": 200, "E2": 100}
    assert gate.gt_edges_by_embryo == {"E1": 180, "E2": 90}
    assert gate.gt_divisions_by_embryo == {"E1": 6, "E2": 3}
    assert len(gate.loeo_fold_names) == 2


def test_rejects_more_than_two_training_embryos():
    train = (
        _record("E1_a", "E1"),
        _record("E2_a", "E2"),
        _record("E3_a", "E3"),
    )
    with pytest.raises(RealDataGateError, match="exactly two"):
        validate_real_inventory(_report(train=train))


def test_rejects_missing_estimated_node_count():
    train = (_record("E1_a", "E1", estimate=None), _record("E2_a", "E2"))
    with pytest.raises(RealDataGateError, match="estimated_number_of_nodes"):
        validate_real_inventory(_report(train=train))


def test_rejects_train_visible_test_name_overlap():
    with pytest.raises(RealDataGateError, match="name overlap"):
        validate_real_inventory(_report(overlap=("E1_a",)))


def test_warns_if_coarse_total_is_below_sparse_gt_count():
    gate = validate_real_inventory(
        _report(train=(_record("E1_a", "E1", estimate=50), _record("E2_a", "E2")))
    )
    assert any("below annotated GT nodes" in warning for warning in gate.warnings)


def test_multiple_physical_scales_are_preserved_but_flagged():
    gate = validate_real_inventory(
        _report(
            train=(
                _record("E1_a", "E1"),
                _record("E2_a", "E2", scale=(2.0, 0.5, 0.5)),
            )
        )
    )
    assert any("multiple physical scales" in warning for warning in gate.warnings)
