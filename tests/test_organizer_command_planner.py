from pathlib import Path

import pytest

from biohub.baselines.runner import (
    OrganizerCommandError,
    OrganizerRunSettings,
    build_organizer_commands,
)


def _layout(tmp_path: Path):
    repo = tmp_path / "repo"
    scripts = repo / "vendor" / "kaggle-cell-tracking-competition" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train_unet_transformer.py").write_text("# train\n")
    (scripts / "predict_unet_transformer.py").write_text("# predict\n")
    data = tmp_path / "train"
    data.mkdir()
    train_splits = tmp_path / "train_splits.json"
    pred_splits = tmp_path / "pred_splits.json"
    train_splits.write_text("[]\n")
    pred_splits.write_text("[]\n")
    return repo, data, train_splits, pred_splits


def test_public_three_epoch_settings_are_explicit(tmp_path: Path):
    repo, data, train_splits, pred_splits = _layout(tmp_path)
    commands = build_organizer_commands(
        repo_root=repo,
        data_dir=data,
        train_splits_path=train_splits,
        predict_splits_path=pred_splits,
        method="baseline-E2-001",
        username="yj",
        python_executable="python",
    )

    train = list(commands.train)
    assert train[:2] == ["python", str(repo / "vendor/kaggle-cell-tracking-competition/scripts/train_unet_transformer.py")]
    assert train[train.index("--epochs") + 1] == "3"
    assert train[train.index("--lr") + 1] == "0.0001"
    assert train[train.index("--det-loss-weight") + 1] == "1.0"
    assert train[train.index("--downsample") + 1] == "1,4,4"
    assert "--data-parallel" in train

    predict = list(commands.predict)
    assert predict[predict.index("--det-threshold") + 1] == "0.99"
    assert predict[predict.index("--unet-batch-size") + 1] == "4"
    assert "--use-ilp" not in predict
    assert commands.weights_path.endswith("weights/baseline-E2-001/split_0/edge_predictor_best.pth")
    assert commands.predictions_dir.endswith("predictions/yj/baseline-E2-001/split_0")


def test_ilp_flags_are_only_added_when_enabled(tmp_path: Path):
    repo, data, train_splits, pred_splits = _layout(tmp_path)
    commands = build_organizer_commands(
        repo_root=repo,
        data_dir=data,
        train_splits_path=train_splits,
        predict_splits_path=pred_splits,
        method="baseline",
        username="yj",
        settings=OrganizerRunSettings(use_ilp=True),
    )
    predict = list(commands.predict)
    assert "--use-ilp" in predict
    assert predict[predict.index("--ilp-division-weight") + 1] == "1.0"


def test_unsafe_method_name_is_rejected(tmp_path: Path):
    repo, data, train_splits, pred_splits = _layout(tmp_path)
    with pytest.raises(OrganizerCommandError, match="filesystem-safe"):
        build_organizer_commands(
            repo_root=repo,
            data_dir=data,
            train_splits_path=train_splits,
            predict_splits_path=pred_splits,
            method="../escape",
            username="yj",
        )


def test_missing_pinned_submodule_scripts_fail_closed(tmp_path: Path):
    data = tmp_path / "train"
    data.mkdir()
    train_splits = tmp_path / "train.json"
    pred_splits = tmp_path / "pred.json"
    train_splits.write_text("[]")
    pred_splits.write_text("[]")
    with pytest.raises(OrganizerCommandError, match="initialize pinned submodules"):
        build_organizer_commands(
            repo_root=tmp_path / "repo",
            data_dir=data,
            train_splits_path=train_splits,
            predict_splits_path=pred_splits,
            method="baseline",
            username="yj",
        )
