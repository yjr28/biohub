import json
from pathlib import Path

import pytest

from biohub.experiments.leaderboard import (
    LeaderboardLedgerError,
    SubmissionObservation,
    SubmissionPlan,
    append_submission_observation,
    append_submission_plan,
)


def _plan(**overrides):
    payload = {
        "submission_id": "sub-001",
        "experiment_ids": ("exp-E1", "exp-E2"),
        "purpose": "test whether robust LOEO gain transfers to public LB",
        "expected_delta_low": -0.001,
        "expected_delta_high": 0.004,
        "decision_if_gain": "retain as viable finalist",
        "decision_if_flat": "inspect slice disagreement before changing anything",
        "decision_if_loss": "do not revert unless local evidence is also weak",
        "public_lb_is_informative_because": "tests transfer beyond the two training embryos",
        "baseline_public_score": 0.930,
    }
    payload.update(overrides)
    return SubmissionPlan(**payload)


def test_plan_must_exist_before_observation(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    obs = SubmissionObservation(submission_id="sub-001", public_score=0.932)
    with pytest.raises(LeaderboardLedgerError, match="prior plan"):
        append_submission_observation(ledger, obs)


def test_observation_derives_delta_from_precommitted_baseline(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    append_submission_plan(ledger, _plan())
    append_submission_observation(
        ledger,
        SubmissionObservation(submission_id="sub-001", public_score=0.9325, public_rank=12),
    )
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert events[0]["event_type"] == "plan"
    assert events[1]["event_type"] == "observation"
    assert events[1]["observed_delta"] == pytest.approx(0.0025)
    assert events[1]["public_rank"] == 12


def test_submission_id_is_single_use(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    append_submission_plan(ledger, _plan())
    with pytest.raises(LeaderboardLedgerError, match="already exists"):
        append_submission_plan(ledger, _plan())


def test_observation_is_single_use(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    append_submission_plan(ledger, _plan())
    obs = SubmissionObservation(submission_id="sub-001", public_score=0.931)
    append_submission_observation(ledger, obs)
    with pytest.raises(LeaderboardLedgerError, match="already recorded"):
        append_submission_observation(ledger, obs)


def test_expected_interval_cannot_be_reversed():
    with pytest.raises(LeaderboardLedgerError, match="cannot exceed"):
        _plan(expected_delta_low=0.01, expected_delta_high=0.0)
