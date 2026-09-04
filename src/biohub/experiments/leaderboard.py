"""Append-only public-leaderboard information ledger.

A Kaggle submission is treated as an information-budget expenditure. The plan
must be recorded before the observed score. Observations are separate events,
so hindsight cannot rewrite the pre-submission expectation or decision rule.
Store the ledger under the gitignored ``leaderboard/`` directory while the
competition is active.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LeaderboardLedgerError(ValueError):
    """Raised when submission provenance or information accounting is invalid."""


def _text(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise LeaderboardLedgerError(f"{name} cannot be empty")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise LeaderboardLedgerError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class SubmissionPlan:
    submission_id: str
    experiment_ids: tuple[str, ...]
    purpose: str
    expected_delta_low: float
    expected_delta_high: float
    decision_if_gain: str
    decision_if_flat: str
    decision_if_loss: str
    public_lb_is_informative_because: str
    baseline_public_score: float | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = field(default="plan", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "submission_id", _text(self.submission_id, "submission_id"))
        ids = tuple(sorted({_text(value, "experiment_ids") for value in self.experiment_ids}))
        if not ids:
            raise LeaderboardLedgerError("experiment_ids cannot be empty")
        object.__setattr__(self, "experiment_ids", ids)
        for name in (
            "purpose",
            "decision_if_gain",
            "decision_if_flat",
            "decision_if_loss",
            "public_lb_is_informative_because",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        low = _finite(self.expected_delta_low, "expected_delta_low")
        high = _finite(self.expected_delta_high, "expected_delta_high")
        if low > high:
            raise LeaderboardLedgerError("expected_delta_low cannot exceed expected_delta_high")
        object.__setattr__(self, "expected_delta_low", low)
        object.__setattr__(self, "expected_delta_high", high)
        if self.baseline_public_score is not None:
            object.__setattr__(
                self,
                "baseline_public_score",
                _finite(self.baseline_public_score, "baseline_public_score"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubmissionObservation:
    submission_id: str
    public_score: float
    public_rank: int | None = None
    observed_delta: float | None = None
    interpretation: str = "recorded; causal interpretation pending local evidence review"
    observed_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = field(default="observation", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "submission_id", _text(self.submission_id, "submission_id"))
        object.__setattr__(self, "public_score", _finite(self.public_score, "public_score"))
        if self.public_rank is not None and int(self.public_rank) <= 0:
            raise LeaderboardLedgerError("public_rank must be positive")
        if self.public_rank is not None:
            object.__setattr__(self, "public_rank", int(self.public_rank))
        if self.observed_delta is not None:
            object.__setattr__(
                self,
                "observed_delta",
                _finite(self.observed_delta, "observed_delta"),
            )
        object.__setattr__(self, "interpretation", _text(self.interpretation, "interpretation"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LeaderboardLedgerError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(event, dict) or event.get("event_type") not in {"plan", "observation"}:
                raise LeaderboardLedgerError(f"invalid ledger event at {path}:{line_number}")
            events.append(event)
    return events


def append_submission_plan(path: str | Path, plan: SubmissionPlan) -> None:
    path = Path(path)
    events = _events(path)
    if any(event.get("submission_id") == plan.submission_id for event in events):
        raise LeaderboardLedgerError(f"submission_id already exists: {plan.submission_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False) + "\n")


def append_submission_observation(path: str | Path, observation: SubmissionObservation) -> None:
    path = Path(path)
    events = _events(path)
    plans = [
        event for event in events
        if event.get("event_type") == "plan" and event.get("submission_id") == observation.submission_id
    ]
    prior = [
        event for event in events
        if event.get("event_type") == "observation" and event.get("submission_id") == observation.submission_id
    ]
    if len(plans) != 1:
        raise LeaderboardLedgerError(
            f"observation requires exactly one prior plan for {observation.submission_id}; found {len(plans)}"
        )
    if prior:
        raise LeaderboardLedgerError(f"observation already recorded: {observation.submission_id}")

    if observation.observed_delta is None:
        baseline = plans[0].get("baseline_public_score")
        if baseline is not None:
            payload = observation.to_dict()
            payload["observed_delta"] = observation.public_score - float(baseline)
        else:
            payload = observation.to_dict()
    else:
        payload = observation.to_dict()

    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
