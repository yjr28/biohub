"""Structured reports around official Biohub metric rows.

All score aggregation delegates to the organizer's pinned ``summarise``
function.  This module only attaches dataset identities, groups rows into
pre-declared diagnostic slices, validates coverage, and serializes results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tracking_cellmot.metrics import summarise as _official_summarise

from .official import EvaluationRun, assert_official_constants


class EvaluationReportError(ValueError):
    """Raised when report metadata does not exactly cover an evaluation run."""


@dataclass(frozen=True)
class EvaluationReport:
    """Portable representation of one exact official-metric evaluation."""

    datasets: tuple[dict[str, Any], ...]
    overall: dict[str, Any]
    groups: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": [dict(row) for row in self.datasets],
            "overall": dict(self.overall),
            "groups": {name: dict(summary) for name, summary in self.groups.items()},
        }


def _validate_run(run: EvaluationRun) -> None:
    if not run.names:
        raise EvaluationReportError("cannot report an empty EvaluationRun")
    if len(run.names) != len(run.rows):
        raise EvaluationReportError(
            f"EvaluationRun names/rows length mismatch: {len(run.names)} != {len(run.rows)}"
        )
    if len(set(run.names)) != len(run.names):
        raise EvaluationReportError("EvaluationRun contains duplicate dataset names")


def build_report(
    run: EvaluationRun,
    *,
    group_by_dataset: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """Attach names and optionally aggregate exact diagnostic groups.

    ``group_by_dataset`` must cover the evaluated dataset set exactly. This
    prevents a slice summary from silently omitting a hard movie. The group
    summaries themselves call the organizer's pinned ``summarise`` function.
    """

    assert_official_constants()
    _validate_run(run)
    named_rows = tuple({"dataset": name, **dict(row)} for name, row in zip(run.names, run.rows))

    groups: dict[str, dict[str, Any]] = {}
    if group_by_dataset is not None:
        expected = set(run.names)
        supplied = set(group_by_dataset)
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            raise EvaluationReportError(
                "group_by_dataset must exactly cover evaluated datasets: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        bucketed: dict[str, list[dict[str, Any]]] = {}
        for name, row in zip(run.names, run.rows):
            group = str(group_by_dataset[name]).strip()
            if not group:
                raise EvaluationReportError(f"empty group label for dataset {name}")
            bucketed.setdefault(group, []).append(dict(row))
        for group, rows in sorted(bucketed.items()):
            groups[group] = _official_summarise(rows)

    return EvaluationReport(datasets=named_rows, overall=dict(run.summary), groups=groups)


def _portable(value: Any) -> Any:
    """Convert tuples/non-finite floats into strict JSON-compatible values."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def write_report(report: EvaluationReport, path: str | Path) -> None:
    """Write strict JSON, replacing metric NaN/Inf sentinels with ``null``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _portable(report.to_dict())
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)
