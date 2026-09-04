import json
from pathlib import Path

import pytest
from tracking_cellmot.metrics import summarise

from biohub.evaluation.official import EvaluationRun
from biohub.evaluation.reporting import EvaluationReportError, build_report, write_report


def _row(edge_tp, edge_fp, edge_fn, *, div_tp=0, div_fp=0, div_fn=0, recall=0.9, adj=0.8):
    denom = edge_tp + edge_fp + edge_fn
    return {
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "num_pred_nodes": 100,
        "node_recall": recall,
        "total_node_ratio": 0.0,
        "edge_jaccard": edge_tp / denom,
        "adj_edge_jaccard": adj,
    }


def _run():
    rows = (
        _row(80, 10, 10, div_tp=2, div_fp=1, div_fn=1, adj=0.80),
        _row(45, 5, 50, div_tp=1, div_fp=0, div_fn=2, adj=0.45),
    )
    return EvaluationRun(
        names=("E2_easy", "E2_hard"),
        rows=rows,
        summary=summarise(list(rows)),
    )


def test_report_attaches_dataset_names_and_uses_official_group_summary():
    run = _run()
    report = build_report(
        run,
        group_by_dataset={"E2_easy": "low_motion", "E2_hard": "high_motion"},
    )
    assert report.datasets[0]["dataset"] == "E2_easy"
    assert report.overall == run.summary
    assert report.groups["low_motion"] == summarise([dict(run.rows[0])])
    assert report.groups["high_motion"] == summarise([dict(run.rows[1])])


def test_group_mapping_must_cover_exact_evaluation_set():
    with pytest.raises(EvaluationReportError, match="exactly cover"):
        build_report(_run(), group_by_dataset={"E2_easy": "only"})


def test_report_rejects_duplicate_names():
    run = _run()
    bad = EvaluationRun(names=("same", "same"), rows=run.rows, summary=run.summary)
    with pytest.raises(EvaluationReportError, match="duplicate"):
        build_report(bad)


def test_strict_json_serialization_converts_nan_to_null(tmp_path: Path):
    row = _row(1, 0, 0, adj=float("nan"))
    run = EvaluationRun(names=("E2_one",), rows=(row,), summary=summarise([row]))
    report = build_report(run)
    out = tmp_path / "report.json"
    write_report(report, out)
    payload = json.loads(out.read_text())
    assert payload["datasets"][0]["adj_edge_jaccard"] is None
    assert payload["overall"]["adj_edge_jaccard"] is None
