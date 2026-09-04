import pytest

from biohub.analysis.comparison import ComparisonError, compare_two_direction_loeo


def _summary(score, *, edge=0.8, div=0.2, recall=0.9):
    return {
        "score": score,
        "adj_edge_jaccard": edge,
        "edge_jaccard": edge + 0.01,
        "division_jaccard": div,
        "node_recall": recall,
        "total_node_ratio": 0.0,
        "edge_tp": 80,
        "edge_fp": 10,
        "edge_fn": 10,
        "division_tp": 2,
        "division_fp": 1,
        "division_fn": 1,
    }


def test_two_direction_comparison_exposes_mean_and_tail():
    baseline = {
        "holdout_E1": _summary(0.90),
        "holdout_E2": _summary(0.91),
    }
    challenger = {
        "holdout_E1": _summary(0.92),
        "holdout_E2": _summary(0.905),
    }
    result = compare_two_direction_loeo(baseline, challenger)
    assert result.score_delta_mean == pytest.approx(0.0075)
    assert result.score_delta_worst == pytest.approx(-0.005)
    assert result.score_delta_best == pytest.approx(0.02)
    assert not result.both_score_directions_positive
    assert not result.both_score_directions_nonnegative


def test_both_direction_gain_is_explicit():
    baseline = {"a": _summary(0.90), "b": _summary(0.90)}
    challenger = {"a": _summary(0.901), "b": _summary(0.903)}
    result = compare_two_direction_loeo(baseline, challenger)
    assert result.both_score_directions_positive
    assert result.score_delta_worst == pytest.approx(0.001)


def test_mismatched_fold_sets_fail_closed():
    with pytest.raises(ComparisonError, match="fold sets differ"):
        compare_two_direction_loeo(
            {"a": _summary(0.9), "b": _summary(0.9)},
            {"a": _summary(0.9), "c": _summary(0.9)},
        )


def test_exactly_two_folds_are_required():
    with pytest.raises(ComparisonError, match="exactly two folds"):
        compare_two_direction_loeo({"a": _summary(0.9)}, {"a": _summary(0.91)})
