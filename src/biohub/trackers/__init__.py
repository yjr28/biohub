"""Tracker adapters and compatibility layers."""

from .hoct_analysis import candidate_edges_in_source_detection_space
from .hoct_compat import (
    HOCT_POINT_API_IMPLEMENTED,
    HOCT_REVISION,
    HOCTPointGraphConfig,
    build_hoct_point_graph,
)
from .hoct_models import (
    HOCT_MODELS,
    HOCTCheckpointError,
    HOCTModelSpec,
    checkpoint_sha256,
    verify_hoct_checkpoint,
)
from .hoct_sweep import (
    HOCTCandidateSweepError,
    HOCTCandidateSweepReport,
    HOCTCandidateTrial,
    aggregate_candidate_sweep_reports,
    candidate_config_id,
    evaluate_hoct_candidate_configs,
    expand_candidate_grid,
    pareto_frontier,
)

__all__ = [
    "HOCT_MODELS",
    "HOCT_POINT_API_IMPLEMENTED",
    "HOCT_REVISION",
    "HOCTCandidateSweepError",
    "HOCTCandidateSweepReport",
    "HOCTCandidateTrial",
    "HOCTCheckpointError",
    "HOCTModelSpec",
    "HOCTPointGraphConfig",
    "aggregate_candidate_sweep_reports",
    "build_hoct_point_graph",
    "candidate_config_id",
    "candidate_edges_in_source_detection_space",
    "checkpoint_sha256",
    "evaluate_hoct_candidate_configs",
    "expand_candidate_grid",
    "pareto_frontier",
    "verify_hoct_checkpoint",
]
