"""Bottleneck, oracle, and cross-embryo comparison analyses."""

from .comparison import (
    ComparisonError,
    FoldComparison,
    PairedLOEOComparison,
    compare_fold,
    compare_two_direction_loeo,
)
from .oracles import (
    BottleneckDecomposition,
    CandidateCoverage,
    FixedDetectionOracleContext,
    OracleAnalysisError,
    decompose_fixed_detections,
    prepare_fixed_detection_oracle,
)

__all__ = [
    "BottleneckDecomposition",
    "CandidateCoverage",
    "ComparisonError",
    "FixedDetectionOracleContext",
    "FoldComparison",
    "OracleAnalysisError",
    "PairedLOEOComparison",
    "compare_fold",
    "compare_two_direction_loeo",
    "decompose_fixed_detections",
    "prepare_fixed_detection_oracle",
]
