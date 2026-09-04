"""Utilities for reconciling HOCT adapter graphs with frozen detector node IDs."""

from __future__ import annotations

import tracksdata as td

from .hoct_compat import HOCTCompatibilityError


def candidate_edges_in_source_detection_space(
    graph: td.graph.BaseGraph,
) -> tuple[tuple[int, int], ...]:
    """Return candidate edges expressed in original fixed-detection node IDs."""

    if "source_detection_id" not in graph.node_attr_keys():
        raise HOCTCompatibilityError(
            "HOCT candidate graph lacks source_detection_id and cannot be reconciled "
            "with the frozen detector graph"
        )

    node_id = td.DEFAULT_ATTR_KEYS.NODE_ID
    source_key = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    target_key = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    nodes = graph.node_attrs(attr_keys=[node_id, "source_detection_id"])
    mapping = {
        int(node): int(source_detection)
        for node, source_detection in zip(
            nodes[node_id].to_list(), nodes["source_detection_id"].to_list(), strict=True
        )
    }
    if len(set(mapping.values())) != len(mapping):
        raise HOCTCompatibilityError("source_detection_id values are not one-to-one")

    edges = graph.edge_attrs(attr_keys=[source_key, target_key])
    result: list[tuple[int, int]] = []
    for source, target in zip(edges[source_key].to_list(), edges[target_key].to_list(), strict=True):
        source = int(source)
        target = int(target)
        if source not in mapping or target not in mapping:
            raise HOCTCompatibilityError(
                f"candidate edge references an unmapped adapter node: {(source, target)}"
            )
        result.append((mapping[source], mapping[target]))
    return tuple(result)
