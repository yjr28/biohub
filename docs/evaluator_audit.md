# Official Evaluator Audit

Last audited: **2026-09-04**

This document records the exact evaluator source we trust, the non-obvious scoring behaviors discovered from source/tests, and the boundary between official code and our adapter.

## 1. Upstream revisions

### Organizer evaluator

Repository: `royerlab/kaggle-cell-tracking-competition`

Pinned commit:

```text
075fc5f5a52d11077f9dc2b074644618f26939e2
```

As of this audit, that commit is still the tip of upstream `main`. Its commit message is the metric-fix merge that patches the weakly-connected-component exploit.

Key upstream blobs at this revision:

| File | Git blob SHA |
|---|---|
| `src/tracking_cellmot/metrics.py` | `e536cdc9f0877542ab227ec701ef0fdbb667189a` |
| `src/tracking_cellmot/division_metrics.py` | `9afa8630f3d7a294f9a25fca81cce1e0c7c7aeca` |
| `scripts/evaluate.py` | `b17afbbfec40a7e477d30b947ed0987625bf59ae` |
| `src/tracking_cellmot/io.py` | `a215f97b5bd0e137dd49d382aa230eb1074fcc4e` |
| `metrics.md` | `df84639194776282af9b788828ad6a9b870e19bb` |
| `tests/test_metrics.py` | `b53bc8da7ff22934e9dd6ce406d7633fc95be3dd` |
| `tests/test_division_metrics.py` | `2b64fc78ecf5a1520af539b2dccfdefddd8dc5c8` |

The upstream project is BSD-3-Clause licensed.

### `tracksdata`

The official evaluator's `pyproject.toml` depends on `tracksdata @ ...@main`, which is a moving reference. To eliminate transitive drift in our experiments, this repository pins a second submodule to the last `tracksdata/main` commit available before the evaluator merge timestamp:

```text
39dccf3a243e44274759468cb31b2ad9e7fc1d09
```

That commit was authored 2026-07-13. This is our reproducibility pin, not a claim that Kaggle's server-side evaluator used exactly this wheel. The exact server environment remains unverified. Current `tracksdata/main` has moved since then, so silently installing `main` would make experiments non-reproducible.

## 2. Integration rule

We do **not** rewrite the metric math.

- `vendor/kaggle-cell-tracking-competition` is a git submodule pinned to the organizer commit above.
- `vendor/tracksdata` is a git submodule pinned to the historical compatibility revision above.
- `biohub.evaluation.official` calls the organizer's `tracking_cellmot.metrics.evaluate`, `node_recall`, `per_sample_metrics`, and `summarise` directly.
- Our code is allowed to add stricter file-set validation and provenance checks, but it must not silently alter TP/FP/FN, node matching, division matching, or score aggregation.

## 3. Audited edge-metric behavior

### 3.1 Node matching

Verified from organizer code plus `tracksdata` matching implementation/tests:

- Matching is timepoint-aware; co-located nodes at different `t` values do not match.
- Spatial matching uses optimal one-to-one bipartite assignment.
- Candidate pairs are accepted at physical distance **<= 7.0 micrometers** by default.
- Spatial coordinates are scaled before Euclidean distance; the official dataset scale is `(z, y, x) = (1.625, 0.40625, 0.40625)` micrometers/voxel unless dataset metadata supplies another scale.
- The threshold is inclusive (`<= max_distance`).

### 3.2 Only consecutive forward edges are scored

Before edge TP/FP counting, the official evaluator keeps only predicted edges satisfying:

```text
t_target - t_source == 1
```

Therefore same-frame edges, backward edges, and skip edges with `dt > 1` are dropped from the edge metric. They may still matter elsewhere (notably raw graph topology/division logic), so this is not permission to emit invalid graphs casually.

### 3.3 Sparse-GT FP semantics

A predicted edge is evaluable when the source matches a GT node known to have an outgoing GT edge **or** the target matches a GT node known to have an incoming GT edge. Other unmatched/background edges can be invisible to edge TP/FP accounting because the GT is sparse.

Implication: ordinary dense edge precision is not the competition metric. Our diagnostics must distinguish `metric-visible FP` from graph edges that are ignored by the sparse evaluator.

### 3.4 Duplicate and merge guards

The patched evaluator contains explicit anti-inflation guards:

- Exact duplicate `(source, target)` prediction edges are deduplicated, preferring a matched copy if duplicate rows disagree.
- If multiple predicted edges collapse onto the same matched GT edge because several predictions match the same GT nodes, only the lowest predicted edge ID is kept for that matched GT edge.

### 3.5 Out-degree cap in the **edge** metric

For edge scoring, a source with more than two outgoing predicted edges is capped to two. The two lowest edge IDs are kept; the cap is blind to correctness/confidence.

This ordering dependence is important. Candidate graph construction must never assume a third edge will merely be an additional harmless option.

### 3.6 Empty graphs

The organizer evaluator returns zero edge score when the predicted graph has zero nodes or zero edges, with a warning. `evaluate()` still returns explicit TP/FP/FN counts and calls the division evaluator.

## 4. Node-count adjustment

Per sample:

```text
ratio = (N_pred - N_total_estimate) / N_total_estimate
J_adj = max(0, J_edge * (1 - 0.1 * ratio))
```

`N_total_estimate` is read from GT GEFF metadata key `estimated_number_of_nodes` by the official evaluation script.

Two non-obvious consequences are source-level facts:

1. Extra unmatched predictions can be invisible to edge FP counting but still hurt through `N_pred`.
2. There is no upper clamp on the multiplicative adjustment. If `N_pred < N_total_estimate`, the factor is greater than 1. We will characterize this with tests rather than assume the adjusted metric is bounded by the raw edge Jaccard.

We do **not** treat this as an exploit recommendation; it is a metric property that affects calibration experiments.

## 5. Run-level aggregation

The final competition-style aggregation is **not** `tracking_cellmot.metrics.evaluate_datasets()`.

For competition-style scoring, the official script performs:

```text
evaluate -> node_recall -> per_sample_metrics -> summarise
```

`summary = summarise(rows)` has these semantics:

- raw edge Jaccard: TP/FP/FN micro-averaged across valid samples;
- adjusted edge Jaccard: each sample's adjusted edge Jaccard is weighted by `edge_tp + edge_fp + edge_fn`;
- division Jaccard: division TP/FP/FN micro-averaged across samples;
- final score: `adjusted_edge_jaccard + 0.1 * division_jaccard`;
- if no divisions exist anywhere in the evaluated split, the division term is dropped.

`evaluate_datasets()` computes a different convenience summary without the node-count adjustment and must not be used as our private-LB proxy.

## 6. Division evaluator behavior

Verified from `division_metrics.py` and its tests:

- Any predicted node with raw graph out-degree >= 2 is a candidate predicted fork.
- Division evaluation matches prediction nodes independently against each GT division's local window.
- The GT local window contains the divider, its immediate predecessor, children, and grandchildren.
- A candidate must connect parent-side evidence to two distinct GT daughter lineages through two distinct predicted child lineages.
- A predicted fork can be one frame early or late when the required local directed topology is satisfied.
- Candidate GT divisions and predicted forks are paired by maximum-cardinality bipartite matching, so one predicted fork can recover at most one GT division.
- Cross-GT-component branch evidence and locally merged/shared branches can make a predicted fork a division FP.
- Division FP categories are unioned by fork ID, so one bad fork is counted once even if several rejection rules apply.

Important asymmetry: the **edge** metric caps out-degree to two before edge counting, but the division metric examines the raw graph and treats `out_degree >= 2` as a fork. A 3+-child node is therefore not equivalent to a clean 2-child division.

## 7. Official CLI behavior we intentionally make stricter

The organizer `scripts/evaluate.py` evaluates only dataset names present in the intersection of prediction and GT directories and skips unreadable pairs after printing an error.

That is convenient for development but dangerous for model selection because an accidentally missing/hard sample could disappear from a local score. Our adapter therefore supports strict expected-name validation and will fail closed by default when the requested evaluation set is incomplete.

This is an infrastructure safety check, not a scoring change.

## 8. Phase 0B conclusions

1. The correct score path is pinned and understood.
2. The evaluator has several order/topology behaviors that make a naive reimplementation unsafe.
3. We should call official functions directly and add only strict orchestration around them.
4. The moving `tracksdata@main` dependency is a reproducibility risk, so our experiments pin a historical revision and CI must verify compatibility.
5. Phase 0C must characterize, at minimum: 7-um boundary, anisotropic scaling, timepoint matching, `dt=1` filtering, sparse-GT ignored edges, duplicate-edge guard, out-degree cap/order dependence, node-count adjustment, and division timing/topology.
