# Competition Contract

Last verified: **2026-09-04**

This document is the machine-development contract for this repository: code may rely on items marked **VERIFIED**, must not silently rely on items marked **UNRESOLVED**, and must distinguish community reports from organizer/official sources.

## 1. Sole competition target — VERIFIED

**Biohub - Cell Tracking During Development**  
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

Organizer goal: detect, track, and link cells through 3D microscopy over time, including divisions/lineage reconstruction.

Primary sources:
- Kaggle competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Official organizer repository: https://github.com/royerlab/kaggle-cell-tracking-competition

## 2. Source hierarchy

When sources disagree, use this order unless a newer official statement explicitly supersedes an older one:

1. Current Kaggle competition rules / evaluation / host clarification.
2. Current official organizer evaluator implementation.
3. Official organizer repository documentation/tests.
4. Kaggle staff or competition-host discussion posts.
5. Public competition notebooks/discussions.
6. External papers/repositories.
7. Our own inference.

Community claims never become repository assumptions merely because they are popular or leaderboard-adjacent.

## 3. Official evaluator revision — VERIFIED / PINNED FOR INSPECTION

Repository: `royerlab/kaggle-cell-tracking-competition`  
Pinned revision for Phase 0B inspection:

```text
075fc5f5a52d11077f9dc2b074644618f26939e2
```

The commit message is `Merge pull request #2 from royerlab/metrics-fix — Updating metric to patch weakly connected component exploit` (2026-07-18).

**Important:** Phase 0A does not vendor or wrap this evaluator yet. Phase 0B must inspect the exact code/tests at this revision and verify whether Kaggle has since published a newer official metric revision before implementation begins.

Pinned documentation:
- `metrics.md`: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md
- `README.md`: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/README.md

## 4. Data / graph representation — VERIFIED

From the official repository at the pinned revision:

- Images are OME-Zarr with dimensions `(T, Z, Y, X)`.
- Spatial voxel scale `(Z, Y, X)` is `(1.625, 0.40625, 0.40625)` micrometers per pixel.
- Tracks use `tracksdata` GEFF graphs.
- Nodes represent approximate cell centers `(t, z, y, x)`.
- Temporal edges link cells through time.
- A division is represented by a source node linked to two daughter nodes.
- Ground-truth annotations are sparse: only a subset of cells is annotated.

Do not hard-code these values outside a single configuration/adapter boundary until Phase 0B confirms how the evaluator obtains scale in practice.

## 5. Edge metric — VERIFIED

From the pinned official `metrics.md`:

### Node matching

- Predicted nodes and GT nodes are matched by centroid distance.
- Maximum matching distance: **7 micrometers**.
- Matching uses an optimal bipartite assignment.
- A predicted node can match at most one GT node.

### Edge TP/FN/FP semantics

- A predicted edge is TP when both endpoints match GT nodes connected by a GT edge.
- A GT edge without such a predicted match is FN.
- Because GT is sparse, not every unmatched predicted edge/node is automatically an FP.
- The official metric defines specific FP cases when one matched endpoint contradicts the annotated GT connection; other predicted edges may be ignored.

This sparse-label behavior is strategically important and must be preserved exactly by our evaluator adapter; we will not replace it with an ordinary dense precision/recall implementation.

### Edge Jaccard

```text
edge_jaccard = TP / (TP + FP + FN)
```

### Predicted-node-count adjustment

```text
adjusted_jaccard = max(
    0,
    jaccard * (1 - a * (T_pred - T_true) / T_true)
)
```

where the official documentation states `a = 0.1`, `T_pred` is total predicted nodes, and `T_true` is a provided coarse estimate of total true nodes including unannotated cells.

## 6. Division metric — VERIFIED AT DOCUMENTATION LEVEL

From the pinned official `metrics.md`:

- A GT division has exactly two outgoing edges.
- During evaluation, a predicted node with at least two outgoing edges is treated as a predicted fork.
- Division matching uses a local lineage window around the split and allows a predicted fork one timepoint before or after the annotated split when the required local topology/evidence conditions are met.
- GT divisions and predicted forks are paired by maximum-cardinality bipartite matching.
- Division Jaccard is `TP / (TP + FP + FN)`.

The patched division rules include directed local topology, distinct daughter branches, component evidence, and anti-merge constraints. We will treat the implementation/tests, not this prose summary, as authoritative in Phase 0B/0C.

## 7. Aggregation / final score — VERIFIED

The official documentation states:

- Division counts are micro-averaged across videos before Division Jaccard is computed.
- Adjusted edge Jaccard is weight-averaged by per-sample size `TP + FP + FN`.
- Final score:

```text
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

## 8. Submission/runtime constraints — VERIFIED FROM CURRENT KAGGLE OVERVIEW

Current Kaggle overview states:

- Submissions must be made through Kaggle Notebooks.
- CPU runtime: **<= 12 hours**.
- GPU runtime: **<= 12 hours**.
- Internet access must be disabled for submission execution.
- Freely/publicly available external data is allowed, including pretrained models.
- Submission file must be named `submission.csv`.
- Every test dataset must appear in the submission.

Primary source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

Additional rule details (licensing, code sharing, team/submission limits, winner obligations, private external services) must be re-read from the current Rules page before they are encoded into tooling or workflow policy.

## 9. Timeline — VERIFIED FROM CURRENT KAGGLE OVERVIEW

All listed deadlines are 11:59 PM UTC unless Kaggle states otherwise:

- Start: 2026-06-29
- Entry deadline: 2026-09-22
- Team merger deadline: 2026-09-22
- Final submission deadline: 2026-09-29

The organizer reserves the right to update the timeline. Any automation or final-submission plan must refresh this source rather than assume this snapshot remains current.

## 10. Official baseline facts — VERIFIED

The pinned official repository documents an end-to-end baseline:

- `TemporalUNet3D` for detection/features.
- Local-max suppression for cell-center recovery.
- `SimpleNodeTransformer` cross-attention linking of `(t, t+1)` node pairs.
- Sparse supervision using GT edges while unannotated/background detections are ignored for training.
- The public baseline training command used **3 epochs**.
- Organizer README explicitly says the released public UNet baseline was **not trained to convergence**.

This is a baseline fact, not a recommendation that we use this architecture.

## 11. Known official metric-patch event — VERIFIED

Kaggle host/staff announced a metric exploit and rescore; the pinned official evaluator revision above is the merge that patches the weakly-connected-component exploit. We therefore must never compare scores produced by unknown/older metric revisions as if they were directly equivalent.

## 12. UNRESOLVED — must not be silently assumed yet

The following may be true based on prior research/community/host discussion, but Phase 0A deliberately does **not** encode them as implementation assumptions until their exact current provenance is pinned:

- Exact number and identity of training embryos.
- Exact relationship/non-overlap of train and hidden-test embryo IDs.
- Public/private leaderboard split proportions.
- Max submissions per day, number of final submissions, and maximum team size.
- Any implementation behavior not explicitly checked against the pinned metric code/tests (including graph sanitization and any out-degree handling beyond the documented fork definition).
- Exact hidden-test size/distribution.
- Community-reported acquisition pathologies (frozen/repeated frames, global jumps).
- Provenance/clean-CV eligibility of public checkpoints and Kaggle notebook artifacts.

These belong in provenance/validation research and must be promoted to VERIFIED only with a primary citation.

## 13. Phase-0A acceptance rule

Phase 0A passes only if:

- repository contains no competition data or private credentials;
- every factual competition assumption in the scaffold has an identifiable primary source;
- uncertain facts are explicitly marked unresolved;
- no model architecture has been prematurely selected;
- the official evaluator revision is recorded but not reimplemented from memory.
