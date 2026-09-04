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

## 3. Official evaluator revision — VERIFIED / PINNED

Repository: `royerlab/kaggle-cell-tracking-competition`

Pinned revision:

```text
075fc5f5a52d11077f9dc2b074644618f26939e2
```

As audited on 2026-09-04, this remained the tip of organizer `main`. The commit is the merge that patches the weakly-connected-component metric exploit.

The exact source is vendored as a git submodule at `vendor/kaggle-cell-tracking-competition`. We call its scoring functions directly; we do not maintain an independent metric implementation.

See:
- `docs/evaluator_audit.md`
- `docs/metric_characterization.md`
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md

## 4. Data / graph representation — VERIFIED

From the official repository at the pinned revision:

- Images are OME-Zarr with dimensions `(T, Z, Y, X)`.
- Spatial voxel scale `(Z, Y, X)` is `(1.625, 0.40625, 0.40625)` micrometers per pixel when not overridden by dataset OME metadata.
- Tracks use `tracksdata` GEFF graphs.
- Nodes represent approximate cell centers `(t, z, y, x)`.
- Temporal edges link cells through time.
- A division is represented by a source node linked to two daughter nodes.
- Ground-truth annotations are sparse: only a subset of cells is annotated.

Metric-adjacent code must use physical scale, not assume isotropic voxels.

## 5. Dataset identity and hidden-domain split — VERIFIED BY COMPETITION HOST

Two host clarifications are now implementation-level facts:

1. Dataset IDs use `<embryo_id>_<crop_id>`. The **embryo ID is only the prefix before the first underscore**; the remainder is the crop ID.
   - Host source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723694
2. The training set contains **exactly two unique embryo IDs**. The host states that the hidden test set has **no embryo-ID overlap with training** and is roughly similar in size.
   - Host source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793

### Validation consequence — VERIFIED DESIGN CONTRACT

Primary model-selection validation must preserve embryo identity. The repository therefore builds **leave-one-embryo-out (LOEO)** folds from the two host-defined train embryos rather than random frame/edge splits.

Because there are only two embryos, LOEO is inherently high-variance; stress slices and secondary diagnostics may supplement it, but they may not mix the held-out embryo back into training/calibration and then be described as clean cross-embryo validation.

## 6. Visible test folder vs hidden scoring set — VERIFIED BY COMPETITION HOST

The four publicly visible `test/` clips are **dummy placeholder files** used to verify that a submission notebook produces a CSV. The host states that the real scoring set is a substantially larger private hidden set and has no overlap with the public training set.

Host source:
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062

Consequences:

- never use the visible test clips as evidence of unseen-embryo generalization;
- an overlap between visible-test and train dataset names is expected placeholder behavior, not hidden-test leakage;
- final runtime engineering must target the hidden-set scale rather than the four visible placeholder clips.

## 7. Edge metric — VERIFIED

### Node matching

- predicted and GT nodes are matched by centroid distance;
- matching is timepoint-aware and one-to-one optimal bipartite assignment;
- maximum physical matching distance is **7 micrometers**, inclusive.

### Edge semantics

- predicted edge TP requires both endpoints to match GT nodes connected by a GT edge;
- a GT edge without such a predicted match is FN;
- sparse GT means many unmatched prediction edges are ignored rather than automatically counted as FP;
- the patched evaluator has explicit duplicate, merge, temporal-direction, and out-degree guards.

The executable details are pinned in `tests/test_metric_characterization.py`.

### Edge Jaccard

```text
edge_jaccard = TP / (TP + FP + FN)
```

### Predicted-node-count adjustment

```text
adjusted_jaccard = max(
    0,
    jaccard * (1 - 0.1 * (T_pred - T_true) / T_true)
)
```

`T_true` is the GT GEFF metadata field `estimated_number_of_nodes`. Phase 0C verified that the official implementation has no upper clamp: underprediction can produce an adjusted edge value larger than raw edge Jaccard. This is a characterization fact, not a recommendation to deliberately underpredict.

## 8. Division metric — VERIFIED AGAINST PINNED CODE/TESTS

- any raw prediction node with out-degree >= 2 is treated as a predicted fork candidate;
- division matching uses a local lineage window around the split;
- a prediction one frame early or late can be accepted when the required directed topology is present;
- two distinct daughter branches are required;
- GT divisions and predicted forks are paired one-to-one by maximum-cardinality bipartite matching;
- cross-component/merged-branch evidence can create division FPs;
- division Jaccard is `TP / (TP + FP + FN)`.

The edge scorer's out-degree cap and the division evaluator's raw-graph fork behavior are not interchangeable; downstream graph construction must respect both.

## 9. Aggregation / final score — VERIFIED

Competition-style local scoring uses the organizer path:

```text
evaluate -> node_recall -> per_sample_metrics -> summarise
```

- adjusted edge Jaccard is sample-size weighted by `edge_tp + edge_fp + edge_fn`;
- division TP/FP/FN are micro-averaged across videos;
- final score:

```text
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

- when an evaluated split contains no divisions at all, the official summary drops the division term.

Do not use the organizer convenience `evaluate_datasets()` as a private-LB proxy because it does not implement the per-sample node-count adjustment path used by `scripts/evaluate.py`.

## 10. Submission/runtime constraints — VERIFIED FROM CURRENT KAGGLE OVERVIEW

Current Kaggle overview states:

- submissions are made through Kaggle Notebooks;
- CPU runtime: **<= 12 hours**;
- GPU runtime: **<= 12 hours**;
- Internet access must be disabled for submission execution;
- freely/publicly available external data is allowed, including pretrained models;
- submission file must be named `submission.csv`;
- every hidden test dataset must appear in the submission.

Primary source:
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview

Additional rule details (licensing, code sharing, team/submission limits, winner obligations, private external services) remain workflow constraints but must be refreshed from the current Rules page before final-submission tooling encodes them.

## 11. Timeline — VERIFIED FROM CURRENT KAGGLE OVERVIEW

All listed deadlines are 11:59 PM UTC unless Kaggle states otherwise:

- Start: 2026-06-29
- Entry deadline: 2026-09-22
- Team merger deadline: 2026-09-22
- Final submission deadline: 2026-09-29

The organizer reserves the right to update the timeline. Final-submission planning must refresh this source.

## 12. Official baseline facts — VERIFIED

The pinned organizer repository documents an end-to-end baseline:

- `TemporalUNet3D` for detection/features;
- local-max suppression for cell-center recovery;
- `SimpleNodeTransformer` cross-attention linking `(t, t+1)` nodes;
- sparse supervision using GT edges while unannotated/background detections are ignored for training;
- the released public baseline training command used **3 epochs**;
- the organizer explicitly says that released model was **not trained to convergence**.

This is a baseline fact, not a recommendation that we use this architecture.

## 13. Known official metric-patch event — VERIFIED

The competition host announced a division-metric exploit and rescore. The pinned organizer revision is the public patch. We therefore must never compare scores produced by unknown/older scorer revisions as if they were directly equivalent.

Host source:
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154

## 14. UNRESOLVED — must not be silently assumed

- public/private leaderboard split proportions;
- exact hidden-test dataset count and per-movie distribution beyond host statements above;
- current max submissions/day, final-submission count, and team size until refreshed from the current Rules page for final tooling;
- community-reported acquisition pathologies (frozen/repeated frames, sudden global motion) until reproduced on our own train inventory or pinned to a host statement;
- provenance/clean-CV eligibility of public checkpoints and Kaggle notebook artifacts;
- whether public-LB rank accurately predicts private-LB rank;
- which pipeline layer (detection, candidates, association, topology/division) is currently our dominant recoverable error.

The last item is deliberately an experiment question. Architecture selection must wait for the bottleneck decomposition.