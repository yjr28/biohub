# Data and Validation Contract

Last verified: **2026-09-04**

This phase establishes what we are allowed to call a clean validation result before we train or compare models.

## 1. Host-verified domain structure

The competition host has publicly clarified two facts that drive our validation design:

1. Dataset IDs use `<embryo_id>_<crop_id>`; only the prefix before the first underscore is the embryo ID.
   - https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723694
2. The training set contains exactly two unique embryo IDs, and the hidden test set has no embryo-ID overlap with training.
   - https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793

The host also states that the four publicly visible test clips are dummy placeholders for submission-pipeline checks; the real scoring set is a larger private hidden set.
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062

## 2. Primary validation regime

Primary model selection uses leave-one-embryo-out (LOEO):

```text
Fold 0: train/calibrate on embryo A -> evaluate all crops from embryo B
Fold 1: train/calibrate on embryo B -> evaluate all crops from embryo A
```

A result is **clean cross-embryo validation** only if the held-out embryo contributes no information to:

- model training;
- detector threshold calibration;
- association threshold calibration;
- post-processing threshold selection;
- early stopping/model selection;
- pseudo-label construction used for that fold;
- ensemble/fusion weight fitting.

Inference-time transformations that do not learn from hidden GT may be evaluated separately, but their parameters must not be selected using the held-out embryo and then reported as untouched LOEO.

## 3. Why random frame/edge splits are secondary only

Frames and crops from the same embryo share biological identity, imaging conditions, global geometry, motion statistics, and acquisition-specific artifacts. A random edge/frame split can therefore leak embryo-specific information into both sides of validation.

Random/time/edge splits may still be useful for debugging or high-sample-size diagnostics, but they cannot override a contradictory LOEO result when choosing the private-LB submission strategy.

## 4. Two-embryo limitation

LOEO is the least-contaminated split available, but with only two train embryos it is high variance. We therefore track all of the following rather than a single mean score:

- `A -> B` score;
- `B -> A` score;
- weighted/pooled summary where appropriate;
- per-crop score distribution;
- worst-crop / lower-tail behavior;
- error decomposition by failure class;
- stress slices such as high motion, dense regions, division neighborhoods, and acquisition anomalies once empirically defined.

A change that improves one fold and materially harms the other is not automatically accepted on mean score alone.

## 5. Metadata-only inventory

Before any baseline run, execute inside a Kaggle notebook/session with the competition dataset mounted:

```bash
python scripts/inventory_competition.py \
  --competition-root /kaggle/input/competitions/biohub-cell-tracking-during-development \
  --json artifacts/data_inventory.json \
  --csv artifacts/data_inventory.csv
```

The inventory intentionally reads metadata/graphs, not full image tensors. It records:

- exact train and visible-placeholder dataset names;
- embryo grouping;
- `(T,Z,Y,X)` image shapes;
- physical `(Z,Y,X)` voxel scale;
- GT node, edge, and division counts;
- GT annotated time range;
- `estimated_number_of_nodes` metadata used by the adjusted score;
- generated LOEO manifests.

Do **not** commit raw competition images, GEFFs, model predictions derived from private competition data, credentials, or Kaggle tokens to this public repository.

## 6. Phase 1A data-gate acceptance criteria

The real mounted dataset passes the gate only when all are true:

- [ ] every training `.zarr` has its paired `.geff`;
- [ ] every training image has 4D `(T,Z,Y,X)` shape;
- [ ] spatial scale is valid and positive;
- [ ] exactly two training embryo IDs are found;
- [ ] every dataset is assigned to exactly one embryo by the host-defined prefix rule;
- [ ] each LOEO fold is exhaustive and train/holdout sets are disjoint;
- [ ] every training GT has a positive `estimated_number_of_nodes` value before adjusted-score experiments;
- [ ] GT annotated time indices lie inside the corresponding image time range;
- [ ] graph temporal-edge statistics are inspected rather than assumed;
- [ ] visible `test/` files are explicitly labeled placeholders and never used as evidence of hidden-domain generalization.

If any item fails, we stop model-selection experiments and investigate the data/source mismatch first.

## 7. What Phase 1A does not decide

This phase does not decide:

- which detector is best;
- whether node selection or association dominates error;
- whether HOCT, Trackastra, Ultrack, NodeTransformer/ILP, or another tracker is preferable;
- whether global motion compensation helps;
- any public-leaderboard threshold;
- any final-submission model.

Those decisions require the real-data baseline and bottleneck decomposition after this contract passes.
