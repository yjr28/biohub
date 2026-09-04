# HOCT candidate calibration protocol

Phase 2E answers a narrower question than “does HOCT win?”:

> **At the exact same fixed detections, which candidate-generation hypotheses retain the recoverable GT edges without exploding graph size?**

The learned HOCT checkpoint and lineage solver are intentionally downstream of this gate. A transformer cannot recover a true association that never enters its candidate graph, and a huge graph can make inference/ILP slower and noisier even when raw proposal recall rises.

## Selection boundary

Candidate hyperparameters are **not** selected on the opposite-embryo LOEO holdout.

For each LOEO direction, Phase 2C's `train-embryo-hash-holdout` protocol already partitions the training embryo into:

- optimizer datasets;
- deterministic checkpoint-monitor datasets;
- the completely separate opposite-embryo LOEO holdout.

Phase 2E uses only `checkpoint_monitor_datasets` for candidate calibration. `biohub.trackers.calibration_scope_from_protocol` fails closed unless that policy and separation are present. The true LOEO datasets are recorded as forbidden in every sweep report.

The checkpoint-monitor set is **not** clean cross-embryo validation. It has already participated in checkpoint selection and comes from the same embryo as optimization data. Its role here is hyperparameter calibration while preserving the opposite embryo as scarce generalization evidence.

## What is frozen

Before a candidate sweep, freeze:

1. the organizer checkpoint;
2. its prediction/detection settings;
3. the calibration dataset set;
4. the fixed detection Parquets extracted from those predictions;
5. the candidate-grid JSON.

A sweep changes only:

- candidate distance space;
- candidate radius;
- number of neighbours;
- optional `max_delta_t` ablation.

No detector threshold, learned edge model, ILP weight, division heuristic, or LOEO holdout observation changes inside the sweep.

## Candidate spaces

Two candidate geometries are deliberately kept separate:

### `hoct_native_voxel`

Raw `(z, y, x)` distance, matching the pinned public HOCT candidate implementation.

### `physical_um`

Anisotropically scaled physical distance using each dataset's OME-Zarr `(z, y, x)` spacing.

Neither is assumed superior. The candidate grid can contain both, and each trial records the active convention.

## Primary measurements

For each candidate config and calibration dataset, official node matching is computed once and reused. We record:

- fixed detections;
- all GT edges;
- GT edges whose two endpoint cells are represented by fixed detections;
- GT edges available in the candidate graph;
- candidate-generation gap;
- `candidate_recall_of_detectable`;
- `candidate_recall_all_gt`;
- total candidate edges;
- candidate edges per detection.

`candidate_recall_of_detectable` is the central proposal metric because it isolates candidate generation from detector misses.

These values are **not** leaderboard scores. They do not measure learned HOCT association quality, solver topology, adjusted-Jaccard count penalties, or division quality.

## Aggregation and Pareto frontier

Identical conceptual configs are micro-aggregated across calibration datasets by summing counts before recomputing recall. Dataset spatial scale is metadata, not part of the conceptual config ID, so a physical-radius hypothesis remains the same hypothesis across datasets.

A config is Pareto-dominated when another config:

- covers at least as many recoverable GT edges;
- uses no more candidate edges;
- is strictly better in at least one of those dimensions.

The frontier is an engineering shortlist, not an automatic winner selector.

## Promotion rule

After the training-side sweep, freeze a small set of candidate configs **before** evaluating the LOEO embryo. A practical promotion order is:

1. near-maximal recoverable-edge coverage;
2. lowest graph cost among configs with effectively equivalent coverage;
3. preserve an alternative distance space only when it offers a meaningfully different cost/coverage tradeoff.

Do not choose a candidate radius from the LOEO result and then report that same LOEO score as unbiased evidence.

## Execution

Create an explicit grid JSON, for example:

```json
{
  "n_neighbors": [3, 5, 8],
  "max_delta_t": [1],
  "physical_um": [2.0, 4.0, 6.0],
  "hoct_native_voxel": [4.0, 8.0, 12.0]
}
```

The numbers above are an **example schema**, not pre-approved competition hyperparameters. The actual grid should be chosen and recorded before observing the LOEO holdout.

Then run:

```bash
python scripts/sweep_hoct_candidates.py \
  --protocol <organizer_baseline_protocol.json> \
  --inventory <data_inventory.json> \
  --grid <candidate_grid.json> \
  --pred-dir <training-side-monitor predictions> \
  --detections-dir <frozen monitor detections> \
  --gt-dir <competition train directory> \
  --out <private candidate sweep report.json>
```

The output records SHA-256 provenance for the protocol, inventory, and grid, plus per-dataset and aggregate Pareto frontiers.

## Next gate

Only after candidate generation is frozen do we spend learned HOCT inference/ILP compute. The first learned comparison remains fixed-detection association quality versus the competition-native NodeTransformer control, evaluated in both LOEO directions without retuning on the held-out embryo.
