# HOCT compatibility layer

This repository treats HOCT as a public external tracker, not as an opaque notebook dependency. The compatibility contract is pinned to public HOCT revision `2ccc5040823bc944ab67790abd1f56eea7cd4f05`.

## Upstream point API status

At the pinned revision, `hoct.create_graph_from_points(...)` is declared in `src/hoct/_api.py` but its implementation is still a TODO followed by `pass`. Therefore Biohub does **not** call that helper or claim that upstream HOCT currently supports our centroid-only path directly.

Instead, `biohub.trackers.build_hoct_point_graph` creates the candidate `tracksdata` graph required by HOCT's documented `predict(model, graph=...)` path.

## Fixed detections

The experiment starts from a frozen detector graph. `scripts/cache_fixed_detections.py` strips every association edge and writes only:

- `detection_id`
- `t`
- `z`
- `y`
- `x`

The resulting Parquet files are deterministic and content-hashed. `detection_id` is carried into the HOCT candidate graph as `source_detection_id`, so candidate/solution edges can be reconciled with the exact baseline nodes for official matching and bottleneck analysis.

## Candidate geometry

Candidate edges are generated in physical `(z, y, x)` coordinates using the per-dataset OME-Zarr scale from the Phase-2A inventory. This is deliberate: the Biohub data are anisotropic and a raw one-voxel z displacement is not equivalent to a raw one-voxel x/y displacement.

The primary compatibility experiment uses `max_delta_t=1`. The official Biohub evaluator directly retains only consecutive-frame edges, so longer-gap candidates are an explicit ablation rather than an accidental default.

## Missing segmentation features

HOCT's `FrameDataset` expects the region-property feature contract used by its pretrained model. A centroid cache has no segmentation mask from which to recover those features without changing the experiment.

For the initial centroid-only ablation, the missing morphology/intensity dimensions are set to the public pretrained feature means from the pinned HOCT inference API. HOCT's own fixed standardization therefore maps those unknown dimensions to approximately zero. The real centroid coordinates and HOCT-compatible border-distance feature remain informative.

This is an intentionally conservative ablation. It does **not** assert that mean-filled morphology is optimal. If the learned edge signal proves useful, image-derived features can be evaluated later as a separate causal change.

## Public checkpoints

The public checkpoint registry is copied from the pinned HOCT model registry and verified before inference:

- `general_v1`: `5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33`
- `ctc_v0`: `b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000`
- `general_v0`: `024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a`

Inference fails closed if a local checkpoint does not match its audited SHA-256. This also makes the path compatible with Kaggle's internet-disabled submission requirement once the public dependency and weight files are attached locally.

## Execution sequence

After a clean organizer baseline has produced prediction GEFFs:

```bash
python scripts/cache_fixed_detections.py \
  --pred-dir <baseline_predictions> \
  --out-dir artifacts/private/fixed_detections
```

For one dataset and one precommitted candidate configuration:

```bash
python scripts/build_hoct_candidate_graph.py \
  --detections artifacts/private/fixed_detections/<dataset>.parquet \
  --inventory artifacts/private/data/inventory.json \
  --dataset <dataset> \
  --distance-threshold-um <radius> \
  --n-neighbors <k> \
  --out artifacts/private/hoct_candidates/<dataset>.geff
```

Candidate recall can be decomposed *before* spending HOCT inference time:

```bash
python scripts/analyze_fixed_detection_bottleneck.py \
  --inventory artifacts/private/data/inventory.json \
  --dataset <dataset> \
  --pred-geff <baseline_prediction.geff> \
  --gt-geff <competition_train>/<dataset>.geff \
  --candidate-geff artifacts/private/hoct_candidates/<dataset>.geff \
  --out artifacts/private/diagnostics/<dataset>.json
```

Only candidates with adequate fixed-detection GT-edge coverage should be promoted to learned HOCT scoring/ILP inference.

## Data handling

Competition image/GT data, frozen detections, generated candidate graphs, model outputs, experiment results, checkpoints, and submissions remain under ignored/private artifact paths. No competition data are required in this public compatibility code.
