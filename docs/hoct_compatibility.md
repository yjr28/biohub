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

## Candidate geometry: two hypotheses, not one hidden assumption

The Biohub images are anisotropic, so physical distance and raw voxel distance are materially different. Our first adapter used physical `(z, y, x)` coordinates after multiplying by the OME-Zarr scale. That remains a valid, biologically motivated hypothesis.

However, a source audit of public HOCT revealed an important implementation detail at the pinned revision: `hoct.features.create_graph(...)` accepts a `scale` argument and constructs temporary `scaled_t/scaled_z/scaled_y/scaled_x` columns, but those columns are not written back into the graph before `tracksdata.edges.DistanceEdges(...)` is called. The pinned `DistanceEdges` defaults to the graph's raw `z/y/x` attributes. Therefore the public HOCT candidate graph is, in practice, generated in **raw voxel space**, not physical microns.

That distinction matters because the public pretrained model was developed around the upstream candidate-generation implementation. We therefore expose two explicit, separately logged candidate spaces:

### `physical_um`

Use `--distance-threshold-um`. Candidate search multiplies `(z,y,x)` by the per-dataset OME-Zarr spatial scale before KD-tree search.

This is the biologically meaningful geometry and may generalize better across anisotropic data.

### `hoct_native_voxel`

Use `--distance-threshold-voxels`. Candidate search uses raw `(z,y,x)` coordinates, matching the candidate-distance convention of public HOCT at the audited revision.

This is the fairest first test of the pretrained HOCT association model because it minimizes candidate-distribution shift relative to the public implementation.

Neither mode is assumed superior. Candidate recall on fixed detections is measured before learned scoring, and both can be retained if their failure sets are complementary.

The primary competition experiment keeps `max_delta_t=1`. Longer temporal-gap candidates are an explicit ablation rather than an accidental default.

## Missing segmentation features

HOCT's `FrameDataset` expects the region-property feature contract used by its pretrained model. A centroid cache has no segmentation mask from which to recover those features without changing the experiment.

For the initial centroid-only ablation, the missing morphology/intensity dimensions are set to the public pretrained feature means from the pinned HOCT inference API. HOCT's own fixed standardization therefore maps those unknown dimensions to approximately zero. The real centroid coordinates and HOCT-compatible border-distance feature remain informative.

This is an intentionally conservative ablation. It does **not** assert that mean-filled morphology is optimal. If the learned edge signal proves useful, image-derived features can be evaluated later as a separate causal change.

## Public-package integration contract

Unit tests against our own adapter are not enough. `.github/workflows/hoct-integration.yml` installs the exact audited public HOCT revision and runs `tests/integration/test_hoct_public_contract.py`.

The integration test exercises the real public `hoct.predict(model, graph=...)` path through HOCT's `FrameDataset` and fixed `Standardize(_MEAN, _STD)` transform. It monkeypatches only the expensive model/ILP boundary, so no public weight download, Gurobi optimization, or competition data are required. The contract verifies that:

- the adapter graph is accepted by the public package;
- HOCT produces the expected 19-wide node feature tensor;
- node/edge positions and edge indices have the expected dimensions;
- all tensors are finite;
- mean-filled unknown morphology/intensity dimensions standardize to approximately zero.

A second characterization test constructs a tiny segmentation with a one-z-voxel displacement and an intentionally large physical z scale. Public HOCT still creates the edge with raw distance `1.0`, locking the raw-voxel candidate-space observation into CI so a future upstream change cannot silently invalidate our experiment design.

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

For one dataset and one precommitted candidate configuration, use exactly one candidate radius convention.

Public-HOCT-native candidate geometry:

```bash
python scripts/build_hoct_candidate_graph.py \
  --detections artifacts/private/fixed_detections/<dataset>.parquet \
  --inventory artifacts/private/data/inventory.json \
  --dataset <dataset> \
  --distance-threshold-voxels <radius> \
  --n-neighbors <k> \
  --out artifacts/private/hoct_candidates/<dataset>.geff
```

Physical-micron candidate geometry:

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
