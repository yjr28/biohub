# Decision Log

This log records strategic/engineering decisions that affect experiment interpretation. Decisions are reversible; when reversed, append a new decision rather than rewriting history.

## D-0001 — Evidence-first phased execution

**Date:** 2026-09-04  
**Status:** active

### Decision

Execute the competition in small, gated phases. Do not begin the next phase until the current phase's explicit acceptance criteria have been reviewed.

### Reason

The competition has a nonstandard sparse-label tracking metric, public artifacts of mixed provenance, and hidden-domain generalization risk. Coupling evaluator, validation, detector, tracker, and leaderboard feedback too early would make causal attribution unreliable.

### Consequence

Phase order begins:

```text
0A repository/evidence foundation
→ 0B official evaluator isolation
→ 0C metric characterization
→ 1A unchanged strong baseline
→ 1B clean validation + bottleneck decomposition
→ architecture-specific attacks
```

---

## D-0002 — No architecture is the default winner

**Date:** 2026-09-04  
**Status:** active

### Decision

HOCT, Trackastra, Ultrack, NodeTransformer/ILP, custom GNNs, detector retraining, and graph-repair methods are hypotheses/challengers, not predetermined implementation priorities.

### Reason

Research suggests association/global reasoning may have upside, but the actual recoverable score-loss layer must be measured first.

### Pivot rule

After a clean baseline exists, classify missing GT-edge opportunity into at least:

- detection availability;
- candidate-generation availability;
- association/ranking;
- global solver/topology.

Concentrate engineering on the largest credible recoverable loss rather than the most interesting architecture.

---

## D-0003 — Public leaderboard is a sensor, not the objective

**Date:** 2026-09-04  
**Status:** active

### Decision

Do not promote a change solely because public leaderboard score rises. Local validation, provenance, error decomposition, and robustness evidence remain part of model selection.

### Reason

The final objective is private-leaderboard performance on hidden data, while public-LB feedback can be noisy and can reward overfitting/correlated public-solution behavior.

---

## D-0004 — Unknown-provenance learned artifacts cannot certify clean CV

**Date:** 2026-09-04  
**Status:** active

### Decision

A learned public checkpoint/notebook artifact with unknown training/sample overlap may be tested for final-solution potential, but its score cannot be used as proof of clean held-out generalization until provenance is resolved.

### Reason

Validation contamination would invalidate comparisons and could systematically favor public artifacts that have seen held-out samples.

---

## D-0005 — HOCT candidate geometry is an experimental factor

**Date:** 2026-09-04  
**Status:** active

### Decision

Do not collapse HOCT candidate generation to a single distance convention. Evaluate at least two explicitly named spaces on the same frozen detections:

- `hoct_native_voxel`: raw z/y/x geometry matching public HOCT's pinned candidate implementation;
- `physical_um`: anisotropically scaled OME-Zarr physical geometry.

Candidate coverage must be measured before learned HOCT scoring so a gain/loss can be attributed to proposal geometry rather than to the transformer or ILP.

### Evidence

At HOCT revision `2ccc5040823bc944ab67790abd1f56eea7cd4f05`, `create_graph(...)` accepts `scale` and creates temporary scaled-coordinate columns, but it invokes `tracksdata.edges.DistanceEdges(...)` without selecting those scaled columns. The pinned tracksdata operator defaults to raw `z/y/x`. A dedicated public-package integration test characterizes this behavior with an anisotropic synthetic example.

### Consequence

The first real-data HOCT sweep starts with candidate-recall calibration in both spaces. It does not spend checkpoint inference time on radius/neighbor settings whose fixed-detection GT-edge coverage is already inadequate.

---

## D-0006 — Preserve the opposite embryo as scarce generalization evidence

**Date:** 2026-09-04  
**Status:** active

### Decision

Do not choose candidate radii, neighbour counts, solver weights, thresholds, or other tracker hyperparameters from the true LOEO embryo. For each direction, tracker selection uses only the deterministic `checkpoint_monitor_datasets` created inside the training embryo by the Phase-2C `train-embryo-hash-holdout` protocol.

### Reason

With only two training embryos, every adaptive look at the opposite embryo spends scarce cross-embryo evidence. Tuning a tracker on that embryo and then reporting its score as clean LOEO would systematically overstate hidden-embryo generalization.

The same-embryo checkpoint monitor is not an independent validation set and has already influenced checkpoint selection. It is nevertheless the correct place to perform training-side hyperparameter calibration while keeping the opposite embryo untouched until a hypothesis is frozen.

### Enforcement

`biohub.trackers.calibration_scope_from_protocol` fails closed unless:

- checkpoint monitor policy is `train-embryo-hash-holdout`;
- calibration datasets are a non-empty subset of declared training datasets;
- calibration and LOEO holdout datasets are disjoint;
- the declared training and LOEO universes are disjoint.

Sweep reports explicitly record the LOEO datasets as forbidden and `loeo_holdout_used=false`.

### Consequence

The experiment order becomes:

```text
training embryo
├── optimizer datasets → train/select baseline checkpoint
└── nested monitor datasets → calibrate candidate/solver hypothesis
                              ↓ freeze configuration
opposite embryo          → clean LOEO evaluation only
```

After both directional hypotheses are frozen independently, compare their cross-embryo results. Do not reverse-fit the calibration grid from those holdout scores.
