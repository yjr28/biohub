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

---

## D-0007 — Freeze the aggregate candidate frontier before learned tracker evaluation

**Date:** 2026-09-04  
**Status:** active

### Decision

The candidate-generation stage may inspect only training-side nested monitor datasets. Its output is the **aggregate Pareto frontier** of candidate proposal coverage versus candidate-graph size. That complete frontier is frozen as the only configuration set allowed to enter learned-HOCT/solver calibration.

The opposite-embryo LOEO set may evaluate the final frozen learned tracker, but it may not add candidate configurations, expand radii, alter neighbour counts, or resurrect dominated configurations.

### Enforcement

`scripts/run_hoct_candidate_calibration.py` derives prediction settings from the selected organizer baseline, redirects prediction into a calibration-specific `USER` namespace, requires exactly the intended monitor GEFF set, freezes those exact detector nodes, runs the oracle-first candidate sweep, and writes `candidate_shortlist.json` with:

- the allowed Pareto-front configuration IDs;
- a deterministic priority ordering for the next training-side stage;
- the maximum-coverage/minimum-cost frontier member;
- hashes for checkpoint, protocol, effective config, split file, inventory, grid, detection index, and sweep report;
- `loeo_may_expand_shortlist=false`.

### Reason

Candidate recall and learned association quality are different causal layers. Running transformer/ILP inference on obviously dominated or coverage-deficient proposal graphs wastes compute and makes failures harder to attribute. Freezing the candidate frontier before learned scoring also prevents the LOEO embryo from becoming an implicit hyperparameter search oracle.

### Next gate

Run learned-HOCT/solver calibration only on the frozen candidate shortlist and only on the same training-side monitor datasets. Select and freeze one learned configuration per LOEO direction before evaluating the opposite embryo.

---

## D-0008 — Freeze the learned tracker against a same-scope organizer control before LOEO

**Date:** 2026-09-04  
**Status:** active

### Decision

Learned association/global-solver selection must be completed entirely on the same training-side nested monitor set used for candidate calibration. Compare every allowed HOCT model/solver trial against the organizer NodeTransformer control on exactly that movie set, using the pinned official metric and a promotion margin declared before execution.

Freeze exactly one winner for the LOEO direction: either one HOCT trial or the organizer control. The opposite embryo may evaluate that frozen winner once, but may not retune or replace it.

### Enforcement

`scripts/run_hoct_learned_calibration.py`:

- consumes only candidate IDs already frozen by Phase 2E;
- requires a fully explicit learned grid for audited model names, window sizes, solver weights, gap policy, and promotion margin;
- verifies the training-side monitor and forbidden-LOEO dataset sets match the Phase-2E artifacts;
- verifies audited HOCT checkpoint SHA-256 values;
- scores the organizer control and all HOCT trials on the identical monitor set with the pinned official evaluator;
- promotes HOCT only when its aggregate monitor score clears `organizer_control_score + hoct_promotion_margin`;
- writes `learned_selection.json` with full provenance and `loeo_may_retune_or_replace_winner=false`.

### Reason

With only two embryos, using the opposite embryo to choose among learned association models would convert the only cross-embryo validation instrument into a tuning set. Requiring a same-scope organizer control also prevents complexity from being rewarded merely for being novel: HOCT must earn promotion on the exact competition objective before consuming scarce LOEO evidence.

### Next gate

Execute the full training-side chain on real competition data for each direction, freeze the Phase-2F winner, and only then run one clean opposite-embryo LOEO evaluation. Subsequent engineering must be driven by the measured bottleneck/error decomposition rather than by reverse-fitting this holdout.

---

## D-0009 — Treat each opposite embryo as a one-shot frozen evidence boundary

**Date:** 2026-09-04  
**Status:** active

### Decision

Once Phase 2F freezes a winner, the opposite-embryo LOEO evaluation must not perform any additional model selection. Score exactly that frozen winner and preserve the result as evidence. Do not alter detector settings, candidate generation, HOCT checkpoint, window size, solver weights, or promotion policy from the resulting holdout score and then reuse the same embryo as if it were fresh validation.

### Enforcement

`scripts/run_frozen_loeo.py`:

- requires Phase-2E and Phase-2F artifacts to agree exactly on monitor and holdout dataset scope;
- requires both upstream artifacts to state that LOEO was unused for selection and cannot expand/replace the frozen winner;
- permits only `organizer_control` or the exact recorded winning HOCT trial;
- verifies a HOCT winner's candidate ID still belongs to the frozen Phase-2E frontier;
- requires the organizer holdout prediction directory to contain exactly the holdout movies and no training-side monitor outputs;
- fingerprints file- or directory-backed GEFF artifacts with deterministic SHA-256 tree hashes before scoring;
- constructs all frozen HOCT predictions before invoking the GT evaluator;
- refuses to overwrite an existing LOEO output directory.

### Reason

With two embryos, the informational value of a clean cross-embryo result is unusually high and unusually easy to destroy. A mechanically enforced boundary prevents architecture or threshold search from quietly migrating into the validation set and gives the two directional LOEO results a defensible interpretation as generalization evidence.

### Next gate

Run the complete baseline → candidate calibration → learned calibration → frozen LOEO chain on real competition data in both directions. Only after those two results exist should the next architecture branch be chosen, using the measured loss decomposition and directional robustness rather than a new broad research sweep.