# Biohub Cell Tracking

Competition workspace for **Biohub - Cell Tracking During Development**.

**Competition target:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

This repository is intentionally built in small, reviewable phases. We do not promote a model, metric assumption, validation scheme, or external artifact until its provenance and evidence are documented.

## Current phase

**Phase 0A — repository and evidence foundation**

No detector, tracker, evaluator wrapper, or competition submission code has been implemented yet. Phase 0A only establishes the repository contract, provenance rules, and project skeleton.

## Operating principles

1. One narrowly defined objective per phase.
2. Official/primary sources outrank community claims.
3. Verified facts, inferences, and unresolved claims remain explicitly separated.
4. Every external checkpoint, dataset, notebook, and code dependency receives a provenance record before it can influence model selection.
5. Competition data, checkpoints, generated predictions, and submissions are not committed to this public repository.
6. Each experiment must state a falsifiable hypothesis and a pass/pivot condition.
7. Public-leaderboard score is evidence, not the optimization target; model selection must be defensible independently of leaderboard probing.

## Planned structure

```text
biohub/
├── configs/                 # experiment configuration (later phases)
├── docs/
│   ├── competition_contract.md
│   ├── decision_log.md
│   └── provenance.md
├── experiments/             # reproducible experiment manifests/results (no raw data)
├── src/biohub/              # implementation (later phases)
└── tests/                   # metric/adapter/invariant tests (later phases)
```

## Phase gates

- **0A:** repository + evidence foundation
- **0B:** isolate and pin the official evaluator; write only a thin adapter
- **0C:** characterize metric behavior with controlled synthetic tests
- **1A:** reproduce a strong baseline unchanged
- **1B:** clean embryo-level validation + bottleneck decomposition

We stop at every gate, review what was verified, and only then proceed.
