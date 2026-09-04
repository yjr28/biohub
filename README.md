# Biohub Cell Tracking

Competition workspace for **Biohub - Cell Tracking During Development**.

**Sole competition target:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

The objective is not public-leaderboard optimization in isolation. The repository is structured to maximize the probability of strong **private-leaderboard** generalization by making metric semantics, validation provenance, and experiment decisions auditable.

## Current phase

**Phase 1B — reproducible experiment harness**

Completed foundations:

- **0A:** evidence/provenance contract and repository scaffold.
- **0B:** exact organizer evaluator pinned as a git submodule and wrapped by a thin adapter; metric math is not reimplemented.
- **0C:** controlled synthetic characterization of scorer behavior against the pinned official implementation.
- **1A:** metadata-only competition inventory tooling plus fail-closed leave-one-embryo-out (LOEO) validation construction based on host-verified embryo semantics.

Phase 1B adds append-only experiment manifests, exact fold scoring, strict result reports, and diagnostic grouping so that later architecture comparisons cannot silently mix code revisions, contaminated validation, or different evaluator/data states.

## Operating principles

1. Official/primary sources outrank community claims.
2. Verified facts, inferences, competitor claims, and unresolved items remain explicitly separated.
3. Competition data, checkpoints, generated predictions, and submissions are not committed to this public repository.
4. Every experiment states a falsifiable hypothesis and records the exact code, evaluator, inventory, split, config, and seed state that produced it.
5. A run is not called clean LOEO if the held-out embryo influenced thresholds, early stopping, pseudo-labels, post-processing, association settings, or ensemble weights.
6. Public-leaderboard score is a noisy observation, not the model-selection objective.
7. Score aggregation always delegates to the pinned organizer evaluator.
8. We attack measured bottlenecks before introducing architectural complexity.

## Repository structure

```text
biohub/
├── configs/                 # experiment/model configuration
├── docs/                    # competition, evaluator, validation and provenance contracts
├── experiments/             # lightweight manifests/results only; no competition data
├── scripts/                 # inventory and exact-fold evaluation entry points
├── src/biohub/
│   ├── data/                # metadata inventory + validation grouping
│   ├── evaluation/          # thin official adapter + reports
│   └── experiments/         # immutable experiment bookkeeping
├── tests/                   # scorer semantics and research-system invariants
└── vendor/                  # pinned organizer evaluator + tracksdata revisions
```

## Immediate execution gates

1. Run `scripts/inventory_competition.py` against the accepted Kaggle data mount and preserve the generated metadata inventory.
2. Verify the real downloaded dataset set, physical scales, GT counts, coarse node-count metadata, and the two exact LOEO folds.
3. Reproduce an unchanged strong/public organizer baseline separately in each LOEO direction.
4. Score every baseline through `scripts/evaluate_loeo.py` and record immutable experiment manifests/results.
5. Perform bottleneck/oracle decomposition on fixed detections before choosing the next model family.
6. Only then begin targeted detector/linker/global-motion/division experiments.

## Phase gates

- **0A:** repository + evidence foundation — complete
- **0B:** pin/wrap official evaluator — complete
- **0C:** characterize scorer semantics — complete
- **1A:** data + embryo-level validation contract — complete in code; real-data inventory still required
- **1B:** experiment registry + exact fold reporting — current
- **2A:** unchanged strong baseline reproduction
- **2B:** bottleneck/oracle decomposition
- **3+:** targeted modeling, robustness, and final private-LB portfolio
