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
