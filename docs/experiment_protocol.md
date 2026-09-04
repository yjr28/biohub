# Experiment Protocol

This document defines how model-selection evidence is created for **Biohub - Cell Tracking During Development**. The target is private-leaderboard generalization to unseen embryos, not maximization of the public leaderboard in isolation.

## 1. Order of operations

Every serious experiment follows this sequence:

```text
inventory real competition data
→ choose one declared LOEO fold
→ write config
→ register immutable manifest
→ train / infer
→ score exact holdout set with pinned organizer evaluator
→ write strict evaluation report
→ attach immutable result
→ interpret against the falsifiable hypothesis
```

The manifest is written **before** looking at the held-out score. If a parameter changes after seeing the score, that is a new experiment ID.

## 2. Clean LOEO definition

A run is clean only when the held-out embryo was excluded from every learned or tuned decision that could affect the prediction:

- detector/feature training;
- early stopping or checkpoint selection;
- confidence/NMS thresholds;
- candidate-radius or motion thresholds;
- association costs/weights;
- division thresholds;
- graph repair/post-processing parameters;
- pseudo-label generation;
- ensemble weights;
- any learned calibration.

Using the holdout for post-hoc error diagnosis is allowed, but the next changed system receives a new experiment ID and the resulting score is no longer independent confirmation of that hypothesis. With only two embryos, this distinction must remain explicit.

## 3. Two-direction evidence

The primary validation object is the pair:

```text
train embryo A → validate embryo B
train embryo B → validate embryo A
```

Report both directions separately. Do not collapse them into a single average until both raw results are visible. A change that gains strongly in one direction and loses in the other is a robustness warning, not automatically a promotion.

Secondary within-embryo/time/movie slices are diagnostic tools. They are not substitutes for cross-embryo validation.

## 4. Required manifest fields

Each registered experiment records:

- unique experiment ID;
- falsifiable hypothesis;
- full git commit SHA;
- SHA-256 of the exact `data_inventory.json`;
- pinned evaluator and `tracksdata` revisions;
- fold name;
- train/validation embryos and exact dataset IDs;
- full config object;
- random seeds;
- concrete leakage controls;
- optional parent experiment;
- notes.

This is deliberately stricter than ordinary Kaggle bookkeeping because small public-LB differences are not interpretable if code, data grouping, or validation contamination differs between runs.

## 5. Score reporting

All official score aggregation delegates to the pinned organizer implementation. The report preserves per-dataset rows and the exact organizer summary.

Primary columns to inspect together:

- `adj_edge_jaccard`;
- `edge_jaccard`;
- `edge_tp`, `edge_fp`, `edge_fn`;
- `node_recall`;
- `total_node_ratio`;
- `division_jaccard` and division TP/FP/FN;
- final `score`.

A combined score alone is insufficient for diagnosis.

## 6. Diagnostic groups

Optional group/slice reports are predeclared mappings from dataset ID to a diagnostic label. The reporting code requires exact coverage of the evaluated set, preventing a difficult movie from disappearing silently.

Once real-data diagnostics exist, useful slice families may include motion magnitude, density, division proximity, acquisition anomalies, and temporal region. These labels must be derived by a documented procedure before being used for repeated model selection.

## 7. Experiment promotion rule

A candidate replaces its parent only when there is a causal explanation consistent with the measured changes. Default evidence order:

1. both clean LOEO directions;
2. exact score decomposition;
3. predeclared stress slices;
4. robustness/repeatability across seeds where stochasticity matters;
5. runtime/memory and Kaggle submission feasibility;
6. public leaderboard as an additional noisy observation.

A public-LB increase without supporting local evidence is not enough to overwrite the robust candidate.

## 8. Baseline rule

Before architecture-specific work, reproduce the selected strong baseline unchanged in both LOEO directions. Do not simultaneously change detector, association model, solver, preprocessing, and post-processing. The first baseline establishes the reference error budget.

## 9. Bottleneck/oracle decomposition

After the baseline, attack the largest recoverable score-loss layer. At minimum distinguish:

1. **node availability** — would the GT endpoint be matchable if association were perfect?
2. **candidate availability** — if nodes exist, does the true edge enter the candidate set?
3. **ranking/association** — candidate exists but loses to an incorrect edge.
4. **global/topology** — local scores are adequate but a global constraint/solver/repair choice breaks the lineage.
5. **division-specific** — fork timing/topology/branch identity errors.
6. **count calibration** — predicted-node total changes adjusted edge score.

An architecture is selected only after this decomposition suggests which layer has the largest credible recoverable private-LB value.

## 10. Submission budget

Kaggle submissions are treated as information-budget expenditures. For each submission record, before seeing the result:

- experiment ID(s) represented;
- expected direction and approximate magnitude;
- reason the public LB is informative beyond local validation;
- what decision will change under gain / flat / loss outcomes.

Do not use submissions merely to search thresholds that can be evaluated locally.

## 11. Final two-submission portfolio

The two final selections should ideally differ in failure mode, not merely in one threshold. Final choice should consider expected private score, variance across embryo directions/stress slices, and error correlation between candidates. A slightly lower public-LB solution can be valuable if it is the robust/orthogonal hedge.

## 12. CLI workflow

Inventory:

```bash
python scripts/inventory_competition.py \
  --competition-root /kaggle/input/competitions/biohub-cell-tracking-during-development \
  --json artifacts/data_inventory.json \
  --csv artifacts/data_inventory.csv
```

Register a run:

```bash
python scripts/register_experiment.py \
  --inventory artifacts/data_inventory.json \
  --fold holdout_<EMBRYO> \
  --experiment-id baseline-<EMBRYO>-001 \
  --hypothesis "unchanged organizer baseline establishes the clean reference" \
  --config configs/baseline.json \
  --seed 7 \
  --leakage-control "no holdout-based checkpoint selection" \
  --leakage-control "no holdout-based threshold tuning"
```

Evaluate exact holdout datasets:

```bash
python scripts/evaluate_loeo.py \
  --inventory artifacts/data_inventory.json \
  --fold holdout_<EMBRYO> \
  --pred-dir artifacts/predictions/baseline-<EMBRYO>-001 \
  --gt-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --out artifacts/reports/baseline-<EMBRYO>-001.json
```

Attach the result:

```bash
python scripts/record_result.py \
  --experiment-id baseline-<EMBRYO>-001 \
  --report artifacts/reports/baseline-<EMBRYO>-001.json \
  --out experiments/results/baseline-<EMBRYO>-001.json
```
