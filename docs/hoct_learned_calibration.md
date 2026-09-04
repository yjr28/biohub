# Learned HOCT calibration protocol

Phase 2F begins only after Phase 2E has frozen a candidate-generation Pareto frontier.

The question is now:

> **Among candidate graphs already approved on training-side proposal coverage, does an audited HOCT checkpoint + solver beat the organizer NodeTransformer control strongly enough to justify promotion before clean LOEO evaluation?**

## Evidence boundary

Phase 2F uses exactly the same nested training-embryo `checkpoint_monitor_datasets` used for Phase-2E calibration. The opposite-embryo LOEO datasets remain explicitly forbidden.

The inputs are:

- `candidate_shortlist.json` from Phase 2E;
- `monitor_prediction_plan.json` from the same calibration run;
- the frozen detector Parquets from that run;
- the selected organizer checkpoint's monitor predictions, used as the NodeTransformer control;
- an explicit learned-HOCT grid JSON;
- local checkpoint files that exactly match the audited SHA-256 registry.

The learned grid cannot add candidate geometries. Only `allowed_config_ids` from the frozen shortlist are expanded.

## What may vary

The learned calibration grid must explicitly name every quality-affecting HOCT setting:

- audited model name;
- window size;
- solver appearance weight;
- solver disappearance weight;
- division weight;
- node weight;
- delta-t weight;
- edge bias;
- solver timeout;
- tracklet-solver mode;
- whether gap candidates are allowed;
- the monitor-side HOCT promotion margin over the organizer control.

No hidden quality defaults are invented by the grid parser.

## Organizer control

Before HOCT trials are ranked, the organizer NodeTransformer monitor predictions are scored with the pinned official evaluator on the exact same monitor movie set and per-dataset physical scales.

This control matters because a more complicated tracker should not be promoted merely because it is different. The fixed detection set is the same causal substrate; the question is whether the association/global-solver layer improves the exact competition metric.

## Promotion rule

1. Rank HOCT trials by aggregate official `score` on the training-side monitor set.
2. Break equal-score ties by higher `adj_edge_jaccard`.
3. If those are still equal, prefer lower measured runtime.
4. Promote HOCT only if its best score is at least:

```text
organizer_control_score + hoct_promotion_margin
```

Otherwise freeze the organizer control.

The margin is part of the predeclared learned-grid JSON. It must be chosen before any LOEO observation. It acts as a complexity/generalization guard, not a claim about statistical significance.

## Example grid

`configs/hoct_learned_calibration.example.json` is only a schema/example. Its values are not automatically approved as final competition hyperparameters.

The example starts narrowly with two audited public checkpoints, one window size, and the solver settings currently exposed as the `run_hoct_candidate.py` defaults. Expand the grid only from a concrete training-side failure hypothesis, never because the LOEO embryo suggests a better setting.

## Execution

```bash
python scripts/run_hoct_learned_calibration.py \
  --candidate-calibration-work-dir <phase2e work dir> \
  --inventory <data_inventory.json> \
  --competition-root <competition root> \
  --learned-grid <explicit learned grid.json> \
  --checkpoint-dir <local audited HOCT checkpoints> \
  --calibration-id <id> \
  --execute
```

Default mode is a dry run that freezes and prints the full trial plan without spending HOCT compute.

Execution then:

1. verifies Phase-2E scope and frozen candidate IDs;
2. verifies the exact organizer control prediction set;
3. verifies audited checkpoint hashes;
4. builds each candidate graph once per monitor movie;
5. reuses that graph across model/solver trials;
6. runs HOCT inference/ILP;
7. evaluates every trial with the pinned official metric;
8. compares against the organizer control using the predeclared margin;
9. writes `learned_selection.json` with the frozen winner and complete provenance.

## Next gate

Only the single frozen winner from `learned_selection.json` is allowed to touch the opposite-embryo LOEO set for that direction.

The LOEO result may diagnose whether the frozen hypothesis generalized. It may **not** change the candidate radius, neighbour count, HOCT checkpoint, solver weights, window size, detector settings, or promotion margin and then reuse that same holdout score as clean evidence.
