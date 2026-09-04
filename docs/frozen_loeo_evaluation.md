# Frozen LOEO evaluation protocol

Phase 2G is the first point where a tracker selected from training-side evidence is allowed to touch the opposite embryo's labels.

The purpose is deliberately narrow:

> **Evaluate exactly one tracker configuration that was frozen before LOEO, then treat the resulting score as evidence rather than as a hyperparameter search signal.**

## Required upstream chain

For one LOEO direction, Phase 2G requires completed artifacts from:

1. a clean organizer baseline run with the opposite embryo excluded from optimization/checkpoint selection;
2. Phase 2E candidate calibration on the nested training-embryo monitor set;
3. Phase 2F learned association/solver selection on that same monitor set.

`learned_selection.json` must explicitly state:

- `loeo_used=false`;
- `loeo_may_retune_or_replace_winner=false`;
- exactly one winner family: `organizer_control` or `hoct`.

The Phase-2E candidate shortlist must also state that LOEO cannot expand the frozen candidate set.

## What is permitted

### Frozen organizer winner

If Phase 2F kept the organizer NodeTransformer control, Phase 2G evaluates the existing organizer holdout predictions directly. No alternate tracker is run.

### Frozen HOCT winner

If Phase 2F promoted HOCT, Phase 2G:

1. reads detections from the already-produced organizer holdout predictions;
2. reconstructs the exact frozen Phase-2E candidate configuration;
3. verifies the exact audited HOCT checkpoint;
4. applies the exact Phase-2F window and solver specification;
5. produces all holdout prediction GEFFs;
6. only after prediction generation is complete, invokes the pinned official evaluator against GT.

This preserves fixed detections and prevents a different detector threshold from entering the LOEO comparison.

## Artifact fingerprints

GEFF artifacts may be directory stores, not regular files. `path_sha256(...)` fingerprints an entire directory tree deterministically by hashing every regular file together with its relative path. The Phase-2G dry-run plan records hashes for all organizer holdout predictions before any score is produced.

For a HOCT winner, the audited checkpoint SHA-256 is also recorded.

## No overwrite

A Phase-2G output directory is immutable by convention. `run_frozen_loeo.py` refuses to use an existing output directory. This prevents a clean evaluation record from being silently replaced after seeing the result.

Dry-run mode creates a plan directory and exits before GT scoring. Delete that dry-run directory and rerun with `--execute` only when the upstream artifacts are final.

## Execution

```bash
python scripts/run_frozen_loeo.py \
  --learned-selection <phase2f/learned_selection.json> \
  --candidate-shortlist <phase2e/candidate_shortlist.json> \
  --monitor-prediction-plan <phase2e/monitor_prediction_plan.json> \
  --inventory <data_inventory.json> \
  --competition-root <competition root> \
  --organizer-holdout-pred-dir <organizer baseline holdout predictions> \
  --checkpoint-dir <audited HOCT checkpoints, if needed> \
  --evaluation-id <unique direction/run id> \
  --execute
```

The command validates that the organizer prediction directory contains exactly the declared opposite-embryo holdout datasets and no training-side monitor movies.

## Interpretation rule

A clean LOEO result answers whether the pre-frozen hypothesis generalized to the opposite embryo. It does **not** authorize changing candidate radius, neighbour count, model checkpoint, window size, solver weights, detector threshold, or promotion margin and then treating the same embryo as fresh validation.

After both directions are evaluated, compare directional score changes and the existing bottleneck/error decomposition. The next engineering branch should be chosen from the largest credible remaining loss category, not by reverse-fitting the holdout result.