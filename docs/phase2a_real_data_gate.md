# Phase 2A — Real-data acceptance gate

This phase adds a fail-closed preflight between the downloaded Kaggle competition mount and any expensive baseline or tracker experiment.

The gate inventories the actual training/test metadata without loading image voxels and verifies the assumptions the validation system depends on:

- exactly two training embryo IDs
- unique dataset identifiers and no train/visible-test name overlap
- valid image shapes and positive physical scales
- GT GEFF availability with positive node/edge counts and nonnegative division counts
- valid annotated time ranges
- positive `estimated_number_of_nodes` metadata required by the adjusted edge score
- two exact, disjoint LOEO folds that each hold out one embryo and cover every dataset exactly once

Surprising but not invalid metadata (for example, multiple physical scales or a coarse node estimate below the sparse annotated-node count) is surfaced as a warning rather than silently normalized.

## Kaggle preflight

With the competition attached to a Kaggle runtime, run:

```bash
python scripts/phase2a_real_data_gate.py
```

The script writes only ignored/private artifacts and prints the accepted embryo IDs and the clean baseline commands to run next. It does not transmit or commit competition data.

## Why this is a hard gate

The two-embryo LOEO strategy is a competition-specific assumption. If the downloaded files or metadata differ from the audited competition state, silently constructing a different validation split would invalidate downstream model-selection evidence. Phase 2A therefore stops immediately and requires re-audit instead of guessing.
