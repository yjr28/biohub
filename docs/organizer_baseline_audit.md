# Organizer Baseline Audit

Pinned organizer revision: `075fc5f5a52d11077f9dc2b074644618f26939e2`  
Source: https://github.com/royerlab/kaggle-cell-tracking-competition

This audit distinguishes **the public organizer baseline implementation** from **a clean LOEO adaptation of that implementation**. They are not the same experiment.

## 1. Public organizer baseline facts

The pinned README describes an end-to-end temporal 3D U-Net + cross-attention node transformer baseline with sparse supervision and gives the public-model training command:

```bash
python scripts/train_unet_transformer.py \
  --data-dir data/train --split 0 --epochs 3
```

The README explicitly says that released model was not trained to convergence.

## 2. Critical validation behavior in the trainer

The pinned `train_unet_transformer.py` does **not** treat its `test` list as a passive final holdout. Every epoch it:

1. evaluates the current model on the supplied `test` datasets;
2. computes `score = test_acc * test_recall`;
3. saves `edge_predictor_best.pth` whenever that score is at least the previous best;
4. reloads that best checkpoint before returning.

Therefore:

> Passing the true LOEO embryo as the organizer trainer's `test` split would contaminate checkpoint selection.

That run could be useful as an upper-bound/debug experiment, but it cannot certify clean cross-embryo generalization.

## 3. Holdout-safe adaptation used by this repository

`biohub.baselines.organizer` creates two different split files for the same organizer fold index:

### Training split

```text
train = all datasets from the training embryo(s)
test  = the same training-embryo dataset set
```

This keeps every held-out-embryo dataset out of the organizer trainer's optimization/checkpoint-selection universe while preserving all available training-embryo datasets for backpropagation.

The monitor is intentionally **not** interpreted as independent validation. It exists only because the pinned trainer requires a `test_loader` to choose its saved checkpoint.

### Prediction split

```text
train = training-embryo dataset set
test  = true LOEO holdout dataset set
```

The prediction script uses its `test` list to choose which datasets to infer. Scoring is then performed externally by our strict LOEO evaluator.

## 4. Reproducibility caveat in the pinned trainer

The `train()` function accepts a `seed` argument and seeds the DataLoader generator/worker NumPy state when supplied, but the pinned CLI does not expose/pass that argument.

More importantly, `FrameWindowDataset.__getitem__` creates an augmentation RNG with:

```python
rng = np.random.default_rng()
```

without an explicit seed. That RNG is initialized from fresh entropy and is not made deterministic merely by calling `np.random.seed(...)` in the worker.

Consequence: the pinned augmentation path is not guaranteed bitwise reproducible from a recorded integer seed. Baseline comparisons involving small score differences should therefore be repeated before treating a tiny change as causal.

## 5. Defaults that must be recorded explicitly

Several defaults differ between function signatures, CLI defaults, comments, and prediction dataclass defaults. A manifest must record the values actually passed through the CLI/runtime rather than relying on prose.

At the pinned revision:

- README public training example: **3 epochs**.
- training CLI default learning rate: `1e-4`.
- `train()` function signature learning-rate default: `1e-3`.
- training CLI `--det-loss-weight` default: `1e0`.
- `train()` function signature detection-loss default: `1e1`.
- training CLI `--downsample`: `1,4,4`.
- training CLI `--window-size`: `2`.
- training CLI `--pool-kernel-um`: `5.0`.
- prediction CLI detection threshold: `0.99`.
- prediction `PredictConfig` class detection threshold default: `0.5`, but `main()` overrides it with the CLI value.
- prediction `PredictConfig.pool_kernel_um`: `3.0`.
- prediction edge activation: `softmax`.
- prediction edge threshold: `0.5`.
- greedy max parents defaults to `1`; max children defaults to `2` when ILP is off.
- ILP is off unless `--use-ilp` is supplied.

These are baseline facts, not tuned recommendations.

## 6. Split-generation behavior if no splits file exists

If `--splits` is not supplied and no `dataset_splits.json` exists under the data directory, the pinned trainer:

1. enumerates datasets that have both `.zarr` and `.geff`;
2. sorts stems;
3. shuffles with Python `random.Random(0)`;
4. uses approximately 10% as its validation/test list and 90% as train.

This dataset-level random split can mix embryo identities and is therefore not appropriate evidence for the hidden-test generalization problem once the host has clarified that hidden test embryos are disjoint from train embryos.

## 7. Baseline experiments we will keep separate

### A. Public-protocol reproduction

Purpose: implementation sanity and comparison with public artifacts.

- reproduce the pinned organizer command/protocol as closely as possible;
- do **not** call its validation score clean LOEO;
- public notebook/checkpoint provenance must be tracked separately.

### B. Clean-LOEO organizer adaptation

Purpose: trustworthy reference for architecture/error-budget work.

- train on one embryo direction;
- prevent held-out embryo from influencing checkpoint selection;
- infer the held-out embryo only after training;
- score exact holdout datasets through the pinned official evaluator;
- run the reverse embryo direction independently.

Architecture and core organizer training/inference code stay unchanged unless a later experiment explicitly declares a modification.

## 8. Promotion rule

The organizer baseline is a reference, not a sacred architecture. Once both clean LOEO directions exist, we measure whether loss is dominated by node availability, candidate generation, association, global topology, division handling, or count calibration. The next engineering move is selected from that decomposition rather than from model popularity.
