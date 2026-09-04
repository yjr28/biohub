# Provenance Ledger

Every external artifact that can influence evaluation or model selection must be recorded here **before** it is used to justify a competition decision.

This includes code, checkpoints, pretrained models, Kaggle notebooks, datasets, pseudo-labels, external labels, synthetic data, and derived caches.

## Rules

1. Unknown provenance is not equivalent to clean provenance.
2. A public artifact may be legal to use but still be invalid for clean cross-validation if its training data overlap is unknown.
3. Competition legality/licensing and validation cleanliness are separate fields.
4. Record immutable identifiers whenever possible: Git commit SHA, Kaggle version, model checksum, dataset DOI/version, or file hash.
5. If an artifact changes, create a new record rather than silently editing its identity.
6. Never commit restricted competition data, private checkpoints, credentials, or raw user-specific Kaggle artifacts to this public repository.

## Status vocabulary

- `candidate` — known artifact; not used yet.
- `verified` — source/version/license/training provenance checked sufficiently for its stated use.
- `restricted` — may be legal for final inference but not eligible for clean validation/model selection.
- `rejected` — provenance, license, rules, or technical issue disqualifies it.

## Clean-validation eligibility

Use one of:

- `yes` — evidence supports no relevant validation contamination for the specified fold/use.
- `no` — known contamination/overlap.
- `unknown` — insufficient provenance; treat as `no` for clean model selection until resolved.
- `n/a` — artifact does not learn from data (for example evaluator code).

## Artifact template

| Field | Value |
|---|---|
| Artifact ID | |
| Type | code / checkpoint / model / dataset / notebook / derived cache |
| Status | candidate / verified / restricted / rejected |
| Source | |
| Author/organization | |
| Immutable version | commit SHA / notebook version / checksum / DOI |
| License | |
| Intended use | |
| Training data known? | yes / no / n/a |
| Competition train data seen? | yes / no / unknown / n/a |
| Embryo/sample overlap known? | yes / no / unknown / n/a |
| External data used | |
| Clean-validation eligibility | yes / no / unknown / n/a |
| Competition-rule review | pending / passed / failed / n/a |
| Local checksum | |
| Evidence links | |
| Notes | |

---

## P-0001 — Official organizer competition/evaluator repository

| Field | Value |
|---|---|
| Artifact ID | `P-0001` |
| Type | code |
| Status | candidate (pin verified; code inspection deferred to Phase 0B) |
| Source | https://github.com/royerlab/kaggle-cell-tracking-competition |
| Author/organization | Royer Lab at Biohub SF |
| Immutable version | `075fc5f5a52d11077f9dc2b074644618f26939e2` |
| License | BSD-3-Clause (repository metadata; Phase 0B must verify local LICENSE text) |
| Intended use | authoritative evaluator/reference baseline |
| Training data known? | n/a for evaluator; baseline provenance reviewed separately if weights are imported |
| Competition train data seen? | n/a for evaluator |
| Embryo/sample overlap known? | n/a for evaluator |
| External data used | n/a |
| Clean-validation eligibility | n/a |
| Competition-rule review | n/a |
| Local checksum | not vendored yet |
| Evidence links | pinned `README.md`, `metrics.md`, Git commit |
| Notes | Commit patches weakly-connected-component metric exploit. Before Phase 0B implementation, refresh official main branch and determine whether this remains the current evaluator revision. |

## Candidate artifacts not yet admitted

Do **not** add public Kaggle checkpoints, Trackastra/HOCT/Ultrack models, synthetic datasets, or public notebook-derived caches to experiments until each receives its own provenance record with immutable version and clean-validation assessment.
