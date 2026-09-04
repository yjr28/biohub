#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git submodule update --init --recursive

python -m pip install --upgrade pip
python -m pip install -e vendor/tracksdata
python -m pip install --no-deps -e vendor/kaggle-cell-tracking-competition
python -m pip install -e '.[dev]'

python - <<'PY'
from biohub.evaluation.official import (
    OFFICIAL_EVALUATOR_COMMIT,
    TRACKSDATA_COMMIT,
    assert_official_constants,
)

assert_official_constants()
print(f"official evaluator: {OFFICIAL_EVALUATOR_COMMIT}")
print(f"tracksdata pin:      {TRACKSDATA_COMMIT}")
PY
