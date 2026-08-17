#!/usr/bin/env bash
# Disposable smoke matrix for kitAb interface (does not touch production data).
#
# Safety:
#   - Writes only under /tmp (or $KITAB_SMOKE_OUT) and reads tests/fixtures + examples/
#   - Runs `kitab validate` only (no predict / minimize / descriptors / AutoML)
#   - Runs pytest (unit + mocked pipeline tests)
#
# Usage (from repo root):
#   bash tests/run_smoke_matrix.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

KITAB_ENV="${KITAB_ENV:-kitab}"
OUT_ROOT="${KITAB_SMOKE_OUT:-/tmp/kitab_smoke_matrix_$$}"
mkdir -p "$OUT_ROOT"
echo "[smoke] output root: $OUT_ROOT"

run_py() {
  conda run --no-capture-output -n "$KITAB_ENV" python "$@"
}

FIX="$REPO_ROOT/tests/fixtures"
if [[ ! -d "$FIX/csv" ]]; then
  echo "[smoke] ERROR: missing fixtures at $FIX" >&2
  exit 1
fi

# Generate ephemeral manifests pointing only at tests/fixtures
KITAB_SMOKE_REPO="$REPO_ROOT" KITAB_SMOKE_OUT_DIR="$OUT_ROOT" run_py - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["KITAB_SMOKE_REPO"])
fix = root / "tests" / "fixtures"
out = Path(os.environ["KITAB_SMOKE_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)

cases = {
    "S_descriptors_only": {
        "inputs": {"structures_dir": str(fix / "structures")},
        "run": {"output_dir": str(out / "S_descriptors_only")},
        "descriptors": {"enabled": True},
        "automl": {"enabled": False},
        "tuning": {"enabled": False},
    },
    "S_structures_automl": {
        "inputs": {
            "datasets_dir": str(fix / "csv"),
            "structures_dir": str(fix / "structures"),
            "split_randomly": ["ab21_mini"],
        },
        "run": {"output_dir": str(out / "S_structures_automl")},
        "descriptors": {"enabled": True},
        "automl": {"enabled": True},
        "tuning": {"enabled": False},
    },
    "S_automl_only": {
        "inputs": {
            "datasets_dir": str(fix / "csv"),
            "predefined_descriptors_dir": str(fix / "descriptors"),
            "split_randomly": ["ab21_mini"],
        },
        "run": {"output_dir": str(out / "S_automl_only")},
        "automl": {"enabled": True},
        "tuning": {"enabled": False},
    },
    "S_full_tuning": {
        "inputs": {
            "datasets_dir": str(fix / "csv"),
            "structures_dir": str(fix / "structures"),
            "split_randomly": ["ab21_mini"],
        },
        "run": {"output_dir": str(out / "S_full_tuning")},
        "descriptors": {"enabled": True},
        "automl": {"enabled": True},
        "tuning": {"enabled": True},
    },
    "S_predict": {
        "inputs": {
            "datasets_dir": str(fix / "csv"),
            "split_randomly": ["ab21_mini"],
        },
        "run": {"output_dir": str(out / "S_predict")},
        "structure_prediction": {"enabled": True, "model": "abb2", "device": "cpu"},
        "descriptors": {"enabled": True},
        "automl": {"enabled": True},
        "tuning": {"enabled": False},
    },
}

for name, payload in cases.items():
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(path)
PY

for cfg in "$OUT_ROOT"/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  echo "[smoke] VALIDATE $name"
  run_py -m kitab validate "$cfg" \
    | tee "$OUT_ROOT/${name}.validate.txt"
done

echo "[smoke] validating example manifests"
: > "$OUT_ROOT/examples_validate.txt"
for cfg in "$REPO_ROOT"/examples/configs/*.yaml; do
  echo "[smoke] validate $(basename "$cfg")"
  {
    echo "=== $(basename "$cfg") ==="
    run_py -m kitab validate "$cfg"
    echo
  } | tee -a "$OUT_ROOT/examples_validate.txt"
done

echo "[smoke] unit/integration tests"
run_py -m pytest tests/ -q --tb=line | tee "$OUT_ROOT/pytest.txt"

echo "[smoke] DONE — artifacts under $OUT_ROOT"
printf '%s\n' "$OUT_ROOT" > "$OUT_ROOT/OUT_ROOT.txt"
